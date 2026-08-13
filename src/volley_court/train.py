from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from .dataset import CourtLineDataset
from .inference import resolve_device
from .loss import CourtLineLoss
from .model import (
    DEFAULT_POSE_ARCHITECTURE,
    DIRECT_LAYOUT_HEAD_VERSION,
    YOLO26CourtLine,
    build_from_pose_checkpoint,
    load_court_line_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dense zero-width court-line segments with optional semantic heads"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Converted zero-width dataset root"
    )
    parser.add_argument(
        "--image-root", type=Path, default=None, help="Override original image root"
    )
    parser.add_argument(
        "--pose-checkpoint", type=Path, help="Pose36 checkpoint used for transfer/rebuild"
    )
    parser.add_argument(
        "--weights", type=Path, help="Existing court-line checkpoint for fine-tuning"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-feature-epochs", type=int, default=2)
    parser.add_argument("--freeze-shared-epochs", type=int, default=0)
    parser.add_argument("--freeze-dense-epochs", type=int, default=0)
    parser.add_argument("--freeze-layout-geometry-epochs", type=int, default=0)
    parser.add_argument("--base-head-gradient-scale", type=float, default=1.0)
    parser.add_argument(
        "--target-mode",
        choices=("center", "dense_votes", "dense_context", "dense_semantic"),
        default=None,
    )
    parser.add_argument("--sample-spacing", type=float, default=16.0)
    parser.add_argument("--intersection-exclusion", type=float, default=4.0)
    parser.add_argument("--hard-negative-probability", type=float, default=0.15)
    parser.add_argument("--family-weight", type=float, default=0.5)
    parser.add_argument("--identity-weight", type=float, default=0.75)
    parser.add_argument("--roi-weight", type=float, default=0.5)
    parser.add_argument(
        "--layout-proposals",
        type=int,
        default=None,
        help="Direct ordered-corner proposals; zero disables the layout head",
    )
    parser.add_argument("--layout-coordinate-weight", type=float, default=5.0)
    parser.add_argument("--layout-validity-weight", type=float, default=1.0)
    parser.add_argument("--layout-softargmax-weight", type=float, default=0.0)
    parser.add_argument("--layout-visibility-positive-weight", type=float, default=1.0)
    parser.add_argument("--layout-validity-positive-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also retain an epoch checkpoint at this interval; zero disables snapshots",
    )
    parser.add_argument("--limit-train", type=int, default=0, help="Debug-only sample limit")
    parser.add_argument("--limit-valid", type=int, default=0, help="Debug-only sample limit")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in target.items()}


def _run_epoch(
    model: YOLO26CourtLine,
    loader: DataLoader,
    criterion: CourtLineLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    feature_frozen: bool,
    shared_frozen: bool,
    dense_frozen: bool,
    layout_geometry_frozen: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and feature_frozen:
        model.feature_layers.eval()
    if training and shared_frozen:
        model.set_shared_features_eval()
    if training and dense_frozen:
        model.set_dense_head_eval()
    if training and layout_geometry_frozen:
        model.set_layout_geometry_eval()
    totals = {
        key: 0.0
        for key in (
            "total",
            "center",
            "offset",
            "orientation",
            "half_length",
            "family",
            "identity",
            "roi",
            "layout_coordinate",
            "layout_validity",
            "layout_softargmax",
        )
    }
    examples = 0
    collisions = 0
    votes = 0
    hard_negatives = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = _to_device(target, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ),
        ):
            prediction, layout_prediction = model.forward_outputs(images)
            loss, parts = criterion(prediction, target, layout_prediction)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {float(loss)}")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        batch_size = images.shape[0]
        examples += batch_size
        collisions += int(target["collision_count"].sum())
        votes += int(target["vote_count"].sum())
        hard_negatives += int(target["hard_negative"].sum())
        for key in totals:
            totals[key] += float(parts[key]) * batch_size
    return {
        **{key: value / max(1, examples) for key, value in totals.items()},
        "target_collisions": float(collisions),
        "target_votes": float(votes),
        "hard_negative_samples": float(hard_negatives),
    }


def _checkpoint_payload(
    model: YOLO26CourtLine,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    args: argparse.Namespace,
    pose_checkpoint: str,
    epoch: int,
    best_validation: float,
    transfer_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "yolo26n-court-line-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation,
        "pose_checkpoint": pose_checkpoint,
        "pose_architecture": DEFAULT_POSE_ARCHITECTURE,
        "fusion_channels": 64,
        "output_channels": int(model.output_channels),
        "layout_proposals": int(model.layout_proposals),
        "layout_head_version": DIRECT_LAYOUT_HEAD_VERSION if model.layout_proposals else 0,
        "stride": 4,
        "target_mode": args.target_mode,
        "sample_spacing": args.sample_spacing,
        "intersection_exclusion": args.intersection_exclusion,
        "representation": (
            "dense votes plus two families, seven semantic line identities, and court ROI"
            if args.target_mode == "dense_semantic"
            else "dense identity-free votes plus two line-family logits and court ROI"
            if args.target_mode == "dense_context"
            else "dense identity-free (x,y,cos2theta,sin2theta) votes"
            if args.target_mode == "dense_votes"
            else "(cx,cy,cos2theta,sin2theta,normalized_half_length)"
        ),
        "prediction_class": "court_line",
        "transfer_report": transfer_report,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def main() -> int:
    args = parse_args()
    if not args.pose_checkpoint and not args.weights:
        raise ValueError("pass --pose-checkpoint for initial transfer or --weights for fine-tuning")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.weights:
        checkpoint_payload = torch.load(
            args.weights.resolve(), map_location="cpu", weights_only=False
        )
        checkpoint_mode = str(checkpoint_payload.get("target_mode", "center"))
        checkpoint_layout_proposals = int(checkpoint_payload.get("layout_proposals", 0))
        requested_layout_proposals = (
            checkpoint_layout_proposals
            if args.layout_proposals is None
            else int(args.layout_proposals)
        )
        requested_mode = args.target_mode or checkpoint_mode
        upgrade_context = checkpoint_mode == "dense_votes" and requested_mode in {
            "dense_context",
            "dense_semantic",
        }
        if requested_mode != checkpoint_mode and not upgrade_context:
            raise ValueError(
                f"--target-mode {requested_mode} cannot upgrade checkpoint mode {checkpoint_mode}"
            )
        model, metadata = load_court_line_checkpoint(
            args.weights.resolve(),
            device=device,
            pose_checkpoint=args.pose_checkpoint.resolve() if args.pose_checkpoint else None,
            output_channels_override=(
                15 if requested_mode == "dense_semantic" else 8 if upgrade_context else None
            ),
            layout_proposals_override=(
                requested_layout_proposals
                if requested_layout_proposals != checkpoint_layout_proposals
                else None
            ),
        )
        pose_checkpoint = str(
            args.pose_checkpoint.resolve()
            if args.pose_checkpoint
            else metadata.get("resolved_pose_source", metadata["pose_checkpoint"])
        )
        args.target_mode = requested_mode
        args.layout_proposals = requested_layout_proposals
        args.sample_spacing = float(metadata.get("sample_spacing", args.sample_spacing))
        args.intersection_exclusion = float(
            metadata.get("intersection_exclusion", args.intersection_exclusion)
        )
        transfer_report = (
            metadata["load_transfer_report"]
            if upgrade_context
            else metadata.get("transfer_report", metadata["load_transfer_report"])
        )
    else:
        args.target_mode = args.target_mode or "dense_votes"
        args.layout_proposals = int(args.layout_proposals or 0)
        model, transfer_report = build_from_pose_checkpoint(
            args.pose_checkpoint.resolve(),
            output_channels={
                "center": 6,
                "dense_votes": 5,
                "dense_context": 8,
                "dense_semantic": 15,
            }[args.target_mode],
            layout_proposals=args.layout_proposals,
        )
        model.to(device)
        pose_checkpoint = str(args.pose_checkpoint.resolve())
    (output / "transfer-report.json").write_text(
        json.dumps(transfer_report, indent=2) + "\n",
        encoding="utf-8",
    )
    train_set = CourtLineDataset(
        args.data,
        "train",
        args.imgsz,
        args.image_root,
        augment=True,
        seed=args.seed,
        limit=args.limit_train,
        target_mode=args.target_mode,
        sample_spacing=args.sample_spacing,
        intersection_exclusion=args.intersection_exclusion,
        hard_negative_probability=(
            args.hard_negative_probability
            if args.target_mode in {"dense_context", "dense_semantic"}
            else 0.0
        ),
    )
    validation_set = CourtLineDataset(
        args.data,
        "valid",
        args.imgsz,
        args.image_root,
        augment=False,
        seed=args.seed,
        limit=args.limit_valid,
        target_mode=args.target_mode,
        sample_spacing=args.sample_spacing,
        intersection_exclusion=args.intersection_exclusion,
        hard_negative_probability=0.0,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        # Workers restart each epoch so the deterministic epoch-specific augmentation seed propagates.
        persistent_workers=False,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    criterion = CourtLineLoss(
        target_mode=args.target_mode,
        length_weight=(
            0.0 if args.target_mode in {"dense_votes", "dense_context", "dense_semantic"} else 2.0
        ),
        family_weight=args.family_weight,
        identity_weight=args.identity_weight,
        roi_weight=args.roi_weight,
        layout_coordinate_weight=args.layout_coordinate_weight,
        layout_validity_weight=args.layout_validity_weight,
        layout_softargmax_weight=args.layout_softargmax_weight,
        layout_visibility_positive_weight=args.layout_visibility_positive_weight,
        layout_validity_positive_weight=args.layout_validity_positive_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    best_validation = math.inf
    history_path = output / "history.csv"
    fieldnames = [
        "epoch",
        "seconds",
        "lr",
        "feature_frozen",
        "shared_frozen",
        "dense_frozen",
        "layout_geometry_frozen",
        *[
            f"train_{key}"
            for key in (
                "total",
                "center",
                "offset",
                "orientation",
                "half_length",
                "family",
                "identity",
                "roi",
                "layout_coordinate",
                "layout_validity",
                "layout_softargmax",
                "target_collisions",
                "target_votes",
                "hard_negative_samples",
            )
        ],
        *[
            f"valid_{key}"
            for key in (
                "total",
                "center",
                "offset",
                "orientation",
                "half_length",
                "family",
                "identity",
                "roi",
                "layout_coordinate",
                "layout_validity",
                "layout_softargmax",
                "target_collisions",
                "target_votes",
                "hard_negative_samples",
            )
        ],
    ]
    with history_path.open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(args.epochs):
            started = time.perf_counter()
            train_set.set_epoch(epoch)
            feature_frozen = epoch < args.freeze_feature_epochs
            shared_frozen = epoch < args.freeze_shared_epochs
            dense_frozen = epoch < args.freeze_dense_epochs
            layout_geometry_frozen = epoch < args.freeze_layout_geometry_epochs
            model.freeze_shared_features(shared_frozen)
            model.freeze_dense_head(dense_frozen)
            model.freeze_layout_geometry(layout_geometry_frozen)
            if not shared_frozen:
                model.freeze_feature_extractor(feature_frozen)
            model.set_base_head_gradient_scale(args.base_head_gradient_scale)
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                amp_enabled,
                amp_dtype,
                feature_frozen,
                shared_frozen,
                dense_frozen,
                layout_geometry_frozen,
            )
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                None,
                scaler,
                amp_enabled,
                amp_dtype,
                False,
                False,
                False,
                False,
            )
            scheduler.step()
            row = {
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - started,
                "lr": optimizer.param_groups[0]["lr"],
                "feature_frozen": feature_frozen,
                "shared_frozen": shared_frozen,
                "dense_frozen": dense_frozen,
                "layout_geometry_frozen": layout_geometry_frozen,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"valid_{key}": value for key, value in validation_metrics.items()},
            }
            writer.writerow(row)
            history.flush()
            improved = validation_metrics["total"] < best_validation
            best_validation = min(best_validation, validation_metrics["total"])
            payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                pose_checkpoint,
                epoch + 1,
                best_validation,
                transfer_report,
            )
            torch.save(payload, output / "last.pt")
            if improved:
                torch.save(payload, output / "best.pt")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                torch.save(payload, output / f"epoch-{epoch + 1:04d}.pt")
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
