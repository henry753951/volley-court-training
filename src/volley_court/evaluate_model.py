from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .api import CourtLineModel, InferenceConfig
from .dataset import ImageTransform, canonicalize_pose36_points, points_to_layout_target
from .evaluate_layout import (
    _empty_accumulator,
    _label_path,
    _read_label,
    _split_directories,
    _summarize_keypoints,
    _update_keypoint_metrics,
)
from .types import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an end-to-end court layout checkpoint")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder", choices=("auto", "spatial", "cuda"), default="auto")
    parser.add_argument("--anchor-confidence", type=float, default=0.0)
    parser.add_argument("--anchor-ransac-threshold", type=float, default=0.01)
    parser.add_argument("--anchor-ransac-max-iters", type=int, default=128)
    parser.add_argument(
        "--anchor-solver",
        choices=(
            "hybrid",
            "ransac",
            "rho",
            "usac_default",
            "usac_accurate",
            "usac_magsac",
            "usac_prosac",
        ),
        default="hybrid",
    )
    parser.add_argument("--minimum-anchor-inlier-ratio", type=float, default=0.4)
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def evaluate_checkpoint(
    dataset_yaml: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    device: str = "auto",
    image_size: int = 512,
    half: bool = True,
    decoder: str = "auto",
    anchor_confidence: float = 0.0,
    anchor_ransac_threshold: float = 0.01,
    anchor_ransac_max_iters: int = 128,
    anchor_solver: str = "hybrid",
    minimum_anchor_inlier_ratio: float = 0.4,
    fuse: bool = True,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    image_directories = _split_directories(dataset_yaml, split)
    images = sorted(
        path for directory in image_directories for path in directory.iterdir() if path.is_file()
    )
    output.mkdir(parents=True, exist_ok=True)
    model = CourtLineModel(
        checkpoint,
        config=InferenceConfig(
            device=device,
            image_size=image_size,
            half=half,
            fuse=fuse,
            decoder=decoder,
            include_layout=True,
            anchor_confidence=anchor_confidence,
            anchor_ransac_threshold=anchor_ransac_threshold,
            anchor_ransac_max_iters=anchor_ransac_max_iters,
            anchor_solver=anchor_solver,
            minimum_anchor_inlier_ratio=minimum_anchor_inlier_ratio,
        ),
    )
    model.warmup(batch_size=min(batch_size, 4))
    accumulator = _empty_accumulator()
    raw_accumulator = _empty_accumulator()
    candidate_accumulator = _empty_accumulator()
    statuses: Counter[str] = Counter()
    ground_truth_symmetries: Counter[str] = Counter()
    per_image = []
    severe_false_accepts = []
    false_accepts = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        paths = images[start : start + batch_size]
        frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
        if any(frame is None for frame in frames):
            failed = paths[next(index for index, frame in enumerate(frames) if frame is None)]
            raise RuntimeError(f"OpenCV could not decode: {failed}")
        typed_frames = [cast(Image, frame) for frame in frames]
        results = model.predict_many(typed_frames, include_layout=True)
        for image_path, frame, result in zip(paths, typed_frames, results, strict=True):
            gt, box = _read_label(_label_path(image_path))
            raw_gt = gt
            identity = np.eye(3, dtype=np.float64)
            gt_layout = points_to_layout_target(
                gt,
                ImageTransform(identity, identity, max(result.width, result.height)),
                result.width,
                result.height,
            )
            symmetry_index = int(gt_layout["layout_symmetry"])
            ground_truth_symmetries[str(symmetry_index)] += 1
            gt = canonicalize_pose36_points(gt, symmetry_index)
            layout = result.layout
            status = layout.status if layout is not None else "unknown"
            statuses[status] += 1
            predicted = [point.to_mapping() for point in layout.keypoints] if layout else []
            candidates = (
                [point.to_mapping() for point in layout.candidate_keypoints] if layout else []
            )
            metrics = _update_keypoint_metrics(
                accumulator,
                gt,
                predicted,
                result.width,
                result.height,
                box,
                2,
                0.0,
            )
            raw_metrics = _update_keypoint_metrics(
                raw_accumulator,
                raw_gt,
                predicted,
                result.width,
                result.height,
                box,
                2,
                0.0,
            )
            candidate_metrics = _update_keypoint_metrics(
                candidate_accumulator,
                gt,
                candidates,
                result.width,
                result.height,
                box,
                2,
                0.0,
            )
            pck_002 = float(metrics["pck@0.02"])
            if status == "ok" and pck_002 < 0.5:
                false_accepts.append(image_path.name)
            if status == "ok" and pck_002 < 0.25:
                severe_false_accepts.append(image_path.name)
            row = {
                "image": str(image_path),
                "status": status,
                "reason": layout.reason if layout else "missing layout result",
                "layout_score": layout.score if layout else 0.0,
                "ground_truth_symmetry": symmetry_index,
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **metrics,
            }
            per_image.append(row)
            overlay = model.visualize(frame, result)
            overlay_path = output / "overlays" / f"{image_path.stem}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise RuntimeError(f"could not write overlay: {overlay_path}")
            result_path = output / "results" / f"{image_path.stem}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(result.to_mapping(), indent=2) + "\n", encoding="utf-8"
            )
    wall_seconds = time.perf_counter() - started
    summary = {
        "schema": "volley-court-direct-layout-evaluation-v1",
        "dataset": str(dataset_yaml.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "split": split,
        "images": len(images),
        "layout_status": dict(statuses),
        "ground_truth_symmetry": dict(ground_truth_symmetries),
        "visible_v2": _summarize_keypoints(accumulator),
        "visible_v2_raw_labels": _summarize_keypoints(raw_accumulator),
        "visible_v2_all_geometric_candidates": _summarize_keypoints(candidate_accumulator),
        "false_accept_pck_below_0.5_at_0.02": false_accepts,
        "severe_false_accept_pck_below_0.25_at_0.02": severe_false_accepts,
        "wall_seconds": wall_seconds,
        "end_to_end_fps": len(images) / max(wall_seconds, 1e-9),
    }
    (output / "per-image.json").write_text(json.dumps(per_image, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = evaluate_checkpoint(
        args.dataset,
        args.checkpoint,
        args.output,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.imgsz,
        half=args.half,
        decoder=args.decoder,
        anchor_confidence=args.anchor_confidence,
        anchor_ransac_threshold=args.anchor_ransac_threshold,
        anchor_ransac_max_iters=args.anchor_ransac_max_iters,
        anchor_solver=args.anchor_solver,
        minimum_anchor_inlier_ratio=args.minimum_anchor_inlier_ratio,
        fuse=args.fuse,
    )
    print(json.dumps(summary, indent=2))
    return 0
