from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFile

from .api import CourtLineModel, InferenceConfig
from .dataset import (
    ImageTransform,
    canonicalize_pose36_points,
    points_to_layout_target,
)
from .evaluate_layout import (
    _empty_accumulator,
    _label_path,
    _read_label,
    _split_directories,
    _summarize_keypoints,
    _update_keypoint_metrics,
)
from .layout import CANONICAL_KEYPOINTS
from .types import Image

_EVALUATION_GRID = np.asarray(
    [(x, y) for y in np.linspace(0.0, 18.0, 19) for x in np.linspace(0.0, 9.0, 10)],
    dtype=np.float32,
)
_OUTER_COURT = np.asarray(
    ((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)),
    dtype=np.float32,
)


def _fit_ground_truth_homography(
    points: list[Any], width: int, height: int
) -> np.ndarray | None:
    canonical = []
    image = []
    for point in points:
        if point.visibility not in {1, 2} or not 0 <= point.index < len(CANONICAL_KEYPOINTS):
            continue
        canonical.append(CANONICAL_KEYPOINTS[point.index])
        image.append((point.x * width, point.y * height))
    if len(canonical) < 4:
        return None
    source = np.asarray(canonical, dtype=np.float32)
    destination = np.asarray(image, dtype=np.float32)
    if abs(float(cv2.contourArea(cv2.convexHull(source)))) < 1.0:
        return None
    homography, _mask = cv2.findHomography(source, destination, method=0)
    return homography if homography is not None and np.isfinite(homography).all() else None


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return projected if np.isfinite(projected).all() else None


def _homography_metrics(
    ground_truth: np.ndarray | None,
    predicted: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, float | None]:
    empty = {
        "grid_reprojection_mean": None,
        "grid_reprojection_p95": None,
        "whole_court_iou": None,
        "visible_court_iou": None,
    }
    if ground_truth is None or predicted is None or not np.isfinite(predicted).all():
        return empty
    gt_grid = _project(ground_truth, _EVALUATION_GRID)
    predicted_grid = _project(predicted, _EVALUATION_GRID)
    gt_outer = _project(ground_truth, _OUTER_COURT)
    predicted_outer = _project(predicted, _OUTER_COURT)
    if gt_grid is None or predicted_grid is None or gt_outer is None or predicted_outer is None:
        return empty
    diagonal = max(float(np.hypot(width, height)), 1.0)
    errors = np.linalg.norm(gt_grid - predicted_grid, axis=1) / diagonal
    gt_area = abs(float(cv2.contourArea(gt_outer)))
    predicted_area = abs(float(cv2.contourArea(predicted_outer)))
    intersection, _polygon = cv2.intersectConvexConvex(
        gt_outer.astype(np.float32), predicted_outer.astype(np.float32)
    )
    union = gt_area + predicted_area - float(intersection)
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    predicted_mask = np.zeros_like(gt_mask)
    cv2.fillConvexPoly(gt_mask, np.rint(gt_outer).astype(np.int32), 1)
    cv2.fillConvexPoly(predicted_mask, np.rint(predicted_outer).astype(np.int32), 1)
    visible_intersection = int(np.logical_and(gt_mask, predicted_mask).sum())
    visible_union = int(np.logical_or(gt_mask, predicted_mask).sum())
    return {
        "grid_reprojection_mean": float(np.mean(errors)),
        "grid_reprojection_p95": float(np.percentile(errors, 95)),
        "whole_court_iou": float(intersection / union) if union > 1e-6 else 0.0,
        "visible_court_iou": (
            visible_intersection / visible_union if visible_union > 0 else 0.0
        ),
    }


def _summarize_homographies(rows: list[dict[str, float | None]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"evaluated": len(rows)}
    for key in (
        "grid_reprojection_mean",
        "grid_reprojection_p95",
        "whole_court_iou",
        "visible_court_iou",
    ):
        values = [float(row[key]) for row in rows if row[key] is not None]
        summary[key] = {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p05": float(np.percentile(values, 5)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    return summary


def _read_image(path: Path) -> Image | None:
    # imdecode is more reliable than imread for long/network-backed Windows
    # paths and still preserves OpenCV's BGR contract.
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is not None:
        return cast(Image, decoded)
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with PILImage.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return cast(Image, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except OSError:
        return None


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
    homography_rows: list[dict[str, float | None]] = []
    accepted_homography_rows: list[dict[str, float | None]] = []
    catastrophic_layout_accepts = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        paths = images[start : start + batch_size]
        frames = [_read_image(path) for path in paths]
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
            gt_homography = _fit_ground_truth_homography(gt, result.width, result.height)
            predicted_homography = (
                np.asarray(layout.homography, dtype=np.float64)
                if layout is not None and layout.homography is not None
                else None
            )
            homography_metrics = _homography_metrics(
                gt_homography,
                predicted_homography,
                result.width,
                result.height,
            )
            if homography_metrics["grid_reprojection_mean"] is not None:
                homography_rows.append(homography_metrics)
                if status == "ok":
                    accepted_homography_rows.append(homography_metrics)
                    if float(homography_metrics["grid_reprojection_mean"]) > 0.05:
                        catastrophic_layout_accepts.append(image_path.name)
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
                **homography_metrics,
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
        "schema": "volley-court-direct-layout-evaluation-v2",
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
        "homography_all_candidates": _summarize_homographies(homography_rows),
        "homography_accepted": _summarize_homographies(accepted_homography_rows),
        "catastrophic_layout_accepts_reprojection_above_0.05": catastrophic_layout_accepts,
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
