from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFile

from .geometry import PointSample, fit_line_group, parse_yolo_pose_line
from .layout import draw_layout_overlay, match_semantic_court_layout
from .topology import load_topology

THRESHOLDS = (0.005, 0.01, 0.02)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))[:, :, ::-1].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recovered Pose36 court identities")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--topology", type=Path, default=Path("configs/court_line_topology.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--minimum-keypoint-score", type=float, default=0.0)
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def _dataset_root(dataset_yaml: Path, payload: dict[str, Any]) -> Path:
    configured = Path(str(payload.get("path", ".")))
    return configured if configured.is_absolute() else (dataset_yaml.parent / configured).resolve()


def _split_directories(dataset_yaml: Path, split: str) -> list[Path]:
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = _dataset_root(dataset_yaml, payload)
    configured = payload.get(split)
    if configured is None:
        raise KeyError(f"dataset has no {split!r} split")
    values = configured if isinstance(configured, list) else [configured]
    return [Path(value) if Path(value).is_absolute() else root / value for value in values]


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    position = len(parts) - 1 - parts[::-1].index("images")
    parts[position] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_label(path: Path) -> tuple[list[PointSample], tuple[float, float, float, float]]:
    rows = [row for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one labelled court: {path}")
    values = rows[0].split()
    if len(values) < 5:
        raise ValueError(f"invalid YOLO pose label: {path}")
    box = tuple(float(value) for value in values[1:5])
    return parse_yolo_pose_line(rows[0]), (box[0], box[1], box[2], box[3])


def _segment_equation(segment: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, segment)
    dx, dy = x2 - x1, y2 - y1
    length = max(math.hypot(dx, dy), 1e-9)
    a, b = -dy / length, dx / length
    return a, b, -(a * x1 + b * y1), math.atan2(dy, dx)


def _segment_pair_cost(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float]:
    a1, b1, c1, theta1 = _segment_equation(first)
    a2, b2, c2, theta2 = _segment_equation(second)
    angle = abs(theta1 - theta2) % math.pi
    angle = min(angle, math.pi - angle)
    first_midpoint = (0.5 * (first[0] + first[2]), 0.5 * (first[1] + first[3]))
    second_midpoint = (0.5 * (second[0] + second[2]), 0.5 * (second[1] + second[3]))
    distance = 0.5 * (
        abs(a1 * second_midpoint[0] + b1 * second_midpoint[1] + c1)
        + abs(a2 * first_midpoint[0] + b2 * first_midpoint[1] + c2)
    )
    return math.degrees(angle) / 15.0 + distance / 20.0, math.degrees(angle), distance


def _family_metrics(
    classified_segments: Sequence[dict[str, Any]],
    points: Sequence[PointSample],
    topology_path: Path,
    width: int,
    height: int,
) -> dict[str, int]:
    topology = load_topology(topology_path)
    gt = []
    for topology_index, group in enumerate(topology.line_groups):
        fitted = fit_line_group(points, group, width, height, topology.usable_visibility)
        if fitted.valid and fitted.segment is not None:
            gt.append(
                {
                    "segment": fitted.segment,
                    "family": "vertical" if topology_index in {0, 2} else "horizontal",
                    "topology_index": topology_index,
                }
            )
    proposals = []
    for prediction_index, prediction in enumerate(classified_segments):
        segment = prediction.get("segment")
        if not isinstance(segment, Sequence) or len(segment) != 4:
            continue
        for gt_index, target in enumerate(gt):
            cost, angle, distance = _segment_pair_cost(segment, target["segment"])
            if angle <= 20.0 and distance <= 28.0:
                proposals.append((cost, prediction_index, gt_index))
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    correct = 0
    matched = 0
    for _cost, prediction_index, gt_index in sorted(proposals):
        if prediction_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(prediction_index)
        used_gt.add(gt_index)
        matched += 1
        if classified_segments[prediction_index].get("family") == gt[gt_index]["family"]:
            correct += 1
    return {
        "gt_lines": len(gt),
        "predicted_lines": len(classified_segments),
        "matched_lines": matched,
        "correct_family": correct,
    }


def _empty_accumulator() -> dict[str, Any]:
    return {
        "gt": 0,
        "emitted": 0,
        "errors": [],
        "pixel_errors": [],
        "tp": {threshold: 0 for threshold in THRESHOLDS},
        "fp": {threshold: 0 for threshold in THRESHOLDS},
        "id_correct": {threshold: 0 for threshold in THRESHOLDS},
    }


def _update_keypoint_metrics(
    accumulator: dict[str, Any],
    gt: Sequence[PointSample],
    predictions: Sequence[dict[str, Any]],
    width: int,
    height: int,
    box: Sequence[float],
    minimum_visibility: int,
    minimum_score: float,
) -> dict[str, Any]:
    valid = [point for point in gt if point.visibility >= minimum_visibility]
    by_id = {
        int(row["id"]): row
        for row in predictions
        if row.get("in_frame") and float(row.get("score", 0.0)) >= minimum_score
    }
    court_diagonal = max(math.hypot(float(box[2]) * width, float(box[3]) * height), 1.0)
    accumulator["gt"] += len(valid)
    accumulator["emitted"] += len(by_id)
    per_image_tp = {threshold: 0 for threshold in THRESHOLDS}
    normalized_by_id: dict[int, float] = {}
    for point in valid:
        prediction = by_id.get(point.index)
        if prediction is None:
            continue
        error_px = math.hypot(
            float(prediction["x"]) - point.x * width,
            float(prediction["y"]) - point.y * height,
        )
        normalized = error_px / court_diagonal
        normalized_by_id[point.index] = normalized
        accumulator["errors"].append(normalized)
        accumulator["pixel_errors"].append(error_px)
        for threshold in THRESHOLDS:
            if normalized <= threshold:
                accumulator["tp"][threshold] += 1
                per_image_tp[threshold] += 1
    valid_ids = {point.index for point in valid}
    for threshold in THRESHOLDS:
        accumulator["fp"][threshold] += sum(
            1
            for point_id in by_id
            if point_id not in valid_ids or normalized_by_id.get(point_id, math.inf) > threshold
        )
    emitted_rows = list(by_id.values())
    if emitted_rows:
        coordinates = np.asarray([[row["x"], row["y"]] for row in emitted_rows], dtype=np.float64)
        for point in valid:
            distances = np.linalg.norm(coordinates - (point.x * width, point.y * height), axis=1)
            nearest = int(np.argmin(distances))
            normalized = float(distances[nearest]) / court_diagonal
            for threshold in THRESHOLDS:
                if normalized <= threshold and int(emitted_rows[nearest]["id"]) == point.index:
                    accumulator["id_correct"][threshold] += 1
    return {
        f"pck@{threshold:g}": per_image_tp[threshold] / max(len(valid), 1)
        for threshold in THRESHOLDS
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _summarize_keypoints(accumulator: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "gt_keypoints": accumulator["gt"],
        "emitted_keypoints": accumulator["emitted"],
        "emission_rate": accumulator["emitted"] / max(accumulator["gt"], 1),
        "normalized_error": {
            "mean": float(np.mean(accumulator["errors"])) if accumulator["errors"] else None,
            "median": _percentile(accumulator["errors"], 50),
            "p95": _percentile(accumulator["errors"], 95),
        },
        "pixel_error": {
            "mean": float(np.mean(accumulator["pixel_errors"]))
            if accumulator["pixel_errors"]
            else None,
            "median": _percentile(accumulator["pixel_errors"], 50),
            "p95": _percentile(accumulator["pixel_errors"], 95),
        },
    }
    for threshold in THRESHOLDS:
        tp = accumulator["tp"][threshold]
        fp = accumulator["fp"][threshold]
        fn = accumulator["gt"] - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        output[f"pck@{threshold:g}"] = recall
        output[f"precision@{threshold:g}"] = precision
        output[f"recall@{threshold:g}"] = recall
        output[f"f1@{threshold:g}"] = 2.0 * precision * recall / max(precision + recall, 1e-12)
        output[f"id_accuracy@{threshold:g}"] = accumulator["id_correct"][threshold] / max(
            accumulator["gt"], 1
        )
    return output


def evaluate(
    dataset_yaml: Path,
    predictions: Path,
    topology_path: Path,
    output: Path,
    *,
    split: str = "test",
    minimum_keypoint_score: float = 0.0,
    write_overlays: bool = True,
) -> dict[str, Any]:
    image_directories = _split_directories(dataset_yaml, split)
    images = sorted(
        path for directory in image_directories for path in directory.iterdir() if path.is_file()
    )
    output.mkdir(parents=True, exist_ok=True)
    accumulators = {"visible": _empty_accumulator(), "annotated": _empty_accumulator()}
    statuses: Counter[str] = Counter()
    family = Counter()
    per_image = []
    missing_predictions = []
    for image_path in images:
        prediction_path = predictions / f"{image_path.stem}.json"
        if not prediction_path.is_file():
            missing_predictions.append(image_path.name)
            prediction_segments: list[dict[str, Any]] = []
        else:
            prediction_segments = json.loads(prediction_path.read_text(encoding="utf-8")).get(
                "segments", []
            )
        image = _read_image(image_path)
        height, width = image.shape[:2]
        gt, box = _read_label(_label_path(image_path))
        layout = match_semantic_court_layout(prediction_segments, width, height)
        statuses[layout["status"]] += 1
        predicted_keypoints = layout.get("keypoints", [])
        visible = _update_keypoint_metrics(
            accumulators["visible"],
            gt,
            predicted_keypoints,
            width,
            height,
            box,
            2,
            minimum_keypoint_score,
        )
        annotated = _update_keypoint_metrics(
            accumulators["annotated"],
            gt,
            predicted_keypoints,
            width,
            height,
            box,
            1,
            minimum_keypoint_score,
        )
        family_row = _family_metrics(layout.get("segments", []), gt, topology_path, width, height)
        family.update(family_row)
        per_image.append(
            {
                "image": str(image_path),
                "prediction": str(prediction_path),
                "status": layout["status"],
                "reason": layout.get("reason"),
                "layout_score": layout.get("layout_score", 0.0),
                "hypothesis_margin": layout.get("hypothesis_margin", 0.0),
                "matched_line_count": layout.get("matched_line_count", 0),
                "visible": visible,
                "annotated": annotated,
                "family": family_row,
            }
        )
        result_path = output / "layouts" / f"{image_path.stem}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        if write_overlays:
            overlay_path = output / "overlays" / f"{image_path.stem}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), draw_layout_overlay(image, layout)):
                raise RuntimeError(f"could not write layout overlay: {overlay_path}")
    summary = {
        "dataset": str(dataset_yaml.resolve()),
        "predictions": str(predictions.resolve()),
        "split": split,
        "images": len(images),
        "missing_predictions": missing_predictions,
        "layout_status": dict(statuses),
        "layout_success_rate": statuses["ok"] / max(len(images), 1),
        "visible_v2": _summarize_keypoints(accumulators["visible"]),
        "annotated_v1_or_v2": _summarize_keypoints(accumulators["annotated"]),
        "line_family": {
            **dict(family),
            "accuracy": family["correct_family"] / max(family["matched_lines"], 1),
            "matched_recall": family["matched_lines"] / max(family["gt_lines"], 1),
        },
        "threshold_definition": "fraction of labelled court bounding-box diagonal",
        "layout_solver": "semantic",
    }
    (output / "per-image.json").write_text(json.dumps(per_image, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = evaluate(
        args.dataset.resolve(),
        args.predictions.resolve(),
        args.topology.resolve(),
        args.output.resolve(),
        split=args.split,
        minimum_keypoint_score=args.minimum_keypoint_score,
        write_overlays=not args.no_overlays,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
