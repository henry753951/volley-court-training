from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .geometry import PointSample, audit_center_collisions, fit_line_group, parse_yolo_pose_line
from .topology import CourtLineTopology, load_topology

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pose36 labels into unordered zero-width line GT"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--preview-count", type=int, default=24)
    return parser.parse_args()


def _dataset_root(dataset_yaml: Path, payload: dict[str, Any]) -> Path:
    configured = Path(str(payload.get("path", ".")))
    if configured.is_absolute():
        return configured
    return (dataset_yaml.parent / configured).resolve()


def _split_paths(root: Path, value: str | list[str]) -> Iterable[Path]:
    values = value if isinstance(value, list) else [value]
    for configured in values:
        path = Path(configured)
        yield path if path.is_absolute() else root / path


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(
            f"image path does not contain an images directory: {image_path}"
        ) from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_points(label_path: Path, topology: CourtLineTopology):
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) > 1:
        raise ValueError(f"expected at most one court instance, got {len(lines)}: {label_path}")
    return parse_yolo_pose_line(lines[0], topology.keypoint_count) if lines else []


def _draw_preview(
    image: np.ndarray,
    rows: list[dict[str, Any]],
    points: list[PointSample],
    usable_visibility: frozenset[int],
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for row in rows:
        if not row["valid"]:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in row["segment"])
        # Display thickness only. It is not a model target and is never saved as a mask.
        cv2.line(output, (x1, y1), (x2, y2), (0, 230, 255), 2, cv2.LINE_AA)
    # Small dots are converter-only Pose36 samples. No point IDs enter model targets.
    for point in points:
        if point.visibility not in usable_visibility:
            continue
        center = (int(round(point.x * width)), int(round(point.y * height)))
        if point.visibility == 2:
            cv2.circle(output, center, 3, (80, 230, 80), -1, cv2.LINE_AA)
        else:
            cv2.circle(output, center, 4, (220, 80, 255), 1, cv2.LINE_AA)
    for row in rows:
        if not row["valid"]:
            continue
        cx, cy = (int(round(value)) for value in row["center"])
        # The larger cross is the line-center heatmap target, not a landmark.
        cv2.drawMarker(
            output,
            (cx, cy),
            (255, 80, 0),
            cv2.MARKER_CROSS,
            11,
            2,
            cv2.LINE_AA,
        )
    return output


def convert_dataset(
    dataset_yaml: Path,
    topology_path: Path,
    output: Path,
    stride: int = 4,
    preview_count: int = 24,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    topology = load_topology(topology_path)
    dataset_payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if dataset_payload.get("kpt_shape") != [topology.keypoint_count, 3]:
        raise ValueError(
            f"dataset kpt_shape {dataset_payload.get('kpt_shape')} does not match topology "
            f"[{topology.keypoint_count}, 3]"
        )
    root = _dataset_root(dataset_yaml, dataset_payload)
    output.mkdir(parents=True, exist_ok=True)
    split_keys = (("train", "train"), ("valid", "val"), ("test", "test"))
    summary: dict[str, Any] = {
        "dataset": str(dataset_yaml.resolve()),
        "topology": str(topology_path.resolve()),
        "representation": "unordered zero-width finite line segments",
        "stride": stride,
        "splits": {},
    }
    all_minimum_distances: list[float] = []
    all_fit_rms_distances: list[float] = []
    total_collisions = 0
    total_images = 0
    total_valid_segments = 0
    images_with_valid_segments = 0
    invalid_reasons: Counter[str] = Counter()
    for output_split, yaml_key in split_keys:
        configured = dataset_payload.get(yaml_key)
        if configured is None:
            continue
        images = sorted(
            path
            for directory in _split_paths(root, configured)
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        preview_indices = set(
            np.linspace(
                0, max(0, len(images) - 1), min(preview_count, len(images)), dtype=int
            ).tolist()
        )
        split_valid = 0
        split_collisions = 0
        split_images_with_valid_segments = 0
        for image_index, image_path in enumerate(images):
            label_path = _label_path(image_path)
            if not label_path.exists():
                raise FileNotFoundError(f"missing label for {image_path}: {label_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV could not decode: {image_path}")
            height, width = image.shape[:2]
            points = _read_points(label_path, topology)
            fitted = (
                [
                    fit_line_group(
                        points,
                        group,
                        width,
                        height,
                        usable_visibility=topology.usable_visibility,
                    )
                    for group in topology.line_groups
                ]
                if points
                else []
            )
            rows = []
            for topology_index, segment in enumerate(fitted):
                row = {"topology_index": topology_index, **segment.to_dict()}
                rows.append(row)
                if segment.valid:
                    split_valid += 1
                    if segment.fit_rms_distance_px is not None:
                        all_fit_rms_distances.append(float(segment.fit_rms_distance_px))
                else:
                    invalid_reasons[segment.reason or "unknown"] += 1
            collision = audit_center_collisions(fitted, stride)
            if collision["valid_segment_count"]:
                split_images_with_valid_segments += 1
            split_collisions += int(collision["grid_collision_count"])
            minimum = collision["minimum_center_distance_px"]
            if minimum is not None:
                all_minimum_distances.append(float(minimum))
            metadata = {
                "image": str(image_path.resolve()),
                "image_relative": str(image_path.resolve().relative_to(root.resolve())).replace(
                    "\\", "/"
                ),
                "pose_label": str(label_path.resolve()),
                "width": width,
                "height": height,
                "prediction_class": "court_line",
                "unordered": True,
                "zero_width": True,
                "keypoints": [
                    {
                        "id": point.index,
                        "x": point.x,
                        "y": point.y,
                        "visibility": point.visibility,
                    }
                    for point in points
                ],
                "line_groups": rows,
                "center_collision": collision,
            }
            destination = output / output_split / "labels" / f"{image_path.stem}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if image_index in preview_indices:
                preview = _draw_preview(image, rows, points, topology.usable_visibility)
                preview_path = output / output_split / "previews" / f"{image_path.stem}.jpg"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise RuntimeError(f"could not write preview: {preview_path}")
        summary["splits"][output_split] = {
            "images": len(images),
            "valid_segments": split_valid,
            "grid_collisions": split_collisions,
            "images_with_valid_segments": split_images_with_valid_segments,
        }
        total_images += len(images)
        total_valid_segments += split_valid
        images_with_valid_segments += split_images_with_valid_segments
        total_collisions += split_collisions
    summary["totals"] = {
        "images": total_images,
        "valid_segments": total_valid_segments,
        "grid_collisions": total_collisions,
        "images_with_possible_segments": total_images,
        "images_with_valid_segments": images_with_valid_segments,
        "minimum_center_distance_px": min(all_minimum_distances) if all_minimum_distances else None,
        "fit_rms_distance_px": {
            "median": float(np.median(all_fit_rms_distances)) if all_fit_rms_distances else None,
            "p95": float(np.percentile(all_fit_rms_distances, 95))
            if all_fit_rms_distances
            else None,
            "max": max(all_fit_rms_distances) if all_fit_rms_distances else None,
        },
        "invalid_reasons": dict(invalid_reasons),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = convert_dataset(
        args.dataset.resolve(),
        args.topology.resolve(),
        args.output.resolve(),
        stride=args.stride,
        preview_count=args.preview_count,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
