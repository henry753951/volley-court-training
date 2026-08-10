
"""Small utilities for inspecting and training the volleyball court keypoint set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

# The source export contains only numeric keypoint indices. These names keep the
# source order intact while making the rendered samples and model output usable.
DEFAULT_POINT_NAMES = [
    "court_left_near_corner",
    "court_left_attack_near",
    "court_left_net_line",
    "court_left_attack_far",
    "court_left_far_corner",
    "court_right_far_corner",
    "court_right_attack_far",
    "court_right_net_line",
    "court_right_attack_near",
    "court_right_near_corner",
    "net_left_bottom",
    "net_left_top",
    "net_right_top",
    "net_right_bottom",
]
POINT_NAME_NOTE = (
    "Point names and keypoint count are loaded from dataset.yaml; numeric label order is unchanged."
)


def load_point_names(root: Path) -> list[str]:
    dataset_yaml = root.resolve() / "dataset.yaml"
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = payload.get("kpt_names") if isinstance(payload, dict) else None
    shape = payload.get("kpt_shape") if isinstance(payload, dict) else None
    if not isinstance(names, list) or not names:
        raise ValueError(f"dataset has no kpt_names: {dataset_yaml}")
    if not isinstance(shape, list) or not shape or int(shape[0]) != len(names):
        raise ValueError(f"kpt_shape and kpt_names disagree: {dataset_yaml}")
    return [str(name) for name in names]
COLORS = [
    (0, 255, 255),
    (0, 200, 255),
    (0, 128, 255),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 128),
    (255, 0, 0),
    (128, 0, 255),
    (0, 255, 0),
    (128, 255, 0),
    (255, 255, 0),
    (255, 128, 0),
    (255, 255, 255),
    (160, 160, 160),
]


def _parse_label(path: Path, keypoint_count: int = 14) -> list[tuple[int, float, float, float, float, np.ndarray]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        expected = 5 + keypoint_count * 3
        if not values:
            continue
        if len(values) != expected:
            raise ValueError(f"{path}:{line_number}: expected {expected} values, got {len(values)}")
        numbers = np.asarray([float(value) for value in values], dtype=np.float32)
        keypoints = numbers[5:].reshape(keypoint_count, 3)
        rows.append((int(numbers[0]), *map(float, numbers[1:5]), keypoints))
    return rows


def _image_label_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label for {image_path.name}: {label_path}")
        pairs.append((image_path, label_path))
    return pairs


def _draw_sample(image_path: Path, label_path: Path, point_names: list[str]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    height, width = image.shape[:2]
    for class_id, cx, cy, box_width, box_height, keypoints in _parse_label(label_path, len(point_names)):
        x1 = int((cx - box_width / 2) * width)
        y1 = int((cy - box_height / 2) * height)
        x2 = int((cx + box_width / 2) * width)
        y2 = int((cy + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(image, f"class {class_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2, cv2.LINE_AA)
        for index, (x, y, visibility) in enumerate(keypoints):
            if visibility <= 0:
                continue
            point = (int(float(x) * width), int(float(y) * height))
            color = COLORS[index % len(COLORS)]
            radius = 7 if visibility >= 2 else 5
            cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, point, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{point_names[index]}({index})",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
    return image


def visualize_dataset(
    root: Path,
    output_dir: Path,
    sample_count: int = 16,
    point_names: list[str] | None = None,
) -> dict:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_names = point_names or load_point_names(root)
    all_pairs = [(split, pair) for split in ("train", "valid", "test") for pair in _image_label_pairs(root, split)]
    if not all_pairs:
        raise ValueError(f"no image/label pairs found under {root}")
    sample_indices = np.linspace(0, len(all_pairs) - 1, min(sample_count, len(all_pairs)), dtype=int)
    samples = []
    for sample_number, index in enumerate(dict.fromkeys(sample_indices), 1):
        split, (image_path, label_path) = all_pairs[int(index)]
        rendered = _draw_sample(image_path, label_path, point_names)
        target = output_dir / f"sample_{sample_number:03d}_{split}_{image_path.stem[:32]}.jpg"
        cv2.imwrite(str(target), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
        samples.append({"split": split, "image": str(image_path), "label": str(label_path), "visualization": str(target)})

    # A single contact sheet makes it quick to inspect the complete sample set.
    rendered_samples = [cv2.imread(item["visualization"]) for item in samples]
    rendered_samples = [image for image in rendered_samples if image is not None]
    if rendered_samples:
        tile_width = 320
        tiles = []
        for image in rendered_samples:
            scale = tile_width / image.shape[1]
            tiles.append(cv2.resize(image, (tile_width, max(1, int(image.shape[0] * scale)))))
        tile_height = max(image.shape[0] for image in tiles)
        columns = min(4, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
        for index, image in enumerate(tiles):
            row, column = divmod(index, columns)
            sheet[row * tile_height : row * tile_height + image.shape[0], column * tile_width : column * tile_width + image.shape[1]] = image
        cv2.imwrite(str(output_dir / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])

    stats = {str(index): {"visible": 0, "x": [], "y": []} for index in range(len(point_names))}
    for _, (_, label_path) in all_pairs:
        for _, _, _, _, _, keypoints in _parse_label(label_path, len(point_names)):
            for index, (x, y, visibility) in enumerate(keypoints):
                if visibility > 0:
                    stats[str(index)]["visible"] += 1
                    stats[str(index)]["x"].append(float(x))
                    stats[str(index)]["y"].append(float(y))
    summary = {
        "root": str(root),
        "image_count": len(all_pairs),
        "point_names": point_names,
        "point_name_note": POINT_NAME_NOTE,
        "samples": samples,
        "keypoints": {
            index: {
                "name": point_names[int(index)],
                "visible": value["visible"],
                "mean_x": float(np.mean(value["x"])) if value["x"] else None,
                "mean_y": float(np.mean(value["y"])) if value["y"] else None,
            }
            for index, value in stats.items()
        },
    }
    (output_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def write_data_yaml(root: Path, point_names: list[str]) -> Path:
    if len(point_names) != 14:
        raise ValueError("YOLO26 court dataset expects exactly 14 point names")
    root = root.resolve()
    lines = [
        f"path: '{root.as_posix()}'",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        "kpt_shape: [14, 3]",
        # The first ten points run along the two sidelines from one court end
        # to the other. A horizontal flip reverses that run on each sideline.
        "flip_idx: [4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 13, 12, 11, 10]",
        "kpt_names:",
    ]
    lines.extend(f"  - {name}" for name in point_names)
    lines.extend(["", "nc: 1", "names: [volleyball-court]", ""])
    path = root / "data.yolo26-pose.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize and prepare the volleyball court YOLO pose dataset")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    output = args.output or args.root / "visualizations" / "indexed"
    point_names = load_point_names(args.root)
    summary = visualize_dataset(args.root, output, args.samples, point_names)
    data_yaml = args.root.resolve() / "dataset.yaml"
    print(json.dumps({
        "images": summary["image_count"],
        "output": str(output),
        "summary": str(output / "visualization_summary.json"),
        "data_yaml": str(data_yaml),
        "point_names": summary["point_names"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
