"""Draw canonical 36-point YOLO labels over Blender synthetic renders."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


BASE_SEGMENTS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0),
    (1, 8),
    (2, 7),
    (3, 6),
)


def expanded_skeleton() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for segment_index, (start, end) in enumerate(BASE_SEGMENTS):
        first = 10 + segment_index * 2
        second = first + 1
        edges.extend(((start, first), (first, second), (second, end)))
    return tuple(edges)


def parse_label(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    values = path.read_text(encoding="utf-8").strip().split()
    if len(values) != 5 + 36 * 3:
        raise ValueError(f"expected 113 YOLO values, got {len(values)}: {path}")
    raw = np.asarray([float(value) for value in values[5:]], dtype=np.float32).reshape(36, 3)
    points = raw[:, :2] * np.asarray([width, height], dtype=np.float32)
    return points, raw[:, 2].astype(np.int32)


def draw(image: np.ndarray, points: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    output = image.copy()
    for start, end in expanded_skeleton():
        if visibility[start] > 0 and visibility[end] > 0:
            color = (235, 235, 235) if visibility[start] == 2 and visibility[end] == 2 else (220, 80, 255)
            cv2.line(output, tuple(np.rint(points[start]).astype(int)), tuple(np.rint(points[end]).astype(int)), color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        state = int(visibility[index])
        if state == 0:
            continue
        center = tuple(np.rint(point).astype(int))
        color = (40, 220, 255) if state == 2 else (220, 60, 255)
        cv2.circle(output, center, 7 if index < 10 else 5, (8, 12, 18), -1, cv2.LINE_AA)
        if state == 2:
            cv2.circle(output, center, 5 if index < 10 else 3, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(output, center, 5 if index < 10 else 3, color, 2, cv2.LINE_AA)
            cv2.line(output, (center[0] - 4, center[1] - 4), (center[0] + 4, center[1] + 4), color, 1, cv2.LINE_AA)
        cv2.putText(output, str(index), (center[0] + 7, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (7, 10, 16), 3, cv2.LINE_AA)
        cv2.putText(output, str(index), (center[0] + 7, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    visible = int((visibility == 2).sum())
    occluded = int((visibility == 1).sum())
    outside = int((visibility == 0).sum())
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (6, 10, 17), -1)
    cv2.putText(output, f"visible={visible}  occluded={occluded}  outside={outside}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 247, 250), 2, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "valid", "test"):
        image_dir = args.dataset / split / "images"
        label_dir = args.dataset / split / "labels"
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                pairs.append((image_path, label_path))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.limit]
    if not pairs:
        raise FileNotFoundError(f"no image/label pairs found under {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)
    previews: list[np.ndarray] = []
    rows: list[dict[str, str]] = []
    for index, (image_path, label_path) in enumerate(pairs):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        points, visibility = parse_label(label_path, image.shape[1], image.shape[0])
        rendered = draw(image, points, visibility)
        destination = args.output / f"sample_{index + 1:03d}_{image_path.stem}.jpg"
        cv2.imwrite(str(destination), rendered, [cv2.IMWRITE_JPEG_QUALITY, 94])
        previews.append(cv2.resize(rendered, (640, 360), interpolation=cv2.INTER_AREA))
        rows.append({"image": str(image_path.resolve()), "label": str(label_path.resolve()), "visualization": str(destination.resolve())})
    blank = np.zeros_like(previews[0])
    while len(previews) % 3:
        previews.append(blank)
    sheet = np.vstack([np.hstack(previews[index : index + 3]) for index in range(0, len(previews), 3)])
    contact_sheet = args.output / "contact-sheet.jpg"
    cv2.imwrite(str(contact_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    (args.output / "visualization.json").write_text(json.dumps({"dataset": str(args.dataset.resolve()), "contact_sheet": str(contact_sheet.resolve()), "samples": rows}, indent=2) + "\n", encoding="utf-8")
    print(contact_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
