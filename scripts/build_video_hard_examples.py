from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from volley_court.api import CourtLineModel, InferenceConfig
from volley_court.types import CourtFrameResult


@dataclass(frozen=True)
class Window:
    video: str
    second: float
    positive: bool


WINDOWS = (
    *(Window("clip.mp4", second, True) for second in (2.0, 5.0, 8.0, 11.0)),
    *(Window("gdrSr-Vso90.mp4", second, True) for second in (119.85, 299.624, 479.399)),
    *(Window("pW66S38FAQM.mp4", second, True) for second in (516.3, 825.6)),
    Window("rMvxEtorQhw.mp4", 119.65, True),
    Window("pW66S38FAQM.mp4", 207.4, False),
    Window("rMvxEtorQhw.mp4", 299.124, False),
    Window("rMvxEtorQhw.mp4", 478.598, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewed video hard examples")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def pose_label(result: CourtFrameResult) -> str | None:
    layout = result.layout
    if layout is None or layout.status != "ok" or len(layout.keypoints) != 36:
        return None
    points = {point.id: point for point in layout.keypoints}
    corners = [points[index] for index in (0, 4, 5, 9)]
    if not (
        0.5 * (corners[0].y + corners[3].y) > 0.5 * (corners[1].y + corners[2].y)
        and corners[0].x < corners[3].x
        and corners[1].x < corners[2].x
    ):
        return None
    visible = [
        point
        for point in layout.keypoints
        if 0.0 <= point.x < result.width and 0.0 <= point.y < result.height
    ]
    if len(visible) < 8:
        return None
    normalized = [(point.x / result.width, point.y / result.height) for point in visible]
    minimum = np.min(np.asarray(normalized), axis=0)
    maximum = np.max(np.asarray(normalized), axis=0)
    center = 0.5 * (minimum + maximum)
    size = np.maximum(maximum - minimum, 1e-4)
    tokens = ["0", *(f"{value:.8f}" for value in (*center, *size))]
    for index in range(36):
        point = points[index]
        x, y = point.x / result.width, point.y / result.height
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            tokens.extend((f"{x:.8f}", f"{y:.8f}", "2"))
        else:
            tokens.extend(("0.00000000", "0.00000000", "0"))
    return " ".join(tokens) + "\n"


def main() -> int:
    args = parse_args()
    image_dir = args.output / "train" / "images"
    label_dir = args.output / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    model = CourtLineModel(
        args.model,
        config=InferenceConfig(device=args.device, decoder="cuda", include_layout=True),
    )
    preview_dir = args.output / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_tiles: dict[str, list[np.ndarray]] = defaultdict(list)
    manifest = []
    for window_index, window in enumerate(WINDOWS):
        source = args.videos / window.video
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        center = int(round(window.second * fps))
        frame_numbers = range(center - 30, center + 31, 6)
        frames = []
        decoded_numbers = []
        for frame_number in frame_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
                decoded_numbers.append(frame_number)
        capture.release()
        results = model.predict_many(frames, include_layout=True)
        for frame_number, frame, result in zip(decoded_numbers, frames, results, strict=True):
            stem = f"{source.stem}-f{frame_number:08d}"
            label = pose_label(result) if window.positive else ""
            if window.positive and label is None:
                continue
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write {image_path}")
            label_path.write_text(label or "", encoding="utf-8")
            manifest.append(
                {
                    "image": str(image_path.resolve()),
                    "video": window.video,
                    "frame": frame_number,
                    "positive": window.positive,
                }
            )
            preview_key = (
                f"{source.stem}-{'positive' if window.positive else 'negative'}-"
                f"{window_index:02d}-{window.second:.3f}s"
            )
            if len(preview_tiles[preview_key]) < 10:
                rendered = model.visualize(frame, result) if window.positive else frame.copy()
                for point in result.layout.keypoints if result.layout is not None else ():
                    if window.positive and point.id in {0, 4, 5, 9} and point.in_frame:
                        cv2.putText(
                            rendered,
                            str(point.id),
                            (int(point.x), int(point.y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                tile = cv2.resize(rendered, (512, 288), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tile, (0, 0), (512, 30), (0, 0, 0), -1)
                cv2.putText(
                    tile,
                    f"{window.video}  frame={frame_number}  "
                    f"{'POSITIVE' if window.positive else 'NEGATIVE'}",
                    (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                preview_tiles[preview_key].append(tile)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "dataset.yaml").write_text(
        "path: .\ntrain: train/images\nval: train/images\ntest: train/images\n"
        "kpt_shape: [36, 3]\nflip_idx: []\nnames:\n  0: volleyball_court\n",
        encoding="utf-8",
    )
    for preview_key, tiles in preview_tiles.items():
        rows = [
            np.concatenate(tiles[index : index + 5], axis=1) for index in range(0, len(tiles), 5)
        ]
        width = max(row.shape[1] for row in rows)
        padded = [
            cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT)
            for row in rows
        ]
        cv2.imwrite(str(preview_dir / f"{preview_key}.jpg"), np.concatenate(padded, axis=0))
    positives = sum(bool(row["positive"]) for row in manifest)
    print(json.dumps({"images": len(manifest), "positives": positives}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
