"""Build an explicitly pseudo-labelled Pose36 dataset from stable video layouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2

from volley_court import CourtLineModel
from volley_court.tracking import CourtLayoutTracker
from volley_court.types import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=512)
    return parser.parse_args()


def _read_batch(capture: cv2.VideoCapture, size: int) -> list[Image]:
    frames: list[Image] = []
    for _ in range(size):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cast(Image, frame))
    return frames


def _pose_row(width: int, height: int, layout: object) -> str:
    points = sorted(layout.keypoints, key=lambda point: point.id)  # type: ignore[attr-defined]
    visible = [point for point in points if point.in_frame]
    if len(points) != 36 or len(visible) < 4:
        raise ValueError("teacher layout does not contain enough Pose36 points")
    xs = [point.x / width for point in visible]
    ys = [point.y / height for point in visible]
    left, right = max(0.0, min(xs)), min(1.0, max(xs))
    top, bottom = max(0.0, min(ys)), min(1.0, max(ys))
    values = ["0", f"{(left + right) / 2:.8f}", f"{(top + bottom) / 2:.8f}"]
    values += [f"{right - left:.8f}", f"{bottom - top:.8f}"]
    for point in points:
        if point.in_frame:
            values += [f"{point.x / width:.8f}", f"{point.y / height:.8f}", "2"]
        else:
            values += ["0", "0", "0"]
    return " ".join(values) + "\n"


def main() -> int:
    args = parse_args()
    if args.sample_every < 1 or args.batch_size < 1:
        raise ValueError("sampling and batch sizes must be positive")
    image_dir = args.output / "train" / "images"
    label_dir = args.output / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = CourtLineModel.from_pretrained(
        args.model,
        device=args.device,
        image_size=args.imgsz,
        half=True,
        fuse=True,
        decoder="cuda",
        include_layout=True,
        anchor_ransac_max_iters=128,
    )
    tracker = CourtLayoutTracker()
    frame_index = 0
    written = 0
    try:
        while frames := _read_batch(capture, args.batch_size):
            results = model.predict_many(frames, include_layout=True)
            for frame, result in zip(frames, results, strict=True):
                layout = tracker.update(result.layout, width=width, height=height, frame=frame)
                if (
                    frame_index % args.sample_every == 0
                    and layout is not None
                    and layout.status == "ok"
                ):
                    stem = f"teacher-{args.source.stem}-f{frame_index:06d}"
                    if not cv2.imwrite(str(image_dir / f"{stem}.jpg"), frame):
                        raise RuntimeError(f"failed to write teacher frame {frame_index}")
                    (label_dir / f"{stem}.txt").write_text(
                        _pose_row(width, height, layout), encoding="utf-8"
                    )
                    written += 1
                frame_index += 1
    finally:
        capture.release()
    (args.output / "dataset.yaml").write_text(
        "path: .\ntrain: train/images\nval: train/images\ntest: train/images\n"
        "kpt_shape: [36, 3]\nnames:\n  0: court\n",
        encoding="utf-8",
    )
    print(f"wrote {written} teacher frames from {frame_index} source frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
