from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .api import CourtLineModel, InferenceConfig
from .types import CourtFrameResult, Image

AUDIT_CORNER_COLORS = {
    0: (80, 255, 80),
    4: (255, 255, 80),
    5: (255, 80, 255),
    9: (80, 200, 255),
}


def _label_layout_corners(image: Image, result: CourtFrameResult) -> None:
    if result.layout is None or result.layout.status != "ok":
        return
    height, width = image.shape[:2]
    for point in result.layout.keypoints:
        if point.id not in AUDIT_CORNER_COLORS:
            continue
        center = (
            int(np.clip(round(point.x), 12, max(width - 72, 12))),
            int(np.clip(round(point.y), 28, max(height - 12, 28))),
        )
        color = AUDIT_CORNER_COLORS[point.id]
        cv2.circle(image, center, 9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(image, center, 7, color, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            f"ID {point.id}{'' if point.in_frame else ' OUT'}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"ID {point.id}{'' if point.in_frame else ' OUT'}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render five-consecutive-frame layout sheets for visual release gating"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timestamps",
        default="",
        help="Comma-separated seconds; defaults to 20, 50, and 80 percent",
    )
    parser.add_argument("--consecutive", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder", choices=("auto", "spatial", "cuda"), default="auto")
    parser.add_argument("--anchor-ransac-max-iters", type=int, default=128)
    return parser.parse_args()


def _timestamps(capture: cv2.VideoCapture, configured: str) -> list[float]:
    if configured.strip():
        return [float(value) for value in configured.split(",") if value.strip()]
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps
    return [duration * fraction for fraction in (0.2, 0.5, 0.8)]


def audit_video(
    source: Path,
    model_path: Path,
    output: Path,
    *,
    timestamps: str = "",
    consecutive: int = 5,
    tile_width: int = 384,
    device: str = "auto",
    image_size: int = 512,
    half: bool = True,
    decoder: str = "auto",
    anchor_ransac_max_iters: int = 128,
) -> dict[str, Any]:
    if consecutive < 5:
        raise ValueError("release audit requires at least five consecutive frames")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    sample_times = _timestamps(capture, timestamps)
    model = CourtLineModel(
        model_path,
        config=InferenceConfig(
            device=device,
            image_size=image_size,
            half=half,
            decoder=decoder,
            include_layout=True,
            anchor_ransac_max_iters=anchor_ransac_max_iters,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    try:
        for group_index, timestamp in enumerate(sample_times):
            start_frame = max(0, int(round(timestamp * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames: list[Image] = []
            for _ in range(consecutive):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cast(Image, frame))
            if len(frames) != consecutive:
                raise RuntimeError(f"could not read {consecutive} frames at {timestamp:.3f}s")
            results = model.predict_many(frames, include_layout=True)
            tiles = []
            rows = []
            for offset, (frame, result) in enumerate(zip(frames, results, strict=True)):
                rendered = model.visualize(frame, result)
                _label_layout_corners(rendered, result)
                scale = tile_width / rendered.shape[1]
                tile = cv2.resize(
                    rendered,
                    (tile_width, max(1, int(round(rendered.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
                status = result.layout.status if result.layout else "unknown"
                cv2.putText(
                    tile,
                    f"frame {start_frame + offset} | {status}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
                rows.append(
                    {
                        "frame": start_frame + offset,
                        "status": status,
                        "layout": result.layout.to_mapping() if result.layout else None,
                    }
                )
            sheet = np.concatenate(tiles, axis=1)
            sheet_path = output / f"group-{group_index + 1:02d}-{timestamp:.3f}s.jpg"
            if not cv2.imwrite(str(sheet_path), sheet):
                raise RuntimeError(f"could not write contact sheet: {sheet_path}")
            groups.append(
                {
                    "timestamp": timestamp,
                    "start_frame": start_frame,
                    "sheet": str(sheet_path.resolve()),
                    "frames": rows,
                }
            )
    finally:
        capture.release()
    report = {
        "schema": "volley-court-five-frame-visual-audit-v1",
        "source": str(source.resolve()),
        "model": str(model_path.resolve()),
        "fps": fps,
        "consecutive": consecutive,
        "groups": groups,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = audit_video(
        args.source,
        args.model,
        args.output,
        timestamps=args.timestamps,
        consecutive=args.consecutive,
        tile_width=args.tile_width,
        device=args.device,
        image_size=args.imgsz,
        half=args.half,
        decoder=args.decoder,
        anchor_ransac_max_iters=args.anchor_ransac_max_iters,
    )
    print(json.dumps(report, indent=2))
    return 0
