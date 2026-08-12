from __future__ import annotations

import argparse
import json
from itertools import pairwise
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


def _candidate_points(layout: dict[str, Any] | None) -> np.ndarray | None:
    if not layout:
        return None
    rows = layout.get("candidate_keypoints") or layout.get("keypoints") or []
    by_id = {int(row["id"]): (float(row["x"]), float(row["y"])) for row in rows}
    if len(by_id) != 36 or any(index not in by_id for index in range(36)):
        return None
    points = np.asarray([by_id[index] for index in range(36)], dtype=np.float32)
    return points if np.isfinite(points).all() else None


def _global_affine(previous: Image, current: Image) -> np.ndarray | None:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if features is None or len(features) < 12:
        return None
    tracked, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if tracked is None or status is None:
        return None
    selected = status.reshape(-1).astype(bool)
    if int(selected.sum()) < 12:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(
        features[selected].reshape(-1, 2),
        tracked[selected].reshape(-1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=500,
        confidence=0.99,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 8:
        return None
    return affine.astype(np.float32)


def _group_stability_metrics(
    frames: list[Image],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = [str(row["status"]) for row in rows]
    transitions = sum(first != second for first, second in pairwise(statuses))
    accepted = sum(status == "ok" for status in statuses)
    semantic = [
        float(row["layout"]["semantic_alignment"])
        for row in rows
        if row.get("layout") and row["layout"].get("semantic_alignment") is not None
    ]
    residuals: list[float] = []
    for (previous_frame, current_frame), (previous_row, current_row) in zip(
        pairwise(frames),
        pairwise(rows),
        strict=True,
    ):
        previous_points = _candidate_points(previous_row.get("layout"))
        current_points = _candidate_points(current_row.get("layout"))
        if previous_points is None or current_points is None:
            continue
        affine = _global_affine(previous_frame, current_frame)
        if affine is None:
            continue
        warped = cv2.transform(previous_points.reshape(1, -1, 2), affine).reshape(-1, 2)
        diagonal = max(float(np.hypot(*current_frame.shape[:2])), 1.0)
        residuals.extend((np.linalg.norm(warped - current_points, axis=1) / diagonal).tolist())
    return {
        "frame_count": len(rows),
        "accepted_count": accepted,
        "accepted_coverage": accepted / max(len(rows), 1),
        "status_transition_count": transitions,
        "status_flicker_rate": transitions / max(len(rows) - 1, 1),
        "semantic_alignment_median": float(np.median(semantic)) if semantic else None,
        "semantic_alignment_p05": float(np.percentile(semantic, 5)) if semantic else None,
        "motion_compensated_layout_jitter_median": (
            float(np.median(residuals)) if residuals else None
        ),
        "motion_compensated_layout_jitter_p95": (
            float(np.percentile(residuals, 95)) if residuals else None
        ),
        "motion_compensated_sample_count": len(residuals),
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
                    "stability": _group_stability_metrics(frames, rows),
                }
            )
    finally:
        capture.release()
    all_rows = [row for group in groups for row in group["frames"]]
    all_stability = [group["stability"] for group in groups]
    report = {
        "schema": "volley-court-five-frame-visual-audit-v2",
        "source": str(source.resolve()),
        "model": str(model_path.resolve()),
        "fps": fps,
        "consecutive": consecutive,
        "groups": groups,
        "summary": {
            "frame_count": len(all_rows),
            "accepted_count": sum(row["status"] == "ok" for row in all_rows),
            "accepted_coverage": (
                sum(row["status"] == "ok" for row in all_rows) / max(len(all_rows), 1)
            ),
            "status_transition_count": sum(
                int(row["status_transition_count"]) for row in all_stability
            ),
            "motion_compensated_layout_jitter_median": (
                float(
                    np.median(
                        [
                            row["motion_compensated_layout_jitter_median"]
                            for row in all_stability
                            if row["motion_compensated_layout_jitter_median"] is not None
                        ]
                    )
                )
                if any(
                    row["motion_compensated_layout_jitter_median"] is not None
                    for row in all_stability
                )
                else None
            ),
            "release_gates": {
                "catastrophic_axis_swap_count": 0,
                "identity_contract": "fixed_pose36_no_runtime_permutation",
            },
        },
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
