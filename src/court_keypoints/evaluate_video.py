
"""Evaluate and visualize a YOLO court-keypoint checkpoint on a video clip.

The report intentionally measures the same first-ten-point homography consumed
by the analysis engine.  It also renders all 36 points and the court skeleton so
index/order failures are visible instead of being hidden behind aggregate mAP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


COURT_WORLD_POINTS = np.asarray(
    [
        # The dataset schema walks the left sideline, far baseline, right
        # sideline, and near baseline. Keep this order identical to kpt_names
        # in dataset.yaml; it is not a row-major rectangle order.
        (0.0, 9.0),
        (0.0, 6.0),
        (0.0, 4.5),
        (0.0, 3.0),
        (0.0, 0.0),
        (18.0, 0.0),
        (18.0, 3.0),
        (18.0, 4.5),
        (18.0, 6.0),
        (18.0, 9.0),
    ],
    dtype=np.float32,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--point-conf", type=float, default=0.25)
    parser.add_argument("--sheet-columns", type=int, default=4)
    parser.add_argument("--preview-count", type=int, default=10)
    return parser.parse_args()


def expanded_skeleton() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for segment_index, (start, end) in enumerate(BASE_SEGMENTS):
        first = 10 + segment_index * 2
        second = first + 1
        edges.extend(((start, first), (first, second), (second, end)))
    return tuple(edges)


def homography_metrics(
    points: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
) -> tuple[bool, int, float | None]:
    valid = (
        np.isfinite(points[:10]).all(axis=1)
        & np.isfinite(confidences[:10])
        & (confidences[:10] >= threshold)
        & (points[:10, 0] > 0)
        & (points[:10, 1] > 0)
    )
    if int(valid.sum()) < 4:
        return False, 0, None
    matrix, mask = cv2.findHomography(
        points[:10][valid].astype(np.float32),
        COURT_WORLD_POINTS[valid],
        cv2.RANSAC,
        1.0,
    )
    if matrix is None or mask is None:
        return False, 0, None
    projected = cv2.perspectiveTransform(
        points[:10][valid].astype(np.float32).reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - COURT_WORLD_POINTS[valid], axis=1)
    inliers = int(mask.sum())
    median_error = float(np.median(errors[mask.reshape(-1).astype(bool)]))
    return inliers >= 4, inliers, median_error


def draw_prediction(
    frame: np.ndarray,
    points: np.ndarray,
    confidences: np.ndarray,
    *,
    point_threshold: float,
    status: str,
) -> np.ndarray:
    output = frame.copy()
    visible = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidences)
        & (confidences >= point_threshold)
        & (points[:, 0] > 0)
        & (points[:, 1] > 0)
    )
    for start, end in expanded_skeleton():
        if start < len(points) and end < len(points) and visible[start] and visible[end]:
            cv2.line(
                output,
                tuple(np.rint(points[start]).astype(int)),
                tuple(np.rint(points[end]).astype(int)),
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if not visible[index]:
            continue
        center = tuple(np.rint(point).astype(int))
        color = (40, 220, 255) if index < 10 else (255, 180, 70)
        cv2.circle(output, center, 7 if index < 10 else 5, (8, 12, 18), -1, cv2.LINE_AA)
        cv2.circle(output, center, 5 if index < 10 else 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(index),
            (center[0] + 7, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (8, 12, 18),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            str(index),
            (center[0] + 7, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 44), (7, 12, 18), -1)
    cv2.putText(
        output,
        status,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 248, 250),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(args.output / "overlay.mp4"),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot create overlay video")

    model = YOLO(str(args.model))
    rows: list[dict[str, Any]] = []
    previews: list[np.ndarray] = []
    preview_count = max(0, args.preview_count)
    preview_indices = (
        set(np.linspace(0, max(0, total_frames - 1), preview_count).astype(int).tolist())
        if preview_count
        else set()
    )
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            predictions = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                verbose=False,
                save=False,
            )
            result = predictions[0] if predictions else None
            detected = bool(
                result is not None
                and result.boxes is not None
                and result.keypoints is not None
                and len(result.boxes) > 0
            )
            points = np.zeros((36, 2), dtype=np.float32)
            confidences = np.zeros(36, dtype=np.float32)
            box_confidence = 0.0
            if detected:
                scores = result.boxes.conf.detach().cpu().numpy().reshape(-1)
                best = int(np.argmax(scores))
                box_confidence = float(scores[best])
                raw_points = result.keypoints.xy[best].detach().cpu().numpy()
                raw_confidences = (
                    result.keypoints.conf[best].detach().cpu().numpy()
                    if result.keypoints.conf is not None
                    else np.ones(len(raw_points), dtype=np.float32)
                )
                count = min(36, len(raw_points))
                points[:count] = raw_points[:count]
                confidences[:count] = raw_confidences[:count]
            homography_ok, inliers, median_error = homography_metrics(
                points,
                confidences,
                args.point_conf,
            )
            visible_points = int((confidences >= args.point_conf).sum())
            status = (
                f"frame {frame_index}/{total_frames - 1}  box={box_confidence:.2f}  "
                f"points={visible_points}/36  H={'OK' if homography_ok else 'FAIL'}  "
                f"inliers={inliers}  err={median_error:.3f}m"
                if median_error is not None
                else (
                    f"frame {frame_index}/{total_frames - 1}  box={box_confidence:.2f}  "
                    f"points={visible_points}/36  H=FAIL  inliers={inliers}"
                )
            )
            rendered = draw_prediction(
                frame,
                points,
                confidences,
                point_threshold=args.point_conf,
                status=status,
            )
            writer.write(rendered)
            if frame_index in preview_indices:
                previews.append(cv2.resize(rendered, (640, 360), interpolation=cv2.INTER_AREA))
            rows.append(
                {
                    "frame_index": frame_index,
                    "detected": detected,
                    "box_confidence": box_confidence,
                    "visible_points": visible_points,
                    "homography_ok": homography_ok,
                    "homography_inliers": inliers,
                    "median_reprojection_error_m": median_error,
                }
            )
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    homography_rows = [row for row in rows if row["homography_ok"]]
    summary = {
        "model": str(args.model.resolve()),
        "video": str(args.video.resolve()),
        "imgsz": args.imgsz,
        "frames": len(rows),
        "detected_frames": sum(bool(row["detected"]) for row in rows),
        "homography_frames": len(homography_rows),
        "homography_rate": len(homography_rows) / max(1, len(rows)),
        "mean_visible_points": float(np.mean([row["visible_points"] for row in rows])),
        "median_box_confidence": float(np.median([row["box_confidence"] for row in rows])),
        "median_homography_inliers": (
            float(np.median([row["homography_inliers"] for row in homography_rows]))
            if homography_rows
            else None
        ),
        "median_reprojection_error_m": (
            float(
                np.median(
                    [
                        row["median_reprojection_error_m"]
                        for row in homography_rows
                        if row["median_reprojection_error_m"] is not None
                    ]
                )
            )
            if homography_rows
            else None
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    if previews:
        blank = np.zeros_like(previews[0])
        while len(previews) % args.sheet_columns:
            previews.append(blank)
        sheet = np.vstack(
            [
                np.hstack(previews[index : index + args.sheet_columns])
                for index in range(0, len(previews), args.sheet_columns)
            ]
        )
        cv2.imwrite(str(args.output / "contact-sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
