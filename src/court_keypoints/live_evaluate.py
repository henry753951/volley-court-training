"""Realtime OpenCV viewer for a 36-point court-keypoint checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .evaluate_video import draw_prediction, homography_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--point-conf", type=float, default=0.25)
    parser.add_argument("--window", default="Court CV realtime - 36 keypoints")
    parser.add_argument("--loop", action="store_true", help="loop the video after the last frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(str(args.model))
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, 1280, 720)
    frame_index = 0
    frame_period = 1.0 / max(1.0, fps)
    next_frame_at = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    continue
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
            points = np.zeros((36, 2), dtype=np.float32)
            confidences = np.zeros(36, dtype=np.float32)
            box_confidence = 0.0
            detected = bool(
                result is not None
                and result.boxes is not None
                and result.keypoints is not None
                and len(result.boxes) > 0
            )
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
                points, confidences, args.point_conf
            )
            visible_points = int((confidences >= args.point_conf).sum())
            status = (
                f"frame {frame_index}/{max(0, total_frames - 1)}  "
                f"box={box_confidence:.2f}  points={visible_points}/36  "
                f"H={'OK' if homography_ok else 'FAIL'}  inliers={inliers}  "
                f"q/Esc: quit"
            )
            rendered = draw_prediction(
                frame,
                points,
                confidences,
                point_threshold=args.point_conf,
                status=status,
            )
            cv2.imshow(args.window, rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_index += 1
            next_frame_at += frame_period
            delay = next_frame_at - time.perf_counter()
            if delay > 0:
                time.sleep(min(delay, 0.05))
            else:
                next_frame_at = time.perf_counter()
    finally:
        capture.release()
        cv2.destroyWindow(args.window)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
