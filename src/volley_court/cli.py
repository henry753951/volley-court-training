from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2

from .api import CourtLineModel, InferenceConfig
from .assets import DEFAULT_MODEL, download_model, sha256sum
from .benchmark import benchmark_model, read_video_frames
from .tracking import CourtLayoutTracker
from .types import Image
from .video import open_video_writer
from .visualization import CourtVisualizer, VisualizationConfig


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="v3", help="Checkpoint path or bundled model name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--decoder", choices=("auto", "spatial", "cuda"), default="auto")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-ransac-max-iters", type=int, default=128)
    parser.add_argument(
        "--anchor-solver",
        choices=(
            "hybrid",
            "ransac",
            "rho",
            "usac_default",
            "usac_accurate",
            "usac_magsac",
            "usac_prosac",
        ),
        default="hybrid",
    )
    parser.add_argument("--minimum-anchor-inlier-ratio", type=float, default=0.4)


def _model(args: argparse.Namespace) -> CourtLineModel:
    return CourtLineModel(
        args.model,
        config=InferenceConfig(
            device=args.device,
            image_size=args.imgsz,
            confidence=args.conf,
            top_k=args.top_k,
            half=args.half,
            fuse=args.fuse,
            decoder=args.decoder,
            anchor_ransac_max_iters=args.anchor_ransac_max_iters,
            anchor_solver=args.anchor_solver,
            minimum_anchor_inlier_ratio=args.minimum_anchor_inlier_ratio,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volley-court")
    parser.add_argument("--version", action="version", version="%(prog)s 0.3.0")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download and verify the default model")
    download.add_argument("--force", action="store_true")

    image = commands.add_parser("predict-image", help="Run typed inference on one image")
    _model_arguments(image)
    image.add_argument("source", type=Path)
    image.add_argument("output", type=Path)
    image.add_argument("--layout", action=argparse.BooleanOptionalAction, default=True)
    image.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True)
    image.add_argument("--keypoint-ids", action="store_true")
    image.add_argument("--overwrite", action="store_true")

    video = commands.add_parser("predict-video", help="Render a court overlay video")
    _model_arguments(video)
    video.add_argument("source", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument("--batch-size", type=int, default=8)
    video.add_argument("--layout-every", type=int, default=10)
    video.add_argument("--labels", action=argparse.BooleanOptionalAction, default=False)
    video.add_argument("--keypoint-ids", action="store_true")
    video.add_argument(
        "--video-codec",
        choices=("web", "mp4v"),
        default="web",
        help="web writes H.264/yuv420p/faststart and preserves source audio",
    )
    video.add_argument("--max-frames", type=int, default=0)
    video.add_argument("--overwrite", action="store_true")

    benchmark = commands.add_parser("benchmark", help="Benchmark preloaded video frames")
    _model_arguments(benchmark)
    benchmark.add_argument("source", type=Path)
    benchmark.add_argument("--frames", type=int, default=512)
    benchmark.add_argument("--batch-sizes", default="1,4,8,16")
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--layout", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--output", type=Path, required=True)
    return parser


def _predict_image(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    frame = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"could not decode image: {args.source}")
    model = _model(args)
    image = cast(Image, frame)
    result = model.predict(image, include_layout=args.layout)
    visualizer = CourtVisualizer(
        VisualizationConfig(
            show_labels=args.labels,
            show_keypoint_ids=args.keypoint_ids,
        )
    )
    rendered = visualizer.draw(image, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered):
        raise RuntimeError(f"could not write image: {args.output}")
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(json.dumps(result.to_mapping(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result.to_mapping()}, indent=2))
    return 0


def _read_batch(capture: cv2.VideoCapture, size: int, remaining: int | None) -> list[Image]:
    requested = min(size, remaining) if remaining is not None else size
    frames: list[Image] = []
    while len(frames) < requested:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cast(Image, frame))
    return frames


def _predict_video(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.layout_every < 0:
        raise ValueError("--layout-every cannot be negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(
        args.output,
        width=width,
        height=height,
        fps=fps,
        codec=args.video_codec,
        audio_source=args.source if args.video_codec == "web" else None,
    )
    model = _model(args)
    direct_layout = bool(getattr(model._model, "layout_proposals", 0))
    visualizer = CourtVisualizer(
        VisualizationConfig(
            show_labels=args.labels,
            show_keypoint_ids=args.keypoint_ids,
        )
    )
    tracker = CourtLayoutTracker()
    frames_written = 0
    inference_seconds = 0.0
    layout_status_counts: dict[str, int] = {}
    started = time.perf_counter()
    try:
        while True:
            remaining = args.max_frames - frames_written if args.max_frames else None
            if remaining is not None and remaining <= 0:
                break
            frames = _read_batch(capture, args.batch_size, remaining)
            if not frames:
                break
            results = model.predict_many(frames, include_layout=direct_layout)
            for frame, result in zip(frames, results, strict=True):
                inference_seconds += result.inference_seconds
                if direct_layout:
                    layout = tracker.update(
                        result.layout,
                        width=width,
                        height=height,
                        frame=frame,
                    )
                elif args.layout_every and frames_written % args.layout_every == 0:
                    result = model.attach_layout(result)
                    layout = tracker.update(result.layout, width=width, height=height, frame=frame)
                else:
                    layout = tracker.update(None, width=width, height=height, frame=frame)
                result = result.with_layout(layout)
                status = layout.status if layout is not None else "evidence"
                layout_status_counts[status] = layout_status_counts.get(status, 0) + 1
                writer.write(visualizer.draw(frame, result))
                frames_written += 1
    finally:
        capture.release()
        writer.close()
    wall_seconds = time.perf_counter() - started
    summary = {
        "schema": "volley-court-lines-video-v1",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "frames": frames_written,
        "input_fps": fps,
        "model_decode_fps": frames_written / max(inference_seconds, 1e-9),
        "end_to_end_fps": frames_written / max(wall_seconds, 1e-9),
        "batch_size": args.batch_size,
        "layout_every": 1 if direct_layout else args.layout_every,
        "layout_status_counts": layout_status_counts,
        "video_codec": writer.codec,
        "device": str(model.device),
        "checkpoint": str(model.checkpoint),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(",") if value)
    frames = read_video_frames(args.source, args.frames)
    report = benchmark_model(
        _model(args),
        frames,
        batch_sizes=batch_sizes,
        warmup_iterations=args.warmup,
        include_layout=args.layout,
    )
    report["source"] = str(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "download":
        path = download_model(DEFAULT_MODEL, force=args.force)
        print(json.dumps({"path": str(path), "sha256": sha256sum(path)}, indent=2))
        return 0
    if args.command == "predict-image":
        return _predict_image(args)
    if args.command == "predict-video":
        return _predict_video(args)
    if args.command == "benchmark":
        return _benchmark(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
