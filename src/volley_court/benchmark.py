from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch

from .api import CourtLineModel
from .types import Image


@dataclass(frozen=True, slots=True)
class BatchBenchmark:
    batch_size: int
    frames: int
    batches: int
    core_fps: float
    pipeline_fps: float
    mean_frame_latency_ms: float
    p50_batch_latency_ms: float
    p95_batch_latency_ms: float
    peak_cuda_memory_bytes: int | None


def read_video_frames(source: str | Path, limit: int) -> list[Image]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open benchmark source: {source}")
    frames: list[Image] = []
    try:
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cast(Image, frame))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"benchmark source contains no readable frames: {source}")
    return frames


def benchmark_model(
    model: CourtLineModel,
    frames: list[Image],
    *,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
    warmup_iterations: int = 3,
    include_layout: bool = False,
) -> dict[str, Any]:
    rows: list[BatchBenchmark] = []
    for batch_size in batch_sizes:
        if batch_size < 1:
            raise ValueError("batch sizes must be positive")
        warmup_frames = frames[: min(batch_size, len(frames))]
        for _ in range(warmup_iterations):
            model.predict_many(warmup_frames, include_layout=include_layout)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
        batch_latencies: list[float] = []
        core_seconds = 0.0
        processed = 0
        started = time.perf_counter()
        for offset in range(0, len(frames), batch_size):
            batch = frames[offset : offset + batch_size]
            batch_started = time.perf_counter()
            results = model.predict_many(batch, include_layout=include_layout)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            batch_latencies.append(time.perf_counter() - batch_started)
            core_seconds += sum(result.inference_seconds for result in results)
            processed += len(results)
        wall_seconds = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda"
            else None
        )
        rows.append(
            BatchBenchmark(
                batch_size=batch_size,
                frames=processed,
                batches=len(batch_latencies),
                core_fps=processed / max(core_seconds, 1e-9),
                pipeline_fps=processed / max(wall_seconds, 1e-9),
                mean_frame_latency_ms=1000.0 * wall_seconds / max(processed, 1),
                p50_batch_latency_ms=1000.0 * float(np.percentile(batch_latencies, 50)),
                p95_batch_latency_ms=1000.0 * float(np.percentile(batch_latencies, 95)),
                peak_cuda_memory_bytes=peak_memory,
            )
        )
    gpu_name = torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None
    return {
        "schema": "volley-court-lines-benchmark-v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(model.device),
        "gpu": gpu_name,
        "checkpoint": str(model.checkpoint),
        "checkpoint_bytes": model.checkpoint_bytes,
        "parameters": model.parameter_count,
        "estimated_gflops_640": 9.16,
        "image_size": model.config.image_size,
        "half": model.config.half,
        "fused": model.config.fuse,
        "decoder": model.config.decoder,
        "include_layout": include_layout,
        "anchor_ransac_max_iters": model.config.anchor_ransac_max_iters,
        "anchor_solver": model.config.anchor_solver,
        "results": [asdict(row) for row in rows],
    }
