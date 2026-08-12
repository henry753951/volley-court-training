from __future__ import annotations

import time
from concurrent.futures import Executor
from typing import Any

import cv2
import numpy as np
import torch

from .dataset import ImageTransform, warp_to_square
from .decode import (
    assign_semantic_line_identities,
    decode_dense_votes,
    decode_dense_votes_cuda,
    decode_predictions,
    merge_duplicates,
)
from .model import YOLO26CourtLine


def resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_decoder(
    configured: str,
    *,
    device_type: str,
    target_mode: str,
    batch_size: int,
) -> str:
    if configured == "auto":
        return (
            "cuda"
            if device_type == "cuda" and batch_size >= 2 and target_mode == "dense_semantic"
            else "spatial"
        )
    if configured == "cuda" and target_mode != "dense_semantic":
        raise ValueError("cuda decoder requires a dense_semantic checkpoint")
    if configured not in {"spatial", "cuda"}:
        raise ValueError(f"unknown decoder: {configured}")
    return configured


def prepare_frame(frame: np.ndarray, image_size: int) -> tuple[torch.Tensor, ImageTransform]:
    warped, transform = warp_to_square(frame, image_size, training=False)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
    return tensor.unsqueeze(0), transform


def prepare_frames(
    frames: list[np.ndarray],
    image_size: int,
    preprocess_executor: Executor | None = None,
) -> list[tuple[torch.Tensor, ImageTransform]]:
    if not frames:
        return []
    if preprocess_executor is None:
        return [prepare_frame(frame, image_size) for frame in frames]
    return list(
        preprocess_executor.map(
            lambda frame: prepare_frame(frame, image_size),
            frames,
        )
    )


@torch.inference_mode()
def infer_prepared_frames(
    model: YOLO26CourtLine,
    frames: list[np.ndarray],
    prepared: list[tuple[torch.Tensor, ImageTransform]],
    device: torch.device,
    *,
    image_size: int,
    confidence: float,
    top_k: int,
    half: bool = False,
    target_mode: str = "center",
    sample_spacing: float = 16.0,
    return_heatmap: bool = True,
    decoder: str = "auto",
) -> tuple[list[list[dict[str, Any]]], list[np.ndarray | None], float]:
    if not frames:
        raise ValueError("infer_prepared_frames requires at least one frame")
    if len(prepared) != len(frames):
        raise ValueError("prepared tensor count must match frame count")
    tensor = torch.cat([row[0] for row in prepared], dim=0).to(device, non_blocking=True)
    if half and device.type == "cuda":
        tensor = tensor.half()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model(tensor)
    selected_decoder = resolve_decoder(
        decoder,
        device_type=prediction.device.type,
        target_mode=target_mode,
        batch_size=prediction.shape[0],
    )
    if selected_decoder == "cuda":
        decoded_batches = (
            [
                assign_semantic_line_identities(rows, image_size, image_size)
                for rows in decode_dense_votes_cuda(
                    prediction,
                    stride=model.stride,
                    confidence=confidence,
                    top_k=top_k,
                    image_size=image_size,
                    sample_spacing=sample_spacing,
                )
            ]
            if target_mode == "dense_semantic"
            else decode_dense_votes_cuda(
                prediction,
                stride=model.stride,
                confidence=confidence,
                top_k=top_k,
                image_size=image_size,
                sample_spacing=sample_spacing,
            )
        )
    elif selected_decoder == "spatial" and target_mode in {
        "dense_votes",
        "dense_context",
        "dense_semantic",
    }:
        decoded_batches = decode_dense_votes(
            prediction,
            stride=model.stride,
            confidence=confidence,
            top_k=top_k,
            image_size=image_size,
            sample_spacing=sample_spacing,
        )
    elif selected_decoder == "spatial":
        decoded_batches = decode_predictions(
            prediction,
            stride=model.stride,
            confidence=confidence,
            top_k=top_k,
            image_size=image_size,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    restored_batches = []
    restored_heatmaps: list[np.ndarray | None] = []
    heatmaps = prediction[:, 0].sigmoid().float().cpu().numpy() if return_heatmap else None
    for batch_index, (frame, (_tensor, transform), decoded) in enumerate(
        zip(frames, prepared, decoded_batches, strict=True)
    ):
        height, width = frame.shape[:2]
        restored = []
        for row in decoded:
            segment = transform.restore_segment(row["segment"], width, height)
            if segment is None:
                continue
            updated = dict(row)
            updated["segment"] = list(segment)
            updated["center"] = [
                0.5 * (segment[0] + segment[2]),
                0.5 * (segment[1] + segment[3]),
            ]
            if row.get("votes"):
                updated["votes"] = [
                    [*transform.restore_point(vote[:2]), vote[2]] for vote in row["votes"]
                ]
            restored.append(updated)
        restored_batches.append(merge_duplicates(restored))
        restored_heatmap = None
        if heatmaps is not None:
            heatmap_canvas = cv2.resize(
                heatmaps[batch_index],
                (image_size, image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            restored_heatmap = cv2.warpPerspective(
                heatmap_canvas,
                transform.inverse,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        restored_heatmaps.append(restored_heatmap)
    return restored_batches, restored_heatmaps, elapsed


@torch.inference_mode()
def infer_frames(
    model: YOLO26CourtLine,
    frames: list[np.ndarray],
    device: torch.device,
    *,
    image_size: int,
    confidence: float,
    top_k: int,
    half: bool = False,
    target_mode: str = "center",
    sample_spacing: float = 16.0,
    return_heatmap: bool = True,
    preprocess_executor: Executor | None = None,
    decoder: str = "auto",
) -> tuple[list[list[dict[str, Any]]], list[np.ndarray | None], float]:
    prepared = prepare_frames(frames, image_size, preprocess_executor)
    return infer_prepared_frames(
        model,
        frames,
        prepared,
        device,
        image_size=image_size,
        confidence=confidence,
        top_k=top_k,
        half=half,
        target_mode=target_mode,
        sample_spacing=sample_spacing,
        return_heatmap=return_heatmap,
        decoder=decoder,
    )


@torch.inference_mode()
def infer_frame(
    model: YOLO26CourtLine,
    frame: np.ndarray,
    device: torch.device,
    *,
    image_size: int,
    confidence: float,
    top_k: int,
    half: bool = False,
    target_mode: str = "center",
    sample_spacing: float = 16.0,
    return_heatmap: bool = True,
    decoder: str = "auto",
) -> tuple[list[dict[str, Any]], np.ndarray | None, float]:
    segments, heatmaps, elapsed = infer_frames(
        model,
        [frame],
        device,
        image_size=image_size,
        confidence=confidence,
        top_k=top_k,
        half=half,
        target_mode=target_mode,
        sample_spacing=sample_spacing,
        return_heatmap=return_heatmap,
        preprocess_executor=None,
        decoder=decoder,
    )
    return segments[0], heatmaps[0], elapsed


def draw_segments(frame: np.ndarray, segments: list[dict[str, Any]]) -> np.ndarray:
    output = frame.copy()
    for row in segments:
        x1, y1, x2, y2 = (int(round(value)) for value in row["segment"])
        center = (int(round(row["center"][0])), int(round(row["center"][1])))
        # Visualization thickness only; the prediction remains a zero-width segment.
        cv2.line(output, (x1, y1), (x2, y2), (0, 230, 255), 2, cv2.LINE_AA)
        for vote in row.get("votes", []):
            cv2.circle(
                output,
                (int(round(vote[0])), int(round(vote[1]))),
                2,
                (80, 255, 220),
                -1,
                cv2.LINE_AA,
            )
        cv2.drawMarker(output, center, (255, 80, 0), cv2.MARKER_CROSS, 9, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{row['score']:.2f}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(
        np.clip(resized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    return cv2.addWeighted(frame, 0.55, colored, 0.45, 0.0)
