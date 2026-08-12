from __future__ import annotations

import time
from concurrent.futures import Executor
from typing import Any

import cv2
import numpy as np
import torch

from .dataset import ImageTransform, warp_to_square
from .decode import (
    decode_dense_votes,
    decode_dense_votes_cuda,
    decode_predictions,
    merge_duplicates,
)
from .layout import (
    CANONICAL_KEYPOINTS,
    court_homography_orientation_candidates,
    resolve_court_homography_symmetry,
)
from .model import DirectLayoutOutput, YOLO26CourtLine

ANCHOR_SOLVERS = {
    "ransac": cv2.RANSAC,
    "rho": cv2.RHO,
    "usac_default": cv2.USAC_DEFAULT,
    "usac_accurate": cv2.USAC_ACCURATE,
    "usac_magsac": cv2.USAC_MAGSAC,
    "usac_prosac": cv2.USAC_PROSAC,
}


def _anchor_solver_passes(
    anchor_solver: str, maximum_iterations: int
) -> tuple[tuple[int, int], ...]:
    if anchor_solver == "hybrid":
        return (
            (cv2.RANSAC, maximum_iterations),
            (cv2.USAC_DEFAULT, max(maximum_iterations, 512)),
        )
    method = ANCHOR_SOLVERS.get(anchor_solver)
    if method is None:
        raise ValueError(f"unsupported anchor solver: {anchor_solver}")
    return ((method, maximum_iterations),)


def decode_layout_anchors(
    prediction: DirectLayoutOutput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode Pose36 anchor cells and sub-cell offsets without copying spatial maps to CPU."""

    batch, keypoints, grid_height, grid_width = prediction.heatmap_logits.shape
    flattened = prediction.heatmap_logits.flatten(2)
    spatial_index = flattened.argmax(dim=2)
    cell_x = spatial_index.remainder(grid_width)
    cell_y = spatial_index.div(grid_width, rounding_mode="floor")
    offset = prediction.offset_logits.reshape(batch, keypoints, 2, grid_height * grid_width).gather(
        3, spatial_index[:, :, None, None].expand(-1, -1, 2, 1)
    )
    offset = offset.squeeze(3).sigmoid()
    points = torch.stack(
        (
            (cell_x + offset[:, :, 0]) / grid_width,
            (cell_y + offset[:, :, 1]) / grid_height,
        ),
        dim=2,
    )
    visibility = prediction.point_visibility_logits.sigmoid()
    validity = prediction.validity_logits.sigmoid()
    spatial_confidence = flattened.softmax(dim=2).amax(dim=2)
    return (
        points.float().cpu().numpy(),
        visibility.float().cpu().numpy(),
        spatial_confidence.float().cpu().numpy(),
        validity.float().cpu().numpy(),
        prediction.orientation_logits.softmax(dim=1).float().cpu().numpy(),
    )


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
    anchor_confidence: float = 0.0,
    anchor_ransac_threshold: float = 0.01,
    anchor_ransac_max_iters: int = 128,
    anchor_solver: str = "ransac",
) -> tuple[
    list[list[dict[str, Any]]],
    list[np.ndarray | None],
    list[dict[str, Any] | None],
    float,
]:
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
    prediction, direct_prediction = model.forward_outputs(tensor)
    selected_decoder = resolve_decoder(
        decoder,
        device_type=prediction.device.type,
        target_mode=target_mode,
        batch_size=prediction.shape[0],
    )
    if selected_decoder == "cuda":
        decoded_batches = decode_dense_votes_cuda(
            prediction,
            stride=model.stride,
            confidence=confidence,
            top_k=top_k,
            image_size=image_size,
            sample_spacing=sample_spacing,
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
    else:
        raise ValueError(
            f"unsupported decoder/target mode combination: {selected_decoder}/{target_mode}"
        )
    direct_rows = (
        decode_layout_anchors(direct_prediction) if direct_prediction is not None else None
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    restored_batches = []
    restored_heatmaps: list[np.ndarray | None] = []
    heatmaps = prediction[:, 0].sigmoid().float().cpu().numpy() if return_heatmap else None
    restored_direct: list[dict[str, Any] | None] = []
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
        restored_segments = merge_duplicates(restored)
        restored_batches.append(restored_segments)
        if direct_rows is None:
            restored_direct.append(None)
        else:
            (
                anchor_points,
                anchor_visibility,
                anchor_spatial_confidence,
                direct_validity,
                orientation_probabilities,
            ) = direct_rows
            square_points = anchor_points[batch_index] * image_size
            homogeneous_points = np.column_stack(
                (square_points, np.ones(len(square_points), dtype=np.float32))
            )
            restored_anchors = (
                (transform.inverse @ homogeneous_points.T).T[:, :2].astype(np.float32)
            )
            selected = (anchor_visibility[batch_index] >= 0.5) & (
                anchor_spatial_confidence[batch_index] >= anchor_confidence
            )
            canonical = np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32)[selected]
            observed = restored_anchors[selected]
            if anchor_solver in {"rho", "usac_prosac"}:
                quality_order = np.argsort(anchor_spatial_confidence[batch_index][selected])[::-1]
                canonical = canonical[quality_order]
                observed = observed[quality_order]
            restored_proposals: list[list[list[float]]] = []
            proposal_orientation_indices: list[int] = []
            inlier_counts: list[int] = []
            anchor_fit_errors: list[float | None] = []
            anchor_symmetry_index = -1
            available: list[int] = []
            if len(canonical) >= 4 and abs(float(cv2.contourArea(cv2.convexHull(canonical)))) > 1.0:
                for method, maximum_iterations in _anchor_solver_passes(
                    anchor_solver, anchor_ransac_max_iters
                ):
                    homography, inlier_mask = cv2.findHomography(
                        canonical,
                        observed,
                        method=method,
                        ransacReprojThreshold=max(width, height) * anchor_ransac_threshold,
                        maxIters=maximum_iterations,
                        confidence=0.99,
                    )
                    if homography is None or not np.isfinite(homography).all():
                        continue
                    outer = np.asarray(
                        [[[0.0, 0.0], [0.0, 18.0], [9.0, 18.0], [9.0, 0.0]]],
                        dtype=np.float32,
                    )
                    oriented = {
                        orientation_index: candidate
                        for candidate, orientation_index in court_homography_orientation_candidates(
                            homography
                        )
                    }
                    available = sorted(oriented)
                    selected_orientation = (
                        max(
                            available,
                            key=lambda index: float(orientation_probabilities[batch_index][index]),
                        )
                        if available
                        else -1
                    )
                    resolved = (
                        (oriented[selected_orientation], selected_orientation)
                        if selected_orientation >= 0
                        else resolve_court_homography_symmetry(homography)
                    )
                    appended = False
                    if resolved is not None:
                        candidate, orientation_index = resolved
                        corners = cv2.perspectiveTransform(outer, candidate)[0]
                        if np.isfinite(corners).all():
                            restored_proposals.append(corners.astype(float).tolist())
                            proposal_orientation_indices.append(orientation_index)
                            appended = True
                    if not appended:
                        continue
                    inlier_count = (
                        int(inlier_mask.sum()) if inlier_mask is not None else len(canonical)
                    )
                    inlier_counts.append(inlier_count)
                    projected = cv2.perspectiveTransform(canonical[None], homography)[0]
                    errors = np.linalg.norm(projected - observed, axis=1)
                    if inlier_mask is not None:
                        errors = errors[inlier_mask.reshape(-1).astype(bool)]
                    if len(errors):
                        anchor_fit_errors.append(float(np.median(errors) / max(width, height, 1)))
                    else:
                        anchor_fit_errors.append(None)
                    if anchor_solver == "hybrid" and method == cv2.RANSAC:
                        candidate_corners = np.asarray(restored_proposals[-1], dtype=np.float32)
                        area_ratio = abs(float(cv2.contourArea(candidate_corners))) / max(
                            float(width * height), 1.0
                        )
                        normalized = candidate_corners / np.asarray(
                            [max(width, 1), max(height, 1)], dtype=np.float32
                        )
                        degenerate = (
                            area_ratio < 0.01
                            or not cv2.isContourConvex(candidate_corners)
                            or float(np.max(np.abs(normalized))) > 3.0
                        )
                        if not degenerate:
                            break
            restored_direct.append(
                {
                    "corner_proposals": restored_proposals,
                    "proposal_probabilities": [1.0] * len(restored_proposals),
                    "proposal_orientation_indices": proposal_orientation_indices,
                    "validity_probability": float(direct_validity[batch_index]),
                    "anchor_count": int(selected.sum()),
                    "anchor_inlier_count": inlier_counts[0] if inlier_counts else 0,
                    "anchor_inlier_counts": inlier_counts,
                    "anchor_fit_error": anchor_fit_errors[0] if anchor_fit_errors else None,
                    "anchor_fit_errors": anchor_fit_errors,
                    "anchor_symmetry_index": anchor_symmetry_index,
                    "orientation_probabilities": orientation_probabilities[batch_index].tolist(),
                    "orientation_margin": (
                        float(
                            np.sort(orientation_probabilities[batch_index][available])[-1]
                            - np.sort(orientation_probabilities[batch_index][available])[-2]
                        )
                        if len(available) >= 2
                        else 1.0
                    ),
                }
            )
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
    return restored_batches, restored_heatmaps, restored_direct, elapsed


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
    anchor_confidence: float = 0.0,
    anchor_ransac_threshold: float = 0.01,
    anchor_ransac_max_iters: int = 128,
    anchor_solver: str = "ransac",
) -> tuple[
    list[list[dict[str, Any]]],
    list[np.ndarray | None],
    list[dict[str, Any] | None],
    float,
]:
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
        anchor_confidence=anchor_confidence,
        anchor_ransac_threshold=anchor_ransac_threshold,
        anchor_ransac_max_iters=anchor_ransac_max_iters,
        anchor_solver=anchor_solver,
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
    anchor_confidence: float = 0.0,
    anchor_ransac_threshold: float = 0.01,
    anchor_ransac_max_iters: int = 128,
    anchor_solver: str = "ransac",
) -> tuple[list[dict[str, Any]], np.ndarray | None, dict[str, Any] | None, float]:
    segments, heatmaps, direct, elapsed = infer_frames(
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
        anchor_confidence=anchor_confidence,
        anchor_ransac_threshold=anchor_ransac_threshold,
        anchor_ransac_max_iters=anchor_ransac_max_iters,
        anchor_solver=anchor_solver,
    )
    return segments[0], heatmaps[0], direct[0], elapsed


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
