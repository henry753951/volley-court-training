from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .geometry import clip_segment_to_image


def decode_predictions(
    prediction: torch.Tensor,
    *,
    stride: int = 4,
    confidence: float = 0.25,
    top_k: int = 12,
    image_size: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Decode identity-free finite segments after 3x3 center NMS."""

    if prediction.ndim != 4 or prediction.shape[1] != 6:
        raise ValueError(f"expected [B,6,H,W], got {tuple(prediction.shape)}")
    batch, _channels, grid_height, grid_width = prediction.shape
    canvas_width = image_size or grid_width * stride
    canvas_height = image_size or grid_height * stride
    diagonal = math.hypot(canvas_width, canvas_height)
    heatmap = prediction[:, 0:1].sigmoid()
    local_maximum = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = heatmap * heatmap.eq(local_maximum)
    k = min(top_k, grid_height * grid_width)
    scores, indices = torch.topk(peaks.flatten(1), k=k, dim=1)
    offset = prediction[:, 1:3].sigmoid()
    orientation = F.normalize(prediction[:, 3:5], dim=1, eps=1e-6)
    half_length = 0.5 * prediction[:, 5:6].sigmoid() * diagonal
    decoded: list[list[dict[str, Any]]] = []
    for batch_index in range(batch):
        rows: list[dict[str, Any]] = []
        for score, flat_index in zip(scores[batch_index], indices[batch_index], strict=True):
            score_value = float(score)
            if score_value < confidence:
                continue
            flat = int(flat_index)
            cell_y, cell_x = divmod(flat, grid_width)
            center_x = (cell_x + float(offset[batch_index, 0, cell_y, cell_x])) * stride
            center_y = (cell_y + float(offset[batch_index, 1, cell_y, cell_x])) * stride
            cos2 = float(orientation[batch_index, 0, cell_y, cell_x])
            sin2 = float(orientation[batch_index, 1, cell_y, cell_x])
            if abs(cos2) + abs(sin2) < 1e-6:
                cos2, sin2 = 1.0, 0.0
            theta = 0.5 * math.atan2(sin2, cos2)
            support = float(half_length[batch_index, 0, cell_y, cell_x])
            dx, dy = support * math.cos(theta), support * math.sin(theta)
            segment = clip_segment_to_image(
                (center_x - dx, center_y - dy, center_x + dx, center_y + dy),
                canvas_width,
                canvas_height,
            )
            if segment is None:
                continue
            rows.append(
                {
                    "class": "court_line",
                    "score": score_value,
                    "center": [center_x, center_y],
                    "orientation": [cos2, sin2],
                    "half_length": support,
                    "segment": list(segment),
                    "cell": [cell_x, cell_y],
                }
            )
        decoded.append(rows)
    return decoded


def decode_dense_votes(
    prediction: torch.Tensor,
    *,
    stride: int = 4,
    confidence: float = 0.25,
    top_k: int = 256,
    image_size: int | None = None,
    sample_spacing: float = 16.0,
    angle_degrees: float = 12.0,
    perpendicular_distance_px: float = 7.0,
    max_gap_factor: float = 3.0,
    minimum_votes: int = 2,
    spatial_hash: bool = True,
) -> list[list[dict[str, Any]]]:
    """Decode dense centerline votes, group them, then robustly reconstruct finite segments."""

    if prediction.ndim != 4 or prediction.shape[1] not in {5, 8, 15}:
        raise ValueError(
            f"expected [B,5,H,W], [B,8,H,W], or [B,15,H,W], got {tuple(prediction.shape)}"
        )
    # Match the validated decoder's numerical behavior: the model may run in FP16/BF16,
    # but sigmoid, local-maximum comparison, offsets, and semantic probabilities use FP32.
    if prediction.dtype in {torch.float16, torch.bfloat16}:
        prediction = prediction.float()
    batch, _channels, grid_height, grid_width = prediction.shape
    canvas_width = image_size or grid_width * stride
    canvas_height = image_size or grid_height * stride
    heatmap = prediction[:, 0:1].sigmoid()
    context_mode = prediction.shape[1] in {8, 15}
    semantic_mode = prediction.shape[1] == 15
    family_probability = prediction[:, 5:7].softmax(dim=1) if context_mode else None
    identity_probability = prediction[:, 7:14].softmax(dim=1) if semantic_mode else None
    roi_probability = (
        prediction[:, 14:15].sigmoid()
        if semantic_mode
        else prediction[:, 7:8].sigmoid()
        if context_mode
        else None
    )
    peaks = heatmap * heatmap.eq(F.max_pool2d(heatmap, 3, stride=1, padding=1))
    k = min(top_k, grid_height * grid_width)
    scores, indices = torch.topk(peaks.flatten(1), k=k, dim=1)
    offset = prediction[:, 1:3].sigmoid()
    orientation = F.normalize(prediction[:, 3:5], dim=1, eps=1e-6)
    gather_indices = indices.unsqueeze(1)

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            tensor.flatten(2),
            2,
            gather_indices.expand(-1, tensor.shape[1], -1),
        ).transpose(1, 2)

    components = [
        scores.unsqueeze(-1),
        indices.to(dtype=prediction.dtype).unsqueeze(-1),
        gather(offset),
        gather(orientation),
    ]
    if family_probability is not None:
        components.append(gather(family_probability))
        if identity_probability is not None:
            components.append(gather(identity_probability))
        components.append(gather(roi_probability))
    # A single device-to-host transfer avoids hundreds of scalar CUDA synchronizations.
    gathered = torch.cat(components, dim=-1).float().cpu().numpy()
    maximum_gap = max_gap_factor * sample_spacing
    maximum_angle_cosine = math.cos(math.radians(angle_degrees))
    decoded: list[list[dict[str, Any]]] = []
    for batch_index in range(batch):
        votes = []
        for values in gathered[batch_index]:
            score_value = float(values[0])
            if score_value < confidence:
                continue
            cell_y, cell_x = divmod(int(values[1]), grid_width)
            x = (cell_x + float(values[2])) * stride
            y = (cell_y + float(values[3])) * stride
            cos2 = float(values[4])
            sin2 = float(values[5])
            theta = 0.5 * math.atan2(sin2, cos2)
            if family_probability is not None:
                probabilities = values[6:8]
                family_index = int(np.argmax(probabilities))
                family_score = float(probabilities[family_index])
                family = "vertical" if family_index == 0 else "horizontal"
                family_probabilities = (float(probabilities[0]), float(probabilities[1]))
                if identity_probability is not None:
                    identity_probabilities = values[8:15]
                    line_identity = int(np.argmax(identity_probabilities))
                    identity_score = float(identity_probabilities[line_identity])
                    identity_values = tuple(float(value) for value in identity_probabilities)
                    roi_score = float(values[15])
                else:
                    line_identity = None
                    identity_score = 0.0
                    identity_values = tuple()
                    roi_score = float(values[8])
            else:
                family_index = -1
                family_score = 0.0
                family = None
                family_probabilities = (0.5, 0.5)
                line_identity = None
                identity_score = 0.0
                identity_values = tuple()
                roi_score = 1.0
            votes.append(
                {
                    "point": (x, y),
                    "unit": (math.cos(theta), math.sin(theta)),
                    "orientation": (cos2, sin2),
                    "score": score_value,
                    "family_index": family_index,
                    "family": family,
                    "family_score": family_score,
                    "family_probabilities": family_probabilities,
                    "line_identity": line_identity,
                    "identity_score": identity_score,
                    "identity_probabilities": identity_values,
                    "roi_score": roi_score,
                }
            )
        parents = list(range(len(votes)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            a, b = find(first), find(second)
            if a != b:
                parents[b] = a

        buckets: dict[tuple[int, int], list[int]] = {}
        if spatial_hash:
            bucket_size = max(maximum_gap, 1.0)
            for index, vote in enumerate(votes):
                key = (
                    int(math.floor(vote["point"][0] / bucket_size)),
                    int(math.floor(vote["point"][1] / bucket_size)),
                )
                buckets.setdefault(key, []).append(index)

        for first_index, first in enumerate(votes):
            if spatial_hash:
                first_bucket = (
                    int(math.floor(first["point"][0] / bucket_size)),
                    int(math.floor(first["point"][1] / bucket_size)),
                )
                candidates = (
                    index
                    for offset_x in (-1, 0, 1)
                    for offset_y in (-1, 0, 1)
                    for index in buckets.get(
                        (first_bucket[0] + offset_x, first_bucket[1] + offset_y),
                        (),
                    )
                    if index > first_index
                )
            else:
                candidates = range(first_index + 1, len(votes))
            for second_index in candidates:
                second = votes[second_index]
                dx = second["point"][0] - first["point"][0]
                dy = second["point"][1] - first["point"][1]
                if math.hypot(dx, dy) > maximum_gap:
                    continue
                direction_similarity = abs(
                    first["unit"][0] * second["unit"][0] + first["unit"][1] * second["unit"][1]
                )
                if direction_similarity < maximum_angle_cosine:
                    continue
                first_perpendicular = abs(dx * -first["unit"][1] + dy * first["unit"][0])
                second_perpendicular = abs(dx * -second["unit"][1] + dy * second["unit"][0])
                if max(first_perpendicular, second_perpendicular) <= perpendicular_distance_px:
                    union(first_index, second_index)
        groups: dict[int, list[dict[str, Any]]] = {}
        for index, vote in enumerate(votes):
            groups.setdefault(find(index), []).append(vote)
        rows = []
        for group in groups.values():
            if len(group) < minimum_votes:
                continue
            coordinates = np.asarray([vote["point"] for vote in group], dtype=np.float64)
            weights = np.asarray([vote["score"] for vote in group], dtype=np.float64)
            family_probabilities = np.average(
                np.asarray([vote["family_probabilities"] for vote in group]),
                axis=0,
                weights=weights,
            )
            family_index = int(np.argmax(family_probabilities))
            family = "vertical" if family_index == 0 else "horizontal"
            family_score = float(family_probabilities[family_index])
            if group[0]["identity_probabilities"]:
                identity_probabilities = np.average(
                    np.asarray([vote["identity_probabilities"] for vote in group]),
                    axis=0,
                    weights=weights,
                )
                line_identity = int(np.argmax(identity_probabilities))
                identity_score = float(identity_probabilities[line_identity])
                identity_values = identity_probabilities.tolist()
            else:
                line_identity = None
                identity_score = 0.0
                identity_values = []
            roi_score = float(np.average([vote["roi_score"] for vote in group], weights=weights))
            center = np.average(coordinates, axis=0, weights=weights)
            centered = coordinates - center
            covariance = (centered * weights[:, None]).T @ centered / max(weights.sum(), 1e-9)
            _values, vectors = np.linalg.eigh(covariance)
            direction = vectors[:, -1]
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            projections = centered @ direction
            low = float(projections.min() - 0.5 * sample_spacing)
            high = float(projections.max() + 0.5 * sample_spacing)
            clipped = clip_segment_to_image(
                (*tuple(center + low * direction), *tuple(center + high * direction)),
                canvas_width,
                canvas_height,
            )
            if clipped is None:
                continue
            theta = math.atan2(float(direction[1]), float(direction[0]))
            segment_center = [
                0.5 * (clipped[0] + clipped[2]),
                0.5 * (clipped[1] + clipped[3]),
            ]
            rows.append(
                {
                    "class": "court_line",
                    "score": float(weights.mean()),
                    "center": segment_center,
                    "orientation": [math.cos(2.0 * theta), math.sin(2.0 * theta)],
                    "segment": list(clipped),
                    "vote_count": len(group),
                    "family": family,
                    "family_score": family_score,
                    "family_probabilities": family_probabilities.tolist(),
                    "line_identity": line_identity,
                    "identity_score": identity_score,
                    "identity_probabilities": identity_values,
                    "roi_score": roi_score,
                    "votes": [
                        [vote["point"][0], vote["point"][1], vote["score"]] for vote in group
                    ],
                }
            )
        decoded.append(rows)
    return decoded


def decode_dense_semantic_cuda(
    prediction: torch.Tensor,
    *,
    stride: int = 4,
    confidence: float = 0.25,
    top_k: int = 256,
    image_size: int | None = None,
    sample_spacing: float = 16.0,
    minimum_votes: int = 2,
    minimum_roi: float = 0.15,
    minimum_identity: float = 0.10,
    angle_degrees: float = 12.0,
    perpendicular_distance_px: float = 7.0,
) -> list[list[dict[str, Any]]]:
    """Decode the semantic head with one vectorized Torch filter per physical line.

    The 15-channel model already predicts a unique identity for each of the seven
    physical court lines.  That makes the CPU spatial connected-components pass
    unnecessary in the realtime path: votes are grouped by identity, filtered by
    ROI/orientation/distance, and reduced to seven fitted lines on the prediction
    device.  Only the compact per-line tensor is copied to the host.
    """

    if prediction.ndim != 4 or prediction.shape[1] != 15:
        raise ValueError(f"expected [B,15,H,W], got {tuple(prediction.shape)}")
    # Keep the quality-safe numerical contract used by decode_dense_votes.  This
    # also prevents FP16 plateaus from creating many equal local maxima.
    if prediction.dtype in {torch.float16, torch.bfloat16}:
        prediction = prediction.float()
    batch, _channels, grid_height, grid_width = prediction.shape
    canvas_width = image_size or grid_width * stride
    canvas_height = image_size or grid_height * stride
    heatmap = prediction[:, 0:1].sigmoid()
    peaks = heatmap * heatmap.eq(F.max_pool2d(heatmap, 3, stride=1, padding=1))
    k = min(top_k, grid_height * grid_width)
    scores, indices = torch.topk(peaks.flatten(1), k=k, dim=1)
    gather_indices = indices.unsqueeze(1)

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            tensor.flatten(2),
            2,
            gather_indices.expand(-1, tensor.shape[1], -1),
        ).transpose(1, 2)

    offset = gather(prediction[:, 1:3].sigmoid())
    orientation = gather(F.normalize(prediction[:, 3:5], dim=1, eps=1e-6))
    family_probability = gather(prediction[:, 5:7].softmax(dim=1))
    identity_probability = gather(prediction[:, 7:14].softmax(dim=1))
    roi_probability = gather(prediction[:, 14:15].sigmoid()).squeeze(-1)
    cell_x = (indices % grid_width).to(dtype=prediction.dtype)
    cell_y = torch.div(indices, grid_width, rounding_mode="floor").to(dtype=prediction.dtype)
    points = torch.stack(
        (
            (cell_x + offset[..., 0]) * stride,
            (cell_y + offset[..., 1]) * stride,
        ),
        dim=-1,
    )
    valid = (scores >= confidence) & (roi_probability >= minimum_roi)
    # Soft semantic assignment is important here.  The small real-data set makes
    # the correct ID frequently rank second, so argmax would discard useful court
    # evidence.  Identity probability participates in both assignment and weight.
    assigned = valid.unsqueeze(-1) & (identity_probability >= minimum_identity)
    base_weights = scores.unsqueeze(-1) * roi_probability.unsqueeze(-1) * identity_probability

    def reduce_groups(group_mask: torch.Tensor):
        weights = base_weights * group_mask
        weight_sum = weights.sum(dim=1).clamp_min(1e-9)
        centers = torch.einsum("bkl,bki->bli", weights, points) / weight_sum.unsqueeze(-1)
        axial = torch.einsum("bkl,bki->bli", weights, orientation)
        axial = F.normalize(axial, dim=-1, eps=1e-6)
        theta = 0.5 * torch.atan2(axial[..., 1], axial[..., 0])
        direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        return weights, weight_sum, centers, axial, direction

    # First pass obtains a stable semantic line.  The second pass removes votes
    # that have the right class but cannot lie on that line geometrically.
    _weights, _weight_sum, centers, axial, direction = reduce_groups(assigned)
    centered = points.unsqueeze(2) - centers.unsqueeze(1)
    perpendicular = torch.abs(
        centered[..., 0] * -direction[:, None, :, 1] + centered[..., 1] * direction[:, None, :, 0]
    )
    axial_similarity = torch.einsum("bki,bli->bkl", orientation, axial)
    refined = (
        assigned
        & (perpendicular <= perpendicular_distance_px)
        & (axial_similarity >= math.cos(math.radians(2.0 * angle_degrees)))
    )
    weights, weight_sum, centers, axial, direction = reduce_groups(refined)
    vote_count = refined.sum(dim=1)
    centered = points.unsqueeze(2) - centers.unsqueeze(1)
    projection = torch.einsum("bkli,bli->bkl", centered, direction)
    positive_infinity = torch.full_like(projection, torch.inf)
    negative_infinity = torch.full_like(projection, -torch.inf)
    low = torch.where(refined, projection, positive_infinity).amin(dim=1)
    high = torch.where(refined, projection, negative_infinity).amax(dim=1)
    low = low - 0.5 * sample_spacing
    high = high + 0.5 * sample_spacing
    first = centers + low.unsqueeze(-1) * direction
    second = centers + high.unsqueeze(-1) * direction
    count_float = vote_count.to(dtype=prediction.dtype).clamp_min(1.0)
    mean_score = (scores.unsqueeze(-1) * refined).sum(dim=1) / count_float
    mean_family = torch.einsum("bkl,bki->bli", weights, family_probability) / weight_sum.unsqueeze(
        -1
    )
    mean_identity = torch.einsum(
        "bkl,bki->bli", weights, identity_probability
    ) / weight_sum.unsqueeze(-1)
    mean_roi = (weights * roi_probability.unsqueeze(-1)).sum(dim=1) / weight_sum

    compact = (
        torch.cat(
            (
                first,
                second,
                centers,
                axial,
                mean_score.unsqueeze(-1),
                count_float.unsqueeze(-1),
                mean_family,
                mean_identity,
                mean_roi.unsqueeze(-1),
            ),
            dim=-1,
        )
        .float()
        .cpu()
        .numpy()
    )
    counts = vote_count.cpu().numpy()
    decoded: list[list[dict[str, Any]]] = []
    for batch_index in range(batch):
        rows: list[dict[str, Any]] = []
        for line_identity in range(7):
            if int(counts[batch_index, line_identity]) < minimum_votes:
                continue
            values = compact[batch_index, line_identity]
            clipped = clip_segment_to_image(
                values[0:4],
                canvas_width,
                canvas_height,
            )
            if clipped is None:
                continue
            family_probabilities = values[10:12]
            family_index = int(np.argmax(family_probabilities))
            identity_probabilities = values[12:19]
            rows.append(
                {
                    "class": "court_line",
                    "score": float(values[8]),
                    "center": [
                        0.5 * (clipped[0] + clipped[2]),
                        0.5 * (clipped[1] + clipped[3]),
                    ],
                    "orientation": [float(values[6]), float(values[7])],
                    "segment": list(clipped),
                    "vote_count": int(counts[batch_index, line_identity]),
                    "family": "vertical" if family_index == 0 else "horizontal",
                    "family_score": float(family_probabilities[family_index]),
                    "family_probabilities": family_probabilities.tolist(),
                    "line_identity": line_identity,
                    "identity_score": float(identity_probabilities[line_identity]),
                    "identity_probabilities": identity_probabilities.tolist(),
                    "roi_score": float(values[19]),
                    "decoder": "semantic_cuda",
                }
            )
        decoded.append(rows)
    return decoded


def decode_dense_votes_cuda(
    prediction: torch.Tensor,
    *,
    stride: int = 4,
    confidence: float = 0.25,
    top_k: int = 256,
    image_size: int | None = None,
    sample_spacing: float = 16.0,
    angle_degrees: float = 12.0,
    perpendicular_distance_px: float = 7.0,
    max_gap_factor: float = 3.0,
    minimum_votes: int = 2,
    propagation_steps: int = 32,
) -> list[list[dict[str, Any]]]:
    """Run the quality-safe spatial vote grouping on the prediction device.

    This is the Torch/CUDA equivalent of ``decode_dense_votes``: it builds the
    same geometric adjacency graph, labels connected components, and fits every
    retained line before copying a compact result tensor to the host.
    """

    if prediction.ndim != 4 or prediction.shape[1] != 15:
        raise ValueError(f"expected [B,15,H,W], got {tuple(prediction.shape)}")
    if prediction.dtype in {torch.float16, torch.bfloat16}:
        prediction = prediction.float()
    batch, _channels, grid_height, grid_width = prediction.shape
    canvas_width = image_size or grid_width * stride
    canvas_height = image_size or grid_height * stride
    heatmap = prediction[:, 0:1].sigmoid()
    peaks = heatmap * heatmap.eq(F.max_pool2d(heatmap, 3, stride=1, padding=1))
    k = min(top_k, grid_height * grid_width)
    scores, indices = torch.topk(peaks.flatten(1), k=k, dim=1)
    gather_indices = indices.unsqueeze(1)

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            tensor.flatten(2),
            2,
            gather_indices.expand(-1, tensor.shape[1], -1),
        ).transpose(1, 2)

    offset = gather(prediction[:, 1:3].sigmoid())
    orientation = gather(F.normalize(prediction[:, 3:5], dim=1, eps=1e-6))
    family_probability = gather(prediction[:, 5:7].softmax(dim=1))
    identity_probability = gather(prediction[:, 7:14].softmax(dim=1))
    roi_probability = gather(prediction[:, 14:15].sigmoid()).squeeze(-1)
    cell_x = (indices % grid_width).to(dtype=prediction.dtype)
    cell_y = torch.div(indices, grid_width, rounding_mode="floor").to(dtype=prediction.dtype)
    points = torch.stack(
        (
            (cell_x + offset[..., 0]) * stride,
            (cell_y + offset[..., 1]) * stride,
        ),
        dim=-1,
    )
    theta = 0.5 * torch.atan2(orientation[..., 1], orientation[..., 0])
    unit = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    valid = scores >= confidence

    # [B,K,K,2] contains p_j - p_i, matching the exhaustive CPU predicate.
    delta = points[:, None, :, :] - points[:, :, None, :]
    maximum_gap = max_gap_factor * sample_spacing
    adjacency = (
        valid[:, :, None]
        & valid[:, None, :]
        & ((delta * delta).sum(dim=-1) <= maximum_gap * maximum_gap)
        & (
            torch.abs(torch.matmul(unit, unit.transpose(1, 2)))
            >= math.cos(math.radians(angle_degrees))
        )
        & (
            torch.abs(delta[..., 0] * -unit[:, :, None, 1] + delta[..., 1] * unit[:, :, None, 0])
            <= perpendicular_distance_px
        )
        & (
            torch.abs(delta[..., 0] * -unit[:, None, :, 1] + delta[..., 1] * unit[:, None, :, 0])
            <= perpendicular_distance_px
        )
    )
    labels = torch.arange(k, device=prediction.device).expand(batch, -1)
    sentinel = torch.full(
        (batch, k, k),
        k,
        device=prediction.device,
        dtype=labels.dtype,
    )
    for _ in range(propagation_steps):
        neighbor_labels = labels[:, None, :].expand(-1, k, -1)
        labels = torch.where(adjacency, neighbor_labels, sentinel).amin(dim=-1)

    group_mask = F.one_hot(labels, num_classes=k + 1)[..., :k].bool()
    group_mask &= valid.unsqueeze(-1)
    vote_count = group_mask.sum(dim=1)
    weights = scores.unsqueeze(-1) * group_mask
    weight_sum = weights.sum(dim=1).clamp_min(1e-9)
    centers = torch.einsum("bkl,bki->bli", weights, points) / weight_sum.unsqueeze(-1)
    centered = points.unsqueeze(2) - centers.unsqueeze(1)
    covariance_xx = (weights * centered[..., 0].square()).sum(dim=1) / weight_sum
    covariance_xy = (weights * centered[..., 0] * centered[..., 1]).sum(dim=1) / weight_sum
    covariance_yy = (weights * centered[..., 1].square()).sum(dim=1) / weight_sum
    fit_theta = 0.5 * torch.atan2(
        2.0 * covariance_xy,
        covariance_xx - covariance_yy,
    )
    direction = torch.stack((torch.cos(fit_theta), torch.sin(fit_theta)), dim=-1)
    fitted_orientation = torch.stack(
        (torch.cos(2.0 * fit_theta), torch.sin(2.0 * fit_theta)),
        dim=-1,
    )
    projection = torch.einsum("bkli,bli->bkl", centered, direction)
    low = torch.where(group_mask, projection, torch.inf).amin(dim=1)
    high = torch.where(group_mask, projection, -torch.inf).amax(dim=1)
    first = centers + (low - 0.5 * sample_spacing).unsqueeze(-1) * direction
    second = centers + (high + 0.5 * sample_spacing).unsqueeze(-1) * direction
    count_float = vote_count.to(dtype=prediction.dtype).clamp_min(1.0)
    mean_score = (scores.unsqueeze(-1) * group_mask).sum(dim=1) / count_float
    mean_family = torch.einsum("bkl,bki->bli", weights, family_probability) / weight_sum.unsqueeze(
        -1
    )
    mean_identity = torch.einsum(
        "bkl,bki->bli", weights, identity_probability
    ) / weight_sum.unsqueeze(-1)
    mean_roi = (weights * roi_probability.unsqueeze(-1)).sum(dim=1) / weight_sum
    compact = (
        torch.cat(
            (
                first,
                second,
                centers,
                fitted_orientation,
                mean_score.unsqueeze(-1),
                count_float.unsqueeze(-1),
                mean_family,
                mean_identity,
                mean_roi.unsqueeze(-1),
            ),
            dim=-1,
        )
        .float()
        .cpu()
        .numpy()
    )
    counts = vote_count.cpu().numpy()
    decoded: list[list[dict[str, Any]]] = []
    for batch_index in range(batch):
        rows: list[dict[str, Any]] = []
        for group_index in range(k):
            if int(counts[batch_index, group_index]) < minimum_votes:
                continue
            values = compact[batch_index, group_index]
            clipped = clip_segment_to_image(
                values[0:4],
                canvas_width,
                canvas_height,
            )
            if clipped is None:
                continue
            family_probabilities = values[10:12]
            family_index = int(np.argmax(family_probabilities))
            identity_probabilities = values[12:19]
            line_identity = int(np.argmax(identity_probabilities))
            rows.append(
                {
                    "class": "court_line",
                    "score": float(values[8]),
                    "center": [
                        0.5 * (clipped[0] + clipped[2]),
                        0.5 * (clipped[1] + clipped[3]),
                    ],
                    "orientation": [float(values[6]), float(values[7])],
                    "segment": list(clipped),
                    "vote_count": int(counts[batch_index, group_index]),
                    "family": "vertical" if family_index == 0 else "horizontal",
                    "family_score": float(family_probabilities[family_index]),
                    "family_probabilities": family_probabilities.tolist(),
                    "line_identity": line_identity,
                    "identity_score": float(identity_probabilities[line_identity]),
                    "identity_probabilities": identity_probabilities.tolist(),
                    "roi_score": float(values[19]),
                    "decoder": "cuda",
                }
            )
        decoded.append(rows)
    return decoded


def _duplicate(
    first: dict[str, Any],
    second: dict[str, Any],
    angle_degrees: float,
    distance: float,
    overlap: float,
) -> bool:
    a = first["segment"]
    b = second["segment"]
    adx, ady = a[2] - a[0], a[3] - a[1]
    bdx, bdy = b[2] - b[0], b[3] - b[1]
    alength, blength = math.hypot(adx, ady), math.hypot(bdx, bdy)
    if alength < 1e-6 or blength < 1e-6:
        return False
    au = (adx / alength, ady / alength)
    bu = (bdx / blength, bdy / blength)
    angle = math.degrees(math.acos(min(1.0, abs(au[0] * bu[0] + au[1] * bu[1]))))
    if angle > angle_degrees:
        return False
    ac = ((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5)
    bc = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
    delta = (bc[0] - ac[0], bc[1] - ac[1])
    perpendicular = abs(delta[0] * -au[1] + delta[1] * au[0])
    if perpendicular > distance:
        return False
    projected = delta[0] * au[0] + delta[1] * au[1]
    a_interval = (-0.5 * alength, 0.5 * alength)
    b_interval = (projected - 0.5 * blength, projected + 0.5 * blength)
    intersection = max(0.0, min(a_interval[1], b_interval[1]) - max(a_interval[0], b_interval[0]))
    return intersection / max(1e-6, min(alength, blength)) >= overlap


def merge_duplicates(
    segments: Sequence[dict[str, Any]],
    *,
    angle_degrees: float = 5.0,
    perpendicular_distance_px: float = 6.0,
    overlap_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for candidate in sorted(segments, key=lambda row: float(row["score"]), reverse=True):
        if any(
            _duplicate(
                candidate,
                existing,
                angle_degrees,
                perpendicular_distance_px,
                overlap_ratio,
            )
            for existing in retained
        ):
            continue
        retained.append(candidate)
    return retained
