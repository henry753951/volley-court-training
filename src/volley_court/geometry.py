from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .topology import LineGroup


@dataclass(frozen=True)
class PointSample:
    index: int
    x: float
    y: float
    visibility: int


@dataclass(frozen=True)
class FittedLineSegment:
    valid: bool
    reason: str | None
    segment: tuple[float, float, float, float] | None
    normalized_segment: tuple[float, float, float, float] | None
    line: tuple[float, float, float] | None
    center: tuple[float, float] | None
    normalized_center: tuple[float, float] | None
    orientation: tuple[float, float] | None
    normalized_half_length: float | None
    fit_rms_distance_px: float | None
    fit_max_distance_px: float | None
    source_indices: tuple[int, ...]
    source_visibility: tuple[int, ...]
    valid_source_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "segment": list(self.segment) if self.segment is not None else None,
            "segment_normalized": (
                list(self.normalized_segment) if self.normalized_segment is not None else None
            ),
            "line": list(self.line) if self.line is not None else None,
            "center": list(self.center) if self.center is not None else None,
            "center_normalized": (
                list(self.normalized_center) if self.normalized_center is not None else None
            ),
            "orientation": list(self.orientation) if self.orientation is not None else None,
            "half_length_normalized": self.normalized_half_length,
            "fit_rms_distance_px": self.fit_rms_distance_px,
            "fit_max_distance_px": self.fit_max_distance_px,
            "source_indices": list(self.source_indices),
            "visibility": list(self.source_visibility),
            "valid_source_indices": list(self.valid_source_indices),
        }


def parse_yolo_pose_line(line: str, keypoint_count: int = 36) -> list[PointSample]:
    values = line.split()
    expected = 5 + keypoint_count * 3
    if len(values) != expected:
        raise ValueError(f"expected {expected} YOLO pose fields, got {len(values)}")
    points: list[PointSample] = []
    for index in range(keypoint_count):
        offset = 5 + index * 3
        x, y, visibility = (
            float(values[offset]),
            float(values[offset + 1]),
            int(float(values[offset + 2])),
        )
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"non-finite keypoint coordinate at index {index}")
        points.append(PointSample(index, x, y, visibility))
    return points


def _invalid(group: LineGroup, points: Sequence[PointSample], reason: str) -> FittedLineSegment:
    by_index = {point.index: point for point in points}
    return FittedLineSegment(
        valid=False,
        reason=reason,
        segment=None,
        normalized_segment=None,
        line=None,
        center=None,
        normalized_center=None,
        orientation=None,
        normalized_half_length=None,
        fit_rms_distance_px=None,
        fit_max_distance_px=None,
        source_indices=group.point_indices,
        source_visibility=tuple(
            by_index[index].visibility if index in by_index else 0 for index in group.point_indices
        ),
        valid_source_indices=(),
    )


def _robust_tls(coordinates: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = weights.astype(np.float64, copy=True)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    center = coordinates.mean(axis=0)
    for _ in range(4):
        center = np.average(coordinates, axis=0, weights=current)
        centered = coordinates - center
        covariance = (centered * current[:, None]).T @ centered / max(float(current.sum()), 1e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        normal = np.array([-direction[1], direction[0]])
        residuals = np.abs(centered @ normal)
        median = float(np.median(residuals))
        scale = max(1e-6, 1.4826 * float(np.median(np.abs(residuals - median))))
        huber = np.minimum(1.0, (1.5 * scale) / np.maximum(residuals, 1e-12))
        current = weights * huber
    return center, direction


def _clip_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    delta = p2 - p1
    t0, t1 = 0.0, 1.0
    limits = (
        (-delta[0], p1[0]),
        (delta[0], (width - 1.0) - p1[0]),
        (-delta[1], p1[1]),
        (delta[1], (height - 1.0) - p1[1]),
    )
    for denominator, numerator in limits:
        if abs(float(denominator)) < 1e-12:
            if numerator < 0.0:
                return None
            continue
        ratio = float(numerator / denominator)
        if denominator < 0.0:
            t0 = max(t0, ratio)
        else:
            t1 = min(t1, ratio)
        if t0 > t1:
            return None
    return p1 + t0 * delta, p1 + t1 * delta


def clip_segment_to_image(
    segment: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if len(segment) != 4:
        raise ValueError(f"segment must contain four values, got {len(segment)}")
    clipped = _clip_segment(
        np.asarray(segment[:2], dtype=np.float64),
        np.asarray(segment[2:], dtype=np.float64),
        width,
        height,
    )
    if clipped is None:
        return None
    first, second = clipped
    return float(first[0]), float(first[1]), float(second[0]), float(second[1])


def fit_line_group(
    points: Sequence[PointSample],
    group: LineGroup,
    width: int,
    height: int,
    usable_visibility: frozenset[int] = frozenset({1, 2}),
) -> FittedLineSegment:
    if width < 2 or height < 2:
        raise ValueError(f"image dimensions must be at least 2x2, got {width}x{height}")
    by_index = {point.index: point for point in points}
    if any(index not in by_index for index in group.point_indices):
        return _invalid(group, points, "missing topology point")
    valid = [
        by_index[index]
        for index in group.point_indices
        if by_index[index].visibility in usable_visibility
        and 0.0 <= by_index[index].x <= 1.0
        and 0.0 <= by_index[index].y <= 1.0
    ]
    if len(valid) < 2:
        return _invalid(group, points, "fewer than two reliable points")
    coordinates = np.asarray(
        [[point.x * width, point.y * height] for point in valid], dtype=np.float64
    )
    base_weights = np.asarray([1.0 if point.visibility == 2 else 0.75 for point in valid])
    center, direction = _robust_tls(coordinates, base_weights)
    projections = (coordinates - center) @ direction
    endpoint_points = [by_index[index] for index in group.endpoint_indices]
    if all(point in valid for point in endpoint_points):
        endpoint_coordinates = np.asarray(
            [[point.x * width, point.y * height] for point in endpoint_points], dtype=np.float64
        )
        support = (endpoint_coordinates - center) @ direction
    else:
        support = projections
    low, high = float(np.min(support)), float(np.max(support))
    if high - low < 1e-6:
        return _invalid(group, points, "degenerate segment extent")
    clipped = _clip_segment(center + low * direction, center + high * direction, width, height)
    if clipped is None:
        return _invalid(group, points, "fitted segment lies outside image bounds")
    p1, p2 = clipped
    delta = p2 - p1
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return _invalid(group, points, "clipped segment has zero length")
    unit = delta / length
    theta = math.atan2(float(unit[1]), float(unit[0]))
    segment_center = (p1 + p2) * 0.5
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    line_c = -float(normal @ segment_center)
    if normal[0] < 0.0 or (abs(float(normal[0])) < 1e-12 and normal[1] < 0.0):
        normal = -normal
        line_c = -line_c
    fit_distances = np.abs((coordinates - segment_center) @ normal)
    segment = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
    return FittedLineSegment(
        valid=True,
        reason=None,
        segment=segment,
        normalized_segment=(
            segment[0] / width,
            segment[1] / height,
            segment[2] / width,
            segment[3] / height,
        ),
        line=(float(normal[0]), float(normal[1]), line_c),
        center=(float(segment_center[0]), float(segment_center[1])),
        normalized_center=(float(segment_center[0] / width), float(segment_center[1] / height)),
        orientation=(math.cos(2.0 * theta), math.sin(2.0 * theta)),
        normalized_half_length=0.5 * length / math.hypot(width, height),
        fit_rms_distance_px=float(np.sqrt(np.mean(np.square(fit_distances)))),
        fit_max_distance_px=float(np.max(fit_distances)),
        source_indices=group.point_indices,
        source_visibility=tuple(by_index[index].visibility for index in group.point_indices),
        valid_source_indices=tuple(point.index for point in valid),
    )


def audit_center_collisions(
    segments: Sequence[FittedLineSegment],
    stride: int = 4,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    centers = [
        segment.center for segment in segments if segment.valid and segment.center is not None
    ]
    minimum_distance = None
    collision_pairs: list[tuple[int, int]] = []
    cells: dict[tuple[int, int], list[int]] = {}
    for index, center in enumerate(centers):
        cell = (int(center[0] // stride), int(center[1] // stride))
        for previous in cells.setdefault(cell, []):
            collision_pairs.append((previous, index))
        cells[cell].append(index)
        for other in centers[:index]:
            distance = math.hypot(center[0] - other[0], center[1] - other[1])
            minimum_distance = (
                distance if minimum_distance is None else min(minimum_distance, distance)
            )
    return {
        "valid_segment_count": len(centers),
        "minimum_center_distance_px": minimum_distance,
        "minimum_center_distance_cells": (
            minimum_distance / stride if minimum_distance is not None else None
        ),
        "stride": stride,
        "grid_collision_count": len(collision_pairs),
        "grid_collision_pairs": [list(pair) for pair in collision_pairs],
    }
