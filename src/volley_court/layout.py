from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import clip_segment_to_image

COURT_WIDTH = 9.0
COURT_LENGTH = 18.0

# Canonical coordinates follow the existing Pose36 contract.  y=0 is the near
# baseline and x=0 is the left sideline in the labelled image convention.
CANONICAL_KEYPOINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 6.0),
    (0.0, 9.0),
    (0.0, 12.0),
    (0.0, 18.0),
    (9.0, 18.0),
    (9.0, 12.0),
    (9.0, 9.0),
    (9.0, 6.0),
    (9.0, 0.0),
    (0.0, 2.0),
    (0.0, 4.0),
    (0.0, 7.0),
    (0.0, 8.0),
    (0.0, 10.0),
    (0.0, 11.0),
    (0.0, 14.0),
    (0.0, 16.0),
    (3.0, 18.0),
    (6.0, 18.0),
    (9.0, 16.0),
    (9.0, 14.0),
    (9.0, 11.0),
    (9.0, 10.0),
    (9.0, 8.0),
    (9.0, 7.0),
    (9.0, 4.0),
    (9.0, 2.0),
    (6.0, 0.0),
    (3.0, 0.0),
    (3.0, 6.0),
    (6.0, 6.0),
    (3.0, 9.0),
    (6.0, 9.0),
    (3.0, 12.0),
    (6.0, 12.0),
)


@dataclass(frozen=True)
class CanonicalLine:
    topology_index: int
    name: str
    family: str
    first: tuple[float, float]
    second: tuple[float, float]


CANONICAL_LINES: tuple[CanonicalLine, ...] = (
    CanonicalLine(0, "left_sideline", "vertical", (0.0, 0.0), (0.0, 18.0)),
    CanonicalLine(1, "far_baseline", "horizontal", (0.0, 18.0), (9.0, 18.0)),
    CanonicalLine(2, "right_sideline", "vertical", (9.0, 18.0), (9.0, 0.0)),
    CanonicalLine(3, "near_baseline", "horizontal", (9.0, 0.0), (0.0, 0.0)),
    CanonicalLine(4, "near_attack", "horizontal", (0.0, 6.0), (9.0, 6.0)),
    CanonicalLine(5, "center", "horizontal", (0.0, 9.0), (9.0, 9.0)),
    CanonicalLine(6, "far_attack", "horizontal", (0.0, 12.0), (9.0, 12.0)),
)

HORIZONTAL_Y = (0.0, 6.0, 9.0, 12.0, 18.0)


@dataclass(frozen=True)
class ObservedLine:
    index: int
    segment: tuple[float, float, float, float]
    equation: tuple[float, float, float]
    direction: tuple[float, float]
    axial: tuple[float, float]
    length: float
    score: float
    family: str | None
    identity_probabilities: tuple[float, ...]
    line_identity: int | None
    identity_score: float


def _observed_lines(rows: Sequence[dict[str, Any]]) -> list[ObservedLine]:
    observed: list[ObservedLine] = []
    for index, row in enumerate(rows):
        raw = row.get("segment")
        if not isinstance(raw, Sequence) or len(raw) != 4:
            continue
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 2.0:
            continue
        ux, uy = dx / length, dy / length
        a, b = -uy, ux
        c = -(a * x1 + b * y1)
        theta = math.atan2(uy, ux)
        observed.append(
            ObservedLine(
                index=index,
                segment=(x1, y1, x2, y2),
                equation=(a, b, c),
                direction=(ux, uy),
                axial=(math.cos(2.0 * theta), math.sin(2.0 * theta)),
                length=length,
                score=float(np.clip(row.get("score", 1.0), 0.0, 1.0)),
                family=(
                    str(row["family"]) if row.get("family") in {"vertical", "horizontal"} else None
                ),
                identity_probabilities=tuple(
                    float(value) for value in row.get("identity_probabilities", [])
                ),
                line_identity=(
                    int(row["line_identity"])
                    if isinstance(row.get("line_identity"), (int, np.integer))
                    and 0 <= int(row["line_identity"]) < 7
                    else None
                ),
                identity_score=float(np.clip(row.get("identity_score", 0.0), 0.0, 1.0)),
            )
        )
    return observed


def _axial_clusters(
    lines: Sequence[ObservedLine],
) -> tuple[list[ObservedLine], list[ObservedLine], float]:
    if len(lines) < 4:
        return [], [], 0.0
    axial = np.asarray([line.axial for line in lines], dtype=np.float64)
    weights = np.asarray(
        [max(line.score, 0.05) * math.sqrt(line.length) for line in lines],
        dtype=np.float64,
    )
    first = int(np.argmax(weights))
    second = int(np.argmin(axial @ axial[first]))
    centers = np.stack((axial[first], axial[second]))
    for _ in range(16):
        labels = np.argmax(axial @ centers.T, axis=1)
        updated = centers.copy()
        for cluster in range(2):
            mask = labels == cluster
            if not np.any(mask):
                return [], [], 0.0
            vector = np.sum(axial[mask] * weights[mask, None], axis=0)
            norm = float(np.linalg.norm(vector))
            if norm < 1e-9:
                return [], [], 0.0
            updated[cluster] = vector / norm
        if np.allclose(updated, centers, atol=1e-6):
            centers = updated
            break
        centers = updated
    labels = np.argmax(axial @ centers.T, axis=1)
    separation = 0.5 * math.acos(float(np.clip(centers[0] @ centers[1], -1.0, 1.0)))
    return (
        [line for line, label in zip(lines, labels, strict=True) if label == 0],
        [line for line, label in zip(lines, labels, strict=True) if label == 1],
        math.degrees(separation),
    )


def _family_assignments(
    lines: Sequence[ObservedLine],
    angle_degrees: float = 15.0,
) -> list[tuple[list[ObservedLine], list[ObservedLine]]]:
    """Return family candidates, including strong-perspective three-angle views.

    The two sidelines can diverge sharply in the image even though they share a
    court-coordinate family.  A plain two-angle clustering therefore is only one
    candidate; dominant coherent transverse groups are also tested against all
    remaining lines.
    """

    assignments: list[tuple[list[ObservedLine], list[ObservedLine]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def retain(first: Sequence[ObservedLine], second: Sequence[ObservedLine]) -> None:
        if len(first) < 2 or len(second) < 2:
            return
        key = (
            tuple(sorted(line.index for line in first)),
            tuple(sorted(line.index for line in second)),
        )
        reverse = (key[1], key[0])
        if key in seen or reverse in seen:
            return
        seen.add(key)
        assignments.append((list(first), list(second)))

    cluster_a, cluster_b, _separation = _axial_clusters(lines)
    retain(cluster_a, cluster_b)
    predicted_vertical = [line for line in lines if line.family == "vertical"]
    predicted_horizontal = [line for line in lines if line.family == "horizontal"]
    retain(predicted_vertical, predicted_horizontal)
    threshold = math.cos(math.radians(2.0 * angle_degrees))
    for seed in lines:
        coherent = [line for line in lines if float(np.dot(seed.axial, line.axial)) >= threshold]
        coherent_indices = {line.index for line in coherent}
        retain(coherent, [line for line in lines if line.index not in coherent_indices])
    return assignments


def _intersection(first: ObservedLine, second: ObservedLine) -> tuple[float, float] | None:
    a1, b1, c1 = first.equation
    a2, b2, c2 = second.equation
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-5:
        return None
    return (
        (b1 * c2 - b2 * c1) / determinant,
        (c1 * a2 - c2 * a1) / determinant,
    )


def _project_points(
    homography: np.ndarray,
    points: Iterable[Sequence[float]],
) -> np.ndarray | None:
    source = np.asarray(list(points), dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected


def _angle_distance(first: ObservedLine, segment: Sequence[float]) -> float:
    dx, dy = float(segment[2]) - float(segment[0]), float(segment[3]) - float(segment[1])
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return math.pi / 2.0
    cosine = abs(first.direction[0] * dx / length + first.direction[1] * dy / length)
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def _point_line_distance(point: Sequence[float], line: ObservedLine) -> float:
    a, b, c = line.equation
    return abs(a * float(point[0]) + b * float(point[1]) + c)


def _line_match_quality(reference: Sequence[float], candidate: ObservedLine) -> float:
    angle = _angle_distance(candidate, reference)
    midpoint = (
        0.5 * (float(reference[0]) + float(reference[2])),
        0.5 * (float(reference[1]) + float(reference[3])),
    )
    candidate_midpoint = (
        0.5 * (candidate.segment[0] + candidate.segment[2]),
        0.5 * (candidate.segment[1] + candidate.segment[3]),
    )
    ref_dx = float(reference[2]) - float(reference[0])
    ref_dy = float(reference[3]) - float(reference[1])
    ref_length = max(math.hypot(ref_dx, ref_dy), 1e-6)
    ref_a, ref_b = -ref_dy / ref_length, ref_dx / ref_length
    ref_c = -(ref_a * float(reference[0]) + ref_b * float(reference[1]))
    distance = 0.5 * (
        _point_line_distance(midpoint, candidate)
        + abs(ref_a * candidate_midpoint[0] + ref_b * candidate_midpoint[1] + ref_c)
    )
    angle_quality = math.exp(-((math.degrees(angle) / 12.0) ** 2))
    distance_quality = math.exp(-((distance / 14.0) ** 2))
    return candidate.score * angle_quality * distance_quality


def _project_line(
    homography: np.ndarray,
    line: CanonicalLine,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    points = _project_points(homography, (line.first, line.second))
    if points is None:
        return None
    return clip_segment_to_image(points.reshape(-1).tolist(), width, height)


def _match_projected_lines(
    homography: np.ndarray,
    vertical: Sequence[ObservedLine],
    horizontal: Sequence[ObservedLine],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], float, int]:
    proposals: list[tuple[float, int, int, tuple[float, float, float, float]]] = []
    projected: dict[int, tuple[float, float, float, float]] = {}
    for canonical in CANONICAL_LINES:
        segment = _project_line(homography, canonical, width, height)
        if segment is None:
            continue
        projected[canonical.topology_index] = segment
        candidates = vertical if canonical.family == "vertical" else horizontal
        for candidate in candidates:
            quality = _line_match_quality(segment, candidate)
            if len(candidate.identity_probabilities) == 7:
                semantic = candidate.identity_probabilities[canonical.topology_index]
                quality *= 0.85 + 0.30 * semantic
            proposals.append((quality, canonical.topology_index, candidate.index, segment))
    retained: dict[int, tuple[float, int]] = {}
    used_observed: set[int] = set()
    for quality, topology_index, observed_index, _segment in sorted(proposals, reverse=True):
        if quality < 0.04 or topology_index in retained or observed_index in used_observed:
            continue
        retained[topology_index] = (quality, observed_index)
        used_observed.add(observed_index)
    rows = []
    for canonical in CANONICAL_LINES:
        segment = projected.get(canonical.topology_index)
        if segment is None:
            continue
        match = retained.get(canonical.topology_index)
        rows.append(
            {
                "topology_index": canonical.topology_index,
                "name": canonical.name,
                "family": canonical.family,
                "segment": list(segment),
                "source_index": match[1] if match else None,
                "match_score": match[0] if match else 0.0,
            }
        )
    matched = len(retained)
    geometry = sum(value[0] for value in retained.values()) / max(matched, 1)
    coverage = matched / len(CANONICAL_LINES)
    return rows, 0.55 * coverage + 0.45 * geometry, matched


def _court_convention_score(homography: np.ndarray) -> float:
    corners = _project_points(homography, ((0.0, 0.0), (9.0, 0.0), (0.0, 18.0), (9.0, 18.0)))
    if corners is None:
        return 0.0
    near_y = float(np.mean(corners[:2, 1]))
    far_y = float(np.mean(corners[2:, 1]))
    near_left_x, near_right_x = float(corners[0, 0]), float(corners[1, 0])
    return float(near_y > far_y) * 0.5 + float(near_left_x < near_right_x) * 0.5


def _candidate_homographies(
    vertical: Sequence[ObservedLine],
    horizontal: Sequence[ObservedLine],
) -> Iterable[np.ndarray]:
    vertical = sorted(vertical, key=lambda row: row.score * math.sqrt(row.length), reverse=True)[:5]
    horizontal = sorted(
        horizontal, key=lambda row: row.score * math.sqrt(row.length), reverse=True
    )[:8]
    for side_a, side_b in itertools.combinations(vertical, 2):
        for cross_a, cross_b in itertools.combinations(horizontal, 2):
            intersections = (
                _intersection(side_a, cross_a),
                _intersection(side_b, cross_a),
                _intersection(side_a, cross_b),
                _intersection(side_b, cross_b),
            )
            if any(point is None for point in intersections):
                continue
            destination = np.asarray(intersections, dtype=np.float32)
            polygon = destination[[0, 1, 3, 2]].reshape(-1, 1, 2)
            if abs(float(cv2.contourArea(polygon))) < 16.0:
                continue
            for y_first, y_second in itertools.combinations(HORIZONTAL_Y, 2):
                for swap_x in (False, True):
                    for swap_y in (False, True):
                        x0, x1 = (9.0, 0.0) if swap_x else (0.0, 9.0)
                        y0, y1 = (y_second, y_first) if swap_y else (y_first, y_second)
                        source = np.asarray(
                            ((x0, y0), (x1, y0), (x0, y1), (x1, y1)),
                            dtype=np.float32,
                        )
                        homography = cv2.getPerspectiveTransform(source, destination)
                        if np.isfinite(homography).all():
                            yield homography


def _prior_refined_homographies(
    prior_homography: np.ndarray,
    vertical: Sequence[ObservedLine],
    horizontal: Sequence[ObservedLine],
    width: int,
    height: int,
) -> Iterable[np.ndarray]:
    """Refit a prior topology to the current frame's observed line equations."""

    rows, _score, _matched = _match_projected_lines(
        prior_homography,
        vertical,
        horizontal,
        width,
        height,
    )
    observed_by_index = {line.index: line for line in (*vertical, *horizontal)}
    matched_by_topology = {
        int(row["topology_index"]): observed_by_index[int(source_index)]
        for row in rows
        if (source_index := row.get("source_index")) is not None
        and int(source_index) in observed_by_index
    }
    left = matched_by_topology.get(0)
    right = matched_by_topology.get(2)
    if left is None or right is None:
        return

    horizontal_y = {1: 18.0, 3: 0.0, 4: 6.0, 5: 9.0, 6: 12.0}
    matched_horizontal = [
        (topology, matched_by_topology[topology])
        for topology in horizontal_y
        if topology in matched_by_topology
    ]
    for (first_topology, first), (second_topology, second) in itertools.combinations(
        matched_horizontal, 2
    ):
        intersections = (
            _intersection(left, first),
            _intersection(right, first),
            _intersection(left, second),
            _intersection(right, second),
        )
        if any(point is None for point in intersections):
            continue
        destination = np.asarray(intersections, dtype=np.float32)
        polygon = destination[[0, 1, 3, 2]].reshape(-1, 1, 2)
        if abs(float(cv2.contourArea(polygon))) < 16.0:
            continue
        source = np.asarray(
            (
                (0.0, horizontal_y[first_topology]),
                (9.0, horizontal_y[first_topology]),
                (0.0, horizontal_y[second_topology]),
                (9.0, horizontal_y[second_topology]),
            ),
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        if np.isfinite(homography).all():
            yield homography


def _keypoint_parents() -> tuple[tuple[int, ...], ...]:
    line_points = (
        (0, 10, 11, 1, 12, 13, 2, 14, 15, 3, 16, 17, 4),
        (4, 18, 19, 5),
        (5, 20, 21, 6, 22, 23, 7, 24, 25, 8, 26, 27, 9),
        (9, 28, 29, 0),
        (1, 30, 31, 8),
        (2, 32, 33, 7),
        (3, 34, 35, 6),
    )
    return tuple(
        tuple(index for index, points in enumerate(line_points) if point in points)
        for point in range(36)
    )


KEYPOINT_PARENTS = _keypoint_parents()


_D2_IDENTITY_PERMUTATIONS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6),
    (2, 1, 0, 3, 4, 5, 6),
    (0, 3, 2, 1, 6, 5, 4),
    (2, 3, 0, 1, 6, 5, 4),
)


def _canonical_line_equation(line: CanonicalLine) -> np.ndarray:
    first = np.asarray((*line.first, 1.0), dtype=np.float64)
    second = np.asarray((*line.second, 1.0), dtype=np.float64)
    equation = np.cross(first, second)
    return equation / max(float(np.linalg.norm(equation[:2])), 1e-12)


def _semantic_probability(line: ObservedLine, identity: int) -> float:
    # CUDA semantic decoding already performs per-identity geometric filtering.
    # Once it emits an explicit identity, probability tails for the other six
    # classes are uncertainty metadata, not additional line observations.
    if line.line_identity is not None:
        if line.line_identity != identity:
            return 0.0
        return line.identity_score if line.identity_score > 0.0 else 1.0
    if len(line.identity_probabilities) == len(CANONICAL_LINES):
        return float(np.clip(line.identity_probabilities[identity], 0.0, 1.0))
    return 0.0


def _fit_semantic_line(
    observed: Sequence[ObservedLine],
    identity: int,
    width: int,
    height: int,
    *,
    minimum_probability: float,
) -> tuple[ObservedLine, float] | None:
    """Aggregate all evidence for one physical line with two Huber TLS updates."""

    canonical = CANONICAL_LINES[identity]
    diagonal = math.hypot(width, height)
    samples: list[tuple[ObservedLine, float]] = []
    for line in observed:
        probability = _semantic_probability(line, identity)
        if probability < minimum_probability:
            continue
        family_factor = 1.0 if line.family in {None, canonical.family} else 0.35
        length_factor = math.sqrt(max(line.length / max(diagonal, 1.0), 1e-3))
        weight = line.score * probability * family_factor * length_factor
        if weight > 1e-6:
            samples.append((line, weight))
    if not samples:
        return None

    points = np.asarray(
        [
            point
            for line, _weight in samples
            for point in ((line.segment[0], line.segment[1]), (line.segment[2], line.segment[3]))
        ],
        dtype=np.float64,
    )
    base_weights = np.repeat(np.asarray([weight for _line, weight in samples]), 2)
    weights = base_weights.copy()
    center = np.zeros(2, dtype=np.float64)
    direction = np.asarray((1.0, 0.0), dtype=np.float64)
    huber_delta = max(2.0, 0.004 * diagonal)
    for _iteration in range(3):
        weight_sum = max(float(weights.sum()), 1e-12)
        center = (points * weights[:, None]).sum(axis=0) / weight_sum
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        residual = np.abs(centered @ normal)
        robust = np.minimum(1.0, huber_delta / np.maximum(residual, 1e-9))
        weights = base_weights * robust

    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    if normal[0] < 0.0 or (abs(float(normal[0])) < 1e-12 and normal[1] < 0.0):
        normal = -normal
        direction = -direction
    equation = (float(normal[0]), float(normal[1]), -float(normal @ center))
    support = 2.0 * diagonal
    clipped = clip_segment_to_image(
        (*list(center - support * direction), *list(center + support * direction)),
        width,
        height,
    )
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    length = math.hypot(x2 - x1, y2 - y1)
    theta = math.atan2(y2 - y1, x2 - x1)
    probabilities = np.asarray(
        [_semantic_probability(line, identity) for line, _weight in samples], dtype=np.float64
    )
    evidence_weights = np.asarray([line.score for line, _weight in samples], dtype=np.float64)
    confidence = float(np.average(probabilities, weights=np.maximum(evidence_weights, 1e-6)))
    source = max(samples, key=lambda row: row[1])[0]
    return (
        ObservedLine(
            index=source.index,
            segment=(float(x1), float(y1), float(x2), float(y2)),
            equation=equation,
            direction=(float(direction[0]), float(direction[1])),
            axial=(math.cos(2.0 * theta), math.sin(2.0 * theta)),
            length=length,
            score=float(np.clip(confidence, 0.0, 1.0)),
            family=canonical.family,
            identity_probabilities=tuple(
                1.0 if index == identity else 0.0 for index in range(len(CANONICAL_LINES))
            ),
            line_identity=identity,
            identity_score=float(np.clip(confidence, 0.0, 1.0)),
        ),
        float(np.clip(confidence, 0.0, 1.0)),
    )


def _weighted_line_dlt(
    correspondences: Sequence[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray | None:
    """Solve image H from canonical-to-image line correspondences in dual space."""

    if len(correspondences) < 4:
        return None
    rows = []
    for canonical, image, weight in correspondences:
        x, y, z = canonical
        u, v, w = image
        scale = math.sqrt(max(weight, 1e-9))
        rows.append(
            scale * np.asarray((0.0, 0.0, 0.0, -w * x, -w * y, -w * z, v * x, v * y, v * z))
        )
        rows.append(
            scale * np.asarray((w * x, w * y, w * z, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z))
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(matrix) < 8:
        return None
    # Four observable lines produce an 8x9 system; the complete right basis is
    # required to retain its one-dimensional nullspace.
    _left, _singular, right = np.linalg.svd(matrix, full_matrices=True)
    dual = right[-1].reshape(3, 3)
    if abs(float(np.linalg.det(dual))) < 1e-12:
        return None
    homography = np.linalg.inv(dual).T
    if not np.isfinite(homography).all() or abs(float(homography[2, 2])) < 1e-12:
        return None
    return homography / homography[2, 2]


def _line_reprojection_residual(
    homography: np.ndarray,
    canonical: np.ndarray,
    observed: np.ndarray,
    width: int,
    height: int,
) -> float:
    projected = np.linalg.inv(homography).T @ canonical
    projected /= max(float(np.linalg.norm(projected[:2])), 1e-12)
    observed = observed / max(float(np.linalg.norm(observed[:2])), 1e-12)
    if float(projected[:2] @ observed[:2]) < 0.0:
        projected = -projected
    angle = math.acos(float(np.clip(projected[:2] @ observed[:2], -1.0, 1.0)))
    center = np.asarray((0.5 * width, 0.5 * height, 1.0), dtype=np.float64)
    distance = abs(float((projected - observed) @ center))
    return math.hypot(
        angle / math.radians(3.0), distance / max(3.0, 0.004 * math.hypot(width, height))
    )


def _semantic_homography(
    fitted: dict[int, tuple[ObservedLine, float]],
    permutation: Sequence[int],
    width: int,
    height: int,
) -> np.ndarray | None:
    # Intersections of the two sidelines with each observed transverse line are
    # considerably better conditioned than a dual line-only DLT.  They also
    # provide the exact Pose36 boundary anchors the downstream metric measures.
    inverse_permutation = {observed: canonical for canonical, observed in enumerate(permutation)}
    left_observed = permutation[0]
    right_observed = permutation[2]
    point_rows: list[tuple[np.ndarray, np.ndarray, float]] = []
    if left_observed in fitted and right_observed in fitted:
        for observed_identity, (transverse, transverse_confidence) in fitted.items():
            canonical_identity = inverse_permutation.get(observed_identity)
            if canonical_identity not in {1, 3, 4, 5, 6}:
                continue
            for canonical_side, observed_side in ((0, left_observed), (2, right_observed)):
                side, side_confidence = fitted[observed_side]
                canonical_point = np.cross(
                    _canonical_line_equation(CANONICAL_LINES[canonical_side]),
                    _canonical_line_equation(CANONICAL_LINES[canonical_identity]),
                )
                image_point = np.cross(
                    np.asarray(side.equation, dtype=np.float64),
                    np.asarray(transverse.equation, dtype=np.float64),
                )
                if abs(float(canonical_point[2])) < 1e-9 or abs(float(image_point[2])) < 1e-9:
                    continue
                point_rows.append(
                    (
                        canonical_point[:2] / canonical_point[2],
                        image_point[:2] / image_point[2],
                        math.sqrt(max(side_confidence * transverse_confidence, 1e-9)),
                    )
                )
    if len(point_rows) >= 4:
        source = np.asarray([row[0] for row in point_rows], dtype=np.float64)
        destination = np.asarray([row[1] for row in point_rows], dtype=np.float64)
        homography, _mask = cv2.findHomography(source, destination, method=0)
        if (
            homography is not None
            and np.isfinite(homography).all()
            and abs(float(homography[2, 2])) > 1e-12
        ):
            return homography / homography[2, 2]

    base: list[tuple[np.ndarray, np.ndarray, float]] = []
    for canonical_identity, observed_identity in enumerate(permutation):
        row = fitted.get(observed_identity)
        if row is None:
            continue
        observed, confidence = row
        canonical = _canonical_line_equation(CANONICAL_LINES[canonical_identity])
        base.append((canonical, np.asarray(observed.equation, dtype=np.float64), confidence))
    weights = np.asarray([row[2] for row in base], dtype=np.float64)
    homography: np.ndarray | None = None
    for iteration in range(3):
        correspondences = [
            (canonical, image, float(weight))
            for (canonical, image, _base_weight), weight in zip(base, weights, strict=True)
        ]
        homography = _weighted_line_dlt(correspondences)
        if homography is None:
            return None
        if iteration == 2:
            break
        residuals = np.asarray(
            [
                _line_reprojection_residual(homography, canonical, image, width, height)
                for canonical, image, _base_weight in base
            ]
        )
        huber = np.minimum(1.0, 1.5 / np.maximum(residuals, 1e-9))
        weights = np.asarray([row[2] for row in base], dtype=np.float64) * huber
    return homography


def match_semantic_court_layout(
    segments: Sequence[dict[str, Any]],
    width: int,
    height: int,
    *,
    minimum_identity_probability: float = 0.05,
    minimum_line_confidence: float = 0.10,
    minimum_layout_score: float = 0.45,
    prior_homography: Sequence[Sequence[float]] | np.ndarray | None = None,
    maximum_prior_displacement_ratio: float = 0.12,
) -> dict[str, Any]:
    """Recover Pose36 with a fixed semantic solver and calibrated abstention."""

    observed = _observed_lines(segments)
    fitted = {
        identity: result
        for identity in range(len(CANONICAL_LINES))
        if (
            result := _fit_semantic_line(
                observed,
                identity,
                width,
                height,
                minimum_probability=minimum_identity_probability,
            )
        )
        is not None
        and result[1] >= minimum_line_confidence
    }
    sidelines = all(identity in fitted for identity in (0, 2))
    transverse = [identity for identity in (1, 3, 4, 5, 6) if identity in fitted]
    separation = _axial_clusters([row[0] for row in fitted.values()])[2]
    if not sidelines or len(transverse) < 2:
        return {
            "status": "abstained",
            "reason": "semantic observability requires two sidelines and two transverse lines",
            "family_separation_degrees": separation,
            "layout_score": 0.0,
            "matcher_mode": "semantic_fixed",
            "hypotheses_evaluated": 0,
            "keypoints": [],
            "lines": [],
            "segments": list(segments),
        }

    prior = None
    prior_points = None
    if prior_homography is not None:
        candidate = np.asarray(prior_homography, dtype=np.float64)
        if candidate.shape == (3, 3) and np.isfinite(candidate).all():
            prior = candidate
            prior_points = _project_points(prior, CANONICAL_KEYPOINTS)
    permutations = (
        _D2_IDENTITY_PERMUTATIONS if prior_points is not None else _D2_IDENTITY_PERMUTATIONS[:1]
    )
    candidates: list[tuple[float, np.ndarray, tuple[int, ...]]] = []
    maximum_displacement = maximum_prior_displacement_ratio * math.hypot(width, height)
    for permutation in permutations:
        homography = _semantic_homography(fitted, permutation, width, height)
        if homography is None or _court_convention_score(homography) < 1.0:
            continue
        projected = _project_points(homography, CANONICAL_KEYPOINTS)
        if projected is None:
            continue
        displacement = 0.0
        if prior_points is not None:
            displacement = float(np.median(np.linalg.norm(projected - prior_points, axis=1)))
            if displacement > maximum_displacement:
                continue
        candidates.append((displacement, homography, tuple(permutation)))
    if not candidates:
        return {
            "status": "abstained",
            "reason": "semantic lines did not produce a valid court homography",
            "family_separation_degrees": separation,
            "layout_score": 0.0,
            "matcher_mode": "semantic_fixed_prior_d2" if prior is not None else "semantic_fixed",
            "hypotheses_evaluated": len(permutations),
            "keypoints": [],
            "lines": [],
            "segments": list(segments),
        }

    _displacement, homography, permutation = min(candidates, key=lambda row: row[0])
    projected = _project_points(homography, CANONICAL_KEYPOINTS)
    assert projected is not None
    line_rows = []
    line_scores: dict[int, float] = {}
    semantic_scores = []
    for canonical_identity, canonical in enumerate(CANONICAL_LINES):
        observed_identity = permutation[canonical_identity]
        row = fitted.get(observed_identity)
        segment = _project_line(homography, canonical, width, height)
        if row is None or segment is None:
            continue
        observed_line, semantic_score = row
        quality = _line_match_quality(segment, observed_line)
        match_score = float(np.clip(0.55 * quality + 0.45 * semantic_score, 0.0, 1.0))
        line_scores[canonical_identity] = match_score
        semantic_scores.append(semantic_score)
        line_rows.append(
            {
                "topology_index": canonical_identity,
                "name": canonical.name,
                "family": canonical.family,
                "segment": list(segment),
                "source_index": observed_line.index,
                "match_score": match_score,
            }
        )
    coverage = len(line_rows) / len(CANONICAL_LINES)
    geometry = float(np.mean(list(line_scores.values()))) if line_scores else 0.0
    semantic_alignment = float(np.mean(semantic_scores)) if semantic_scores else 0.0
    score = 0.45 * coverage + 0.35 * geometry + 0.20 * semantic_alignment
    status = "ok" if score >= minimum_layout_score else "abstained"
    reason = None if status == "ok" else "semantic layout score below threshold"
    keypoints = []
    for index, point in enumerate(projected):
        parents = KEYPOINT_PARENTS[index]
        parent_score = sum(line_scores.get(parent, 0.0) for parent in parents) / len(parents)
        keypoints.append(
            {
                "id": index,
                "x": float(point[0]),
                "y": float(point[1]),
                "score": float(np.clip(score * parent_score, 0.0, 1.0)),
                "in_frame": bool(0.0 <= point[0] < width and 0.0 <= point[1] < height),
                "source": "line_intersection" if len(parents) == 2 else "homography_projection",
            }
        )
    classified_segments = []
    for observed_index, source in enumerate(segments):
        identity = None
        probabilities = source.get("identity_probabilities", [])
        if isinstance(probabilities, Sequence) and len(probabilities) == len(CANONICAL_LINES):
            identity = int(np.argmax(np.asarray(probabilities, dtype=np.float64)))
        elif isinstance(source.get("line_identity"), (int, np.integer)):
            identity = int(source["line_identity"])
        family = (
            CANONICAL_LINES[identity].family
            if identity is not None and 0 <= identity < 7
            else "unknown"
        )
        classified_segments.append({**source, "family": family, "source_index": observed_index})
    return {
        "status": status,
        "reason": reason,
        "layout_score": float(score),
        "hypothesis_margin": 1.0,
        "required_hypothesis_margin": 0.0,
        "semantic_alignment": semantic_alignment,
        "family_separation_degrees": float(separation),
        "matched_line_count": len(line_rows),
        "matcher_mode": "semantic_fixed_prior_d2" if prior is not None else "semantic_fixed",
        "hypotheses_evaluated": len(permutations),
        "homography": homography.tolist(),
        "lines": line_rows,
        "segments": classified_segments,
        "keypoints": keypoints if status == "ok" else [],
        "candidate_keypoints": keypoints,
    }


def match_court_layout(
    segments: Sequence[dict[str, Any]],
    width: int,
    height: int,
    *,
    minimum_family_separation_degrees: float = 18.0,
    minimum_matched_lines: int = 4,
    minimum_layout_score: float = 0.45,
    minimum_hypothesis_margin: float | None = None,
    minimum_semantic_alignment: float = 0.22,
    minimum_semantic_matched_lines: int = 5,
    prior_homography: Sequence[Sequence[float]] | np.ndarray | None = None,
    maximum_prior_displacement_ratio: float = 0.12,
) -> dict[str, Any]:
    """Fit the seven-line court template and recover the original Pose36 indices."""

    observed = _observed_lines(segments)
    _cluster_a, _cluster_b, separation = _axial_clusters(observed)
    assignments = _family_assignments(observed)
    if not assignments:
        return {
            "status": "abstained",
            "reason": "two stable line families were not found",
            "family_separation_degrees": separation,
            "layout_score": 0.0,
            "keypoints": [],
            "lines": [],
        }
    hypotheses: list[
        tuple[float, int, np.ndarray, list[dict[str, Any]], list[ObservedLine], list[ObservedLine]]
    ] = []
    directed_assignments = [
        directed for first, second in assignments for directed in ((first, second), (second, first))
    ]
    directed_assignments.sort(
        key=lambda row: (
            0.25 * sum(line.family not in {None, "vertical"} for line in row[0])
            + 0.25 * sum(line.family not in {None, "horizontal"} for line in row[1])
            + abs(len(row[0]) - 2)
            + 0.5 * abs(len(row[1]) - 5)
        )
    )
    matcher_mode = "full_search"
    hypotheses_evaluated = 0
    prior = None
    prior_points = None
    maximum_displacement = maximum_prior_displacement_ratio * math.hypot(width, height)
    if prior_homography is not None:
        candidate_prior = np.asarray(prior_homography, dtype=np.float64)
        if candidate_prior.shape == (3, 3) and np.isfinite(candidate_prior).all():
            prior = candidate_prior

    if prior is not None:
        prior_points = _project_points(prior, CANONICAL_KEYPOINTS)
        for vertical, horizontal in directed_assignments[:8]:
            for homography in _prior_refined_homographies(
                prior,
                vertical,
                horizontal,
                width,
                height,
            ):
                hypotheses_evaluated += 1
                convention = _court_convention_score(homography)
                if convention < 1.0:
                    continue
                candidate_points = _project_points(homography, CANONICAL_KEYPOINTS)
                if prior_points is None or candidate_points is None:
                    continue
                displacement = float(
                    np.median(np.linalg.norm(candidate_points - prior_points, axis=1))
                )
                if displacement > maximum_displacement:
                    continue
                line_rows, line_score, matched = _match_projected_lines(
                    homography,
                    vertical,
                    horizontal,
                    width,
                    height,
                )
                score = 0.92 * line_score + 0.08 * convention
                hypotheses.append(
                    (score, matched, homography, line_rows, list(vertical), list(horizontal))
                )
        if hypotheses and max(row[1] for row in hypotheses) >= minimum_matched_lines:
            matcher_mode = "prior_refined"
        else:
            hypotheses.clear()

    if not hypotheses:
        for vertical, horizontal in directed_assignments[:8]:
            for homography in _candidate_homographies(vertical, horizontal):
                hypotheses_evaluated += 1
                convention = _court_convention_score(homography)
                if convention < 1.0:
                    continue
                if prior_points is not None:
                    candidate_points = _project_points(homography, CANONICAL_KEYPOINTS)
                    if candidate_points is None:
                        continue
                    displacement = float(
                        np.median(np.linalg.norm(candidate_points - prior_points, axis=1))
                    )
                    if displacement > maximum_displacement:
                        continue
                line_rows, line_score, matched = _match_projected_lines(
                    homography,
                    vertical,
                    horizontal,
                    width,
                    height,
                )
                score = 0.92 * line_score + 0.08 * convention
                hypotheses.append(
                    (score, matched, homography, line_rows, list(vertical), list(horizontal))
                )
    if not hypotheses:
        return {
            "status": "abstained",
            "reason": "no valid court homography hypothesis",
            "family_separation_degrees": separation,
            "layout_score": 0.0,
            "matcher_mode": matcher_mode,
            "hypotheses_evaluated": hypotheses_evaluated,
            "keypoints": [],
            "lines": [],
        }
    hypotheses.sort(key=lambda row: (row[0], row[1]), reverse=True)
    best = hypotheses[0]
    best_points = _project_points(best[2], CANONICAL_KEYPOINTS)
    best_signature = tuple(row.get("source_index") for row in best[3])
    runner_up = next(
        (
            row
            for row in hypotheses[1:]
            if (
                tuple(item.get("source_index") for item in row[3]) != best_signature
                and (
                    best_points is None
                    or (candidate_points := _project_points(row[2], CANONICAL_KEYPOINTS)) is None
                    or float(np.max(np.linalg.norm(candidate_points - best_points, axis=1))) > 2.0
                )
            )
        ),
        None,
    )
    margin = (
        best[0]
        if matcher_mode == "prior_refined"
        else best[0] - runner_up[0]
        if runner_up is not None
        else best[0]
    )
    observed_by_index = {line.index: line for line in observed}
    semantic_scores = []
    for row in best[3]:
        source_index = row.get("source_index")
        if source_index is None:
            continue
        candidate = observed_by_index.get(int(source_index))
        topology_index = int(row["topology_index"])
        if candidate is not None and len(candidate.identity_probabilities) == 7:
            semantic_scores.append(candidate.identity_probabilities[topology_index])
    semantic_alignment = float(np.mean(semantic_scores)) if semantic_scores else None
    required_margin = (
        float(minimum_hypothesis_margin)
        if minimum_hypothesis_margin is not None
        else 0.005
        if semantic_alignment is not None
        else 0.015
    )
    status = "ok"
    reason = None
    if best[1] < minimum_matched_lines:
        status, reason = "abstained", "fewer than four template lines matched"
    elif best[0] < minimum_layout_score:
        status, reason = "abstained", "layout score below threshold"
    elif margin < required_margin:
        status, reason = "ambiguous", "multiple court identities have similar scores"
    elif semantic_alignment is not None and best[1] < minimum_semantic_matched_lines:
        status, reason = "ambiguous", "semantic layout has fewer than five matched lines"
    elif semantic_alignment is not None and semantic_alignment < minimum_semantic_alignment:
        status, reason = "ambiguous", "semantic line identities do not support the layout"

    projected = _project_points(best[2], CANONICAL_KEYPOINTS)
    if projected is None:
        status, reason = "abstained", "homography produced invalid keypoints"
        projected = np.empty((0, 2), dtype=np.float64)
    line_scores = {int(row["topology_index"]): float(row["match_score"]) for row in best[3]}
    keypoints = []
    for index, point in enumerate(projected):
        parents = KEYPOINT_PARENTS[index]
        parent_score = sum(line_scores.get(parent, 0.0) for parent in parents) / len(parents)
        in_frame = bool(0.0 <= point[0] < width and 0.0 <= point[1] < height)
        keypoints.append(
            {
                "id": index,
                "x": float(point[0]),
                "y": float(point[1]),
                "score": float(np.clip(best[0] * parent_score, 0.0, 1.0)),
                "in_frame": in_frame,
                "source": "line_intersection" if len(parents) == 2 else "homography_projection",
            }
        )
    family_by_index = {
        line.index: family
        for family, rows in (("vertical", best[4]), ("horizontal", best[5]))
        for line in rows
    }
    classified_segments = [
        {**row, "family": family_by_index.get(index, "unknown")}
        for index, row in enumerate(segments)
    ]
    return {
        "status": status,
        "reason": reason,
        "layout_score": float(best[0]),
        "hypothesis_margin": float(margin),
        "required_hypothesis_margin": required_margin,
        "semantic_alignment": semantic_alignment,
        "family_separation_degrees": float(separation),
        "matched_line_count": int(best[1]),
        "matcher_mode": matcher_mode,
        "hypotheses_evaluated": hypotheses_evaluated,
        "homography": best[2].tolist(),
        "lines": best[3],
        "segments": classified_segments,
        "keypoints": keypoints if status == "ok" else [],
        "candidate_keypoints": keypoints,
    }


def draw_layout_overlay(image: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    output = image.copy()
    family_colors = {"vertical": (255, 120, 30), "horizontal": (30, 220, 255)}
    for row in layout.get("segments", []):
        segment = row.get("segment")
        if not isinstance(segment, Sequence) or len(segment) != 4:
            continue
        first = (int(round(segment[0])), int(round(segment[1])))
        second = (int(round(segment[2])), int(round(segment[3])))
        cv2.line(
            output,
            first,
            second,
            family_colors.get(row.get("family"), (140, 140, 140)),
            2,
            cv2.LINE_AA,
        )
    status = str(layout.get("status", "unknown"))
    points = (
        layout.get("keypoints", []) if status == "ok" else layout.get("candidate_keypoints", [])
    )
    point_color = (60, 240, 80) if status == "ok" else (0, 170, 255)
    for row in points:
        if not row.get("in_frame"):
            continue
        center = (int(round(row["x"])), int(round(row["y"])))
        cv2.circle(output, center, 3, point_color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(row["id"]),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            point_color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"layout={status} score={float(layout.get('layout_score', 0.0)):.3f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
