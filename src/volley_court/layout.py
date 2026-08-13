from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

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

COURT_SYMMETRY_TRANSFORMS: tuple[NDArray[np.float64], ...] = (
    np.eye(3, dtype=np.float64),
    np.asarray([[-1.0, 0.0, 9.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 18.0], [0.0, 0.0, 1.0]]),
    np.asarray([[-1.0, 0.0, 9.0], [0.0, -1.0, 18.0], [0.0, 0.0, 1.0]]),
)


def _pose36_symmetry_maps() -> tuple[tuple[int, ...], ...]:
    coordinate_to_index = {
        coordinate: index for index, coordinate in enumerate(CANONICAL_KEYPOINTS)
    }
    maps = []
    for transform in COURT_SYMMETRY_TRANSFORMS:
        permutation = []
        for x, y in CANONICAL_KEYPOINTS:
            transformed = transform @ np.asarray((x, y, 1.0), dtype=np.float64)
            permutation.append(coordinate_to_index[(float(transformed[0]), float(transformed[1]))])
        maps.append(tuple(permutation))
    return tuple(maps)


POSE36_SYMMETRY_MAPS = _pose36_symmetry_maps()


def _court_orientation_corner_permutations() -> tuple[tuple[int, ...], ...]:
    corners = ((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0))
    coordinate_to_index = {coordinate: index for index, coordinate in enumerate(corners)}
    flips = (
        lambda x, y: (x, y),
        lambda x, y: (9.0 - x, y),
        lambda x, y: (x, 18.0 - y),
        lambda x, y: (9.0 - x, 18.0 - y),
    )
    rows = [tuple(coordinate_to_index[flip(x, y)] for x, y in corners) for flip in flips]
    rows.extend(
        tuple(coordinate_to_index[(0.5 * flip(x, y)[1], 2.0 * flip(x, y)[0])] for x, y in corners)
        for flip in flips
    )
    return tuple(rows)


COURT_ORIENTATION_CORNER_PERMUTATIONS = _court_orientation_corner_permutations()


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


def _refine_homography_from_matched_lines(
    homography: np.ndarray,
    line_rows: Sequence[dict[str, Any]],
    observed: Sequence[ObservedLine],
    width: int,
    height: int,
) -> np.ndarray | None:
    """Snap an anchor-solved homography to identity-matched dense line intersections."""

    observed_by_index = {line.index: line for line in observed}
    matched = {
        int(row["topology_index"]): observed_by_index[int(row["source_index"])]
        for row in line_rows
        if row.get("source_index") is not None and int(row["source_index"]) in observed_by_index
    }
    if not {0, 2}.issubset(matched):
        return None
    horizontal_y = {1: 18.0, 3: 0.0, 4: 6.0, 5: 9.0, 6: 12.0}
    source = []
    destination = []
    maximum_deviation = 0.08 * math.hypot(width, height)
    for vertical_index, court_x in ((0, 0.0), (2, 9.0)):
        for horizontal_index, court_y in horizontal_y.items():
            if horizontal_index not in matched:
                continue
            point = _intersection(matched[vertical_index], matched[horizontal_index])
            if point is None:
                continue
            projected = _project_points(homography, ((court_x, court_y),))
            if projected is None or np.linalg.norm(projected[0] - point) > maximum_deviation:
                continue
            source.append((court_x, court_y))
            destination.append(point)
    if len(source) < 4:
        return None
    source_array = np.asarray(source, dtype=np.float32)
    destination_array = np.asarray(destination, dtype=np.float32)
    if abs(float(cv2.contourArea(cv2.convexHull(source_array)))) < 1.0:
        return None
    refined, _mask = cv2.findHomography(source_array, destination_array, method=0)
    if refined is None or not np.isfinite(refined).all() or _court_convention_score(refined) < 1.0:
        return None
    return refined


def _court_convention_score(homography: np.ndarray) -> float:
    corners = _project_points(homography, ((0.0, 0.0), (9.0, 0.0), (0.0, 18.0), (9.0, 18.0)))
    if corners is None:
        return 0.0
    near_y = float(np.mean(corners[:2, 1]))
    far_y = float(np.mean(corners[2:, 1]))
    near_left_x, near_right_x = float(corners[0, 0]), float(corners[1, 0])
    return float(near_y > far_y) * 0.5 + float(near_left_x < near_right_x) * 0.5


def resolve_court_homography_symmetry(
    homography: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    """Resolve the four legal court symmetries to the fixed image-facing convention."""

    ranked: list[tuple[float, int, np.ndarray]] = []
    for symmetry_index, symmetry in enumerate(COURT_SYMMETRY_TRANSFORMS):
        candidate = np.asarray(homography, dtype=np.float64) @ symmetry
        corners = _project_points(candidate, ((0.0, 0.0), (9.0, 0.0), (0.0, 18.0), (9.0, 18.0)))
        if corners is None:
            continue
        near_y = float(np.mean(corners[:2, 1]))
        far_y = float(np.mean(corners[2:, 1]))
        near_left_x, near_right_x = float(corners[0, 0]), float(corners[1, 0])
        if near_y <= far_y or near_left_x >= near_right_x:
            continue
        ranked.append(((near_y - far_y) + (near_right_x - near_left_x), symmetry_index, candidate))
    if not ranked:
        return None
    _, symmetry_index, normalized = max(ranked, key=lambda row: row[0])
    scale = float(normalized[2, 2])
    normalized = normalized / scale if abs(scale) > 1e-12 else normalized
    return normalized, symmetry_index


def court_homography_orientation_candidates(
    homography: np.ndarray,
) -> tuple[tuple[np.ndarray, int], ...]:
    """Return fixed-cost D4-like axis/orientation candidates in image convention."""

    flips = (
        np.eye(3, dtype=np.float64),
        np.asarray([[-1.0, 0.0, 9.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 18.0], [0.0, 0.0, 1.0]]),
        np.asarray([[-1.0, 0.0, 9.0], [0.0, -1.0, 18.0], [0.0, 0.0, 1.0]]),
    )
    axis_swap = np.asarray([[0.0, 0.5, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    candidates = []
    for index, transform in enumerate((*flips, *(axis_swap @ flip for flip in flips))):
        candidate = np.asarray(homography, dtype=np.float64) @ transform
        if _court_convention_score(candidate) < 1.0:
            continue
        scale = float(candidate[2, 2])
        candidate = candidate / scale if abs(scale) > 1e-12 else candidate
        candidates.append((candidate, index))
    return tuple(candidates)


def canonicalize_court_homography(homography: np.ndarray) -> np.ndarray | None:
    resolved = resolve_court_homography_symmetry(homography)
    return resolved[0] if resolved is not None else None


def layout_from_direct_corners(
    corner_proposals: np.ndarray,
    proposal_probabilities: np.ndarray,
    validity_probability: float,
    segments: Sequence[dict[str, Any]],
    width: int,
    height: int,
    *,
    symmetry_index: int = 0,
    minimum_validity: float = 0.5,
    minimum_matched_lines: int = 5,
    minimum_evidence_score: float = 0.5,
    minimum_semantic_alignment: float = 0.665,
) -> dict[str, Any]:
    """Verify a model-proposed layout against fixed-cost dense line evidence."""

    empty = {
        "status": "abstained",
        "reason": "direct layout is not observable",
        "layout_score": 0.0,
        "hypothesis_margin": 0.0,
        "semantic_alignment": None,
        "matched_line_count": 0,
        "homography": None,
        "lines": [],
        "keypoints": [],
        "candidate_keypoints": [],
        "direct_validity": float(validity_probability),
    }
    proposals = np.asarray(corner_proposals, dtype=np.float64)
    probabilities = np.asarray(proposal_probabilities, dtype=np.float64).reshape(-1)
    if proposals.ndim != 3 or proposals.shape[1:] != (4, 2):
        return {**empty, "reason": "invalid direct layout tensor shape"}
    if len(probabilities) != len(proposals) or not np.isfinite(proposals).all():
        return {**empty, "reason": "non-finite direct layout proposal"}
    if validity_probability < minimum_validity:
        return empty
    best_index = int(np.argmax(probabilities))
    corners = proposals[best_index]
    frame_area = max(float(width * height), 1.0)
    polygon_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if polygon_area / frame_area < 0.01 or not cv2.isContourConvex(corners.astype(np.float32)):
        return {**empty, "reason": "direct layout quadrilateral is degenerate"}
    normalized = corners / np.asarray([max(width, 1), max(height, 1)], dtype=np.float64)
    if float(np.max(np.abs(normalized))) > 3.0:
        return {**empty, "reason": "direct layout is implausibly far outside the image"}

    canonical_corners = np.asarray(
        [[0.0, 0.0], [0.0, 18.0], [9.0, 18.0], [9.0, 0.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(canonical_corners, corners.astype(np.float32))
    if not np.isfinite(homography).all() or _court_convention_score(homography) < 1.0:
        return {**empty, "reason": "direct layout violates the fixed court orientation"}

    observed = _observed_lines(segments)
    line_rows, evidence_score, matched = _match_projected_lines(
        homography,
        observed,
        observed,
        width,
        height,
    )
    refined = _refine_homography_from_matched_lines(homography, line_rows, observed, width, height)
    if refined is not None:
        homography = refined
        line_rows, evidence_score, matched = _match_projected_lines(
            homography,
            observed,
            observed,
            width,
            height,
        )
    observed_by_index = {line.index: line for line in observed}
    semantic_scores = []
    for row in line_rows:
        source_index = row.get("source_index")
        if source_index is None:
            continue
        candidate = observed_by_index.get(int(source_index))
        topology_index = int(row["topology_index"])
        if candidate is not None and len(candidate.identity_probabilities) == 7:
            semantic_scores.append(candidate.identity_probabilities[topology_index])
    semantic_alignment = float(np.mean(semantic_scores)) if semantic_scores else None
    semantic_threshold = minimum_semantic_alignment
    if matched == len(CANONICAL_LINES) and evidence_score >= 0.70:
        # A complete seven-line geometric match is much stronger evidence than
        # one weak identity score at an occluded net/post intersection. Keep
        # partial layouts on the strict threshold so unrelated lines cannot
        # manufacture a connected court.
        semantic_threshold = min(semantic_threshold, 0.55)

    projected = _project_points(homography, CANONICAL_KEYPOINTS)
    if projected is None:
        return {**empty, "reason": "direct homography produced invalid keypoints"}

    def horizontal_fraction(first: np.ndarray, second: np.ndarray) -> float:
        delta = second - first
        return abs(float(delta[0])) / max(float(np.linalg.norm(delta)), 1e-9)

    sorted_probabilities = np.sort(probabilities)[::-1]
    margin = float(
        sorted_probabilities[0] - sorted_probabilities[1]
        if len(sorted_probabilities) > 1
        else sorted_probabilities[0]
    )
    disagreement = 0.0
    if len(proposals) > 1:
        runner_up = int(np.argsort(probabilities)[-2])
        disagreement = float(
            np.max(
                np.linalg.norm(
                    normalized - proposals[runner_up] / np.asarray([width, height]),
                    axis=1,
                )
            )
        )
    baseline_horizontal = 0.5 * (
        horizontal_fraction(projected[0], projected[9])
        + horizontal_fraction(projected[4], projected[5])
    )
    sideline_horizontal = 0.5 * (
        horizontal_fraction(projected[0], projected[4])
        + horizontal_fraction(projected[9], projected[5])
    )
    axis_alignment_score = baseline_horizontal - sideline_horizontal
    status = "ok"
    reason = None
    if symmetry_index not in range(4):
        status, reason = "abstained", "runtime court-axis swaps are not supported"
    elif matched < minimum_matched_lines:
        status, reason = "abstained", "direct layout has insufficient line evidence"
    elif evidence_score < minimum_evidence_score:
        status, reason = "abstained", "direct layout evidence score is below threshold"
    elif disagreement > 0.05 and margin < 0.15:
        status, reason = "ambiguous", "direct layout proposals disagree"
    elif semantic_alignment is not None and semantic_alignment < semantic_threshold:
        status, reason = "ambiguous", "semantic line identities reject the direct layout"

    score = float(np.clip(validity_probability * evidence_score, 0.0, 1.0))

    line_scores = {int(row["topology_index"]): float(row["match_score"]) for row in line_rows}
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
                "source": "direct_layout_homography",
            }
        )
    return {
        "status": status,
        "reason": reason,
        "layout_score": score,
        "hypothesis_margin": margin,
        "proposal_disagreement": disagreement,
        "semantic_alignment": semantic_alignment,
        "matched_line_count": int(matched),
        "homography": homography.tolist(),
        "lines": line_rows,
        "keypoints": keypoints if status == "ok" else [],
        "candidate_keypoints": keypoints,
        "direct_validity": float(validity_probability),
        "direct_proposal_probability": float(probabilities[best_index]),
        "axis_alignment_score": axis_alignment_score,
    }


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
    for vertical, horizontal in directed_assignments[:8]:
        for homography in _candidate_homographies(vertical, horizontal):
            convention = _court_convention_score(homography)
            if convention < 1.0:
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
    margin = best[0] - runner_up[0] if runner_up is not None else best[0]
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
