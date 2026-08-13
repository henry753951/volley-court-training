from __future__ import annotations

import cv2
import numpy as np

from volley_court.layout import (
    CANONICAL_KEYPOINTS,
    CANONICAL_LINES,
    COURT_ORIENTATION_CORNER_PERMUTATIONS,
    canonicalize_court_homography,
    layout_from_direct_corners,
    match_court_layout,
)


def test_orientation_candidates_preserve_all_four_outer_corners() -> None:
    assert len(COURT_ORIENTATION_CORNER_PERMUTATIONS) == 8
    assert all(sorted(row) == [0, 1, 2, 3] for row in COURT_ORIENTATION_CORNER_PERMUTATIONS)


def _perspective_segments() -> tuple[list[dict[str, object]], np.ndarray]:
    source = np.asarray(((0.0, 0.0), (9.0, 0.0), (0.0, 18.0), (9.0, 18.0)), dtype=np.float32)
    destination = np.asarray(
        ((80.0, 580.0), (560.0, 570.0), (210.0, 120.0), (440.0, 125.0)), dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    rows = []
    for line in CANONICAL_LINES:
        points = cv2.perspectiveTransform(
            np.asarray((line.first, line.second), dtype=np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        rows.append(
            {
                "segment": points.reshape(-1).tolist(),
                "score": 0.9,
            }
        )
    return rows, homography


def test_canonical_pose36_contract_has_expected_subdivision_coordinates() -> None:
    assert len(CANONICAL_KEYPOINTS) == 36
    assert CANONICAL_KEYPOINTS[0] == (0.0, 0.0)
    assert CANONICAL_KEYPOINTS[9] == (9.0, 0.0)
    assert CANONICAL_KEYPOINTS[18:20] == ((3.0, 18.0), (6.0, 18.0))
    assert CANONICAL_KEYPOINTS[28:30] == ((6.0, 0.0), (3.0, 0.0))


def test_perfect_seven_lines_recover_all_numbered_keypoints() -> None:
    rows, expected_homography = _perspective_segments()
    result = match_court_layout(rows, 640, 640, minimum_hypothesis_margin=0.0)
    assert result["status"] == "ok"
    assert result["matched_line_count"] == 7
    assert {row["family"] for row in result["segments"]} == {"vertical", "horizontal"}
    expected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2),
        expected_homography,
    ).reshape(-1, 2)
    actual = np.asarray([[row["x"], row["y"]] for row in result["keypoints"]])
    assert float(np.max(np.linalg.norm(actual - expected, axis=1))) < 0.1


def test_one_corner_cannot_emit_numbered_keypoints() -> None:
    rows, _homography = _perspective_segments()
    result = match_court_layout([rows[0], rows[3]], 640, 640)
    assert result["status"] == "abstained"
    assert result["keypoints"] == []


def test_semantic_line_identities_support_a_complete_layout() -> None:
    rows, _homography = _perspective_segments()
    for row, line in zip(rows, CANONICAL_LINES, strict=True):
        probabilities = [0.0] * 7
        probabilities[line.topology_index] = 1.0
        row["identity_probabilities"] = probabilities
    result = match_court_layout(rows, 640, 640)
    assert result["status"] == "ok"
    assert result["required_hypothesis_margin"] == 0.005
    assert result["semantic_alignment"] is not None
    assert result["semantic_alignment"] > 0.99


def test_semantic_guard_rejects_identity_unsupported_layout() -> None:
    rows, _homography = _perspective_segments()
    for row in rows:
        row["identity_probabilities"] = [1.0 / 7.0] * 7
    result = match_court_layout(rows, 640, 640)
    assert result["status"] == "ambiguous"
    assert result["reason"] == "semantic line identities do not support the layout"
    assert result["keypoints"] == []
    assert result["candidate_keypoints"]


def test_direct_layout_uses_fixed_order_and_dense_evidence() -> None:
    rows, homography = _perspective_segments()
    corners = cv2.perspectiveTransform(
        np.asarray(((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)), dtype=np.float32).reshape(
            -1, 1, 2
        ),
        homography,
    ).reshape(1, 4, 2)
    result = layout_from_direct_corners(corners, np.asarray([1.0]), 0.99, rows, 640, 640)
    assert result["status"] == "ok"
    assert result["matched_line_count"] == 7
    assert len(result["keypoints"]) == 36


def test_direct_layout_uses_complete_geometry_to_tolerate_one_weak_identity() -> None:
    rows, homography = _perspective_segments()
    for index, row in enumerate(rows):
        probabilities = [0.44 / 6.0] * 7
        probabilities[index] = 0.56
        row["identity_probabilities"] = probabilities
    corners = cv2.perspectiveTransform(
        np.asarray(((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)), dtype=np.float32).reshape(
            -1, 1, 2
        ),
        homography,
    ).reshape(1, 4, 2)

    complete = layout_from_direct_corners(corners, np.asarray([1.0]), 0.99, rows, 640, 640)
    partial = layout_from_direct_corners(corners, np.asarray([1.0]), 0.99, rows[:-1], 640, 640)

    assert complete["status"] == "ok"
    assert complete["matched_line_count"] == 7
    assert partial["status"] == "ambiguous"
    assert partial["matched_line_count"] == 6


def test_direct_layout_rejects_near_far_flip() -> None:
    rows, homography = _perspective_segments()
    corners = cv2.perspectiveTransform(
        np.asarray(((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)), dtype=np.float32).reshape(
            -1, 1, 2
        ),
        homography,
    ).reshape(1, 4, 2)
    corners = corners[:, [1, 0, 3, 2]]
    result = layout_from_direct_corners(corners, np.asarray([1.0]), 0.99, rows, 640, 640)
    assert result["status"] == "abstained"
    assert "orientation" in result["reason"]


def test_direct_layout_rejects_runtime_axis_swap_even_with_strong_evidence() -> None:
    rows, homography = _perspective_segments()
    corners = cv2.perspectiveTransform(
        np.asarray(((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)), dtype=np.float32).reshape(
            -1, 1, 2
        ),
        homography,
    ).reshape(1, 4, 2)
    result = layout_from_direct_corners(
        corners,
        np.asarray([1.0]),
        0.99,
        rows,
        640,
        640,
        symmetry_index=4,
    )
    assert result["status"] == "abstained"
    assert result["reason"] == "runtime court-axis swaps are not supported"


def test_direct_layout_accepts_topology_preserving_runtime_symmetry() -> None:
    rows, homography = _perspective_segments()
    corners = cv2.perspectiveTransform(
        np.asarray(((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)), dtype=np.float32).reshape(
            -1, 1, 2
        ),
        homography,
    ).reshape(1, 4, 2)
    result = layout_from_direct_corners(
        corners,
        np.asarray([1.0]),
        0.99,
        rows,
        640,
        640,
        symmetry_index=3,
    )
    assert result["status"] == "ok"


def test_homography_canonicalization_resolves_width_and_length_symmetry() -> None:
    _rows, homography = _perspective_segments()
    symmetry = np.asarray([[-1.0, 0.0, 9.0], [0.0, -1.0, 18.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    canonicalized = canonicalize_court_homography(homography @ symmetry)
    assert canonicalized is not None
    points = np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2)
    expected = cv2.perspectiveTransform(points, homography)
    actual = cv2.perspectiveTransform(points, canonicalized)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
