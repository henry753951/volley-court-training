from __future__ import annotations

import cv2
import numpy as np

from volley_court.layout import (
    CANONICAL_KEYPOINTS,
    CANONICAL_LINES,
    match_court_layout,
    match_semantic_court_layout,
)


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


def _segments_for_homography(homography: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for line in CANONICAL_LINES:
        points = cv2.perspectiveTransform(
            np.asarray((line.first, line.second), dtype=np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        rows.append({"segment": points.reshape(-1).tolist(), "score": 0.9})
    return rows


def _with_semantics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row, line in zip(rows, CANONICAL_LINES, strict=True):
        probabilities = [0.001] * 7
        probabilities[line.topology_index] = 0.994
        row.update(
            line_identity=line.topology_index,
            identity_score=0.994,
            identity_probabilities=probabilities,
            family=line.family,
        )
    return rows


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


def test_prior_refines_identity_from_every_current_frame() -> None:
    rows, prior_homography = _perspective_segments()
    initial = match_court_layout(rows, 640, 640, minimum_hypothesis_margin=0.0)
    translation = np.asarray(((1.0, 0.0, 12.0), (0.0, 1.0, 5.0), (0.0, 0.0, 1.0)))
    current_homography = translation @ prior_homography

    current = match_court_layout(
        _segments_for_homography(current_homography),
        640,
        640,
        minimum_hypothesis_margin=0.0,
        prior_homography=initial["homography"],
    )

    assert current["status"] == "ok"
    assert current["matcher_mode"] == "prior_refined"
    assert current["hypotheses_evaluated"] < initial["hypotheses_evaluated"]
    expected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2),
        current_homography,
    ).reshape(-1, 2)
    actual = np.asarray([[row["x"], row["y"]] for row in current["keypoints"]])
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


def test_fixed_semantic_solver_recovers_pose36_without_topology_search() -> None:
    rows, expected_homography = _perspective_segments()
    result = match_semantic_court_layout(_with_semantics(rows), 640, 640)

    assert result["status"] == "ok"
    assert result["matcher_mode"] == "semantic_fixed"
    assert result["hypotheses_evaluated"] == 1
    assert result["matched_line_count"] == 7
    expected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2),
        expected_homography,
    ).reshape(-1, 2)
    actual = np.asarray([[row["x"], row["y"]] for row in result["keypoints"]])
    assert float(np.max(np.linalg.norm(actual - expected, axis=1))) < 0.1


def test_fixed_semantic_solver_accepts_minimal_observable_four_lines() -> None:
    rows, expected_homography = _perspective_segments()
    semantic = _with_semantics(rows)
    result = match_semantic_court_layout([semantic[index] for index in (0, 2, 3, 5)], 640, 640)

    assert result["status"] == "ok"
    assert result["matched_line_count"] == 4
    expected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2),
        expected_homography,
    ).reshape(-1, 2)
    actual = np.asarray([[row["x"], row["y"]] for row in result["keypoints"]])
    assert float(np.max(np.linalg.norm(actual - expected, axis=1))) < 0.1


def test_fixed_semantic_solver_does_not_mix_explicit_identity_probability_tails() -> None:
    rows, expected_homography = _perspective_segments()
    semantic = _with_semantics(rows)
    for row, line in zip(semantic, CANONICAL_LINES, strict=True):
        probabilities = [0.08] * 7
        probabilities[line.topology_index] = 0.52
        row["identity_probabilities"] = probabilities
        row["identity_score"] = 0.52
    result = match_semantic_court_layout(semantic, 640, 640)

    assert result["status"] == "ok"
    expected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2),
        expected_homography,
    ).reshape(-1, 2)
    actual = np.asarray([[row["x"], row["y"]] for row in result["keypoints"]])
    assert float(np.max(np.linalg.norm(actual - expected, axis=1))) < 0.1


def test_fixed_semantic_solver_abstains_without_both_sidelines() -> None:
    rows, _homography = _perspective_segments()
    semantic = _with_semantics(rows)
    result = match_semantic_court_layout([semantic[index] for index in (0, 1, 3, 5)], 640, 640)

    assert result["status"] == "abstained"
    assert result["keypoints"] == []
    assert "two sidelines" in result["reason"]
