from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from volley_court import CourtKeypoint, CourtLayout, CourtLayoutTracker, LayoutTrackingConfig
from volley_court.layout import CANONICAL_KEYPOINTS, COURT_SYMMETRY_TRANSFORMS


def _layout(*, offset: float = 0.0, status: str = "ok") -> CourtLayout:
    points = tuple(
        CourtKeypoint(
            id=index,
            x=10.0 * court_x + offset,
            y=5.0 * court_y + offset,
            score=0.9,
            in_frame=True,
            source="test",
        )
        for index, (court_x, court_y) in enumerate(CANONICAL_KEYPOINTS)
    )
    return CourtLayout(
        status=status,  # type: ignore[arg-type]
        score=0.8,
        reason="test",
        keypoints=points if status == "ok" else (),
        candidate_keypoints=points,
        matched_line_count=7,
        hypothesis_margin=0.2,
        semantic_alignment=0.8,
        homography=None,
    )


def _projected_layout(homography: np.ndarray) -> CourtLayout:
    projected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float64).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    base = _layout()
    return replace(
        base,
        keypoints=tuple(
            replace(point, x=float(position[0]), y=float(position[1]))
            for point, position in zip(base.keypoints, projected, strict=True)
        ),
        homography=tuple(tuple(float(value) for value in row) for row in homography),
    )


def test_tracker_smooths_every_accepted_frame() -> None:
    tracker = CourtLayoutTracker(
        LayoutTrackingConfig(
            smoothing=0.5,
            fast_motion_threshold_px=1_000.0,
            max_hold_frames=2,
        )
    )
    tracker.update(_layout(), width=200, height=200)

    tracked = tracker.update(_layout(offset=10.0), width=200, height=200)

    assert tracked is not None
    assert tracked.status == "ok"
    assert tracked.keypoints[0].x == pytest.approx(5.071, abs=0.01)
    assert tracked.keypoints[0].source == "temporally_smoothed_homography"


def test_tracker_holds_then_expires_confirmed_layout() -> None:
    tracker = CourtLayoutTracker(LayoutTrackingConfig(max_hold_frames=1))
    accepted = tracker.update(_layout(), width=200, height=200)
    assert accepted is not None

    held = tracker.update(_layout(status="ambiguous"), width=200, height=200)
    expired = tracker.update(_layout(status="ambiguous"), width=200, height=200)

    assert held is not None and held.status == "ok"
    assert "tracked static hold" in held.reason
    assert expired is not None and expired.status == "ambiguous"
    assert tracker.current is None


def test_tracker_recomputes_visibility_after_smoothing() -> None:
    tracker = CourtLayoutTracker(
        LayoutTrackingConfig(smoothing=1.0, reacquire_confirmation_frames=1)
    )
    tracker.update(_layout(), width=20, height=20)
    shifted = _layout(offset=100.0)

    tracked = tracker.update(shifted, width=20, height=20)

    assert tracked is not None
    assert not tracked.keypoints[0].in_frame


def test_tracker_canonicalizes_legal_identity_flip_before_smoothing() -> None:
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=1.0, max_identity_jump_ratio=0.01))
    homography = np.asarray(
        [[0.0, -5.0, 100.0], [5.0, 0.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    first = _projected_layout(homography)
    tracker.update(first, width=200, height=100)
    flipped = _projected_layout(homography @ COURT_SYMMETRY_TRANSFORMS[1])

    tracked = tracker.update(flipped, width=200, height=100)

    assert tracked is not None
    assert [point.id for point in tracked.keypoints] == list(range(36))
    assert [point.x for point in tracked.keypoints] == pytest.approx(
        [point.x for point in first.keypoints]
    )


def test_tracker_matches_identity_flip_even_when_direct_jump_is_under_limit() -> None:
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=1.0))
    homography = np.asarray(
        [[0.0, -5.0, 100.0], [5.0, 0.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    first = _projected_layout(homography)
    tracker.update(first, width=10_000, height=10_000)
    flipped = _projected_layout(homography @ COURT_SYMMETRY_TRANSFORMS[3])

    tracked = tracker.update(flipped, width=10_000, height=10_000)

    assert tracked is not None
    assert [point.x for point in tracked.keypoints] == pytest.approx(
        [point.x for point in first.keypoints]
    )


def test_tracker_requires_consistent_large_jump_before_reacquiring() -> None:
    tracker = CourtLayoutTracker(
        LayoutTrackingConfig(
            smoothing=0.5,
            max_static_hold_frames=2,
            reacquire_confirmation_frames=3,
        )
    )
    tracker.update(_layout(), width=200, height=200)

    first = tracker.update(_layout(offset=100.0), width=200, height=200)
    second = tracker.update(_layout(offset=100.0), width=200, height=200)
    tracked = tracker.update(_layout(offset=100.0), width=200, height=200)

    assert first is not None and first.keypoints[0].x == pytest.approx(0.0)
    assert second is not None and second.keypoints[0].x == pytest.approx(0.0)
    assert tracked is not None
    assert tracked.keypoints[0].x == pytest.approx(100.0)
    assert tracked.reason == "reacquired canonical layout after 3 confirmations"


def test_tracker_continues_only_nearby_semantic_ambiguity_after_lock() -> None:
    homography = np.asarray(
        [[0.0, -5.0, 100.0], [5.0, 0.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=1.0))
    tracker.update(_projected_layout(homography), width=200, height=100)
    candidate = _projected_layout(homography.copy())
    ambiguous = replace(
        candidate,
        status="ambiguous",
        reason="semantic line identities reject the direct layout",
        keypoints=(),
        matched_line_count=6,
    )

    continued = tracker.update(ambiguous, width=200, height=100)

    assert continued is not None and continued.status == "ok"
    assert continued.reason == "continued from temporally confirmed ambiguous geometry"


def test_tracker_rejects_distant_semantic_ambiguity() -> None:
    homography = np.asarray(
        [[0.0, -5.0, 100.0], [5.0, 0.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=1.0, max_static_hold_frames=0))
    tracker.update(_projected_layout(homography), width=400, height=200)
    distant_homography = homography.copy()
    distant_homography[0, 2] += 200.0
    candidate = _projected_layout(distant_homography)
    ambiguous = replace(
        candidate,
        status="ambiguous",
        reason="semantic line identities reject the direct layout",
        keypoints=(),
        matched_line_count=6,
    )

    rejected = tracker.update(ambiguous, width=400, height=200)

    assert rejected is not None and rejected.status == "ambiguous"
    assert tracker.current is None


def test_tracker_keeps_homography_and_smoothed_keypoints_synchronized() -> None:
    def projected_layout(tx: float) -> CourtLayout:
        homography = np.asarray(
            [[10.0, 0.0, 20.0 + tx], [0.0, -5.0, 150.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return _projected_layout(homography)

    tracker = CourtLayoutTracker(
        LayoutTrackingConfig(smoothing=0.5, fast_motion_threshold_px=1_000.0)
    )
    tracker.update(projected_layout(0.0), width=400, height=200)
    tracked = tracker.update(projected_layout(20.0), width=400, height=200)

    assert tracked is not None and tracked.homography is not None
    projected = cv2.perspectiveTransform(
        np.asarray(CANONICAL_KEYPOINTS, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(tracked.homography, dtype=np.float64),
    ).reshape(-1, 2)
    assert np.allclose(
        projected,
        np.asarray([(point.x, point.y) for point in tracked.keypoints]),
        atol=1e-6,
    )
    assert tracked.keypoints[0].x < tracked.keypoints[9].x
    assert tracked.keypoints[0].y > tracked.keypoints[4].y


def test_tracker_uses_image_motion_as_prediction_before_model_measurement() -> None:
    rng = np.random.default_rng(7)
    first_frame = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    image_motion = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    second_frame = cv2.warpAffine(first_frame, image_motion, (300, 200))
    first_homography = np.asarray(
        [[10.0, 0.0, 20.0], [0.0, -5.0, 150.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    jittered_homography = first_homography.copy()
    jittered_homography[0, 2] += 15.0
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=0.25))
    tracker.update(_projected_layout(first_homography), width=300, height=200, frame=first_frame)

    tracked = tracker.update(
        _projected_layout(jittered_homography),
        width=300,
        height=200,
        frame=second_frame,
    )

    assert tracked is not None
    # The camera predicts x=25. The noisy model says x=35, so the 25% measurement
    # correction should land near 27.5 instead of following the raw jump.
    assert tracked.keypoints[0].x == pytest.approx(27.5, abs=1.0)
