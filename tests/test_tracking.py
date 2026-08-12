from __future__ import annotations

from dataclasses import replace

import pytest

from volley_court import CourtKeypoint, CourtLayout, CourtLayoutTracker, LayoutTrackingConfig
from volley_court.layout import POSE36_SYMMETRY_MAPS


def _layout(*, offset: float = 0.0, status: str = "ok") -> CourtLayout:
    points = tuple(
        CourtKeypoint(
            id=index,
            x=float(index) + offset,
            y=float(index) + offset,
            score=0.9,
            in_frame=True,
            source="test",
        )
        for index in range(36)
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
    assert "tracked hold" in held.reason
    assert expired is not None and expired.status == "ambiguous"
    assert tracker.current is None


def test_tracker_recomputes_visibility_after_smoothing() -> None:
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=1.0))
    tracker.update(_layout(), width=20, height=20)
    shifted = _layout(offset=100.0)

    tracked = tracker.update(shifted, width=20, height=20)

    assert tracked is not None
    assert not tracked.keypoints[0].in_frame


def test_tracker_matches_legal_identity_flip_before_smoothing() -> None:
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=1.0, max_identity_jump_ratio=0.01))
    first = _layout()
    tracker.update(first, width=200, height=200)
    permutation = POSE36_SYMMETRY_MAPS[1]
    flipped = replace(
        first,
        keypoints=tuple(replace(point, id=permutation[point.id]) for point in first.keypoints),
    )

    tracked = tracker.update(flipped, width=200, height=200)

    assert tracked is not None
    assert [point.id for point in tracked.keypoints] == list(range(36))
    assert [point.x for point in tracked.keypoints] == pytest.approx(
        [point.x for point in first.keypoints]
    )


def test_tracker_reacquires_large_unmatched_jump_without_interpolation() -> None:
    tracker = CourtLayoutTracker(LayoutTrackingConfig(smoothing=0.5))
    tracker.update(_layout(), width=200, height=200)

    tracked = tracker.update(_layout(offset=100.0), width=200, height=200)

    assert tracked is not None
    assert tracked.keypoints[0].x == pytest.approx(100.0)
