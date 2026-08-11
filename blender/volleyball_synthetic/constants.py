"""Canonical 36-point volleyball-court geometry contract."""

from __future__ import annotations

from mathutils import Vector

COURT_WIDTH = 9.0
COURT_LENGTH = 18.0
COURT_CENTER = Vector((COURT_WIDTH / 2.0, COURT_LENGTH / 2.0, 0.0))

KEYPOINT_NAMES = (
    "court_left_near_corner",
    "court_left_attack_near",
    "court_left_net_line",
    "court_left_attack_far",
    "court_left_far_corner",
    "court_right_far_corner",
    "court_right_attack_far",
    "court_right_net_line",
    "court_right_attack_near",
    "court_right_near_corner",
    "court_left_near_sideline_third_1",
    "court_left_near_sideline_third_2",
    "court_left_near_attack_zone_third_1",
    "court_left_near_attack_zone_third_2",
    "court_left_far_attack_zone_third_1",
    "court_left_far_attack_zone_third_2",
    "court_left_far_sideline_third_1",
    "court_left_far_sideline_third_2",
    "court_far_baseline_third_1",
    "court_far_baseline_third_2",
    "court_right_far_sideline_third_1",
    "court_right_far_sideline_third_2",
    "court_right_far_attack_zone_third_1",
    "court_right_far_attack_zone_third_2",
    "court_right_near_attack_zone_third_1",
    "court_right_near_attack_zone_third_2",
    "court_right_near_sideline_third_1",
    "court_right_near_sideline_third_2",
    "court_near_baseline_third_1",
    "court_near_baseline_third_2",
    "court_near_attack_line_third_1",
    "court_near_attack_line_third_2",
    "court_center_line_third_1",
    "court_center_line_third_2",
    "court_far_attack_line_third_1",
    "court_far_attack_line_third_2",
)

BASE_POINTS = (
    (0.0, 18.0, 0.04),
    (0.0, 12.0, 0.04),
    (0.0, 9.0, 0.04),
    (0.0, 6.0, 0.04),
    (0.0, 0.0, 0.04),
    (9.0, 0.0, 0.04),
    (9.0, 6.0, 0.04),
    (9.0, 9.0, 0.04),
    (9.0, 12.0, 0.04),
    (9.0, 18.0, 0.04),
)

BASE_SEGMENTS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0),
    (1, 8),
    (2, 7),
    (3, 6),
)


def _lerp_point(start: tuple[float, float, float], end: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(float(a + (b - a) * t) for a, b in zip(start, end))


KEYPOINT_WORLD = BASE_POINTS + tuple(
    point
    for start_index, end_index in BASE_SEGMENTS
    for point in (
        _lerp_point(BASE_POINTS[start_index], BASE_POINTS[end_index], 1.0 / 3.0),
        _lerp_point(BASE_POINTS[start_index], BASE_POINTS[end_index], 2.0 / 3.0),
    )
)

if len(KEYPOINT_NAMES) != 36 or len(KEYPOINT_WORLD) != 36:
    raise RuntimeError("the synthetic court must preserve exactly 36 keypoints")
