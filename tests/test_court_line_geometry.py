from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from volley_court.geometry import (
    PointSample,
    audit_center_collisions,
    fit_line_group,
    parse_yolo_pose_line,
)
from volley_court.topology import LineGroup, load_topology

ROOT = Path(__file__).resolve().parents[1]


def _blank_points() -> list[PointSample]:
    return [PointSample(index, 0.0, 0.0, 0) for index in range(36)]


def test_topology_groups_all_36_points_into_seven_physical_lines() -> None:
    topology = load_topology(ROOT / "configs" / "court_line_topology.yaml")
    assert topology.keypoint_count == 36
    assert len(topology.line_groups) == 7
    assert {index for group in topology.line_groups for index in group.point_indices} == set(
        range(36)
    )


def test_pose_parser_preserves_occluded_coordinates() -> None:
    values = ["0", "0.5", "0.5", "1", "1"] + ["0", "0", "0"] * 36
    values[5:8] = ["0.25", "0.75", "1"]
    points = parse_yolo_pose_line(" ".join(values))
    assert points[0] == PointSample(0, 0.25, 0.75, 1)


def test_fit_is_endpoint_swap_and_direction_invariant() -> None:
    points = _blank_points()
    points[0] = PointSample(0, 0.1, 0.2, 2)
    points[1] = PointSample(1, 0.5, 0.5, 1)
    points[2] = PointSample(2, 0.9, 0.8, 2)
    forward = fit_line_group(points, LineGroup((0, 1, 2), (0, 2)), 1000, 500)
    reverse = fit_line_group(points, LineGroup((2, 1, 0), (2, 0)), 1000, 500)
    assert forward.valid and reverse.valid
    assert np.allclose(forward.center, reverse.center, atol=1e-6)
    assert np.allclose(forward.orientation, reverse.orientation, atol=1e-6)
    assert math.isclose(
        float(forward.normalized_half_length),
        float(reverse.normalized_half_length),
        abs_tol=1e-9,
    )


def test_reliable_endpoints_define_finite_support_despite_middle_outlier() -> None:
    points = _blank_points()
    points[0] = PointSample(0, 0.2, 0.5, 2)
    points[1] = PointSample(1, 0.5, 0.505, 1)
    points[2] = PointSample(2, 0.8, 0.5, 2)
    points[3] = PointSample(3, 0.95, 0.7, 1)
    fitted = fit_line_group(points, LineGroup((0, 1, 2, 3), (0, 2)), 1000, 500)
    assert fitted.valid
    assert fitted.segment is not None
    assert fitted.fit_rms_distance_px is not None
    xs = sorted((fitted.segment[0], fitted.segment[2]))
    assert 180.0 < xs[0] < 240.0
    assert 760.0 < xs[1] < 820.0


def test_fewer_than_two_reliable_points_is_invalid() -> None:
    points = _blank_points()
    points[0] = PointSample(0, 0.2, 0.5, 2)
    fitted = fit_line_group(points, LineGroup((0, 1), (0, 1)), 1000, 500)
    assert not fitted.valid
    assert fitted.reason == "fewer than two reliable points"


def test_missing_topology_point_is_invalid_instead_of_raising() -> None:
    points = [
        PointSample(0, 0.2, 0.5, 2),
        PointSample(1, 0.8, 0.5, 2),
    ]
    fitted = fit_line_group(points, LineGroup((0, 1, 2), (0, 2)), 1000, 500)
    assert not fitted.valid
    assert fitted.reason == "missing topology point"
    assert fitted.source_visibility == (2, 2, 0)


def test_stride_four_collision_audit_counts_same_cell_centers() -> None:
    points_a = _blank_points()
    points_b = _blank_points()
    points_a[0] = PointSample(0, 0.10, 0.10, 2)
    points_a[1] = PointSample(1, 0.12, 0.10, 2)
    points_b[0] = PointSample(0, 0.10, 0.102, 2)
    points_b[1] = PointSample(1, 0.12, 0.102, 2)
    a = fit_line_group(points_a, LineGroup((0, 1), (0, 1)), 100, 100)
    b = fit_line_group(points_b, LineGroup((0, 1), (0, 1)), 100, 100)
    audit = audit_center_collisions([a, b], stride=4)
    assert audit["grid_collision_count"] == 1
