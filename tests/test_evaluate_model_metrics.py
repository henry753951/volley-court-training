from __future__ import annotations

import cv2
import numpy as np

from volley_court.evaluate_model import _homography_metrics


def _homography() -> np.ndarray:
    source = np.asarray(((0.0, 0.0), (0.0, 18.0), (9.0, 18.0), (9.0, 0.0)), dtype=np.float32)
    destination = np.asarray(((100.0, 600.0), (220.0, 100.0), (780.0, 110.0), (900.0, 590.0)), dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def test_homography_metrics_are_perfect_for_identical_layouts() -> None:
    homography = _homography()
    metrics = _homography_metrics(homography, homography.copy(), 1000, 700)
    assert metrics["grid_reprojection_p95"] < 1e-8
    assert metrics["whole_court_iou"] == 1.0
    assert metrics["visible_court_iou"] == 1.0


def test_homography_metrics_expose_axis_swap_hidden_by_outer_polygon_iou() -> None:
    homography = _homography()
    axis_swap = np.asarray(
        ((0.0, 0.5, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    metrics = _homography_metrics(homography, homography @ axis_swap, 1000, 700)
    assert metrics["whole_court_iou"] > 0.99
    assert metrics["grid_reprojection_mean"] > 0.1
