from __future__ import annotations

import cv2
import numpy as np

from volley_court.audit_video import _group_stability_metrics


def _layout(dx: float = 0.0, *, status: str = "ok") -> dict[str, object]:
    return {
        "status": status,
        "semantic_alignment": 0.8,
        "candidate_keypoints": [
            {"id": index, "x": float(20 + index * 2 + dx), "y": float(30 + index)}
            for index in range(36)
        ],
    }


def test_stability_metric_compensates_global_camera_translation() -> None:
    first = np.zeros((160, 240, 3), dtype=np.uint8)
    for y in range(20, 150, 20):
        for x in range(20, 230, 20):
            cv2.circle(first, (x, y), 2, (255, 255, 255), -1)
    transform = np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    second = cv2.warpAffine(first, transform, (240, 160))
    rows = [
        {"status": "ok", "layout": _layout()},
        {"status": "ok", "layout": _layout(3.0)},
    ]
    metrics = _group_stability_metrics([first, second], rows)
    assert metrics["accepted_coverage"] == 1.0
    assert metrics["status_flicker_rate"] == 0.0
    assert metrics["motion_compensated_sample_count"] == 36
    assert metrics["motion_compensated_layout_jitter_p95"] < 1e-4


def test_stability_metric_reports_status_flicker() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    rows = [
        {"status": "ok", "layout": _layout()},
        {"status": "abstained", "layout": _layout(status="abstained")},
        {"status": "ok", "layout": _layout()},
    ]
    metrics = _group_stability_metrics([frame, frame, frame], rows)
    assert metrics["status_transition_count"] == 2
    assert metrics["status_flicker_rate"] == 1.0
