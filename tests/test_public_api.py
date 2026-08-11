from __future__ import annotations

from dataclasses import replace

import numpy as np

from volley_court import (
    CourtFrameResult,
    CourtKeypoint,
    CourtLayout,
    CourtLine,
    CourtVisualizer,
    VisualizationConfig,
)


def _line() -> CourtLine:
    return CourtLine.from_mapping(
        {
            "segment": [8.0, 12.0, 88.0, 12.0],
            "score": 0.91,
            "family": "horizontal",
            "family_score": 0.96,
            "line_identity": 5,
            "identity_score": 0.84,
            "vote_count": 14,
            "roi_score": 0.88,
            "orientation": [1.0, 0.0],
        }
    )


def test_court_line_mapping_round_trip() -> None:
    original = _line()
    restored = CourtLine.from_mapping(original.to_mapping())

    assert restored == original
    assert restored.name == "center"
    assert restored.center == (48.0, 12.0)


def test_visualizer_returns_a_copy() -> None:
    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    before = frame.copy()
    result = CourtFrameResult(
        lines=(_line(),),
        width=96,
        height=72,
        inference_seconds=0.01,
    )

    rendered = CourtVisualizer().draw(frame, result)

    assert np.array_equal(frame, before)
    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0


def _complete_layout(status: str = "ok") -> CourtLayout:
    coordinates = {
        0: (10.0, 90.0),
        1: (10.0, 65.0),
        2: (10.0, 50.0),
        3: (10.0, 35.0),
        4: (20.0, 10.0),
        5: (80.0, 10.0),
        6: (90.0, 35.0),
        7: (90.0, 50.0),
        8: (90.0, 65.0),
        9: (90.0, 90.0),
    }
    points = tuple(
        CourtKeypoint(
            id=index,
            x=coordinates.get(index, (50.0, 50.0))[0],
            y=coordinates.get(index, (50.0, 50.0))[1],
            score=0.9,
            in_frame=True,
            source="test",
        )
        for index in range(36)
    )
    return CourtLayout(
        status=status,  # type: ignore[arg-type]
        score=0.9,
        reason="test",
        keypoints=points if status == "ok" else (),
        candidate_keypoints=points,
        matched_line_count=7,
        hypothesis_margin=0.2,
        semantic_alignment=0.8,
        homography=None,
    )


def test_visualizer_only_draws_full_lines_for_accepted_layout() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    line = replace(_line(), segment=(10.0, 80.0, 90.0, 80.0))
    config = VisualizationConfig(show_panel=False, show_keypoints=False)
    visualizer = CourtVisualizer(config)
    ambiguous = CourtFrameResult(
        lines=(line,),
        width=100,
        height=100,
        inference_seconds=0.01,
        layout=_complete_layout("ambiguous"),
    )
    accepted = replace(ambiguous, layout=_complete_layout())

    evidence_only = visualizer.draw(frame, ambiguous)
    full_layout = visualizer.draw(frame, accepted)

    assert np.count_nonzero(evidence_only[90, 25]) == 0
    assert np.count_nonzero(full_layout[90, 25]) > 0
    assert np.count_nonzero(evidence_only[80, 50]) > 0
    assert np.count_nonzero(full_layout[80, 50]) == 0
