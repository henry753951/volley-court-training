from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

type Point = tuple[float, float]
type Segment = tuple[float, float, float, float]
type LineFamily = Literal["vertical", "horizontal", "unknown"]
type LayoutStatus = Literal["ok", "ambiguous", "abstained", "unknown"]
type Image = NDArray[np.uint8]


LINE_NAMES: tuple[str, ...] = (
    "left_sideline",
    "far_baseline",
    "right_sideline",
    "near_baseline",
    "near_attack",
    "center",
    "far_attack",
)


def _float_tuple(values: Sequence[Any], length: int) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {len(values)}")
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class CourtLine:
    segment: Segment
    score: float
    family: LineFamily
    family_score: float
    identity: int | None
    identity_score: float
    vote_count: int
    roi_score: float
    orientation: Point
    family_probabilities: tuple[float, float] = (0.5, 0.5)
    identity_probabilities: tuple[float, ...] = ()

    @property
    def name(self) -> str | None:
        if self.identity is None or not 0 <= self.identity < len(LINE_NAMES):
            return None
        return LINE_NAMES[self.identity]

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.segment
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> CourtLine:
        raw_segment = row.get("segment", ())
        raw_orientation = row.get("orientation", (1.0, 0.0))
        raw_family = str(row.get("family", "unknown"))
        family = cast(
            LineFamily,
            raw_family if raw_family in {"vertical", "horizontal"} else "unknown",
        )
        raw_identity = row.get("line_identity")
        return cls(
            segment=cast(Segment, _float_tuple(cast(Sequence[Any], raw_segment), 4)),
            score=float(row.get("score", 0.0)),
            family=family,
            family_score=float(row.get("family_score", 0.0)),
            identity=int(raw_identity) if raw_identity is not None else None,
            identity_score=float(row.get("identity_score", 0.0)),
            vote_count=int(row.get("vote_count", 0)),
            roi_score=float(row.get("roi_score", 1.0)),
            orientation=cast(
                Point,
                _float_tuple(cast(Sequence[Any], raw_orientation), 2),
            ),
            family_probabilities=cast(
                tuple[float, float],
                _float_tuple(
                    cast(Sequence[Any], row.get("family_probabilities", (0.5, 0.5))),
                    2,
                ),
            ),
            identity_probabilities=tuple(
                float(value) for value in row.get("identity_probabilities", ())
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "class": "court_line",
            "segment": list(self.segment),
            "center": list(self.center),
            "orientation": list(self.orientation),
            "score": self.score,
            "family": self.family,
            "family_score": self.family_score,
            "family_probabilities": list(self.family_probabilities),
            "line_identity": self.identity,
            "identity_score": self.identity_score,
            "identity_probabilities": list(self.identity_probabilities),
            "vote_count": self.vote_count,
            "roi_score": self.roi_score,
        }


@dataclass(frozen=True, slots=True)
class CourtKeypoint:
    id: int
    x: float
    y: float
    score: float
    in_frame: bool
    source: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> CourtKeypoint:
        return cls(
            id=int(row["id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            score=float(row.get("score", 0.0)),
            in_frame=bool(row.get("in_frame", False)),
            source=str(row.get("source", "unknown")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "score": self.score,
            "in_frame": self.in_frame,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CourtLayout:
    status: LayoutStatus
    score: float
    reason: str
    keypoints: tuple[CourtKeypoint, ...]
    candidate_keypoints: tuple[CourtKeypoint, ...]
    matched_line_count: int
    hypothesis_margin: float
    semantic_alignment: float | None
    homography: tuple[tuple[float, float, float], ...] | None
    matcher_mode: str = "unknown"
    hypotheses_evaluated: int = 0

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> CourtLayout:
        raw_status = str(row.get("status", "unknown"))
        status = cast(
            LayoutStatus,
            raw_status if raw_status in {"ok", "ambiguous", "abstained"} else "unknown",
        )
        raw_homography = row.get("homography")
        homography = (
            cast(
                tuple[tuple[float, float, float], ...],
                tuple(_float_tuple(cast(Sequence[Any], values), 3) for values in raw_homography),
            )
            if isinstance(raw_homography, Sequence)
            else None
        )
        return cls(
            status=status,
            score=float(row.get("layout_score", 0.0)),
            reason=str(row.get("reason", "")),
            keypoints=tuple(
                CourtKeypoint.from_mapping(value) for value in row.get("keypoints", ())
            ),
            candidate_keypoints=tuple(
                CourtKeypoint.from_mapping(value) for value in row.get("candidate_keypoints", ())
            ),
            matched_line_count=int(row.get("matched_line_count", 0)),
            hypothesis_margin=float(row.get("hypothesis_margin", 0.0)),
            semantic_alignment=(
                float(row["semantic_alignment"])
                if row.get("semantic_alignment") is not None
                else None
            ),
            homography=homography,
            matcher_mode=str(row.get("matcher_mode", "unknown")),
            hypotheses_evaluated=int(row.get("hypotheses_evaluated", 0)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "layout_score": self.score,
            "matched_line_count": self.matched_line_count,
            "hypothesis_margin": self.hypothesis_margin,
            "semantic_alignment": self.semantic_alignment,
            "homography": [list(row) for row in self.homography] if self.homography else None,
            "matcher_mode": self.matcher_mode,
            "hypotheses_evaluated": self.hypotheses_evaluated,
            "keypoints": [point.to_mapping() for point in self.keypoints],
            "candidate_keypoints": [point.to_mapping() for point in self.candidate_keypoints],
        }


@dataclass(frozen=True, slots=True)
class CourtFrameResult:
    lines: tuple[CourtLine, ...]
    width: int
    height: int
    inference_seconds: float
    layout: CourtLayout | None = None
    heatmap: NDArray[np.float32] | None = None

    @property
    def inference_fps(self) -> float:
        return 1.0 / max(self.inference_seconds, 1e-9)

    def with_layout(self, layout: CourtLayout | None) -> CourtFrameResult:
        return replace(self, layout=layout)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "inference_seconds": self.inference_seconds,
            "inference_fps": self.inference_fps,
            "lines": [line.to_mapping() for line in self.lines],
            "layout": self.layout.to_mapping() if self.layout else None,
        }
