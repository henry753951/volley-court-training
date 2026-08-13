from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import median
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .layout import COURT_SYMMETRY_TRANSFORMS, POSE36_SYMMETRY_MAPS
from .types import CourtKeypoint, CourtLayout, Image


@dataclass(frozen=True, slots=True)
class LayoutTrackingConfig:
    """Temporal smoothing for accepted court layouts."""

    smoothing: float = 0.65
    fast_motion_threshold_px: float = 18.0
    max_identity_jump_ratio: float = 0.08
    max_hold_frames: int = 25

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.fast_motion_threshold_px <= 0.0:
            raise ValueError("fast_motion_threshold_px must be positive")
        if not 0.0 < self.max_identity_jump_ratio < 1.0:
            raise ValueError("max_identity_jump_ratio must be in (0, 1)")
        if self.max_hold_frames < 0:
            raise ValueError("max_hold_frames cannot be negative")


class CourtLayoutTracker:
    """Keep a confirmed layout stable without reusing stale sparse detections."""

    def __init__(self, config: LayoutTrackingConfig | None = None) -> None:
        self.config = config or LayoutTrackingConfig()
        self._current: CourtLayout | None = None
        self._missed_frames = 0
        self._previous_gray: NDArray[np.uint8] | None = None

    @property
    def current(self) -> CourtLayout | None:
        return self._current

    def reset(self) -> None:
        self._current = None
        self._missed_frames = 0
        self._previous_gray = None

    def update(
        self,
        layout: CourtLayout | None,
        *,
        width: int,
        height: int,
        frame: Image | None = None,
    ) -> CourtLayout | None:
        gray = self._gray(frame) if frame is not None else None
        if layout is not None and layout.status == "ok" and len(layout.keypoints) == 36:
            identity_match = self._match_identity(layout, width=width, height=height)
            if identity_match is not None:
                symmetry_index, direct_distance, best_distance = identity_match
                if (
                    symmetry_index != 0
                    and direct_distance > self.config.max_identity_jump_ratio
                    and best_distance <= self.config.max_identity_jump_ratio
                ):
                    layout = self._align_identity(layout, symmetry_index)
                elif best_distance > self.config.max_identity_jump_ratio:
                    tracked = self._track_frame(gray, width=width, height=height)
                    if tracked is not None:
                        self._current = replace(
                            tracked,
                            reason="tracked through rejected layout discontinuity",
                        )
                        self._missed_frames = 0
                        self._previous_gray = gray
                        return self._current
                    self._current = self._recompute_visibility(
                        layout,
                        width=width,
                        height=height,
                    )
                    self._missed_frames = 0
                    self._previous_gray = gray
                    return self._current
            self._current = self._smooth(layout, width=width, height=height)
            self._missed_frames = 0
            self._previous_gray = gray
            return self._current
        if self._current is not None and self._missed_frames < self.config.max_hold_frames:
            self._missed_frames += 1
            tracked = self._track_frame(gray, width=width, height=height)
            self._current = tracked or replace(
                self._current,
                score=self._current.score * 0.985,
                reason=f"tracked hold {self._missed_frames}/{self.config.max_hold_frames}",
            )
            self._previous_gray = gray
            return self._current
        self.reset()
        self._previous_gray = gray
        return layout

    def _match_identity(
        self,
        layout: CourtLayout,
        *,
        width: int,
        height: int,
    ) -> tuple[int, float, float] | None:
        if self._current is None or len(self._current.keypoints) != 36:
            return None
        previous = {point.id: point for point in self._current.keypoints}
        candidate = {point.id: point for point in layout.keypoints}
        if len(previous) != 36 or len(candidate) != 36:
            return None
        diagonal = math.hypot(width, height)
        if diagonal <= 0.0:
            return None
        distances = tuple(
            median(
                math.hypot(
                    previous[index].x - candidate[permutation[index]].x,
                    previous[index].y - candidate[permutation[index]].y,
                )
                / diagonal
                for index in range(36)
            )
            for permutation in POSE36_SYMMETRY_MAPS
        )
        symmetry_index = min(range(len(distances)), key=distances.__getitem__)
        return symmetry_index, distances[0], distances[symmetry_index]

    @staticmethod
    def _align_identity(layout: CourtLayout, symmetry_index: int) -> CourtLayout:
        permutation = POSE36_SYMMETRY_MAPS[symmetry_index]
        by_id = {point.id: point for point in layout.keypoints}
        keypoints = tuple(
            replace(by_id[permutation[index]], id=index, source="temporal_identity_match")
            for index in range(36)
        )
        inverse = {source_id: target_id for target_id, source_id in enumerate(permutation)}
        candidate_keypoints = tuple(
            sorted(
                (
                    replace(
                        point,
                        id=inverse.get(point.id, point.id),
                        source="temporal_identity_match",
                    )
                    for point in layout.candidate_keypoints
                ),
                key=lambda point: point.id,
            )
        )
        homography = layout.homography
        if homography is not None:
            aligned = (
                np.asarray(homography, dtype=np.float64) @ COURT_SYMMETRY_TRANSFORMS[symmetry_index]
            )
            scale = float(aligned[2, 2])
            if abs(scale) > 1e-12:
                aligned /= scale
            homography = cast(
                tuple[tuple[float, float, float], ...],
                tuple(tuple(float(value) for value in row) for row in aligned),
            )
        return replace(
            layout,
            keypoints=keypoints,
            candidate_keypoints=candidate_keypoints,
            homography=homography,
            reason=f"temporally matched layout identity symmetry={symmetry_index}",
        )

    @staticmethod
    def _recompute_visibility(
        layout: CourtLayout,
        *,
        width: int,
        height: int,
    ) -> CourtLayout:
        return replace(
            layout,
            keypoints=tuple(
                replace(
                    point,
                    in_frame=0.0 <= point.x < width and 0.0 <= point.y < height,
                )
                for point in layout.keypoints
            ),
        )

    @staticmethod
    def _gray(frame: Image) -> NDArray[np.uint8]:
        return np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)

    def _track_frame(
        self,
        gray: NDArray[np.uint8] | None,
        *,
        width: int,
        height: int,
    ) -> CourtLayout | None:
        if self._current is None or self._previous_gray is None or gray is None:
            return None
        visible = [
            point
            for point in self._current.keypoints
            if 2.0 <= point.x < width - 2.0 and 2.0 <= point.y < height - 2.0
        ]
        if len(visible) < 4:
            return None
        previous_points = np.asarray([(point.x, point.y) for point in visible], dtype=np.float32)
        next_guess = np.empty_like(previous_points.reshape(-1, 1, 2))
        next_points_raw, status_raw, _errors = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            previous_points.reshape(-1, 1, 2),
            next_guess,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        )
        if next_points_raw is None or status_raw is None:
            return None
        next_points = np.asarray(next_points_raw, dtype=np.float32).reshape(-1, 2)
        status = np.asarray(status_raw, dtype=np.uint8).reshape(-1).astype(bool)
        valid_previous = previous_points[status]
        valid_next = next_points[status]
        if len(valid_previous) < 4:
            return None
        delta_raw, mask_raw = cv2.findHomography(
            valid_previous,
            valid_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if delta_raw is None or mask_raw is None:
            return None
        inliers = int(np.count_nonzero(np.asarray(mask_raw).reshape(-1)))
        if inliers < max(4, math.ceil(0.5 * len(valid_previous))):
            return None
        median_flow = median(
            float(math.hypot(*(second - first)))
            for first, second in zip(valid_previous, valid_next, strict=True)
        )
        if median_flow > 0.12 * math.hypot(width, height):
            return None
        delta = np.asarray(delta_raw, dtype=np.float64)
        all_points = np.asarray(
            [(point.x, point.y) for point in self._current.keypoints],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        projected = np.asarray(
            cv2.perspectiveTransform(all_points, delta), dtype=np.float64
        ).reshape(-1, 2)
        if not np.isfinite(projected).all():
            return None
        points = tuple(
            replace(
                point,
                x=float(position[0]),
                y=float(position[1]),
                in_frame=(0.0 <= float(position[0]) < width and 0.0 <= float(position[1]) < height),
                source="temporal_optical_flow",
            )
            for point, position in zip(self._current.keypoints, projected, strict=True)
        )
        homography = self._current.homography
        if homography is not None:
            composed = delta @ np.asarray(homography, dtype=np.float64)
            homography = cast(
                tuple[tuple[float, float, float], ...],
                tuple(tuple(float(value) for value in row) for row in composed),
            )
        return replace(
            self._current,
            keypoints=points,
            homography=homography,
            score=self._current.score * 0.995,
            reason="tracked with optical flow",
        )

    def _smooth(self, layout: CourtLayout, *, width: int, height: int) -> CourtLayout:
        if self._current is None or len(self._current.keypoints) != len(layout.keypoints):
            return layout
        previous = {point.id: point for point in self._current.keypoints}
        displacements = [
            math.hypot(point.x - previous[point.id].x, point.y - previous[point.id].y)
            for point in layout.keypoints
            if point.id in previous
        ]
        motion = median(displacements) if displacements else 0.0
        alpha = min(
            0.95,
            self.config.smoothing
            + (1.0 - self.config.smoothing)
            * min(motion / self.config.fast_motion_threshold_px, 1.0),
        )
        points = tuple(
            self._smooth_point(
                point,
                previous.get(point.id),
                alpha=alpha,
                width=width,
                height=height,
            )
            for point in layout.keypoints
        )
        return replace(layout, keypoints=points, reason="temporally tracked layout")

    @staticmethod
    def _smooth_point(
        point: CourtKeypoint,
        previous: CourtKeypoint | None,
        *,
        alpha: float,
        width: int,
        height: int,
    ) -> CourtKeypoint:
        if previous is None:
            return point
        x = alpha * point.x + (1.0 - alpha) * previous.x
        y = alpha * point.y + (1.0 - alpha) * previous.y
        return replace(
            point,
            x=x,
            y=y,
            in_frame=0.0 <= x < width and 0.0 <= y < height,
            source="temporally_smoothed_homography",
        )
