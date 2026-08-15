from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import median
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .layout import CANONICAL_KEYPOINTS, COURT_SYMMETRY_TRANSFORMS
from .types import CourtKeypoint, CourtLayout, Image


@dataclass(frozen=True, slots=True)
class LayoutTrackingConfig:
    """Temporal smoothing for accepted court layouts."""

    smoothing: float = 0.65
    fast_motion_threshold_px: float = 18.0
    max_identity_jump_ratio: float = 0.08
    max_hold_frames: int = 60
    max_static_hold_frames: int = 2
    reacquire_confirmation_frames: int = 3
    max_ambiguous_continuation_ratio: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.fast_motion_threshold_px <= 0.0:
            raise ValueError("fast_motion_threshold_px must be positive")
        if not 0.0 < self.max_identity_jump_ratio < 1.0:
            raise ValueError("max_identity_jump_ratio must be in (0, 1)")
        if self.max_hold_frames < 0:
            raise ValueError("max_hold_frames cannot be negative")
        if self.max_static_hold_frames < 0:
            raise ValueError("max_static_hold_frames cannot be negative")
        if self.reacquire_confirmation_frames < 1:
            raise ValueError("reacquire_confirmation_frames must be positive")
        if not 0.0 < self.max_ambiguous_continuation_ratio < 1.0:
            raise ValueError("max_ambiguous_continuation_ratio must be in (0, 1)")


class CourtLayoutTracker:
    """Keep a confirmed layout stable without reusing stale sparse detections."""

    def __init__(self, config: LayoutTrackingConfig | None = None) -> None:
        self.config = config or LayoutTrackingConfig()
        self._current: CourtLayout | None = None
        self._missed_frames = 0
        self._previous_gray: NDArray[np.uint8] | None = None
        self._pending_layout: CourtLayout | None = None
        self._pending_count = 0

    @property
    def current(self) -> CourtLayout | None:
        return self._current

    def reset(self) -> None:
        self._current = None
        self._missed_frames = 0
        self._previous_gray = None
        self._clear_pending()

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
            layout = self._canonicalize_layout(layout, width=width, height=height)
        if layout is not None and layout.status == "ok" and len(layout.keypoints) == 36:
            if self._pending_layout is not None and self._current is None:
                return self._handle_discontinuity(
                    layout,
                    gray=gray,
                    width=width,
                    height=height,
                )
            identity_match = self._match_identity(layout, width=width, height=height)
            if identity_match is not None:
                direct_distance = identity_match
                if direct_distance > self.config.max_identity_jump_ratio:
                    return self._handle_discontinuity(
                        layout,
                        gray=gray,
                        width=width,
                        height=height,
                    )
            self._clear_pending()
            self._current = self._smooth(layout, width=width, height=height)
            self._missed_frames = 0
            self._previous_gray = gray
            return self._current
        continuation = self._ambiguous_continuation(layout, width=width, height=height)
        if continuation is not None:
            self._clear_pending()
            self._current = self._smooth(continuation, width=width, height=height)
            self._current = replace(
                self._current,
                score=min(self._current.score, continuation.score) * 0.995,
                reason="continued from temporally confirmed ambiguous geometry",
            )
            self._missed_frames = 0
            self._previous_gray = gray
            return self._current
        self._clear_pending()
        if self._current is not None and self._missed_frames < self.config.max_hold_frames:
            self._missed_frames += 1
            tracked = self._track_frame(gray, width=width, height=height)
            if tracked is not None:
                self._current = tracked
                self._previous_gray = gray
                return self._current
            if self._missed_frames <= self.config.max_static_hold_frames:
                self._current = replace(
                    self._current,
                    score=self._current.score * 0.985,
                    reason=(
                        f"tracked static hold {self._missed_frames}/"
                        f"{self.config.max_static_hold_frames}"
                    ),
                )
                self._previous_gray = gray
                return self._current
        self.reset()
        self._previous_gray = gray
        return layout

    def _ambiguous_continuation(
        self,
        layout: CourtLayout | None,
        *,
        width: int,
        height: int,
    ) -> CourtLayout | None:
        if (
            self._current is None
            or layout is None
            or layout.status != "ambiguous"
            or layout.reason != "semantic line identities reject the direct layout"
            or layout.matched_line_count < 5
            or layout.homography is None
            or len(layout.candidate_keypoints) != 36
        ):
            return None
        promoted = replace(
            layout,
            status="ok",
            keypoints=layout.candidate_keypoints,
            reason="candidate for temporal continuation",
        )
        canonical = self._canonicalize_layout(promoted, width=width, height=height)
        if canonical is None:
            return None
        distance = self._match_identity(canonical, width=width, height=height)
        if distance is None or distance > self.config.max_ambiguous_continuation_ratio:
            return None
        return canonical

    def _handle_discontinuity(
        self,
        layout: CourtLayout,
        *,
        gray: NDArray[np.uint8] | None,
        width: int,
        height: int,
    ) -> CourtLayout | None:
        if self._pending_layout is None or not self._layouts_are_close(
            self._pending_layout,
            layout,
            width=width,
            height=height,
        ):
            self._pending_count = 1
        else:
            self._pending_count += 1
        self._pending_layout = layout
        if self._pending_count >= self.config.reacquire_confirmation_frames:
            self._current = self._recompute_visibility(layout, width=width, height=height)
            self._current = replace(
                self._current,
                reason=f"reacquired canonical layout after {self._pending_count} confirmations",
            )
            self._missed_frames = 0
            self._clear_pending()
            self._previous_gray = gray
            return self._current

        self._missed_frames += 1
        tracked = self._track_frame(gray, width=width, height=height)
        if tracked is not None:
            self._current = replace(
                tracked,
                reason="tracked through pending layout discontinuity",
            )
        elif self._current is not None and self._missed_frames <= self.config.max_static_hold_frames:
            self._current = replace(
                self._current,
                score=self._current.score * 0.985,
                reason=(
                    f"pending layout discontinuity {self._pending_count}/"
                    f"{self.config.reacquire_confirmation_frames}"
                ),
            )
        else:
            self._current = None
        self._previous_gray = gray
        return self._current

    def _layouts_are_close(
        self,
        first: CourtLayout,
        second: CourtLayout,
        *,
        width: int,
        height: int,
    ) -> bool:
        first_points = {point.id: point for point in first.keypoints}
        second_points = {point.id: point for point in second.keypoints}
        diagonal = math.hypot(width, height)
        if len(first_points) != 36 or len(second_points) != 36 or diagonal <= 0.0:
            return False
        distance = median(
            math.hypot(
                first_points[index].x - second_points[index].x,
                first_points[index].y - second_points[index].y,
            )
            / diagonal
            for index in range(36)
        )
        return distance <= self.config.max_identity_jump_ratio

    def _clear_pending(self) -> None:
        self._pending_layout = None
        self._pending_count = 0

    def _match_identity(
        self,
        layout: CourtLayout,
        *,
        width: int,
        height: int,
    ) -> float | None:
        if self._current is None or len(self._current.keypoints) != 36:
            return None
        previous = {point.id: point for point in self._current.keypoints}
        candidate = {point.id: point for point in layout.keypoints}
        if len(previous) != 36 or len(candidate) != 36:
            return None
        diagonal = math.hypot(width, height)
        if diagonal <= 0.0:
            return None
        return median(
            math.hypot(
                previous[index].x - candidate[index].x,
                previous[index].y - candidate[index].y,
            )
            / diagonal
            for index in range(36)
        )

    def _canonicalize_layout(
        self,
        layout: CourtLayout,
        *,
        width: int,
        height: int,
    ) -> CourtLayout | None:
        if layout.homography is None:
            return layout
        resolved = self._resolve_image_facing_symmetry(
            np.asarray(layout.homography, dtype=np.float64)
        )
        if resolved is None:
            return None
        homography, symmetry_index = resolved
        canonical = self._replace_from_homography(
            layout,
            homography,
            width=width,
            height=height,
            source="canonical_identity_lock",
            reason=(
                "canonical image-facing layout"
                if symmetry_index == 0
                else f"canonicalized layout symmetry={symmetry_index}"
            ),
        )
        return canonical

    @staticmethod
    def _resolve_image_facing_symmetry(
        homography: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], int] | None:
        canonical_corners = np.asarray(
            ((0.0, 0.0), (9.0, 0.0), (0.0, 18.0), (9.0, 18.0)),
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        for symmetry_index, symmetry in enumerate(COURT_SYMMETRY_TRANSFORMS):
            candidate = np.asarray(homography, dtype=np.float64) @ symmetry
            corners = np.asarray(
                cv2.perspectiveTransform(canonical_corners, candidate), dtype=np.float64
            ).reshape(-1, 2)
            if not np.isfinite(corners).all():
                continue
            length_axis = np.mean(corners[2:], axis=0) - np.mean(corners[:2], axis=0)
            width_axis = np.mean(corners[[1, 3]], axis=0) - np.mean(corners[[0, 2]], axis=0)
            length_component = length_axis[int(abs(length_axis[1]) > abs(length_axis[0]))]
            width_component = width_axis[int(abs(width_axis[1]) > abs(width_axis[0]))]
            if length_component >= 0.0 or width_component <= 0.0:
                continue
            scale = float(candidate[2, 2])
            normalized = candidate / scale if abs(scale) > 1e-12 else candidate
            return np.asarray(normalized, dtype=np.float64), symmetry_index
        return None

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
            synchronized = self._replace_from_homography(
                self._current,
                composed,
                width=width,
                height=height,
                source="temporal_optical_flow",
                reason="tracked with optical flow",
            )
            if synchronized is not None:
                return replace(synchronized, score=self._current.score * 0.995)
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
        if layout.homography is not None and self._current.homography is not None:
            by_id = {point.id: point for point in points}
            outer_ids = (0, 4, 5, 9)
            if all(index in by_id for index in outer_ids):
                source = np.asarray(
                    [CANONICAL_KEYPOINTS[index] for index in outer_ids], dtype=np.float32
                )
                destination = np.asarray(
                    [(by_id[index].x, by_id[index].y) for index in outer_ids], dtype=np.float32
                )
                homography = cv2.getPerspectiveTransform(source, destination)
                synchronized = self._replace_from_homography(
                    layout,
                    np.asarray(homography, dtype=np.float64),
                    width=width,
                    height=height,
                    source="temporally_smoothed_homography",
                    reason="temporally tracked layout",
                )
                if synchronized is not None:
                    return synchronized
        return replace(layout, keypoints=points, reason="temporally tracked layout")

    @staticmethod
    def _replace_from_homography(
        layout: CourtLayout,
        homography: NDArray[np.float64],
        *,
        width: int,
        height: int,
        source: str,
        reason: str,
    ) -> CourtLayout | None:
        matrix = np.asarray(homography, dtype=np.float64).copy()
        scale = float(matrix[2, 2])
        if abs(scale) <= 1e-12:
            return None
        matrix /= scale
        canonical = np.asarray(CANONICAL_KEYPOINTS, dtype=np.float64).reshape(-1, 1, 2)
        projected = np.asarray(cv2.perspectiveTransform(canonical, matrix), dtype=np.float64).reshape(
            -1, 2
        )
        if not np.isfinite(projected).all():
            return None
        by_id = {point.id: point for point in layout.keypoints}
        if len(by_id) != 36:
            return None
        keypoints = tuple(
            replace(
                by_id[index],
                x=float(position[0]),
                y=float(position[1]),
                in_frame=(0.0 <= float(position[0]) < width and 0.0 <= float(position[1]) < height),
                source=source,
            )
            for index, position in enumerate(projected)
        )
        homography_rows = cast(
            tuple[tuple[float, float, float], ...],
            tuple(tuple(float(value) for value in row) for row in matrix),
        )
        return replace(
            layout,
            keypoints=keypoints,
            homography=homography_rows,
            reason=reason,
        )

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
