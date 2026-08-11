from __future__ import annotations

import math
from dataclasses import dataclass

import cv2

from .geometry import clip_segment_to_image
from .types import CourtFrameResult, CourtKeypoint, CourtLine, Image


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    vertical_color: tuple[int, int, int] = (255, 210, 72)
    horizontal_color: tuple[int, int, int] = (52, 184, 255)
    evidence_color: tuple[int, int, int] = (105, 128, 148)
    unknown_color: tuple[int, int, int] = (145, 150, 158)
    keypoint_color: tuple[int, int, int] = (96, 235, 132)
    candidate_color: tuple[int, int, int] = (66, 170, 255)
    layout_line_thickness: int = 2
    evidence_tick_length: int = 8
    show_evidence: bool = True
    show_layout: bool = True
    show_labels: bool = False
    show_keypoints: bool = True
    show_keypoint_ids: bool = False
    show_candidates: bool = False
    show_panel: bool = True


@dataclass(frozen=True, slots=True)
class _LayoutLine:
    name: str
    family: str
    first_id: int
    second_id: int


_LAYOUT_LINES: tuple[_LayoutLine, ...] = (
    _LayoutLine("left_sideline", "vertical", 0, 4),
    _LayoutLine("far_baseline", "horizontal", 4, 5),
    _LayoutLine("right_sideline", "vertical", 5, 9),
    _LayoutLine("near_baseline", "horizontal", 9, 0),
    _LayoutLine("near_attack", "horizontal", 1, 8),
    _LayoutLine("center", "horizontal", 2, 7),
    _LayoutLine("far_attack", "horizontal", 3, 6),
)


class CourtVisualizer:
    """Render local evidence and accepted virtual layouts as distinct layers."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()

    def draw(
        self,
        frame: Image,
        result: CourtFrameResult,
        *,
        copy: bool = True,
    ) -> Image:
        output = frame.copy() if copy else frame
        accepted = result.layout is not None and result.layout.status == "ok"
        if self.config.show_evidence and not accepted:
            self._draw_evidence(output, result.lines)
        if self.config.show_layout and accepted:
            self._draw_layout(output, result)
        if self.config.show_keypoints and result.layout is not None:
            self._draw_keypoints(output, result, accepted=accepted)
        if self.config.show_panel:
            self._panel(output, result)
        return output

    def _draw_evidence(self, image: Image, lines: tuple[CourtLine, ...]) -> None:
        for line in lines:
            cx, cy = line.center
            ux, uy = line.orientation
            norm = max(math.hypot(ux, uy), 1e-6)
            half = 0.5 * self.config.evidence_tick_length
            dx, dy = half * ux / norm, half * uy / norm
            first = int(round(cx - dx)), int(round(cy - dy))
            second = int(round(cx + dx)), int(round(cy + dy))
            color = (
                self._family_color(line.family)
                if line.score >= 0.45
                else self.config.evidence_color
            )
            cv2.line(image, first, second, (16, 22, 30), 3, cv2.LINE_AA)
            cv2.line(image, first, second, color, 1, cv2.LINE_AA)
            cv2.circle(image, (int(round(cx)), int(round(cy))), 2, color, -1, cv2.LINE_AA)

    def _draw_layout(self, image: Image, result: CourtFrameResult) -> None:
        assert result.layout is not None
        points = {point.id: point for point in result.layout.keypoints}
        for line in _LAYOUT_LINES:
            first, second = points.get(line.first_id), points.get(line.second_id)
            if first is None or second is None:
                continue
            segment = clip_segment_to_image(
                (first.x, first.y, second.x, second.y),
                result.width,
                result.height,
            )
            if segment is None:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in segment)
            color = self._family_color(line.family)
            cv2.line(image, (x1, y1), (x2, y2), (12, 18, 25), 7, cv2.LINE_AA)
            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                color,
                self.config.layout_line_thickness,
                cv2.LINE_AA,
            )
            if self.config.show_labels:
                self._label(
                    image,
                    (0.5 * (x1 + x2), 0.5 * (y1 + y2)),
                    line.name,
                    color,
                )

    def _draw_keypoints(
        self,
        image: Image,
        result: CourtFrameResult,
        *,
        accepted: bool,
    ) -> None:
        assert result.layout is not None
        if accepted:
            points = result.layout.keypoints
            color = self.config.keypoint_color
        elif self.config.show_candidates:
            points = result.layout.candidate_keypoints
            color = self.config.candidate_color
        else:
            return
        for point in points:
            if not point.in_frame:
                continue
            self._virtual_point(image, point, color)

    def _virtual_point(
        self,
        image: Image,
        point: CourtKeypoint,
        color: tuple[int, int, int],
    ) -> None:
        center = int(round(point.x)), int(round(point.y))
        cv2.circle(image, center, 4, (12, 18, 24), -1, cv2.LINE_AA)
        cv2.circle(image, center, 3, color, 1, cv2.LINE_AA)
        if self.config.show_keypoint_ids:
            cv2.putText(
                image,
                str(point.id),
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )

    def _family_color(self, family: str) -> tuple[int, int, int]:
        if family == "vertical":
            return self.config.vertical_color
        if family == "horizontal":
            return self.config.horizontal_color
        return self.config.unknown_color

    @staticmethod
    def _label(
        image: Image,
        center: tuple[float, float],
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        origin = int(round(center[0])) + 6, int(round(center[1])) - 7
        (width, height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(
            image,
            (origin[0] - 3, origin[1] - height - 3),
            (origin[0] + width + 3, origin[1] + 4),
            (20, 25, 33),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _panel(image: Image, result: CourtFrameResult) -> None:
        status = result.layout.status if result.layout else "evidence"
        layout_row = (
            f"{status.upper()}   7 virtual lines / 36 points"
            if status == "ok"
            else f"{status.upper()}   {len(result.lines)} evidence votes"
        )
        rows = (
            "VOLLEY COURT / YOLO26N",
            layout_row,
            f"{result.inference_seconds * 1000.0:.2f} ms   {result.inference_fps:.1f} FPS",
        )
        overlay = image.copy()
        cv2.rectangle(overlay, (14, 14), (342, 92), (13, 18, 26), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, dst=image)
        for index, text in enumerate(rows):
            cv2.putText(
                image,
                text,
                (28, 38 + index * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48 if index == 0 else 0.42,
                (238, 244, 250) if index == 0 else (182, 197, 214),
                1,
                cv2.LINE_AA,
            )
