from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .assets import ModelSpec, resolve_model_path
from .inference import infer_frame, infer_frames, resolve_device
from .layout import match_court_layout, match_semantic_court_layout
from .model import YOLO26CourtLine, load_court_line_checkpoint
from .types import CourtFrameResult, CourtLayout, CourtLine, Image


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    device: str = "auto"
    image_size: int = 640
    confidence: float = 0.25
    top_k: int = 256
    half: bool = True
    decoder: str = "auto"
    include_layout: bool = False

    def __post_init__(self) -> None:
        if self.image_size < 64 or self.image_size % 4:
            raise ValueError("image_size must be >=64 and divisible by four")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.decoder not in {"auto", "spatial", "cuda"}:
            raise ValueError(f"unsupported decoder: {self.decoder}")


class CourtLineModel:
    """Typed inference facade for the YOLO26n court-line model."""

    def __init__(
        self,
        checkpoint: str | Path | ModelSpec | None = None,
        *,
        config: InferenceConfig | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.checkpoint = resolve_model_path(checkpoint)
        self.device = resolve_device(self.config.device)
        model, metadata = load_court_line_checkpoint(self.checkpoint, device=self.device)
        self._model: YOLO26CourtLine = model.eval()
        self.metadata: dict[str, Any] = metadata
        if self.config.half and self.device.type == "cuda":
            self._model.half()
        self.target_mode = str(metadata.get("target_mode", "dense_semantic"))
        self.sample_spacing = float(metadata.get("sample_spacing", 16.0))
        self.semantic_layout_v2 = metadata.get("release") == "semantic-layout-v2"

    @property
    def _decoder(self) -> str:
        if self.config.decoder != "auto":
            return self.config.decoder
        return "cuda" if self.semantic_layout_v2 else "spatial"

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path | ModelSpec | None = None,
        **config: Any,
    ) -> CourtLineModel:
        return cls(model, config=InferenceConfig(**config))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._model.parameters())

    @property
    def checkpoint_bytes(self) -> int:
        return self.checkpoint.stat().st_size

    @property
    def torch_model(self) -> torch.nn.Module:
        return self._model

    def warmup(self, *, batch_size: int = 1, iterations: int = 3) -> None:
        frames = [
            np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        for _ in range(iterations):
            self.predict_many(frames, include_layout=False)

    def predict(
        self,
        frame: Image,
        *,
        include_layout: bool | None = None,
        return_heatmap: bool = False,
    ) -> CourtFrameResult:
        segments, heatmap, seconds = infer_frame(
            self._model,
            frame,
            self.device,
            image_size=self.config.image_size,
            confidence=self.config.confidence,
            top_k=self.config.top_k,
            half=self.config.half,
            target_mode=self.target_mode,
            sample_spacing=self.sample_spacing,
            return_heatmap=return_heatmap,
            decoder=self._decoder,
        )
        result = self._result(frame, segments, seconds, heatmap)
        should_layout = self.config.include_layout if include_layout is None else include_layout
        return self.attach_layout(result) if should_layout else result

    def predict_many(
        self,
        frames: Sequence[Image],
        *,
        include_layout: bool | None = None,
        return_heatmap: bool = False,
    ) -> list[CourtFrameResult]:
        if not frames:
            return []
        segments, heatmaps, seconds = infer_frames(
            self._model,
            list(frames),
            self.device,
            image_size=self.config.image_size,
            confidence=self.config.confidence,
            top_k=self.config.top_k,
            half=self.config.half,
            target_mode=self.target_mode,
            sample_spacing=self.sample_spacing,
            return_heatmap=return_heatmap,
            decoder=self._decoder,
        )
        per_frame_seconds = seconds / len(frames)
        results = [
            self._result(frame, rows, per_frame_seconds, heatmap)
            for frame, rows, heatmap in zip(frames, segments, heatmaps, strict=True)
        ]
        should_layout = self.config.include_layout if include_layout is None else include_layout
        return [self.attach_layout(result) for result in results] if should_layout else results

    def attach_layout(
        self,
        result: CourtFrameResult,
        *,
        prior_layout: CourtLayout | None = None,
    ) -> CourtFrameResult:
        matcher = match_semantic_court_layout if self.semantic_layout_v2 else match_court_layout
        raw = matcher(
            [line.to_mapping() for line in result.lines],
            result.width,
            result.height,
            prior_homography=prior_layout.homography if prior_layout is not None else None,
        )
        return result.with_layout(CourtLayout.from_mapping(raw))

    def visualize(
        self,
        frame: Image,
        result: CourtFrameResult,
        *,
        copy: bool = True,
    ) -> Image:
        from .visualization import CourtVisualizer

        return CourtVisualizer().draw(frame, result, copy=copy)

    def __call__(self, frame: Image) -> CourtFrameResult:
        return self.predict(frame)

    @staticmethod
    def _result(
        frame: Image,
        rows: Sequence[dict[str, Any]],
        seconds: float,
        heatmap: np.ndarray | None,
    ) -> CourtFrameResult:
        height, width = frame.shape[:2]
        return CourtFrameResult(
            lines=tuple(CourtLine.from_mapping(row) for row in rows),
            width=width,
            height=height,
            inference_seconds=seconds,
            heatmap=heatmap,
        )
