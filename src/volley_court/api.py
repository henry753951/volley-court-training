from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .assets import ModelSpec, resolve_model_path
from .inference import infer_frame, infer_frames, resolve_device
from .layout import layout_from_direct_corners, match_court_layout
from .model import YOLO26CourtLine, load_court_line_checkpoint
from .types import CourtFrameResult, CourtLayout, CourtLine, Image


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    device: str = "auto"
    image_size: int = 512
    confidence: float = 0.25
    top_k: int = 256
    half: bool = True
    fuse: bool = True
    decoder: str = "auto"
    include_layout: bool = True
    anchor_confidence: float = 0.0
    anchor_ransac_threshold: float = 0.01
    anchor_ransac_max_iters: int = 128
    anchor_solver: str = "hybrid"
    minimum_anchor_inlier_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.image_size < 64 or self.image_size % 4:
            raise ValueError("image_size must be >=64 and divisible by four")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.decoder not in {"auto", "spatial", "cuda"}:
            raise ValueError(f"unsupported decoder: {self.decoder}")
        if not 0.0 <= self.anchor_confidence <= 1.0:
            raise ValueError("anchor_confidence must be between zero and one")
        if not 0.0 < self.anchor_ransac_threshold <= 0.1:
            raise ValueError("anchor_ransac_threshold must be between zero and 0.1")
        if self.anchor_ransac_max_iters < 16:
            raise ValueError("anchor_ransac_max_iters must be at least 16")
        if self.anchor_solver not in {
            "hybrid",
            "ransac",
            "rho",
            "usac_default",
            "usac_accurate",
            "usac_magsac",
            "usac_prosac",
        }:
            raise ValueError(f"unsupported anchor_solver: {self.anchor_solver}")
        if not 0.0 <= self.minimum_anchor_inlier_ratio <= 1.0:
            raise ValueError("minimum_anchor_inlier_ratio must be between zero and one")


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
        if self.config.fuse:
            self._model.fuse()
        if self.config.half and self.device.type == "cuda":
            self._model.half()
        self.target_mode = str(metadata.get("target_mode", "dense_semantic"))
        self.sample_spacing = float(metadata.get("sample_spacing", 16.0))

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
        segments, heatmap, direct, seconds = infer_frame(
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
            decoder=self.config.decoder,
            anchor_confidence=self.config.anchor_confidence,
            anchor_ransac_threshold=self.config.anchor_ransac_threshold,
            anchor_ransac_max_iters=self.config.anchor_ransac_max_iters,
            anchor_solver=self.config.anchor_solver,
        )
        result = self._result(frame, segments, seconds, heatmap)
        should_layout = self.config.include_layout if include_layout is None else include_layout
        return self.attach_layout(result, direct) if should_layout else result

    def predict_many(
        self,
        frames: Sequence[Image],
        *,
        include_layout: bool | None = None,
        return_heatmap: bool = False,
    ) -> list[CourtFrameResult]:
        if not frames:
            return []
        segments, heatmaps, direct, seconds = infer_frames(
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
            decoder=self.config.decoder,
            anchor_confidence=self.config.anchor_confidence,
            anchor_ransac_threshold=self.config.anchor_ransac_threshold,
            anchor_ransac_max_iters=self.config.anchor_ransac_max_iters,
            anchor_solver=self.config.anchor_solver,
        )
        per_frame_seconds = seconds / len(frames)
        results = [
            self._result(frame, rows, per_frame_seconds, heatmap)
            for frame, rows, heatmap in zip(frames, segments, heatmaps, strict=True)
        ]
        should_layout = self.config.include_layout if include_layout is None else include_layout
        return (
            [
                self.attach_layout(result, direct_row)
                for result, direct_row in zip(results, direct, strict=True)
            ]
            if should_layout
            else results
        )

    def attach_layout(
        self,
        result: CourtFrameResult,
        direct_prediction: dict[str, Any] | None = None,
    ) -> CourtFrameResult:
        rows = [line.to_mapping() for line in result.lines]
        if direct_prediction is not None:
            proposals = np.asarray(direct_prediction["corner_proposals"], dtype=np.float64)
            probabilities = np.asarray(
                direct_prediction["proposal_probabilities"], dtype=np.float64
            )
            orientation_indices = direct_prediction.get("proposal_orientation_indices", [])
            anchor_count = int(direct_prediction.get("anchor_count", 0))
            inlier_counts = direct_prediction.get("anchor_inlier_counts", [])
            fit_errors = direct_prediction.get("anchor_fit_errors", [])
            candidates = []
            for index, proposal in enumerate(proposals):
                candidate = layout_from_direct_corners(
                    proposal[None],
                    np.asarray([probabilities[index]], dtype=np.float64),
                    float(direct_prediction["validity_probability"]),
                    rows,
                    result.width,
                    result.height,
                    orientation_index=int(
                        orientation_indices[index] if index < len(orientation_indices) else 0
                    ),
                    orientation_margin=float(direct_prediction.get("orientation_margin", 1.0)),
                )
                inlier_count = int(inlier_counts[index] if index < len(inlier_counts) else 0)
                candidate["anchor_count"] = anchor_count
                candidate["anchor_inlier_count"] = inlier_count
                candidate["anchor_fit_error"] = (
                    fit_errors[index] if index < len(fit_errors) else None
                )
                candidate["anchor_symmetry_index"] = int(
                    orientation_indices[index] if index < len(orientation_indices) else -1
                )
                inlier_ratio = float(inlier_count) / anchor_count if anchor_count else 0.0
                candidate["anchor_inlier_ratio"] = inlier_ratio
                if (
                    candidate.get("status") == "ok"
                    and anchor_count >= 4
                    and inlier_ratio < self.config.minimum_anchor_inlier_ratio
                ):
                    candidate["status"] = "abstained"
                    candidate["reason"] = "direct anchors have insufficient geometric consensus"
                    candidate["keypoints"] = []
                candidates.append(candidate)
                if candidate.get("status") == "ok":
                    break
            raw = (
                candidates[-1]
                if candidates and candidates[-1].get("status") == "ok"
                else candidates[0]
                if candidates
                else layout_from_direct_corners(
                    proposals,
                    probabilities,
                    float(direct_prediction["validity_probability"]),
                    rows,
                    result.width,
                    result.height,
                )
            )
            raw.setdefault("anchor_count", anchor_count)
            raw.setdefault("anchor_inlier_count", 0)
            raw.setdefault("anchor_inlier_ratio", 0.0)
            raw.setdefault("anchor_fit_error", None)
            raw.setdefault("anchor_symmetry_index", -1)
        else:
            raw = match_court_layout(rows, result.width, result.height)
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
