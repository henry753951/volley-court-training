from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Conv

FEATURE_LAST_INDEX = 16
P2_INDEX = 2
DEEP_P3_INDEX = 16
DEFAULT_OUTPUT_CHANNELS = 6
DEFAULT_POSE_ARCHITECTURE = "yolo26n-pose.yaml"


class YOLO26CourtLine(nn.Module):
    """YOLO26n feature graph with a lightweight stride-4 center or dense-vote head."""

    stride = 4

    def __init__(
        self,
        feature_layers: nn.ModuleList,
        p2_channels: int,
        deep_channels: int,
        fusion_channels: int = 64,
        output_channels: int = DEFAULT_OUTPUT_CHANNELS,
    ) -> None:
        super().__init__()
        self.feature_layers = feature_layers
        self.deep_reduce = Conv(deep_channels, p2_channels, 1, 1)
        self.fusion = C3k2(
            p2_channels * 2,
            fusion_channels,
            n=1,
            c3k=False,
            e=0.5,
            shortcut=True,
        )
        self.head_stem = Conv(fusion_channels, fusion_channels, 3, 1)
        if output_channels not in {5, 6, 8, 15}:
            raise ValueError(
                f"court-line head supports five, six, eight, or fifteen channels, got {output_channels}"
            )
        self.output_channels = output_channels
        self.head = nn.Conv2d(fusion_channels, 5 if output_channels == 15 else output_channels, 1)
        self.context_stem = (
            Conv(fusion_channels, fusion_channels, 3, 1) if output_channels == 15 else None
        )
        self.context_head = nn.Conv2d(fusion_channels, 10, 1) if output_channels == 15 else None
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.head.bias)
        if self.context_head is not None:
            nn.init.normal_(self.context_head.weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.context_head.bias)
        with torch.no_grad():
            self.head.bias[0] = -2.19
        self._head_gradient_handles: list[Any] = []

    def _feature_forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: list[torch.Tensor] = []
        value: Any = image
        for module in self.feature_layers:
            source = getattr(module, "f", -1)
            if source != -1:
                value = (
                    outputs[source]
                    if isinstance(source, int)
                    else [value if index == -1 else outputs[index] for index in source]
                )
            value = module(value)
            outputs.append(value)
        return outputs[P2_INDEX], outputs[DEEP_P3_INDEX]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        p2, deep_p3 = self._feature_forward(image)
        deep_p3 = self.deep_reduce(deep_p3)
        deep_p3 = F.interpolate(deep_p3, size=p2.shape[-2:], mode="nearest")
        fused = self.fusion(torch.cat((p2, deep_p3), dim=1))
        features = self.head_stem(fused)
        dense = self.head(features)
        if self.context_head is None or self.context_stem is None:
            return dense
        return torch.cat((dense, self.context_head(self.context_stem(features))), dim=1)

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        for parameter in self.feature_layers.parameters():
            parameter.requires_grad_(not freeze)

    def freeze_shared_features(self, freeze: bool = True) -> None:
        for module in (self.feature_layers, self.deep_reduce, self.fusion, self.head_stem):
            for parameter in module.parameters():
                parameter.requires_grad_(not freeze)

    def set_shared_features_eval(self) -> None:
        for module in (self.feature_layers, self.deep_reduce, self.fusion, self.head_stem):
            module.eval()

    def set_base_head_gradient_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("base-head gradient scale must be between zero and one")
        for handle in self._head_gradient_handles:
            handle.remove()
        self._head_gradient_handles = []
        if self.output_channels not in {8, 15} or scale == 1.0:
            return

        def scale_base_rows(gradient: torch.Tensor) -> torch.Tensor:
            output = gradient.clone()
            output[:5].mul_(scale)
            return output

        self._head_gradient_handles = [
            self.head.weight.register_hook(scale_base_rows),
            self.head.bias.register_hook(scale_base_rows),
        ]


def _module_output_channels(module: nn.Module) -> int:
    cv2 = getattr(module, "cv2", None)
    conv = getattr(cv2, "conv", None)
    channels = getattr(conv, "out_channels", None)
    if channels is None:
        raise RuntimeError(f"cannot determine output channels for {type(module).__name__}")
    return int(channels)


def build_from_pose_checkpoint(
    checkpoint: str | Path,
    fusion_channels: int = 64,
    output_channels: int = DEFAULT_OUTPUT_CHANNELS,
) -> tuple[YOLO26CourtLine, dict[str, Any]]:
    """Transfer layers 0..16 from a real Pose36 checkpoint and discard its pose branch."""

    source_path = Path(checkpoint)
    architecture_only = source_path.suffix.lower() in {".yaml", ".yml"}
    if architecture_only:
        source = str(source_path.resolve()) if source_path.exists() else str(checkpoint)
    else:
        source = str(source_path.resolve())
    pose = YOLO(source).model.float()
    if type(pose).__name__ != "PoseModel":
        raise ValueError(f"expected an Ultralytics PoseModel, got {type(pose).__name__}")
    if not architecture_only and list(getattr(pose, "kpt_shape", ())) != [36, 3]:
        raise ValueError(
            f"expected Pose36 checkpoint, got kpt_shape={getattr(pose, 'kpt_shape', None)}"
        )
    layers = pose.model
    if len(layers) <= FEATURE_LAST_INDEX or type(layers[DEEP_P3_INDEX]).__name__ != "C3k2":
        raise ValueError("checkpoint does not match the inspected YOLO26n Pose feature graph")
    feature_layers = nn.ModuleList(copy.deepcopy(list(layers[: FEATURE_LAST_INDEX + 1])))
    model = YOLO26CourtLine(
        feature_layers,
        p2_channels=_module_output_channels(layers[P2_INDEX]),
        deep_channels=_module_output_channels(layers[DEEP_P3_INDEX]),
        fusion_channels=fusion_channels,
        output_channels=output_channels,
    )

    source_keys = list(layers.state_dict())
    transferred_keys = [
        key for key in source_keys if int(key.split(".", 1)[0]) <= FEATURE_LAST_INDEX
    ]
    discarded_keys = [key for key in source_keys if key not in transferred_keys]
    new_keys = [key for key in model.state_dict() if not key.startswith("feature_layers.")]
    report = {
        "source_checkpoint": source,
        "source_kind": "architecture" if architecture_only else "pose36_checkpoint",
        "source_model": type(pose).__name__,
        "source_kpt_shape": list(getattr(pose, "kpt_shape", ())),
        "feature_last_index": FEATURE_LAST_INDEX,
        "p2_index": P2_INDEX,
        "deep_p3_index": DEEP_P3_INDEX,
        "transferred_key_count": len(transferred_keys),
        "transferred_parameter_count": sum(p.numel() for p in model.feature_layers.parameters()),
        "missing_keys": new_keys,
        "unexpected_keys": discarded_keys,
        "new_random_parameter_count": sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.startswith("feature_layers.")
        ),
        "output_channels": output_channels,
        "major_stages": {
            "stem": True,
            "backbone_p2": True,
            "backbone_p3_p4_p5": True,
            "sppf_c2psa": True,
            "neck_to_fused_p3": True,
            "pose_head": False,
        },
    }
    return model, report


def load_court_line_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    pose_checkpoint: str | Path | None = None,
    output_channels_override: int | None = None,
) -> tuple[YOLO26CourtLine, dict[str, Any]]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"not a court-line checkpoint: {checkpoint}")
    checkpoint_path = Path(checkpoint).resolve()
    if pose_checkpoint is not None:
        source_pose: str | Path = Path(pose_checkpoint).resolve()
        if not Path(source_pose).exists():
            raise FileNotFoundError(f"explicit pose checkpoint does not exist: {source_pose}")
    else:
        source_pose = ""
        recorded_pose = payload.get("pose_checkpoint")
        if recorded_pose:
            recorded_path = Path(str(recorded_pose))
            candidates = [recorded_path]
            candidates.append(checkpoint_path.parent / recorded_path.name)
            for candidate in candidates:
                resolved_candidate = candidate.resolve()
                if resolved_candidate != checkpoint_path and resolved_candidate.exists():
                    source_pose = resolved_candidate
                    break
        if not source_pose:
            source_pose = str(payload.get("pose_architecture", DEFAULT_POSE_ARCHITECTURE))
    model, transfer_report = build_from_pose_checkpoint(
        source_pose,
        fusion_channels=int(payload.get("fusion_channels", 64)),
        output_channels=(
            int(output_channels_override)
            if output_channels_override is not None
            else int(payload.get("output_channels", DEFAULT_OUTPUT_CHANNELS))
        ),
    )
    source_state = payload["model"]
    if output_channels_override is None:
        state = source_state
    else:
        state = model.state_dict()
        copied = []
        expanded = []
        for key, value in source_state.items():
            if key not in state:
                continue
            if state[key].shape == value.shape:
                state[key] = value
                copied.append(key)
            elif key in {"head.weight", "head.bias"} and value.shape[0] < state[key].shape[0]:
                state[key][: value.shape[0]] = value
                expanded.append(key)
        transfer_report["checkpoint_upgrade"] = {
            "source_output_channels": int(payload.get("output_channels", DEFAULT_OUTPUT_CHANNELS)),
            "target_output_channels": int(output_channels_override),
            "copied_key_count": len(copied),
            "expanded_keys": expanded,
        }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"court-line state mismatch: {incompatible}")
    model.to(device)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "scheduler", "scaler"}
    }
    metadata["load_transfer_report"] = transfer_report
    metadata["resolved_pose_source"] = str(source_pose)
    return model, metadata
