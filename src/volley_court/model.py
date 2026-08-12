from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Conv, Conv2, ConvTranspose, DWConv, RepConv, RepVGGDW
from ultralytics.utils.torch_utils import fuse_conv_and_bn, fuse_deconv_and_bn

FEATURE_LAST_INDEX = 16
P2_INDEX = 2
DEEP_P3_INDEX = 16
DEFAULT_OUTPUT_CHANNELS = 6
DEFAULT_POSE_ARCHITECTURE = "yolo26n-pose.yaml"
DIRECT_LAYOUT_HEAD_VERSION = 5


@dataclass(frozen=True)
class DirectLayoutOutput:
    heatmap_logits: torch.Tensor
    offset_logits: torch.Tensor
    point_visibility_logits: torch.Tensor
    validity_logits: torch.Tensor
    orientation_logits: torch.Tensor


class DirectLayoutHead(nn.Module):
    """Stride-4 Pose36 anchor head used to solve a complete court homography."""

    keypoint_count = 36

    def __init__(
        self,
        input_channels: int,
        proposals: int = 4,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if proposals < 1:
            raise ValueError("direct layout head requires at least one proposal")
        self.proposals = proposals
        self.stem = nn.Sequential(
            Conv(input_channels, hidden_channels, 3, 2),
            Conv(hidden_channels, hidden_channels, 3, 2),
            Conv(hidden_channels, hidden_channels, 3, 1),
        )
        self.heatmap = nn.Conv2d(hidden_channels, self.keypoint_count, 1)
        self.offset = nn.Conv2d(hidden_channels, self.keypoint_count * 2, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, self.keypoint_count + 1),
        )
        self.orientation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 16, 8),
        )
        for layer in (self.heatmap, self.offset):
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            assert layer.bias is not None
            nn.init.zeros_(layer.bias)
        final = self.global_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=0.001)
        assert final.bias is not None
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[-1] = -2.0
        orientation_final = self.orientation_head[-1]
        assert isinstance(orientation_final, nn.Linear)
        nn.init.zeros_(orientation_final.weight)
        assert orientation_final.bias is not None
        nn.init.zeros_(orientation_final.bias)

    def forward(self, features: torch.Tensor) -> DirectLayoutOutput:
        spatial = self.stem(features)
        global_logits = self.global_head(spatial)
        return DirectLayoutOutput(
            heatmap_logits=self.heatmap(spatial),
            offset_logits=self.offset(spatial),
            point_visibility_logits=global_logits[:, :-1],
            validity_logits=global_logits[:, -1],
            orientation_logits=self.orientation_head(spatial),
        )


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
        layout_proposals: int = 0,
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
        self.layout_proposals = layout_proposals
        self.head = nn.Conv2d(fusion_channels, 5 if output_channels == 15 else output_channels, 1)
        self.context_stem = (
            Conv(fusion_channels, fusion_channels, 3, 1) if output_channels == 15 else None
        )
        self.context_head = nn.Conv2d(fusion_channels, 10, 1) if output_channels == 15 else None
        self.layout_head = (
            DirectLayoutHead(fusion_channels, proposals=layout_proposals)
            if layout_proposals > 0
            else None
        )
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)
        if self.context_head is not None:
            nn.init.normal_(self.context_head.weight, mean=0.0, std=0.001)
            assert self.context_head.bias is not None
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

    def forward_outputs(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, DirectLayoutOutput | None]:
        p2, deep_p3 = self._feature_forward(image)
        reduced = self.deep_reduce(deep_p3)
        reduced = F.interpolate(reduced, size=p2.shape[-2:], mode="nearest")
        fused = self.fusion(torch.cat((p2, reduced), dim=1))
        features = self.head_stem(fused)
        layout = self.layout_head(features) if self.layout_head is not None else None
        dense = self.head(features)
        if self.context_head is not None and self.context_stem is not None:
            dense = torch.cat((dense, self.context_head(self.context_stem(features))), dim=1)
        return dense, layout

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dense, _layout = self.forward_outputs(image)
        return dense

    def fuse(self) -> YOLO26CourtLine:
        """Fold inference-only normalization and reparameterized blocks in place."""

        for module in self.modules():
            if isinstance(module, (Conv, Conv2, DWConv)) and hasattr(module, "bn"):
                if isinstance(module, Conv2):
                    module.fuse_convs()
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, "bn")
                module.forward = module.forward_fuse
            elif isinstance(module, ConvTranspose) and hasattr(module, "bn"):
                module.conv_transpose = fuse_deconv_and_bn(module.conv_transpose, module.bn)
                delattr(module, "bn")
                module.forward = module.forward_fuse
            elif isinstance(module, RepConv):
                module.fuse_convs()
                module.forward = module.forward_fuse
            elif isinstance(module, RepVGGDW):
                module.fuse()
                module.forward = module.forward_fuse
        return self

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        for parameter in self.feature_layers.parameters():
            parameter.requires_grad_(not freeze)

    def freeze_shared_features(self, freeze: bool = True) -> None:
        for module in (self.feature_layers, self.deep_reduce, self.fusion, self.head_stem):
            for parameter in module.parameters():
                parameter.requires_grad_(not freeze)

    def freeze_dense_head(self, freeze: bool = True) -> None:
        for module in (self.head, self.context_stem, self.context_head):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(not freeze)

    def freeze_layout_geometry(self, freeze: bool = True) -> None:
        if self.layout_head is None:
            return
        for module in (
            self.layout_head.stem,
            self.layout_head.heatmap,
            self.layout_head.offset,
            self.layout_head.global_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(not freeze)

    def set_layout_geometry_eval(self) -> None:
        if self.layout_head is None:
            return
        for module in (
            self.layout_head.stem,
            self.layout_head.heatmap,
            self.layout_head.offset,
            self.layout_head.global_head,
        ):
            module.eval()

    def set_shared_features_eval(self) -> None:
        for module in (self.feature_layers, self.deep_reduce, self.fusion, self.head_stem):
            module.eval()

    def set_dense_head_eval(self) -> None:
        for module in (self.head, self.context_stem, self.context_head):
            if module is not None:
                module.eval()

    def set_base_head_gradient_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("base-head gradient scale must be between zero and one")
        for handle in self._head_gradient_handles:
            handle.remove()
        self._head_gradient_handles = []
        if (
            self.output_channels not in {8, 15}
            or scale == 1.0
            or not self.head.weight.requires_grad
        ):
            return

        def scale_base_rows(gradient: torch.Tensor) -> torch.Tensor:
            output = gradient.clone()
            output[:5].mul_(scale)
            return output

        self._head_gradient_handles = [
            self.head.weight.register_hook(scale_base_rows),
            cast(torch.Tensor, self.head.bias).register_hook(scale_base_rows),
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
    layout_proposals: int = 0,
) -> tuple[YOLO26CourtLine, dict[str, Any]]:
    """Transfer layers 0..16 from a real Pose36 checkpoint and discard its pose branch."""

    source_path = Path(checkpoint)
    architecture_only = source_path.suffix.lower() in {".yaml", ".yml"}
    if architecture_only:
        source = str(source_path.resolve()) if source_path.exists() else str(checkpoint)
    else:
        source = str(source_path.resolve())
    pose = cast(nn.Module, YOLO(source).model).float()
    if type(pose).__name__ != "PoseModel":
        raise ValueError(f"expected an Ultralytics PoseModel, got {type(pose).__name__}")
    if not architecture_only and list(getattr(pose, "kpt_shape", ())) != [36, 3]:
        raise ValueError(
            f"expected Pose36 checkpoint, got kpt_shape={getattr(pose, 'kpt_shape', None)}"
        )
    layers_module = getattr(pose, "model", None)
    if not isinstance(layers_module, (nn.Sequential, nn.ModuleList)):
        raise ValueError("checkpoint does not expose a sequential YOLO feature graph")
    layers = list(layers_module.children())
    if len(layers) <= FEATURE_LAST_INDEX or type(layers[DEEP_P3_INDEX]).__name__ != "C3k2":
        raise ValueError("checkpoint does not match the inspected YOLO26n Pose feature graph")
    feature_layers = nn.ModuleList(copy.deepcopy(list(layers[: FEATURE_LAST_INDEX + 1])))
    model = YOLO26CourtLine(
        feature_layers,
        p2_channels=_module_output_channels(layers[P2_INDEX]),
        deep_channels=_module_output_channels(layers[DEEP_P3_INDEX]),
        fusion_channels=fusion_channels,
        output_channels=output_channels,
        layout_proposals=layout_proposals,
    )

    source_keys = list(layers_module.state_dict())
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
        "layout_proposals": layout_proposals,
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
    layout_proposals_override: int | None = None,
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
        layout_proposals=(
            int(layout_proposals_override)
            if layout_proposals_override is not None
            else int(payload.get("layout_proposals", 0))
        ),
    )
    source_state = payload["model"]
    layout_upgrade = layout_proposals_override is not None and int(
        layout_proposals_override
    ) != int(payload.get("layout_proposals", 0))
    layout_version_upgrade = int(payload.get("layout_head_version", 0)) < DIRECT_LAYOUT_HEAD_VERSION
    if output_channels_override is None and not layout_upgrade and not layout_version_upgrade:
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
            "target_output_channels": int(model.output_channels),
            "copied_key_count": len(copied),
            "expanded_keys": expanded,
            "source_layout_proposals": int(payload.get("layout_proposals", 0)),
            "target_layout_proposals": int(model.layout_proposals),
            "source_layout_head_version": int(payload.get("layout_head_version", 0)),
            "target_layout_head_version": DIRECT_LAYOUT_HEAD_VERSION,
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
