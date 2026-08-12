from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class CourtLineLoss(nn.Module):
    def __init__(
        self,
        center_weight: float = 1.0,
        offset_weight: float = 1.0,
        orientation_weight: float = 1.0,
        length_weight: float = 2.0,
        family_weight: float = 0.5,
        identity_weight: float = 0.75,
        identity_focal_weight: float = 0.0,
        roi_weight: float = 0.5,
        target_mode: str = "center",
    ) -> None:
        super().__init__()
        self.weights = (
            center_weight,
            offset_weight,
            orientation_weight,
            length_weight,
            family_weight,
            identity_weight,
            roi_weight,
        )
        if target_mode not in {"center", "dense_votes", "dense_context", "dense_semantic"}:
            raise ValueError(f"unsupported target mode: {target_mode}")
        self.target_mode = target_mode
        if identity_focal_weight < 0.0:
            raise ValueError("identity focal weight must be non-negative")
        self.identity_focal_weight = identity_focal_weight

    @staticmethod
    def _center_focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BF16 cannot represent 1 - 1e-5. Promote before sigmoid/clamp so
        # saturated semantic logits never produce log(0) during AMP training.
        probability = logits.float().sigmoid().clamp(1e-5, 1.0 - 1e-5)
        target = target.float()
        positive = target.eq(1.0)
        negative = target.lt(1.0)
        positive_loss = -(1.0 - probability).pow(2.0) * probability.log() * positive
        negative_loss = -probability.pow(2.0) * (1.0 - probability).log() * negative
        count = positive.sum().clamp_min(1)
        return (positive_loss.sum() + negative_loss.sum()) / count

    def forward(
        self,
        prediction: torch.Tensor,
        target: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_channels = {
            "center": 6,
            "dense_votes": 5,
            "dense_context": 8,
            "dense_semantic": 15,
        }[self.target_mode]
        if prediction.shape[1] != expected_channels:
            raise ValueError(
                f"expected {expected_channels} prediction channels for {self.target_mode}, "
                f"got {prediction.shape}"
            )
        mask = target["regression_mask"]
        denominator = mask.sum().clamp_min(1.0)
        center = self._center_focal(prediction[:, 0:1], target["heatmap"])
        predicted_offset = prediction[:, 1:3].sigmoid()
        offset = (
            F.smooth_l1_loss(predicted_offset, target["offset"], reduction="none") * mask
        ).sum() / denominator
        predicted_orientation = F.normalize(prediction[:, 3:5], dim=1, eps=1e-6)
        orientation_cosine = (predicted_orientation * target["orientation"]).sum(
            dim=1, keepdim=True
        )
        orientation = ((1.0 - orientation_cosine) * mask).sum() / denominator
        if self.target_mode == "center":
            predicted_length = 0.5 * prediction[:, 5:6].sigmoid()
            half_length = (
                F.smooth_l1_loss(
                    predicted_length,
                    target["half_length"],
                    reduction="none",
                )
                * mask
            ).sum() / denominator
        else:
            half_length = prediction.new_zeros(())
        if self.target_mode in {"dense_context", "dense_semantic"}:
            family_loss = F.cross_entropy(
                prediction[:, 5:7],
                target["family"],
                reduction="none",
            )
            family = (family_loss * mask[:, 0]).sum() / denominator
            if self.target_mode == "dense_semantic":
                identity_ce = F.cross_entropy(
                    prediction[:, 7:14],
                    target["identity"],
                    reduction="none",
                )
                identity_ce = (identity_ce * mask[:, 0]).sum() / denominator
                identity_focal = torch.stack(
                    [
                        self._center_focal(
                            prediction[:, identity_index + 7 : identity_index + 8],
                            target["identity_heatmap"][:, identity_index : identity_index + 1],
                        )
                        for identity_index in range(7)
                    ]
                ).mean()
                identity = identity_ce + self.identity_focal_weight * identity_focal
                identity_accuracy = (
                    (prediction[:, 7:14].argmax(dim=1) == target["identity"]).float() * mask[:, 0]
                ).sum() / denominator
                roi_logits = prediction[:, 14:15]
            else:
                identity = prediction.new_zeros(())
                identity_accuracy = prediction.new_zeros(())
                roi_logits = prediction[:, 7:8]
            roi_valid = target["roi_valid"].reshape(-1)
            roi_target = target["court_roi"]
            roi_bce = F.binary_cross_entropy_with_logits(
                roi_logits,
                roi_target,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            roi_probability = roi_logits.sigmoid()
            intersection = (roi_probability * roi_target).sum(dim=(1, 2, 3))
            union = roi_probability.sum(dim=(1, 2, 3)) + roi_target.sum(dim=(1, 2, 3))
            roi_dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
            roi = ((roi_bce + roi_dice) * roi_valid).sum() / roi_valid.sum().clamp_min(1.0)
        else:
            family = prediction.new_zeros(())
            identity = prediction.new_zeros(())
            identity_accuracy = prediction.new_zeros(())
            roi = prediction.new_zeros(())
        wc, wo, wr, wl, wf, wid, wi = self.weights
        total = (
            wc * center
            + wo * offset
            + wr * orientation
            + wl * half_length
            + wf * family
            + wid * identity
            + wi * roi
        )
        return total, {
            "total": total.detach(),
            "center": center.detach(),
            "offset": offset.detach(),
            "orientation": orientation.detach(),
            "half_length": half_length.detach(),
            "family": family.detach(),
            "identity": identity.detach(),
            "identity_accuracy": identity_accuracy.detach(),
            "roi": roi.detach(),
        }
