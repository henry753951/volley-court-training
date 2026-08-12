from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .layout import COURT_ORIENTATION_CORNER_PERMUTATIONS
from .model import DirectLayoutOutput


class CourtLineLoss(nn.Module):
    def __init__(
        self,
        center_weight: float = 1.0,
        offset_weight: float = 1.0,
        orientation_weight: float = 1.0,
        length_weight: float = 2.0,
        family_weight: float = 0.5,
        identity_weight: float = 0.75,
        roi_weight: float = 0.5,
        layout_coordinate_weight: float = 5.0,
        layout_validity_weight: float = 1.0,
        layout_proposal_weight: float = 0.25,
        layout_orientation_class_weights: Sequence[float] | None = None,
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
        self.layout_weights = (
            layout_coordinate_weight,
            layout_validity_weight,
            layout_proposal_weight,
        )
        if layout_orientation_class_weights is not None and (
            len(layout_orientation_class_weights) != 8
            or any(weight <= 0.0 for weight in layout_orientation_class_weights)
        ):
            raise ValueError("layout orientation class weights must contain eight positive values")
        self.layout_orientation_class_weights = (
            tuple(float(weight) for weight in layout_orientation_class_weights)
            if layout_orientation_class_weights is not None
            else None
        )

    @staticmethod
    def _center_focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
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
        layout_prediction: DirectLayoutOutput | None = None,
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
                identity_loss = F.cross_entropy(
                    prediction[:, 7:14],
                    target["identity"],
                    reduction="none",
                )
                identity = (identity_loss * mask[:, 0]).sum() / denominator
                roi_logits = prediction[:, 14:15]
            else:
                identity = prediction.new_zeros(())
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
            roi = prediction.new_zeros(())
        layout_coordinate = prediction.new_zeros(())
        layout_validity = prediction.new_zeros(())
        layout_proposal = prediction.new_zeros(())
        if layout_prediction is not None:
            valid = target["layout_valid"].reshape(-1)
            point_valid = target["layout_point_valid"]
            heatmap = layout_prediction.heatmap_logits
            batch, keypoints, grid_height, grid_width = heatmap.shape
            if point_valid.shape != (batch, keypoints):
                raise ValueError(
                    "layout point target does not match anchor head: "
                    f"{point_valid.shape} versus {heatmap.shape}"
                )
            layout_validity = F.binary_cross_entropy_with_logits(
                layout_prediction.validity_logits, valid
            ) + 0.5 * F.binary_cross_entropy_with_logits(
                layout_prediction.point_visibility_logits,
                point_valid,
            )
            points = target["layout_points"]
            grid_x = (points[..., 0] * grid_width).clamp(0.0, grid_width - 1e-4)
            grid_y = (points[..., 1] * grid_height).clamp(0.0, grid_height - 1e-4)
            cell_x = grid_x.floor().long()
            cell_y = grid_y.floor().long()
            spatial_index = cell_y * grid_width + cell_x
            valid_points = point_valid > 0.5
            if bool(valid_points.any()):
                flattened_heatmap = heatmap.flatten(2)
                spatial_ce = F.cross_entropy(
                    flattened_heatmap[valid_points],
                    spatial_index[valid_points],
                )
                offset_logits = layout_prediction.offset_logits.reshape(
                    batch, keypoints, 2, grid_height * grid_width
                )
                gather_index = spatial_index[:, :, None, None].expand(-1, -1, 2, 1)
                predicted_offset = offset_logits.gather(3, gather_index).squeeze(3).sigmoid()
                target_offset = torch.stack((grid_x - cell_x, grid_y - cell_y), dim=2)
                offset_error = F.smooth_l1_loss(
                    predicted_offset,
                    target_offset,
                    reduction="none",
                    beta=0.05,
                ).mean(dim=2)
                layout_coordinate = spatial_ce + 2.0 * (
                    offset_error * point_valid
                ).sum() / point_valid.sum().clamp_min(1.0)
            valid_orientation = (valid > 0.5) & (point_valid.sum(dim=1) >= 4)
            if bool(valid_orientation.any()):
                with torch.no_grad():
                    flattened = heatmap.flatten(2)
                    predicted_index = flattened.argmax(dim=2)
                    predicted_x = predicted_index.remainder(grid_width).float() / grid_width
                    predicted_y = (
                        predicted_index.div(grid_width, rounding_mode="floor").float() / grid_height
                    )
                    predicted_points = torch.stack((predicted_x, predicted_y), dim=2)
                    predicted_corners = predicted_points[:, (0, 4, 5, 9)]
                    permutations = torch.as_tensor(
                        COURT_ORIENTATION_CORNER_PERMUTATIONS,
                        dtype=torch.long,
                        device=predicted_points.device,
                    )
                    candidates = predicted_corners[:, permutations]
                    target_corners = target.get("layout_corners", points[:, (0, 4, 5, 9)])
                    corner_errors = torch.linalg.vector_norm(
                        candidates - target_corners[:, None], dim=3
                    ).mean(dim=2)
                    orientation_target = corner_errors.argmin(dim=1)
                layout_proposal = F.cross_entropy(
                    layout_prediction.orientation_logits[valid_orientation],
                    orientation_target[valid_orientation],
                    weight=(
                        layout_prediction.orientation_logits.new_tensor(
                            self.layout_orientation_class_weights
                        )
                        if self.layout_orientation_class_weights is not None
                        else None
                    ),
                )
        wc, wo, wr, wl, wf, wid, wi = self.weights
        wlc, wlv, wlp = self.layout_weights
        total = (
            wc * center
            + wo * offset
            + wr * orientation
            + wl * half_length
            + wf * family
            + wid * identity
            + wi * roi
            + wlc * layout_coordinate
            + wlv * layout_validity
            + wlp * layout_proposal
        )
        return total, {
            "total": total.detach(),
            "center": center.detach(),
            "offset": offset.detach(),
            "orientation": orientation.detach(),
            "half_length": half_length.detach(),
            "family": family.detach(),
            "identity": identity.detach(),
            "roi": roi.detach(),
            "layout_coordinate": layout_coordinate.detach(),
            "layout_validity": layout_validity.detach(),
            "layout_proposal": layout_proposal.detach(),
        }
