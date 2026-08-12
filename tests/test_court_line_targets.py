from __future__ import annotations

import math

import numpy as np
import torch

from volley_court.dataset import (
    ImageTransform,
    sample_segment_points,
    segments_to_targets,
)
from volley_court.decode import (
    decode_dense_semantic_cuda,
    decode_dense_votes,
    decode_dense_votes_cuda,
    decode_predictions,
    merge_duplicates,
)
from volley_court.inference import resolve_decoder
from volley_court.loss import CourtLineLoss


def test_target_is_endpoint_swap_invariant() -> None:
    forward = segments_to_targets([(10.0, 20.0, 90.0, 60.0)], 128)
    reverse = segments_to_targets([(90.0, 60.0, 10.0, 20.0)], 128)
    for key in ("heatmap", "offset", "orientation", "half_length", "regression_mask"):
        assert torch.allclose(forward[key], reverse[key], atol=1e-6)


def test_target_collision_keeps_longer_segment() -> None:
    target = segments_to_targets(
        [(8.0, 10.0, 16.0, 10.0), (2.0, 10.0, 22.0, 10.0)],
        64,
        stride=4,
    )
    assert int(target["collision_count"]) == 1
    assert int(target["regression_mask"].sum()) == 1


def test_image_transform_round_trip() -> None:
    matrix = np.asarray([[0.5, 0.0, 8.0], [0.0, 0.5, 16.0], [0.0, 0.0, 1.0]])
    transform = ImageTransform(matrix, np.linalg.inv(matrix), 128)
    original = (20.0, 30.0, 100.0, 80.0)
    transformed = transform.apply_segment(original)
    assert transformed is not None
    restored = transform.restore_segment(transformed, 200, 150)
    assert restored is not None
    assert np.allclose(restored, original, atol=1e-6)


def test_decode_restores_center_and_unordered_orientation() -> None:
    raw = torch.full((1, 6, 16, 16), -10.0)
    raw[0, 0, 5, 6] = 10.0
    raw[0, 1, 5, 6] = math.log(0.25 / 0.75)
    raw[0, 2, 5, 6] = math.log(0.75 / 0.25)
    raw[0, 3, 5, 6] = 1.0
    raw[0, 4, 5, 6] = 0.0
    raw[0, 5, 5, 6] = 0.0
    decoded = decode_predictions(raw, confidence=0.5, top_k=4, image_size=64)[0]
    assert len(decoded) == 1
    assert np.allclose(decoded[0]["center"], (25.0, 23.0), atol=1e-4)
    assert np.allclose(decoded[0]["orientation"], (1.0, 0.0), atol=1e-4)


def test_duplicate_merge_keeps_higher_confidence() -> None:
    rows = [
        {"score": 0.9, "segment": [10.0, 20.0, 90.0, 20.0]},
        {"score": 0.6, "segment": [12.0, 22.0, 88.0, 22.0]},
    ]
    merged = merge_duplicates(rows)
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9


def test_dense_samples_are_uniform_and_midpoint_phased() -> None:
    points = sample_segment_points((0.0, 0.0, 64.0, 0.0), spacing=16.0)
    assert np.allclose(points, [(8.0, 0.0), (24.0, 0.0), (40.0, 0.0), (56.0, 0.0)])


def test_dense_targets_remain_endpoint_swap_invariant() -> None:
    forward = segments_to_targets(
        [(0.0, 20.0, 64.0, 20.0)],
        128,
        target_mode="dense_votes",
        sample_spacing=16.0,
    )
    reverse = segments_to_targets(
        [(64.0, 20.0, 0.0, 20.0)],
        128,
        target_mode="dense_votes",
        sample_spacing=16.0,
    )
    assert int(forward["vote_count"]) == 4
    for key in ("heatmap", "offset", "orientation", "regression_mask"):
        assert torch.allclose(forward[key], reverse[key], atol=1e-6)


def test_dense_vote_decode_groups_collinear_points() -> None:
    raw = torch.full((1, 5, 16, 16), -10.0)
    for cell_x in (2, 6, 10):
        raw[0, 0, 8, cell_x] = 10.0
        raw[0, 1, 8, cell_x] = 0.0
        raw[0, 2, 8, cell_x] = 0.0
        raw[0, 3, 8, cell_x] = 1.0
        raw[0, 4, 8, cell_x] = 0.0
    decoded = decode_dense_votes(
        raw,
        confidence=0.5,
        top_k=12,
        image_size=64,
        sample_spacing=16.0,
    )[0]
    assert len(decoded) == 1
    assert decoded[0]["vote_count"] == 3
    assert abs(decoded[0]["segment"][1] - decoded[0]["segment"][3]) < 1e-6


def test_spatial_hash_dense_decode_matches_exhaustive_grouping() -> None:
    raw = torch.full((1, 5, 32, 32), -10.0)
    for cell_x in (2, 6, 10, 14, 18):
        raw[0, 0, 8, cell_x] = 10.0
        raw[0, 1:3, 8, cell_x] = 0.0
        raw[0, 3, 8, cell_x] = 1.0
        raw[0, 4, 8, cell_x] = 0.0
    for cell_y in (3, 7, 11, 15, 19):
        raw[0, 0, cell_y, 24] = 9.0
        raw[0, 1:3, cell_y, 24] = 0.0
        raw[0, 3, cell_y, 24] = -1.0
        raw[0, 4, cell_y, 24] = 0.0
    fast = decode_dense_votes(
        raw,
        confidence=0.5,
        top_k=32,
        image_size=128,
        spatial_hash=True,
    )[0]
    exhaustive = decode_dense_votes(
        raw,
        confidence=0.5,
        top_k=32,
        image_size=128,
        spatial_hash=False,
    )[0]
    fast = sorted(fast, key=lambda row: tuple(row["center"]))
    exhaustive = sorted(exhaustive, key=lambda row: tuple(row["center"]))
    assert len(fast) == len(exhaustive)
    for actual, expected in zip(fast, exhaustive, strict=True):
        assert actual["vote_count"] == expected["vote_count"]
        assert np.allclose(actual["segment"], expected["segment"], atol=1e-6)


def test_dense_loss_accepts_five_channel_head() -> None:
    target = segments_to_targets(
        [(8.0, 16.0, 56.0, 16.0)],
        64,
        target_mode="dense_votes",
    )
    batch_target = {key: value.unsqueeze(0) for key, value in target.items()}
    prediction = torch.zeros((1, 5, 16, 16), requires_grad=True)
    loss, parts = CourtLineLoss(target_mode="dense_votes", length_weight=0.0)(
        prediction,
        batch_target,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(parts["half_length"]) == 0.0


def test_dense_context_targets_include_family_and_court_roi() -> None:
    roi = np.zeros((64, 64), dtype=np.float32)
    roi[16:48, 8:56] = 1.0
    target = segments_to_targets(
        [(8.0, 12.0, 56.0, 12.0), (20.0, 8.0, 20.0, 56.0)],
        64,
        target_mode="dense_context",
        segment_families=[1, 0],
        court_roi=roi,
        roi_valid=True,
    )
    labelled_families = set(target["family"][target["regression_mask"][0].bool()].tolist())
    assert labelled_families == {0, 1}
    assert float(target["court_roi"].max()) == 1.0
    assert float(target["roi_valid"]) == 1.0


def test_dense_context_loss_backpropagates_family_and_roi() -> None:
    target = segments_to_targets(
        [(8.0, 12.0, 56.0, 12.0)],
        64,
        target_mode="dense_context",
        segment_families=[1],
        court_roi=np.ones((64, 64), dtype=np.float32),
        roi_valid=True,
    )
    prediction = torch.zeros((1, 8, 16, 16), requires_grad=True)
    loss, parts = CourtLineLoss(target_mode="dense_context")(
        prediction,
        {key: value.unsqueeze(0) for key, value in target.items()},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(parts["family"]) > 0.0
    assert float(parts["roi"]) > 0.0


def test_dense_context_decode_emits_horizontal_family() -> None:
    prediction = torch.full((1, 8, 16, 16), -10.0)
    for cell_x in (4, 8):
        prediction[0, 0, 4, cell_x] = 10.0
        prediction[0, 1:3, 4, cell_x] = 0.0
        prediction[0, 3, 4, cell_x] = 1.0
        prediction[0, 4, 4, cell_x] = 0.0
        prediction[0, 5, 4, cell_x] = -4.0
        prediction[0, 6, 4, cell_x] = 4.0
        prediction[0, 7, 4, cell_x] = 10.0
    rows = decode_dense_votes(prediction, image_size=64, confidence=0.25)[0]
    assert len(rows) == 1
    assert rows[0]["family"] == "horizontal"
    assert rows[0]["family_score"] > 0.99


def test_dense_semantic_loss_and_decode_emit_line_identity() -> None:
    target = segments_to_targets(
        [(8.0, 12.0, 56.0, 12.0)],
        64,
        target_mode="dense_semantic",
        segment_families=[1],
        segment_identities=[4],
        court_roi=np.ones((64, 64), dtype=np.float32),
        roi_valid=True,
    )
    prediction = torch.zeros((1, 15, 16, 16), requires_grad=True)
    loss, parts = CourtLineLoss(target_mode="dense_semantic")(
        prediction,
        {key: value.unsqueeze(0) for key, value in target.items()},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert float(parts["identity"]) > 0.0

    raw = torch.full((1, 15, 16, 16), -10.0)
    for cell_x in (4, 8):
        raw[0, 0, 4, cell_x] = 10.0
        raw[0, 1:3, 4, cell_x] = 0.0
        raw[0, 3, 4, cell_x] = 1.0
        raw[0, 4, 4, cell_x] = 0.0
        raw[0, 5, 4, cell_x] = -4.0
        raw[0, 6, 4, cell_x] = 4.0
        raw[0, 7:14, 4, cell_x] = -4.0
        raw[0, 11, 4, cell_x] = 4.0
        raw[0, 14, 4, cell_x] = 10.0
    rows = decode_dense_votes(raw, image_size=64, confidence=0.25)[0]
    assert len(rows) == 1
    assert rows[0]["line_identity"] == 4
    assert rows[0]["identity_score"] > 0.99


def test_dense_decoder_promotes_half_precision_head_to_float() -> None:
    raw = torch.full((1, 15, 16, 16), -10.0)
    for cell_x in (3, 7, 11):
        raw[0, 0, 6, cell_x] = 3.25
        raw[0, 1:3, 6, cell_x] = torch.tensor((0.2, -0.3))
        raw[0, 3:5, 6, cell_x] = torch.tensor((0.8, 0.15))
        raw[0, 5:7, 6, cell_x] = torch.tensor((-0.4, 0.7))
        raw[0, 7:14, 6, cell_x] = -0.5
        raw[0, 12, 6, cell_x] = 1.25
        raw[0, 14, 6, cell_x] = 0.9
    half = raw.half()
    actual = decode_dense_votes(half, image_size=64, confidence=0.25)[0]
    expected = decode_dense_votes(half.float(), image_size=64, confidence=0.25)[0]
    assert actual == expected


def test_dense_decoder_batch_matches_individual_predictions() -> None:
    first = torch.full((15, 16, 16), -10.0)
    second = torch.full((15, 16, 16), -10.0)
    for raw, cell_y in ((first, 4), (second, 10)):
        for cell_x in (3, 7, 11):
            raw[0, cell_y, cell_x] = 8.0
            raw[1:3, cell_y, cell_x] = 0.0
            raw[3, cell_y, cell_x] = 1.0
            raw[4, cell_y, cell_x] = 0.0
            raw[5:7, cell_y, cell_x] = torch.tensor((-2.0, 2.0))
            raw[7:14, cell_y, cell_x] = -2.0
            raw[9, cell_y, cell_x] = 2.0
            raw[14, cell_y, cell_x] = 4.0
    batched = decode_dense_votes(
        torch.stack((first, second)),
        image_size=64,
        confidence=0.25,
    )
    assert batched[0] == decode_dense_votes(first.unsqueeze(0), image_size=64)[0]
    assert batched[1] == decode_dense_votes(second.unsqueeze(0), image_size=64)[0]


def _semantic_line_logits(cell_y: int, identity: int) -> torch.Tensor:
    raw = torch.full((15, 16, 16), -10.0)
    for cell_x in (3, 7, 11):
        raw[0, cell_y, cell_x] = 8.0
        raw[1:3, cell_y, cell_x] = 0.0
        raw[3:5, cell_y, cell_x] = torch.tensor((1.0, 0.0))
        raw[5:7, cell_y, cell_x] = torch.tensor((-4.0, 4.0))
        raw[7:14, cell_y, cell_x] = -4.0
        raw[7 + identity, cell_y, cell_x] = 4.0
        raw[14, cell_y, cell_x] = 8.0
    return raw


def test_semantic_cuda_decoder_groups_by_identity_and_filters_geometry() -> None:
    raw = _semantic_line_logits(cell_y=6, identity=4)
    # A confident same-ID vote with an incompatible axial orientation must not
    # pull the fitted line away from the three coherent votes.
    raw[0, 2, 7] = 8.0
    raw[1:3, 2, 7] = 0.0
    raw[3:5, 2, 7] = torch.tensor((-1.0, 0.0))
    raw[5:7, 2, 7] = torch.tensor((-4.0, 4.0))
    raw[7:14, 2, 7] = -4.0
    raw[11, 2, 7] = 4.0
    raw[14, 2, 7] = 8.0
    rows = decode_dense_semantic_cuda(raw.unsqueeze(0), image_size=64)[0]
    assert len(rows) == 1
    assert rows[0]["decoder"] == "semantic_cuda"
    assert rows[0]["line_identity"] == 4
    assert rows[0]["family"] == "horizontal"
    assert rows[0]["vote_count"] == 3
    assert abs(rows[0]["segment"][1] - rows[0]["segment"][3]) < 1e-5


def test_semantic_cuda_decoder_batch_and_half_are_consistent() -> None:
    first = _semantic_line_logits(cell_y=4, identity=3)
    second = _semantic_line_logits(cell_y=10, identity=6)
    raw = torch.stack((first, second))
    batched = decode_dense_semantic_cuda(raw, image_size=64)
    half = decode_dense_semantic_cuda(raw.half(), image_size=64)
    assert batched == half
    assert [rows[0]["line_identity"] for rows in batched] == [3, 6]


def test_cuda_spatial_decoder_matches_reference_line_fit() -> None:
    raw = _semantic_line_logits(cell_y=6, identity=4).unsqueeze(0)
    reference = decode_dense_votes(raw, image_size=64)[0]
    actual = decode_dense_votes_cuda(raw, image_size=64)[0]
    assert len(reference) == len(actual) == 1
    assert actual[0]["decoder"] == "cuda"
    for key in (
        "segment",
        "center",
        "orientation",
        "family_probabilities",
        "identity_probabilities",
    ):
        assert np.allclose(actual[0][key], reference[0][key], atol=1e-5)
    for key in ("score", "family_score", "identity_score", "roi_score"):
        assert math.isclose(actual[0][key], reference[0][key], abs_tol=1e-5)
    assert actual[0]["vote_count"] == reference[0]["vote_count"]
    assert actual[0]["family"] == reference[0]["family"]
    assert actual[0]["line_identity"] == reference[0]["line_identity"]


def test_cuda_spatial_decoder_batch_and_half_are_consistent() -> None:
    raw = torch.stack(
        (
            _semantic_line_logits(cell_y=4, identity=3),
            _semantic_line_logits(cell_y=10, identity=6),
        )
    )
    batched = decode_dense_votes_cuda(raw, image_size=64)
    half = decode_dense_votes_cuda(raw.half(), image_size=64)
    assert batched == half
    assert [rows[0]["line_identity"] for rows in batched] == [3, 6]


def test_auto_decoder_uses_cuda_only_when_batch_amortizes_it() -> None:
    assert (
        resolve_decoder("auto", device_type="cuda", target_mode="dense_semantic", batch_size=1)
        == "spatial"
    )
    assert (
        resolve_decoder("auto", device_type="cuda", target_mode="dense_semantic", batch_size=2)
        == "cuda"
    )
    assert (
        resolve_decoder("auto", device_type="cpu", target_mode="dense_semantic", batch_size=16)
        == "spatial"
    )
