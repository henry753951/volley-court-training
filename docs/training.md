# Training

## Environment

```bash
uv sync --group dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

The project follows the [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
with an explicit CUDA 12.8 wheel index. Recorded H100 runs use BF16.

## Data

- synthetic: `court36-synthetic-combined-2000-camera-mode-v2`, 1,600/200/200;
- real: `court36-unified`, 283/37/37;
- topology: 36 identity-preserving keypoint slots and seven zero-width semantic court lines.

Convert both datasets before training and inspect their previews. Width/length/both symmetry variants
must preserve YOLO keypoint slot identity.

```bash
uv run volley-court-convert \
  --dataset datasets/court36-synthetic-combined-2000-camera-mode-v2/dataset.yaml \
  --topology configs/court_line_topology.yaml \
  --output .work/synthetic-lines --stride 4 --preview-count 24

uv run volley-court-convert \
  --dataset datasets/court36-unified/dataset.yaml \
  --topology configs/court_line_topology.yaml \
  --output .work/real-lines --stride 4 --preview-count 24
```

## Stage design

1. Synthetic dense context teaches court geometry and camera coverage.
2. Real adaptation corrects appearance and broadcast-domain shift.
3. Direct Pose36 anchors learn a single-frame homography proposal.
4. V2 established the direct layout and strict semantic verifier.
5. V3 adds a global soft-argmax layout loss, low-gradient ablations, and video-teacher consistency.
   The final checkpoint is selected by PCK, precision, severe-accept, and video gates rather than
   total validation loss.

The supervised V3 ablation is preserved in
[`scripts/train_v3_accuracy_ablation.sh`](../scripts/train_v3_accuracy_ablation.sh). The selected
`joint-low-gradient/epoch-0005.pt` checkpoint is regularized with the exact release command:

```bash
uv run volley-court-train \
  --data artifacts/converted-teacher-clip-v2-20260813 \
  --weights runs/v3-accuracy-ablation-20260813/joint-low-gradient/epoch-0005.pt \
  --output runs/v3-e5-teacher-consistency-20260813 \
  --epochs 3 --save-every 1 --batch 32 --workers 8 --device cuda:0 \
  --imgsz 512 --lr 0.00001 --weight-decay 0.0005 \
  --target-mode dense_semantic --layout-proposals 1 \
  --freeze-feature-epochs 3 --freeze-shared-epochs 3 --freeze-dense-epochs 3 \
  --base-head-gradient-scale 0 --hard-negative-probability 0 \
  --family-weight 0.5 --identity-weight 1.25 --roi-weight 0.5 \
  --layout-coordinate-weight 8 --layout-validity-weight 1.5 \
  --layout-softargmax-weight 0 --seed 36 --amp-dtype bf16
```

For V3, the best supervised candidate was regularized for two epochs against 255 stable V2 layouts
sampled every three frames from `clip.mp4`. Those predictions are consistency data, not evaluation
ground truth. `scripts/build_video_teacher_dataset.py` records the extraction step, and
`scripts/interpolate_layout_checkpoints.py` supports validation-gated model-soup ablations.
The released V3 file is epoch 2 of this consistency run; epoch 3 was evaluated and rejected.

Longer runs were rejected: after roughly five epochs the small real split overfit even while loss
continued to look reasonable. The fixed real-test, latency, and consecutive-frame visual gates
decide the release.

## Evaluation and release gate

```bash
uv run volley-court-evaluate-model \
  --dataset datasets/court36-unified/dataset.yaml \
  --checkpoint path/to/checkpoint.pt \
  --split test --imgsz 512 --device cuda:0 \
  --output benchmarks/quality/candidate

uv run volley-court-audit-video \
  test-videos/clip.mp4 path/to/checkpoint.pt artifacts/audit/clip \
  --imgsz 512 --consecutive 5 --device cuda:0
```

Do not release from loss curves alone. The required gates are PCK, precision, zero severe accepted
layouts, batch-1 latency, and visual inspection of consecutive frames from all fixed videos.
