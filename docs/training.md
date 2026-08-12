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
4. The released S6 epoch trains only the spatial orientation classifier. Geometry, shared features,
   and dense heads remain frozen so orientation training cannot destroy the accepted layout shape.

The exact released S6 command is preserved in
[`scripts/train_direct_layout_s6_spatial_orientation.sh`](../scripts/train_direct_layout_s6_spatial_orientation.sh).
Its essential settings are:

```bash
uv run volley-court-train \
  --data artifacts/converted-real-video-hard-v2 \
  --weights runs/direct-layout-s2-anchor40-v8-20260812/best.pt \
  --output runs/direct-layout-s6-spatial-orientation-v1-20260813 \
  --epochs 60 --save-every 2 --batch 32 --workers 4 --device cuda:0 \
  --target-mode dense_semantic --layout-proposals 1 \
  --freeze-feature-epochs 60 --freeze-shared-epochs 60 \
  --freeze-dense-epochs 60 --freeze-layout-geometry-epochs 60 \
  --base-head-gradient-scale 0 --hard-negative-probability 0 --lr 0.001 \
  --layout-coordinate-weight 0 --layout-validity-weight 0 \
  --layout-proposal-weight 1 \
  --layout-orientation-class-weights 1.0,4.7,5.2,8.0,3.1,4.6,8.0,3.2
```

Epoch 20 was selected. Later epochs were not preferred merely because they trained longer; the fixed
real-test, latency, and consecutive-frame visual gates decide the release.

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
