# Training

## Environment

```bash
uv sync --group dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

The repository follows the [official uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
with an explicit CUDA 12.8 index in `pyproject.toml`. BF16 was used on the
H100 NVL. RTX 5070 inference uses FP16; training parameters should be retuned before treating a
consumer-GPU run as equivalent.

## Prepare targets

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

Inspect the generated previews before training. This is where Pose36 ordering or topology errors
should be caught.

## Recorded semantic-v4 stages

The release was initialized from an internal dense-votes checkpoint. Replace `$DENSE_BASE` with
that checkpoint to reproduce the historical lineage. For a new fine-tuning run, the published v1
checkpoint can be passed to `--weights` instead. The release run used the converted target roots
created above and retained all 2,000 synthetic and 357 real images.

### S1: synthetic context

```bash
uv run volley-court-train \
  --data .work/synthetic-lines \
  --image-root datasets/court36-synthetic-combined-2000-camera-mode-v2 \
  --weights "$DENSE_BASE" \
  --output runs/s1-semantic-v4 \
  --target-mode dense_semantic \
  --epochs 60 --batch 64 --workers 12 --imgsz 640 --device cuda:0 \
  --amp-dtype bf16 \
  --freeze-feature-epochs 12 --freeze-shared-epochs 4 \
  --base-head-gradient-scale 0.1 \
  --hard-negative-probability 0.20 \
  --family-weight 0.5 --identity-weight 2.0 --identity-focal-weight 0.002 \
  --roi-weight 0.5 --weight-decay 0.0001 --lr 0.0002 --seed 36
```

### S2: real adaptation

```bash
uv run volley-court-train \
  --data .work/real-lines \
  --image-root datasets/court36-unified \
  --replay-data .work/synthetic-lines \
  --replay-image-root datasets/court36-synthetic-combined-2000-camera-mode-v2 \
  --replay-ratio 0.35 \
  --weights runs/s1-semantic-v4/best.pt \
  --output runs/s2-semantic-v4 \
  --target-mode dense_semantic \
  --epochs 40 --batch 32 --workers 12 --imgsz 640 --device cuda:0 \
  --amp-dtype bf16 \
  --freeze-feature-epochs 8 --freeze-shared-epochs 3 \
  --base-head-gradient-scale 0.05 \
  --hard-negative-probability 0.25 \
  --family-weight 0.5 --identity-weight 1.5 --identity-focal-weight 0.001 \
  --roi-weight 0.5 --weight-decay 0.0001 --lr 0.00008 --seed 36
```

The staged freeze protects the transferred dense geometry early, then permits controlled
fine-tuning. S2 samples the synthetic replay set for 35% of each epoch. Two later coordinate-head
pilots were rejected because they reduced validation identity accuracy; they are not part of the
released architecture or checkpoint.

## Evaluation

```bash
uv run volley-court-evaluate \
  --dataset datasets/court36-unified/dataset.yaml \
  --predictions path/to/predictions \
  --split test \
  --output benchmarks/quality/new-run.json
```

Use line recall, family accuracy, precision, PCK, and the layout status distribution together.
Do not gate the release solely on training loss or call line recall “keypoint accuracy.”

For the fixed solver comparison:

```bash
uv run python benchmarks/compare_layout_solvers.py \
  --checkpoint runs/s2-semantic-v4/best.pt \
  --dataset datasets/court36-unified/dataset.yaml \
  --topology configs/court_line_topology.yaml \
  --output benchmarks/layout-ab \
  --device cuda:0 --batch-size 16 --confidence 0.25 --repeats 20
```

## Extending the architecture

The package keeps three seams explicit:

1. `YOLO26CourtLine` owns the learned dense heads.
2. `decode.py` turns dense tensors into typed short segments.
3. `layout.py` performs court-topology matching and may abstain.

This allows a future learned validity/context head, TensorRT export, or CUDA layout kernel without
changing the public `CourtFrameResult` contract.
