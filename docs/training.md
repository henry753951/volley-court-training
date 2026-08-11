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

## Recorded v1 stages

The release was initialized from an internal dense-votes checkpoint. Replace `$DENSE_BASE` with
that checkpoint to reproduce the historical lineage. For a new fine-tuning run, the published v1
checkpoint can be passed to `--weights` instead.

### S1: synthetic context

```bash
uv run volley-court-train \
  --data .work/synthetic-lines \
  --image-root datasets/court36-synthetic-combined-2000-camera-mode-v2 \
  --weights "$DENSE_BASE" \
  --output runs/s1-synthetic \
  --target-mode dense_semantic \
  --epochs 20 --batch 64 --workers 12 --imgsz 640 --device cuda:0 \
  --amp-dtype bf16 \
  --freeze-feature-epochs 20 --freeze-shared-epochs 20 \
  --base-head-gradient-scale 0 \
  --hard-negative-probability 0.15 \
  --family-weight 0.5 --identity-weight 0.75 --roi-weight 0.5 \
  --weight-decay 0 --lr 0.0005 --seed 36
```

### S2: real adaptation

```bash
uv run volley-court-train \
  --data .work/real-lines \
  --image-root datasets/court36-unified \
  --weights runs/s1-synthetic/best.pt \
  --output runs/s2-real \
  --epochs 15 --batch 32 --workers 12 --imgsz 640 --device cuda:0 \
  --amp-dtype bf16 \
  --freeze-feature-epochs 15 --freeze-shared-epochs 15 \
  --base-head-gradient-scale 0 \
  --hard-negative-probability 0.20 \
  --family-weight 0.5 --identity-weight 0.75 --roi-weight 0.5 \
  --weight-decay 0 --lr 0.0002 --seed 36
```

The frozen shared backbone keeps the geometry learned by the dense-vote model while S1/S2 train
the family, identity, and court-ROI context heads. This is a short adaptation recipe, not a claim
that 15 epochs are universally optimal.

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

## Extending the architecture

The package keeps three seams explicit:

1. `YOLO26CourtLine` owns the learned dense heads.
2. `decode.py` turns dense tensors into typed short segments.
3. `layout.py` performs court-topology matching and may abstain.

This allows a future learned validity/context head, TensorRT export, or CUDA layout kernel without
changing the public `CourtFrameResult` contract.
