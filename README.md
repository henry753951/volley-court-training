# Volleyball court keypoint training

This standalone workspace exports 36-point labels from
`H:\Repos\volley-court-annotator`, canonicalizes their semantic order,
and trains/evaluates the volleyball-court pose model.

## Dataset contract

- The annotator Docker PostgreSQL database is the source of truth.
- Only `COMPLETED` and `REVIEWED` annotations are exported by default.
- The 36 keypoints remain ordered because YOLO Pose and homography projection
  require stable semantic slots.
- Rectangle symmetry is handled by testing four valid mappings: identity,
  length mirror, width mirror, and both axes. The scorer adapts between
  sideline and endline viewpoints.
- If the best two symmetry mappings tie, or the canonicalized label has less
  than 90% image-axis agreement, the image is quarantined instead of silently
  training on an uncertain/arbitrary point order. Every decision is recorded
  in `orientation-audit.csv`.
- Built-in YOLO horizontal flip is disabled for the adaptive dataset because a
  sideline flip reverses court length while an endline flip reverses court
  width. Correct per-view flips are generated during dataset preparation.
- Original color images are preserved; this workflow does not create grayscale
  copies.
- Color, brightness, rotation, translation, scaling, shear, and perspective
  distortion are enabled. The 36 labels transform together with each image, so
  augmentation never changes point semantics.

## Reproducible commands

```powershell
Set-Location H:\Repos\volley-court-training
uv sync
uv run court-download-eval-videos

uv run court-prepare-dataset `
  --annotator-env H:\Repos\volley-court-annotator\.env `
  --output datasets\court36-color-canonical-v2 `
  --side-camera-canonicalize `
  --grayscale-copies 0

uv run court-train `
  --data datasets\court36-color-canonical-v2\dataset.yaml `
  --model H:\Repos\volley-ai\yolo26n-pose.pt `
  --epochs 100 `
  --imgsz 1280 `
  --batch 2 `
  --project runs `
  --name court36-color-distortion-v2
```

Every `court-train` run requires all videos in `eval-videos.yaml` and, after
training, writes an overlay video, contact sheet, and frame metrics for each
video under the run's `evaluations` directory.

The command above starts from the generic YOLO pose checkpoint, not an earlier
volleyball-court checkpoint. Built-in `fliplr` must remain zero: horizontal
flip means a length swap for sideline views but a width swap for endline views,
so the exporter creates view-aware flipped copies with the correct 36-point
permutation.

## H100 training

Generated datasets, evaluation videos, model weights, and runs are intentionally
excluded from Git. Synchronize those artifacts separately, then run:

```bash
cd ~/volley-court-training
uv sync --frozen
./scripts/train-h100.sh
```

The H100 preset uses `yolo26m-pose.pt`, 1280 px inputs, automatic batch sizing
targeting 85% of GPU memory, 16 data-loader workers, disk cache, cosine learning
rate, `torch.compile`, and 300 epochs. Override any major sizing choice without
editing the script:

```bash
BATCH=32 EPOCHS=200 MODEL=yolo26l-pose.pt ./scripts/train-h100.sh
```

To move an interrupted run to H100 while preserving the optimizer, scheduler,
EMA, scaler, and completed epoch count, synchronize the full run directory and
execute:

```bash
./scripts/resume-h100.sh
```

Resume keeps the checkpoint's architecture and augmentation schedule. It only
overrides settings Ultralytics explicitly permits during resume: GPU device,
automatic batch sizing, data-loader workers, disk cache, patience, and the
Linux save directory. After completion, the same three fixed evaluation videos
are rendered automatically.
