# Volley Court Lines

Fast, typed volleyball court-line inference and layout recovery built on a compact YOLO26n
backbone. The package detects dense zero-width court-line segments, classifies their direction
and semantic identity, then optionally reconstructs the original 36 court keypoints.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![typed](https://img.shields.io/badge/typing-py.typed-2F80ED)](src/volley_court/py.typed)

[![Court-line demo](https://assets.hsulab.net/demos/volley-court-lines/v1/demo-07s.jpg)](https://assets.hsulab.net/demos/volley-court-lines/v1/clip-volley-court-lines-v1.mp4)

**[Play the 14.8 s H.264 demo](https://assets.hsulab.net/demos/volley-court-lines/v1/clip-volley-court-lines-v1.mp4)**
— generated locally from `test-videos/clip.mp4` through the public module, including layout
matching and the built-in visualizer.

The visualizer keeps raw detections and recovered geometry deliberately separate. Unresolved
frames show only small evidence ticks; an accepted layout shows the complete seven-line virtual
court and all 36 projected points. Video mode tracks that accepted geometry on every source frame
with optical flow, so the overlay does not lag between layout-matcher passes.

## Install

The published wheel is the simplest internal installation path:

```bash
uv add "volley-court-lines @ https://assets.hsulab.net/packages/volley-court-lines/v0.1.1/volley_court_lines-0.1.1-py3-none-any.whl"
```

```bash
pip install "volley-court-lines @ https://assets.hsulab.net/packages/volley-court-lines/v0.1.1/volley_court_lines-0.1.1-py3-none-any.whl"
```

For development:

```bash
git clone https://github.com/henry753951/volley-court-training.git
cd volley-court-training
uv sync --group dev
```

This repository pins PyTorch to the explicit CUDA 12.8 wheel index on Windows and Linux.
Downstream uv projects should copy the PyTorch index/source block from `pyproject.toml` so the
accelerator choice remains explicit.

## Python API

The model is downloaded once, verified with SHA-256, and cached outside the repository.
OpenCV frames are BGR `numpy.uint8` arrays.

```python
import cv2

from volley_court import CourtLineModel, CourtVisualizer

frame = cv2.imread("frame.jpg")
model = CourtLineModel.from_pretrained(device="cuda:0", decoder="auto")
result = model.predict(frame, include_layout=True)

for line in result.lines:
    print(line.name, line.family, line.segment, line.score)

if result.layout and result.layout.status == "ok":
    for point in result.layout.keypoints:
        print(point.id, point.x, point.y, point.score)

rendered = CourtVisualizer().draw(frame, result)
cv2.imwrite("frame-court.jpg", rendered)
```

The result objects are immutable typed dataclasses. `to_mapping()` provides a JSON-ready bridge
for services that do not want to depend on those classes.

## CLI

```bash
uv run volley-court download

uv run volley-court predict-image frame.jpg frame-court.jpg --device cuda:0

uv run volley-court predict-video input.mp4 output.mp4 \
  --device cuda:0 --batch-size 8 --layout-every 10
```

Video output defaults to browser-safe H.264 High Profile, `yuv420p`, AAC audio, and MP4
`faststart`. It requires `ffmpeg`. The legacy `--video-codec mp4v` path is available only as a
fallback and is not recommended for web previews.

## Published assets

| Asset | Contents | SHA-256 |
| --- | --- | --- |
| [v2 semantic checkpoint](https://assets.hsulab.net/models/volley-court-lines/v2/court-line-yolo26n-semantic-v4.pt) | 1.62 M parameters, 6.39 MiB | `b4aed9...12b2b` |
| [v1 checkpoint](https://assets.hsulab.net/models/volley-court-lines/v1/court-line-yolo26n-v3.pt) | rollback artifact | `b0392c...19e86` |
| [Synthetic dataset](https://assets.hsulab.net/datasets/volley-court-lines/v1/court36-synthetic-combined-2000-camera-mode-v2.tar.gz) | 2,000 Blender images | `5f8fa0...78569` |
| [Real dataset](https://assets.hsulab.net/datasets/volley-court-lines/v1/court36-unified.tar.gz) | 357 real images | `a40be9...12a4c` |
| [Demo video](https://assets.hsulab.net/demos/volley-court-lines/v1/clip-volley-court-lines-v1.mp4) | 1080p H.264/AAC | `b30846...5b4be` |

Full checksums and dataset structure are in [docs/datasets.md](docs/datasets.md).

## Performance snapshot

All model tests use 640 px FP16/BF16 inference and the CUDA spatial decoder.

| GPU / path | Batch | Core FPS | Pipeline FPS | Latency |
| --- | ---: | ---: | ---: | ---: |
| RTX 5070, packaged benchmark | 1 | 81.05 | 62.85 | 15.32 ms p50/batch |
| RTX 5070, packaged benchmark | 16 | 830.56 | 240.90 | 4.15 ms mean/frame |
| H100 NVL, 2,000-frame video benchmark | 16 | 763.01 | 254.77 | 3.92 ms throughput/frame |
| H100 NVL, overlay + MP4 | 16 | 738.58 | 112.30 | 8.90 ms throughput/frame |
| RTX 5070, tracked 1080p H.264 demo | 8 | 301.24 | 25.34 | optical flow + layout + CPU encode |

The H100 and RTX 5070 rows use different source videos and benchmark harness revisions; they are
operational measurements, not a claim that one GPU is universally faster. See
[docs/benchmarks.md](docs/benchmarks.md) for methodology, raw JSON, memory use, and hardware
specifications.

## Quality snapshot

Controlled H100 evaluation on the same 37-image real test split:

| Path | PCK@1% | Precision@1% | Line recall | Family accuracy | Layout p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 semantic model + fixed solver | **28.05%** | **79.79%** | **78.87%** | **95.83%** | **2.33 ms** |
| v1 production model + search solver | 24.69% | 74.16% | 72.77% | 88.39% | 307.83 ms |

The fixed solver accepts 8/37 frames versus 11/37 for v1: it intentionally trades some coverage
for higher accepted-layout precision and bounded latency. Missing/abstained points count against
PCK. Full machine-readable reports are in `benchmarks/semantic-layout-v2-real-test.json` and
`benchmarks/v3-layout-baseline-real-test.json`.

## Training design

The released v2 model follows two semantic stages after dense-vote initialization:

1. **S1 synthetic:** 2,000 Blender images, 60 epochs, batch 64, BF16.
2. **S2 real:** 357 real images, 40 epochs, batch 32, BF16, with 35% synthetic replay.

This ordering teaches court geometry and camera coverage first, then adapts appearance to real
broadcasts without discarding the learned topology. Exact commands and the limitations of the
small real split are documented in [docs/training.md](docs/training.md).

## Repository layout

```text
src/volley_court/   installable inference, layout, visualization, training
configs/            topology and recorded stage parameters
datasets/           retained working datasets (not included in wheels)
blender/            synthetic-data generator and calibration tools
benchmarks/         reproducible reports and raw evidence
video-evals/        retained evaluation and demo videos
test-videos/        local regression inputs
docs/               model card, datasets, training, integration, benchmarks
```

Start with the [model card](docs/model-card.md), [training guide](docs/training.md), or
[analysis-engine integration guide](docs/integration.md).

## Distribution note

No standalone repository license has been selected yet. The repository is source-visible, but
no permission to redistribute code, weights, datasets, or demos is granted until the owner adds
explicit license terms.
