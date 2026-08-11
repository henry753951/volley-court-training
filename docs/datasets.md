# Datasets

## Published archives

| Dataset | Split counts | Archive size | SHA-256 |
| --- | --- | ---: | --- |
| [court36-synthetic-combined-2000-camera-mode-v2](https://assets.hsulab.net/datasets/volley-court-lines/v1/court36-synthetic-combined-2000-camera-mode-v2.tar.gz) | 1,600 / 200 / 200 | 93,086,625 B | `5f8fa01c83d324636679e61d4eb9603ae46848c936a01c25a007441a2fe78569` |
| [court36-unified](https://assets.hsulab.net/datasets/volley-court-lines/v1/court36-unified.tar.gz) | 283 / 37 / 37 | 38,140,355 B | `a40be925ecb2aa145635171a24c32cc3331c99086717cfb4a81acf6f12912a4c` |

Counts are train / validation / test images. Both archives preserve the dataset root directory.

```bash
curl -LO https://assets.hsulab.net/datasets/volley-court-lines/v1/court36-unified.tar.gz
tar -xzf court36-unified.tar.gz
```

## Pose36 source labels

Each image uses a YOLO pose label with 36 ordered court points. Visibility values `1` and `2`
are geometrically usable. The topology in `configs/court_line_topology.yaml` maps these points to
seven physical court lines.

The conversion command does not rasterize a fixed-width semantic mask. It builds zero-width
line centers, orientations, and dense uniform samples at the configured spacing:

```bash
uv run volley-court-convert \
  --dataset datasets/court36-unified/dataset.yaml \
  --topology configs/court_line_topology.yaml \
  --output .work/court36-unified-lines \
  --stride 4 --preview-count 24
```

Converted targets are generated artifacts and may live outside `datasets/`; they can always be
recreated from the Pose36 labels.

## Synthetic dataset

The synthetic set contains 2,000 Blender renders across arena, skysphere, black-floor, camera
position, focal length, and framing variations. Camera-mode v2 intentionally includes partial
courts and edge crops so the network learns local geometry without requiring every landmark to
be visible.

Relevant entry points:

- `blender/generate_combined_arena_dataset.py`
- `blender/generate_sky_sphere_dataset.py`
- `blender/sky_sphere_calibrator.py`
- `blender/randomization-config.yaml`
- `blender/README.md`

Run generation through Blender's Python interpreter, not the package's normal Python runtime.
The `.blend` scene and external skysphere assets are deliberately kept under `blender/`.

## Real dataset

The unified real set contains 357 labelled broadcast/practice images. The held-out 37-image test
split is used for the reported line and layout metrics. Because the set has few pure negatives,
deployment should retain the ROI head and geometric abstention instead of accepting every local
line response as a court.

## Data governance

The archive checksums describe the exact internal release used for this model card. No standalone
dataset license has been selected; do not redistribute outside HSULab until provenance and terms
are reviewed.
