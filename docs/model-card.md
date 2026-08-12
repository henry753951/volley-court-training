# Model card: court-line-yolo26n-semantic-v4

## Summary

`court-line-yolo26n-semantic-v4` is a compact volleyball-court geometry model derived from a
YOLO26n feature extractor. It predicts dense, zero-width line evidence instead of a semantic
mask with an invented line thickness.

The network emits 15 channels at stride 4:

- one dense line-center confidence channel;
- two offset channels that vote back to the zero-width centerline;
- two orientation channels;
- two family channels: `vertical` and `horizontal` in court topology;
- seven semantic line identities;
- one court-ROI context channel.

The CUDA decoder consolidates dense votes. A fixed semantic solver assigns the two sidelines and
ordered transverse lines, applies an observability gate, and reconstructs the original Pose36
contract only for accepted layouts.

## Artifact

- URL: <https://assets.hsulab.net/models/volley-court-lines/v2/court-line-yolo26n-semantic-v4.pt>
- SHA-256: `b4aed936446e262518c927bf17dcf04877c7ba0f96e0db7687e84c3b2ea12b2b`
- size: 6,703,451 bytes (6.39 MiB)
- parameters: 1,622,671
- estimated compute at 640 px: 9.16 GFLOPs
- target mode: `dense_semantic`
- sample spacing: 16 px
- intersection exclusion radius: 4 px

## Intended use

- broadcast and fixed-camera volleyball court-line detection;
- recovery of court keypoints for projection, calibration, and overlay;
- offline batched analysis and low-latency edge inference;
- initialization for later court-geometry research.

The model is not a calibrated arbitrary-image court classifier. The layout observability gate is
the guardrail for incomplete or inconsistent evidence.

## Evaluation

The held-out real split contains 37 images.

| Metric | Result |
| --- | ---: |
| matched line recall | 0.7887 |
| family accuracy on matched lines | 0.9583 |
| visible PCK@0.5% | 0.2419 |
| visible PCK@1% | 0.2805 |
| visible precision@1% | 0.7979 |
| visible F1@1% | 0.4151 |
| median visible error | 1.72 px |
| layout accepted | 8 / 37 |
| layout solver p99 | 2.33 ms |

PCK thresholds are fractions of the labelled court bounding-box diagonal. Missing/abstained
keypoints count against recall.

## Training data and lineage

The release was trained in two stages:

- synthetic S1: 1,600 train / 200 validation / 200 test Blender renders, 60 epochs;
- real S2: 283 train / 37 validation / 37 test annotated images, 40 epochs, with 35% synthetic
  replay.

The released history CSV files are stored next to the checkpoint on the asset share. Exact stage
parameters are in [`docs/training.md`](training.md).

## Limitations

- The real dataset is small and has limited negative-only imagery.
- Advertising boards, floor seams, and unrelated straight lines can trigger local evidence.
- A single visible right angle is geometrically underdetermined and should abstain.
- Semantic identity can still be mirrored when camera-side evidence is insufficient.
- Video tracking reduces jitter but does not create missing geometric evidence.
- The 37-image test split is too small for a narrow confidence interval.

Keep the typed layout status in downstream systems; do not coerce `abstained` results into
accepted keypoints.

## Distribution

No standalone license has been selected. Treat the artifact as HSULab-internal until explicit
code, weight, dataset, and demo licenses are published.
