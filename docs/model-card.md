# Model card: court-line-yolo26n-v3

## Summary

`court-line-yolo26n-v3` is a compact volleyball-court geometry model derived from a YOLO26n
feature extractor. It predicts dense, zero-width line evidence rather than a semantic mask with
an invented line thickness.

The network emits 15 channels at stride 4:

- one dense line-center confidence channel;
- two offset channels that vote back to the zero-width centerline;
- two orientation channels;
- two family channels: `vertical` and `horizontal` in court topology;
- seven semantic identity channels: left sideline, far baseline, right sideline, near baseline,
  near attack, center, and far attack;
- one court-ROI context channel.

The CUDA/spatial decoder clusters the dense votes into short line segments. A separate geometric
matcher can then reconstruct the original Pose36 keypoints and emits an explicit `ok`,
`ambiguous`, or `abstained` state.

## Artifact

- URL: <https://assets.hsulab.net/models/volley-court-lines/v1/court-line-yolo26n-v3.pt>
- SHA-256: `b0392c221978c87405f2646f41f14c1b66d4e7940d07c4a19c170b8321119e86`
- size: 6,987,346 bytes (6.67 MiB)
- parameters: 1,622,671
- estimated compute at 640 px: 9.16 GFLOPs
- checkpoint format: `yolo26n-court-line-v1`
- target mode: `dense_semantic`
- sample spacing: 16 px
- intersection exclusion radius: 4 px

## Intended use

- broadcast and fixed-camera volleyball court-line detection;
- recovery of court keypoints for projection, calibration, and overlay;
- offline batched analysis and low-latency edge inference;
- initialization for later court-geometry research.

The model is not intended to determine whether an arbitrary image contains a valid volleyball
court with calibrated confidence. The layout matcher is the current guardrail for incomplete or
inconsistent evidence.

## Evaluation

The held-out real split contains 37 images. The raw report is
[`benchmarks/quality/court36-unified-test.json`](../benchmarks/quality/court36-unified-test.json).

| Metric | Value |
| --- | ---: |
| matched line recall | 0.9061 |
| family accuracy on matched lines | 0.8135 |
| visible PCK@0.5% | 0.4576 |
| visible PCK@1% | 0.4850 |
| visible precision@1% | 0.8104 |
| visible F1@1% | 0.6069 |
| median visible error | 1.67 px |
| layout `ok` / `ambiguous` / `abstained` | 18 / 18 / 1 |

PCK thresholds are fractions of the labelled court bounding-box diagonal. Missing/abstained
keypoints count against recall. Precision answers a different question and is therefore higher.

## Training data and lineage

The context-v3 release was trained in two stages:

- synthetic S1: 1,600 train / 200 validation / 200 test Blender renders;
- real S2: 283 train / 37 validation / 37 test annotated images.

S1 was warm-started from an internal dense-votes checkpoint, and S2 from the best S1 checkpoint.
The released history CSV files are stored next to the checkpoint on the asset share. Exact stage
parameters are in [`docs/training.md`](training.md).

## Limitations

- The real dataset is small and has limited negative-only imagery.
- Advertising boards, floor seams, and unrelated straight lines can trigger local line evidence.
- A single visible right angle is often geometrically underdetermined; the matcher should report
  ambiguity instead of inventing a unique court.
- Semantic line identity can be mirrored when camera-side evidence is insufficient.
- Single-image layout recovery has no temporal context. Video mode provides an optional tracker
  that smooths accepted layouts and advances all 36 points between matcher passes using optical
  flow with RANSAC rejection and bounded state expiry.
- The 37-image test split is too small for a narrow confidence interval.

For deployment, keep the typed layout status and do not coerce `ambiguous` or `abstained` results
into accepted keypoints.

## Distribution

No standalone license has been selected. Treat the artifact as HSULab-internal until explicit
code, weight, dataset, and demo licenses are published.
