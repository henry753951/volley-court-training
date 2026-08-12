# Model card: court-line-yolo26n-semantic-v4

## Summary

`court-line-yolo26n-semantic-v4` is a compact volleyball-court geometry model derived from a YOLO26n
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

The CUDA decoder clusters dense votes, consolidates fragments, assigns two sidelines and an
ordered transverse-line sequence with fixed O(NK) work, and sends only observable layouts to an
intersection-anchored homography solver. The public result still emits an explicit `ok` or
`abstained` state and reconstructs the original Pose36 contract only after acceptance.

## Artifact

- URL: <https://assets.hsulab.net/models/volley-court-lines/v2/court-line-yolo26n-semantic-v4.pt>
- SHA-256: `b4aed936446e262518c927bf17dcf04877c7ba0f96e0db7687e84c3b2ea12b2b`
- size: 6,703,451 bytes (6.39 MiB)
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

The held-out real split contains 37 images. The comparison uses the identical split and H100
environment for both paths.

| Metric | semantic v4 fixed | production v3 search |
| --- | ---: | ---: |
| matched line recall | **0.7887** | 0.7277 |
| family accuracy on matched lines | **0.9583** | 0.8839 |
| visible PCK@0.5% | **0.2419** | 0.2257 |
| visible PCK@1% | **0.2805** | 0.2469 |
| visible precision@1% | **0.7979** | 0.7416 |
| visible F1@1% | **0.4151** | 0.3704 |
| median visible error | 1.72 px | 2.29 px |
| layout accepted | 8 / 37 | 11 / 37 |
| layout solver p99 | **2.33 ms** | 307.83 ms |

PCK thresholds are fractions of the labelled court bounding-box diagonal. Missing/abstained
keypoints count against recall. Precision answers a different question and is therefore higher.

## Training data and lineage

The semantic-v4 release was trained in two stages:

- synthetic S1: 1,600 train / 200 validation / 200 test Blender renders;
- real S2: 283 train / 37 validation / 37 test annotated images.

S1 ran for 60 epochs. S2 ran for 40 epochs from the best S1 checkpoint with 35% synthetic replay
to reduce catastrophic forgetting on court topology.
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
