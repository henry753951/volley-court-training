# Benchmarks

## Model footprint

| Item | Value |
| --- | ---: |
| checkpoint | 6,703,451 bytes / 6.39 MiB |
| parameters | 1,622,671 |
| estimated compute at 640 px | 9.16 GFLOPs |
| output stride | 4 |
| output channels | 15 |

## Controlled packaged benchmark

Both GPUs processed the same 512 preloaded frames from `test-videos/clip.mp4` with the v2
checkpoint, 640 px inference, FP16, and the automatic CUDA decoder. Core FPS times model forward
plus dense decode. Pipeline FPS also includes preprocessing and device transfer.

| GPU | Batch | Core FPS | Pipeline FPS | Mean frame latency | p50 batch | p95 batch | Peak CUDA memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 | 1 | 59.44 | 48.07 | 20.80 ms | 20.17 ms | 26.51 ms | 65.7 MiB |
| RTX 5070 | 4 | 230.04 | 127.51 | 7.84 ms | 30.59 ms | 40.27 ms | 158.1 MiB |
| RTX 5070 | 8 | 435.91 | 174.88 | 5.72 ms | 44.85 ms | 51.61 ms | 282.0 MiB |
| RTX 5070 | 16 | 569.11 | 188.54 | 5.30 ms | 84.31 ms | 96.38 ms | 525.2 MiB |
| H100 NVL | 1 | 135.98 | 81.53 | 12.27 ms | 10.36 ms | 27.31 ms | 66.5 MiB |
| H100 NVL | 4 | 463.74 | 176.17 | 5.68 ms | 21.55 ms | 24.22 ms | 158.1 MiB |
| H100 NVL | 8 | 822.80 | 183.50 | 5.45 ms | 41.33 ms | 46.63 ms | 282.5 MiB |
| H100 NVL | 16 | 1,302.25 | 200.41 | 4.99 ms | 77.96 ms | 85.30 ms | 525.2 MiB |

Raw reports: [`benchmarks/rtx5070.json`](../benchmarks/rtx5070.json) and
[`benchmarks/h100.json`](../benchmarks/h100.json).

## Browser-compatible demo

The 14.75-second 1080p demo contains 884 frames and was rendered on the RTX 5070 with batch 8.
The detector and fixed semantic layout solver both ran on every source frame (`layout_every=1`).
It achieved 256.85 model/decode FPS and 41.73 end-to-end FPS. End-to-end includes file decode,
layout tracking, OpenCV drawing, source audio, and CPU H.264 encoding, so it is deliberately lower
than the preloaded service benchmark.

The output is H.264 High Profile, `yuv420p`, AAC, and MP4 `faststart`:

<https://assets.hsulab.net/demos/volley-court-lines/v2/clip-volley-court-lines-v2.mp4>

## Fixed semantic layout quality

Environment: H100 NVL, 37-image `court36-unified` test split, 640 px, confidence 0.25. PCK
includes missing output as failure; precision is conditional on emitted keypoints.

| PCK@1% | Precision@1% | Line recall | Family accuracy | Layout accepted | Solver p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28.05% | 79.79% | 78.87% | 95.83% | 8/37 | 2.33 ms |

The production path uses CUDA dense proposals, fixed semantic sequence assignment, an
observability gate, and an intersection-anchored homography. It does not enumerate topology
permutations. Raw report:
[`benchmarks/semantic-layout-v2-real-test.json`](../benchmarks/semantic-layout-v2-real-test.json).

## Measurement notes

- CUDA operations are synchronized around timed regions.
- Batch throughput is not single-frame request latency.
- Video end-to-end FPS includes visualization and encoding; packaged pipeline FPS does not.
- Vendor peak compute is not directly comparable to measured application FPS.
