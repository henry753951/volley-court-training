# Benchmarks

## Model footprint

| Item | Value |
| --- | ---: |
| checkpoint | 6,703,451 bytes / 6.39 MiB |
| parameters | 1,622,671 |
| estimated compute at 640 px | 9.16 GFLOPs |
| output stride | 4 |
| output channels | 15 |

## RTX 5070 local package benchmark

Environment: Windows 11, Python 3.12.13, PyTorch 2.11.0+cu128, CUDA decoder, FP16, 512
preloaded frames from `test-videos/clip.mp4`.

| Batch | Core FPS | Pipeline FPS | Mean frame latency | p50 batch | p95 batch | Peak CUDA memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 81.05 | 62.85 | 15.91 ms | 15.32 ms | 20.28 ms | 65.7 MiB |
| 4 | 329.52 | 169.67 | 5.89 ms | 23.12 ms | 25.91 ms | 158.1 MiB |
| 8 | 581.99 | 219.81 | 4.55 ms | 35.52 ms | 40.49 ms | 282.0 MiB |
| 16 | 830.56 | 240.90 | 4.15 ms | 64.19 ms | 80.22 ms | 525.2 MiB |

Raw report: [`benchmarks/rtx5070.json`](../benchmarks/rtx5070.json).

The final 14.8-second 1080p demo achieved 301.24 model/decode FPS and 25.34 end-to-end FPS at
batch 8. End-to-end includes OpenCV drawing, geometric layout matching every 10 frames,
per-frame optical-flow tracking of the accepted 36-point layout, source audio, and CPU H.264
encoding; it is not the correct number for a no-render service.

## H100 NVL operational benchmark

Environment: Linux, PyTorch 2.11.0+cu128, CUDA decoder, FP16, batch 16, 2,000 frames from
`pW66S38FAQM.mp4`.

| Path | Core FPS | End-to-end FPS | Reciprocal throughput |
| --- | ---: | ---: | ---: |
| preloaded forward + decode | 763.01 | — | 1.31 ms/frame |
| async read + preprocess + inference | 763.01 | 254.77 | 3.92 ms/frame |
| read + inference + overlay + MP4 | 738.58 | 112.30 | 8.90 ms/frame |

Raw reports:

- [`benchmarks/raw/h100-video-core-and-pipeline.json`](../benchmarks/raw/h100-video-core-and-pipeline.json)
- [`benchmarks/raw/h100-video-overlay.json`](../benchmarks/raw/h100-video-overlay.json)

The reciprocal values are throughput per frame at batch 16, not single-frame request latency.

## Fixed semantic layout A/B

Environment: H100 NVL, 37-image `court36-unified` test split, 640 px, confidence 0.25.
PCK includes missing output as failure; precision is conditional on emitted keypoints.

| Path | PCK@1% | Precision@1% | Line recall | Family accuracy | Layout ok | Solver p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-v4 + ordered fixed solver | **28.05%** | **79.79%** | **78.87%** | **95.83%** | 8/37 | **2.33 ms** |
| production v3 + topology search | 24.69% | 74.16% | 72.77% | 88.39% | 11/37 | 307.83 ms |

The new path runs CUDA dense proposals, fixed O(NK) semantic sequence alignment, an observability
gate, and an intersection-anchored homography. It does not enumerate line-layout permutations.
The stricter gate reduces coverage by three frames while improving PCK, accepted precision, line
recall, family accuracy, and p99 latency. Raw reports:

- [`benchmarks/semantic-layout-v2-real-test.json`](../benchmarks/semantic-layout-v2-real-test.json)
- [`benchmarks/v3-layout-baseline-real-test.json`](../benchmarks/v3-layout-baseline-real-test.json)

## Hardware context

| GPU | Memory | Vendor peak compute | Power context |
| --- | ---: | ---: | ---: |
| [NVIDIA H100 NVL](https://www.nvidia.com/en-us/data-center/h100/) | 94 GB HBM3, 3.9 TB/s | 1,671 FP16 Tensor TFLOPS with sparsity | 350–400 W configurable |
| [GeForce RTX 5070](https://www.nvidia.com/en-gb/geforce/graphics-cards/compare/) | 12 GB GDDR7 | 988 AI TOPS | local board power limit 250 W |

Vendor peak units are architecture marketing/specification units and are not comparable to the
model's measured FPS. This workload is small enough that preprocessing, launch overhead, decode,
layout matching, and video encoding dominate many paths.

## Measurement definitions

- **Core FPS:** timed model forward plus dense decoder after warm-up.
- **Pipeline FPS:** preprocessing, device transfer, forward, and decoder over preloaded frames.
- **Video end-to-end FPS:** file decode and/or visualization/encode as named by the row.
- CUDA operations are synchronized around timed regions.
- The H100 and RTX 5070 runs used different source videos and harness revisions; use each report
  for capacity planning on its own host, not as a controlled hardware comparison.
