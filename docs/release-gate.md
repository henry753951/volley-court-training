# Direct layout release gate

Production v1 remains available as the rollback checkpoint. Direct-layout v2 became the default only
after passing every gate below; a high line score alone is not evidence of correct geometry.

## Fixed still-image gate

Evaluate `court36-unified` test with `volley-court-evaluate-model`.

- Visible PCK@1% must be at least the v1 baseline: `0.4850`.
- Visible precision@1% must be at least the v1 baseline: `0.8104`.
- Accepted layouts with PCK@2% below `0.25`: exactly `0`.
- Any accepted overlay with crossed, flipped, collapsed, or ray-like court geometry blocks release.
- An unexpected abstention on a clearly observable full court also blocks release.

## Single-image latency gate

Benchmark batch size one after warmup, including preprocessing, model, decode, direct-layout verification,
and result construction.

- RTX 5070 p50 must not exceed the v1 pipeline baseline of `15.32 ms`.
- RTX 5070 p95 must not exceed the v1 baseline of `20.28 ms`.
- Report H100 and RTX 5070 independently; batched throughput is not a substitute for single-image latency.

## Fixed-video visual gate

Run all locally available fixed inputs: `clip.mp4`, `pW66S38FAQM.mp4`, `rMvxEtorQhw.mp4`, and
`gdrSr-Vso90.mp4`.

For each input, generate at least three contact sheets. Every sheet must contain at least five consecutive
source frames, not temporally spaced samples. Inspect the connected layout across the entire five-frame
group for:

- missing layout while the full court is observable;
- a complete layout accepted from unrelated or insufficient evidence;
- left/right or near/far identity flips;
- crossed, collapsed, converging-ray, or off-court geometry;
- one-frame jumps or drift relative to the camera motion.

Any one of these failures blocks model, package, documentation, and asset publication.

## v2 release result

- real test at 512 px: PCK@1% `0.5100`, precision@1% `0.8196`, severe false accepts `0`;
- RTX 5070 batch 1: p50 `12.93 ms`, p95 `18.41 ms` in the median-timing repeated run;
- H100 NVL batch 1: p50 `10.16 ms`, p95 `26.17 ms` on the shared host;
- visual audit: 4 videos, 3 sheets per video, 5 consecutive frames per sheet, 60 frames total;
- accepted results inspected: no wrong rotation, near/far flip, crossed polygon, collapse, or ray fan;
- `clip`: 15 `ok`; `gdrSr-Vso90`: 14 `ok`, 1 `ambiguous`; `pW66S38FAQM`: 6 `ok`,
  9 `abstained`; `rMvxEtorQhw`: 5 `ok`, 10 `abstained`;
- conservative abstention remains on close-ups and insufficient views and is not converted to a layout.
