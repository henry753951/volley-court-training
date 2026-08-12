# Integration with volleyball-analysis-engine

Keep one `CourtLineModel` alive per worker process. Model construction loads the checkpoint and
allocates CUDA state; it should not happen per frame.

```python
from volley_court import CourtLineModel
from volleyball_analysis_engine.inference import COURT_WORLD_POINTS
from volleyball_analysis_engine.records import CourtFrame, CourtKeypoint


class CourtEstimator:
    def __init__(self) -> None:
        self.model = CourtLineModel.from_pretrained(
            device="cuda:0",
            decoder="auto",
            include_layout=False,
        )

    def infer(self, frame, frame_index: int) -> CourtFrame | None:
        result = self.model.predict(frame)
        result = self.model.attach_layout(result)
        if result.layout is None or result.layout.status != "ok":
            return None
        keypoints = tuple(
            CourtKeypoint(
                index=point.id,
                frame_pos_px=(point.x, point.y),
                confidence=point.score,
                world_pos_m=(
                    COURT_WORLD_POINTS[point.id] if point.id < len(COURT_WORLD_POINTS) else None
                ),
            )
            for point in result.layout.keypoints
        )
        return CourtFrame(frame_index=frame_index, available=True, keypoints=keypoints)
```

This is the existing engine's `CourtFrame` / `CourtKeypoint` shape, so `_infer_court` can replace
its current Ultralytics pose call without changing projection and overlay consumers. Line and
layout mappings can be attached separately when the engine schema is ready for richer evidence.
Do not publish candidate keypoints as accepted points when status is `ambiguous` or `abstained`.

## Frame synchronization

For overlays that must track every presented source frame, infer at `court_stride=1` or carry an
explicit source-frame timestamp with every result. Batching improves throughput but does not
remove the need to map each result back to the corresponding frame.

For offline clips:

```python
results = model.predict_many(frames, include_layout=False)
```

Layout matching is CPU geometry and can run less often for video. The package's
`CourtLayoutTracker` advances the accepted 36-point layout on every frame with optical flow,
rejects inconsistent motion with RANSAC, smooths fresh matches, and expires stale state. Feed it
frames in source order; do not reuse a result across frames without tracking it.

## Deployment knobs

- `VOLLEY_COURT_MODEL`: override the checkpoint path.
- `VOLLEY_COURT_CACHE`: override the verified download cache.
- batch 1: lowest request latency;
- batch 4–16: higher offline throughput;
- `decoder="auto"`: spatial CPU decoder for latency-sensitive batch 1, CUDA decoder when batching;
- `decoder="cuda"`: force the CUDA path for a known batched service.

The public package deliberately returns dataclasses and mappings instead of importing the
analysis engine. The engine owns transport schemas, timestamps, queueing, and persistence.
