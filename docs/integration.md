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
            image_size=512,
            include_layout=True,
        )

    def infer(self, frame, frame_index: int) -> CourtFrame | None:
        result = self.model.predict(frame)
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

For overlays that must track every presented source frame, infer at `court_stride=1` and carry an
explicit source-frame timestamp with every result. The v2 layout head runs for every frame in the
batch; batching changes scheduling, not sampling. Map each result back to its source frame.

For offline clips:

```python
results = model.predict_many(frames, include_layout=True)
```

Direct layout verification runs independently for every frame. The optional `CourtLayoutTracker`
smooths fresh accepted matches, rejects inconsistent optical flow with RANSAC, and expires stale
state. Feed it frames in source order; do not reuse a result across frames without tracking it.

## Deployment knobs

- `VOLLEY_COURT_MODEL`: override the checkpoint path.
- `VOLLEY_COURT_CACHE`: override the verified download cache.
- 512 px batch 1: released low-latency path;
- batch 4–16: higher offline throughput;
- `decoder="auto"`: chooses the available spatial/CUDA decoder;
- `decoder="cuda"`: measured RTX 5070 batch-1 path when the worker owns CUDA state.

The public package deliberately returns dataclasses and mappings instead of importing the
analysis engine. The engine owns transport schemas, timestamps, queueing, and persistence.
