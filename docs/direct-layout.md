# Direct single-frame layout head

The candidate architecture keeps the production v1 stride-4 semantic line and ROI heads, then adds a
structured Pose36 anchor head to the same fused stride-4 YOLO26n feature map. The head predicts:

- one spatial heatmap and sub-cell offset for each of the 36 canonical court anchors;
- one visibility logit for each anchor;
- one observability logit saying whether the frame contains enough two-dimensional evidence to solve a
  complete court;
- eight orientation logits from a spatial 4x4 pooled feature map. Global average pooling is not used
  here because it erases the left/right and near/far spatial identity needed by the court topology.

At inference, visible identity-bearing anchors solve a bounded homography and therefore all original
Pose36 keypoints. The primary pass is 128-iteration RANSAC; a 512-iteration USAC pass is attempted only
when the primary projected quadrilateral is geometrically degenerate. This avoids both low-resolution
global corner regression and combinatorial line identity search without charging the fallback on normal
frames. Dense line evidence remains an independent verification signal: a solved
homography is accepted only when projected semantic court lines agree with observed line evidence.

## Training targets

Pose36 labels are converted to zero-width center-line segments. For the direct head, all reliable labelled
points are used to fit a ground-plane homography. A target is observable only when it has non-collinear
support, low reprojection error, a non-degenerate quadrilateral, and the fixed image convention used by
the runtime. Pose36 width/length/both symmetry variants are topology-preserving labels of the same court;
the target builder reorders all four variants into `(left-near, left-far, right-far, right-near)` instead
of teaching the observability head that a mirrored label means no court. Frames with genuinely
insufficient two-dimensional evidence train the validity head to abstain and do not provide a
corner-regression target.

Synthetic and real hard negatives crop outside the court ROI and add distractor lines. They train the
model not to construct a complete court from unrelated venue markings.

## Inference contract

`CourtLineModel.predict(..., include_layout=True)` runs the backbone, semantic evidence head, and direct
layout head once. It then performs a fixed-cost seven-line evidence check and returns either:

- `ok`, with a homography and all 36 canonical keypoints;
- `ambiguous`, when layout proposals or semantic identities disagree;
- `abstained`, when the court is not observable or the geometric evidence is insufficient.

The orientation head ranks only geometrically valid topology permutations. The verifier never repairs
or invents a layout; it rejects unsupported output. Video tracking may smooth accepted layouts, but
release evaluation always inspects raw single-frame results first.

## H100 stages

```bash
./scripts/train_direct_layout_s1.sh <synthetic-run-name>
./scripts/train_direct_layout_s2.sh \
  runs/<synthetic-run-name>/best.pt \
  <real-run-name>
```

S1 trains only the direct head on synthetic data, leaving every production v1 parameter frozen. S2 keeps
the existing dense output layers frozen, warms the direct head on real data, then allows the shared
features to adapt under both dense and layout losses.

The released v2 checkpoint adds S6: geometry and dense heads remain frozen while the spatial eight-way
orientation classifier is trained with class-balanced loss. Epoch 20 is selected by the fixed real-test,
latency, and consecutive-frame visual gates rather than training loss alone.
