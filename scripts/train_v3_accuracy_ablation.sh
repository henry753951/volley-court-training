#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/home/nckusoc/volley-court-training-v3-metric-first}"
base_checkpoint="${2:-/mnt/hsulab-assets/models/volley-court-lines/v2/court-line-yolo26n-layout-v2.pt}"
data_root="${3:-/home/nckusoc/volley-court-training-direct-layout/artifacts/converted-real-v1}"
python_bin="${repo_root}/.venv/bin/python"
run_root="${repo_root}/runs/v3-accuracy-ablation-20260813"

cd "${repo_root}"
mkdir -p "${run_root}"
export PYTHONPATH="${repo_root}/src"

train() {
  local name="$1"
  shift
  "${python_bin}" -c 'from volley_court.train import main; raise SystemExit(main())' \
    --data "${data_root}" \
    --weights "${base_checkpoint}" \
    --output "${run_root}/${name}" \
    --epochs 40 \
    --batch 32 \
    --workers 8 \
    --imgsz 512 \
    --device cuda:0 \
    --lr 0.00005 \
    --weight-decay 0.0005 \
    --freeze-feature-epochs 5 \
    --target-mode dense_semantic \
    --sample-spacing 16 \
    --intersection-exclusion 4 \
    --hard-negative-probability 0.20 \
    --family-weight 0.5 \
    --identity-weight 1.25 \
    --roi-weight 0.5 \
    --layout-proposals 1 \
    --layout-coordinate-weight 8.0 \
    --layout-validity-weight 1.5 \
    --layout-softargmax-weight 4.0 \
    --layout-visibility-positive-weight 1.5 \
    --layout-validity-positive-weight 1.5 \
    --seed 36 \
    --save-every 5 \
    --amp-dtype bf16 \
    "$@" \
    2>&1 | tee "${run_root}/${name}.log"
}

# First isolate the layout-learning change. The released dense semantic head is frozen.
train layout-only --freeze-dense-epochs 40 --base-head-gradient-scale 0.0

# Then allow low-gradient semantic adaptation for the verifier's line-identity evidence.
train joint-low-gradient --freeze-dense-epochs 5 --base-head-gradient-scale 0.25

# This sweep is exploratory: release selection is performed with fixed PCK,
# precision, severe-accept, and consecutive-video gates rather than loss alone.
