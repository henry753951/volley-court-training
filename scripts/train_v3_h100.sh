#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/home/nckusoc/volley-court-training-v3-metric-first}"
python_bin="${repo_root}/.venv/bin/python"
base_checkpoint="/mnt/hsulab-assets/models/volley-court-lines/v2/court-line-yolo26n-layout-v2.pt"
converted_root="/home/nckusoc/volley-court-training-direct-layout/artifacts"
run_root="${repo_root}/runs"

cd "${repo_root}"
mkdir -p "${run_root}"
export PYTHONPATH="${repo_root}/src"

run_training() {
  "${python_bin}" -c 'from volley_court.train import main; raise SystemExit(main())' "$@"
}

run_training \
  --data "${converted_root}/converted-synthetic-v2" \
  --weights "${base_checkpoint}" \
  --output "${run_root}/v3-s1-synthetic" \
  --epochs 80 \
  --batch 64 \
  --workers 8 \
  --imgsz 512 \
  --device cuda \
  --lr 0.0002 \
  --weight-decay 0.0005 \
  --freeze-feature-epochs 5 \
  --base-head-gradient-scale 0.5 \
  --target-mode dense_semantic \
  --sample-spacing 16 \
  --intersection-exclusion 4 \
  --hard-negative-probability 0.2 \
  --family-weight 0.5 \
  --identity-weight 1.0 \
  --roi-weight 0.5 \
  --layout-proposals 1 \
  --layout-coordinate-weight 8.0 \
  --layout-validity-weight 1.0 \
  --seed 36 \
  --save-every 10 \
  --amp-dtype bf16 \
  2>&1 | tee "${run_root}/v3-s1-synthetic.log"

run_training \
  --data "${converted_root}/converted-real-v1" \
  --weights "${run_root}/v3-s1-synthetic/best.pt" \
  --output "${run_root}/v3-s2-real" \
  --epochs 120 \
  --batch 32 \
  --workers 8 \
  --imgsz 512 \
  --device cuda \
  --lr 0.0001 \
  --weight-decay 0.0005 \
  --freeze-feature-epochs 10 \
  --base-head-gradient-scale 0.5 \
  --target-mode dense_semantic \
  --sample-spacing 16 \
  --intersection-exclusion 4 \
  --hard-negative-probability 0.15 \
  --family-weight 0.5 \
  --identity-weight 1.25 \
  --roi-weight 0.5 \
  --layout-proposals 1 \
  --layout-coordinate-weight 10.0 \
  --layout-validity-weight 1.5 \
  --seed 36 \
  --save-every 10 \
  --amp-dtype bf16 \
  2>&1 | tee "${run_root}/v3-s2-real.log"
