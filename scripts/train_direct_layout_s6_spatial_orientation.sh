#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
weights="${1:-$repo_root/runs/direct-layout-s2-anchor40-v8-20260812/best.pt}"
data="${2:-$repo_root/artifacts/converted-real-video-hard-v2}"
run_name="${3:-direct-layout-s6-spatial-orientation-v1-20260813}"
cd "$repo_root"

exec "$HOME/.local/bin/uv" run volley-court-train \
  --data "$data" \
  --weights "$weights" \
  --output "runs/$run_name" \
  --epochs 60 \
  --save-every 2 \
  --batch 32 \
  --workers 4 \
  --device cuda:0 \
  --target-mode dense_semantic \
  --layout-proposals 1 \
  --freeze-feature-epochs 60 \
  --freeze-shared-epochs 60 \
  --freeze-dense-epochs 60 \
  --freeze-layout-geometry-epochs 60 \
  --base-head-gradient-scale 0.0 \
  --hard-negative-probability 0.0 \
  --lr 1e-3 \
  --layout-coordinate-weight 0.0 \
  --layout-validity-weight 0.0 \
  --layout-proposal-weight 1.0 \
  --layout-orientation-class-weights 1.0,4.7,5.2,8.0,3.1,4.6,8.0,3.2
