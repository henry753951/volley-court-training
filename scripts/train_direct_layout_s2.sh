#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage1="${1:-$repo_root/runs/direct-layout-s1-synthetic-20260812/best.pt}"
run_name="${2:-direct-layout-s2-real-20260812}"
cd "$repo_root"

exec "$HOME/.local/bin/uv" run volley-court-train \
  --data artifacts/converted-real-v1 \
  --weights "$stage1" \
  --output "runs/$run_name" \
  --epochs 120 \
  --batch 32 \
  --workers 4 \
  --device cuda:0 \
  --target-mode dense_semantic \
  --layout-proposals 1 \
  --freeze-shared-epochs 0 \
  --freeze-dense-epochs 120 \
  --hard-negative-probability 0.10 \
  --lr 1e-4 \
  --layout-coordinate-weight 1.0 \
  --layout-validity-weight 1.0 \
  --layout-proposal-weight 0.0
