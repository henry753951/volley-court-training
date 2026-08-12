#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${1:-direct-layout-s1-synthetic-20260812}"
cd "$repo_root"

exec "$HOME/.local/bin/uv" run volley-court-train \
  --data artifacts/converted-synthetic-v2 \
  --weights /mnt/hsulab-assets/models/volley-court-lines/v1/court-line-yolo26n-v3.pt \
  --output "runs/$run_name" \
  --epochs 80 \
  --batch 64 \
  --workers 8 \
  --device cuda:0 \
  --target-mode dense_semantic \
  --layout-proposals 1 \
  --freeze-shared-epochs 0 \
  --freeze-dense-epochs 80 \
  --hard-negative-probability 0.10 \
  --lr 5e-4 \
  --layout-coordinate-weight 1.0 \
  --layout-validity-weight 1.0 \
  --layout-proposal-weight 0.0
