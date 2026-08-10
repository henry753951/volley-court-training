#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv was not found; set UV_BIN or install uv under ~/.local/bin" >&2
  exit 1
fi
data="${DATASET_YAML:-$project_root/datasets/court36-color-canonical-v2/dataset.yaml}"
model="${MODEL:-yolo26m-pose.pt}"
epochs="${EPOCHS:-300}"
batch="${BATCH:-0.85}"
workers="${WORKERS:-16}"
run_name="${RUN_NAME:-court36-h100-m-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "$data" ]]; then
  echo "dataset not found: $data" >&2
  exit 1
fi

cd "$project_root"
exec "$uv_bin" run court-train \
  --data "$data" \
  --model "$model" \
  --epochs "$epochs" \
  --imgsz 1280 \
  --batch "$batch" \
  --device 0 \
  --workers "$workers" \
  --cache disk \
  --project "$project_root/runs" \
  --name "$run_name" \
  --patience 75 \
  --cos-lr \
  --compile \
  --close-mosaic 20 \
  --mosaic 0.75 \
  --multi-scale 0.20
