#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${DATASET_YAML:-$project_root/datasets/court36-color-canonical-v2/dataset.yaml}"
checkpoint="${RESUME_CHECKPOINT:-$project_root/runs/court36-color-distortion-20260810-194718/weights/last.pt}"
run_name="${RUN_NAME:-court36-color-distortion-20260810-194718}"
batch="${BATCH:-0.85}"
workers="${WORKERS:-16}"

for required in "$data" "$checkpoint"; do
  if [[ ! -f "$required" ]]; then
    echo "required file not found: $required" >&2
    exit 1
  fi
done

cd "$project_root"
exec uv run court-train \
  --data "$data" \
  --resume "$checkpoint" \
  --batch "$batch" \
  --device 0 \
  --workers "$workers" \
  --cache disk \
  --project "$project_root/runs" \
  --name "$run_name" \
  --patience 75
