#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_ROOT}/data/tactile_ae_dataset_256x144zhiying"
VAE_DIR="${PROJECT_ROOT}/outputs/zhiying-scratch-256x144-eventindex"

python "${PROJECT_ROOT}/scripts/compute_tactile_latent_stats.py" \
  --vae-path "${VAE_DIR}" \
  --image-dir "${DATASET_DIR}/train" \
  --index-file "${DATASET_DIR}/train_event_indexzhiying.txt" \
  --output "${VAE_DIR}/latent_stats.json" \
  --image-width 256 \
  --image-height 144 \
  --batch-size 32 \
  --num-workers 8
