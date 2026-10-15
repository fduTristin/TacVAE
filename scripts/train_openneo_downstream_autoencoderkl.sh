#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VTLA_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"

export TACVAE_PYTHON="${TACVAE_PYTHON:-${VTLA_ROOT}/.venv/bin/python}"
export DATASET_DIR="${DATASET_DIR:-/data/workspace/VTLA/tacvae_openneo_downstream_256x144}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/workspace/VTLA/tacvae_openneo_downstream_vae_100k}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29638}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2000}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-openneo-tacvae}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-openneo-downstream-vae-100k}"
export WANDB_NAME="${WANDB_NAME:-openneo-downstream-vae-4gpu-bs32}"

exec "${SCRIPT_DIR}/train_tactile_autoencoderkl.sh" "$@"
