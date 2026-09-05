#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  printf '%s\n' "conda was not found in PATH" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tacvae

# Edit these values for another dataset, GPU, or experiment.
DATASET_DIR="${PROJECT_ROOT}/data/tactile_ae_dataset_full"
TRAIN_DATA_DIR="${DATASET_DIR}/train"
VALIDATION_IMAGES_FILE="${DATASET_DIR}/validation_images.txt"
SUMMARY_FILE="${DATASET_DIR}/summary.json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/tactile-aekl-formal-bs64"

PRETRAINED_MODEL="stabilityai/sd-vae-ft-mse"
GPU_IDS="5"
NUM_PROCESSES="1"
MAIN_PROCESS_PORT="29617"

RESOLUTION="224"
TRAIN_BATCH_SIZE="64"
DATALOADER_NUM_WORKERS="8"
MAX_TRAIN_STEPS="30000"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="1e-6"
DISC_LEARNING_RATE="1e-6"
LR_SCHEDULER="constant_with_warmup"
DISC_LR_SCHEDULER="constant_with_warmup"
LR_WARMUP_STEPS="1000"

REC_LOSS="l2"
KL_SCALE="1e-6"
PERCEPTUAL_SCALE="0.05"
DISC_START="1000000000"
DISC_FACTOR="1.0"

VALIDATION_STEPS="5000"
CHECKPOINTING_STEPS="5000"
CHECKPOINTS_TOTAL_LIMIT="4"

WANDB_DIR="${PROJECT_ROOT}/wandb"
WANDB_PROJECT="tactile-aekl-formal"
WANDB_RUN_GROUP="tactile-aekl-formal"
WANDB_NAME="tactile-aekl-full-stride2-lpips005-bs64"

if [[ ! -d "${TRAIN_DATA_DIR}" ]]; then
  printf 'Training directory does not exist: %s\n' "${TRAIN_DATA_DIR}" >&2
  printf '%s\n' 'Generate the dataset first with scripts/prepare_tactile_ae_dataset.sh.' >&2
  exit 1
fi
if [[ ! -f "${VALIDATION_IMAGES_FILE}" ]]; then
  printf 'Validation image list does not exist: %s\n' "${VALIDATION_IMAGES_FILE}" >&2
  printf '%s\n' 'Generate the dataset first with scripts/prepare_tactile_ae_dataset.sh.' >&2
  exit 1
fi
if [[ ! -f "${SUMMARY_FILE}" ]]; then
  printf 'Dataset summary does not exist: %s\n' "${SUMMARY_FILE}" >&2
  printf '%s\n' 'Generate the dataset first with scripts/prepare_tactile_ae_dataset.sh.' >&2
  exit 1
fi

if ! python - "${SUMMARY_FILE}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)

reference = summary.get("algorithm", {}).get("reference")
episode_count = summary.get("episode_count", 0)
expected_reference = "first frame of each episode from LeRobot episode metadata"
if reference != expected_reference or episode_count <= 0:
    raise SystemExit(
        f"Dataset was not generated with per-episode references: {summary_path}. "
        "Regenerate it before training."
    )
PY
then
  exit 1
fi

mapfile -t VALIDATION_IMAGES < "${VALIDATION_IMAGES_FILE}"
if [[ "${#VALIDATION_IMAGES[@]}" -eq 0 ]]; then
  printf 'Validation image list is empty: %s\n' "${VALIDATION_IMAGES_FILE}" >&2
  exit 1
fi

for validation_image in "${VALIDATION_IMAGES[@]}"; do
  if [[ ! -f "${validation_image}" ]]; then
    printf 'Validation image does not exist: %s\n' "${validation_image}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export WANDB_DIR
export WANDB_PROJECT
export WANDB_RUN_GROUP
export WANDB_NAME

exec accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${PROJECT_ROOT}/diffusers/examples/research_projects/autoencoderkl/train_autoencoderkl.py" \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --train_data_dir "${TRAIN_DATA_DIR}" \
  --image_column image \
  --validation_image "${VALIDATION_IMAGES[@]}" \
  --resolution "${RESOLUTION}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --disc_learning_rate "${DISC_LEARNING_RATE}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --disc_lr_scheduler "${DISC_LR_SCHEDULER}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS}" \
  --rec_loss "${REC_LOSS}" \
  --kl_scale "${KL_SCALE}" \
  --perceptual_scale "${PERCEPTUAL_SCALE}" \
  --disc_start "${DISC_START}" \
  --disc_factor "${DISC_FACTOR}" \
  --validation_steps "${VALIDATION_STEPS}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb \
  --tracker_project_name "${WANDB_PROJECT}" \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --allow_tf32 \
  --set_grads_to_none \
  --seed 42 \
  "$@"
