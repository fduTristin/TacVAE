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

# Edit these values for another video or another dataset split.
DATASET_ROOT="/home/xhz/OpenNeoData/aloha"
VIDEO_KEY="observation.images.left_wrist_left_tactile"
VIDEO_CHUNK="all"
VIDEO_FILE="all"
# Set both values to "all" to process every video in VIDEO_KEY.

EPISODES_DIR="${DATASET_ROOT}/meta/episodes"

if [[ "${VIDEO_CHUNK}" == "all" && "${VIDEO_FILE}" == "all" ]]; then
  SOURCE="${DATASET_ROOT}/videos/${VIDEO_KEY}"
  OUTPUT_DIR="${PROJECT_ROOT}/data/tactile_ae_dataset_256x144zhiying"
  VALIDATION_RATIO="0.1"
  VALIDATION_UNIT="episode"
elif [[ "${VIDEO_CHUNK}" != "all" && "${VIDEO_FILE}" != "all" ]]; then
  SOURCE="${DATASET_ROOT}/videos/${VIDEO_KEY}/chunk-${VIDEO_CHUNK}/file-${VIDEO_FILE}.mp4"
  OUTPUT_DIR="${PROJECT_ROOT}/data/tactile_ae_dataset_file${VIDEO_FILE}_256x144"
  VALIDATION_RATIO="0.1"
  VALIDATION_UNIT="episode"
else
  printf '%s\n' 'VIDEO_CHUNK and VIDEO_FILE must both be all or both identify one file' >&2
  exit 1
fi

THRESHOLD="0.01"
FRAME_STRIDE="2"
MIN_DEFORMED_GAP="1"
NEUTRAL_RATIO="0.05"
OUTPUT_WIDTH="256"
OUTPUT_HEIGHT="144"
RESIZE_MODE="letterbox"
IMAGE_FORMAT="jpeg"
JPEG_QUALITY="95"
VALIDATION_NEUTRAL_COUNT="5"
VALIDATION_DEFORMED_COUNT="15"

# The extraction manifest contains all selected frames. This second, event-aware
# index is used by the tactile autoencoder training script to reduce redundancy.
EVENT_INDEX_FILE="${OUTPUT_DIR}/train_event_indexzhiying.txt"
EVENT_INDEX_SUMMARY="${OUTPUT_DIR}/train_event_indexzhiying_summary.json"
EVENT_MIN_DEFORMED_GAP="15"
EVENT_RUN_GAP="2"
EVENT_NEUTRAL_RATIO="0.10"

ARGS=("$@")
EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR}"
OUTPUT_ARGS=(--output-size "${OUTPUT_WIDTH}" "${OUTPUT_HEIGHT}")
DRY_RUN=0
SHOW_HELP=0
for ((arg_index = 0; arg_index < ${#ARGS[@]}; arg_index++)); do
  case "${ARGS[arg_index]}" in
    --output-dir)
      arg_index=$((arg_index + 1))
      if ((arg_index >= ${#ARGS[@]})); then
        printf '%s\n' '--output-dir requires a value' >&2
        exit 1
      fi
      EFFECTIVE_OUTPUT_DIR="${ARGS[arg_index]}"
      ;;
    --output-dir=*)
      EFFECTIVE_OUTPUT_DIR="${ARGS[arg_index]#*=}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --output-resolution|--output-size)
      OUTPUT_ARGS=()
      ;;
    --output-resolution=*|--output-size=*)
      OUTPUT_ARGS=()
      ;;
    -h|--help)
      SHOW_HELP=1
      ;;
  esac
done

if ((DRY_RUN == 0 && SHOW_HELP == 0)); then
  EVENT_INDEX_FILE="${EFFECTIVE_OUTPUT_DIR}/train_event_indexzhiying.txt"
  EVENT_INDEX_SUMMARY="${EFFECTIVE_OUTPUT_DIR}/train_event_indexzhiying_summary.json"
fi

python "${PROJECT_ROOT}/data/utils/extract_tactile_video_frames.py" \
  --source "${SOURCE}" \
  --episodes-dir "${EPISODES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --threshold "${THRESHOLD}" \
  --frame-stride "${FRAME_STRIDE}" \
  --min-deformed-gap "${MIN_DEFORMED_GAP}" \
  --neutral-ratio "${NEUTRAL_RATIO}" \
  --validation-ratio "${VALIDATION_RATIO}" \
  --validation-unit "${VALIDATION_UNIT}" \
  "${OUTPUT_ARGS[@]}" \
  --resize-mode "${RESIZE_MODE}" \
  --image-format "${IMAGE_FORMAT}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --validation-neutral-count "${VALIDATION_NEUTRAL_COUNT}" \
  --validation-deformed-count "${VALIDATION_DEFORMED_COUNT}" \
  --overwrite \
  "$@"

if ((DRY_RUN == 0 && SHOW_HELP == 0)); then
  python "${PROJECT_ROOT}/scripts/build_tactile_index.py" \
    --manifest "${EFFECTIVE_OUTPUT_DIR}/manifest.jsonl" \
    --output "${EVENT_INDEX_FILE}" \
    --summary "${EVENT_INDEX_SUMMARY}" \
    --min-deformed-gap "${EVENT_MIN_DEFORMED_GAP}" \
    --event-run-gap "${EVENT_RUN_GAP}" \
    --neutral-ratio "${EVENT_NEUTRAL_RATIO}"
fi
