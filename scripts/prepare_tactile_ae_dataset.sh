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
VIDEO_CHUNK="000"
VIDEO_FILE="138"
# Set both values to "all" to process every video in VIDEO_KEY.

EPISODES_DIR="${DATASET_ROOT}/meta/episodes"

if [[ "${VIDEO_CHUNK}" == "all" && "${VIDEO_FILE}" == "all" ]]; then
  SOURCE="${DATASET_ROOT}/videos/${VIDEO_KEY}"
  OUTPUT_DIR="${PROJECT_ROOT}/data/tactile_ae_dataset_full"
  VALIDATION_RATIO="0.1"
  VALIDATION_UNIT="video"
elif [[ "${VIDEO_CHUNK}" != "all" && "${VIDEO_FILE}" != "all" ]]; then
  SOURCE="${DATASET_ROOT}/videos/${VIDEO_KEY}/chunk-${VIDEO_CHUNK}/file-${VIDEO_FILE}.mp4"
  OUTPUT_DIR="${PROJECT_ROOT}/data/tactile_ae_dataset_file${VIDEO_FILE}"
  VALIDATION_RATIO="0.1"
  VALIDATION_UNIT="episode"
else
  printf '%s\n' 'VIDEO_CHUNK and VIDEO_FILE must both be all or both identify one file' >&2
  exit 1
fi

THRESHOLD="0.015"
FRAME_STRIDE="2"
MIN_DEFORMED_GAP="15"
NEUTRAL_RATIO="0.05"
OUTPUT_RESOLUTION="224"
RESIZE_MODE="letterbox"
IMAGE_FORMAT="jpeg"
JPEG_QUALITY="95"
VALIDATION_IMAGE_COUNT="0"

exec python "${PROJECT_ROOT}/data/utils/extract_tactile_video_frames.py" \
  --source "${SOURCE}" \
  --episodes-dir "${EPISODES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --threshold "${THRESHOLD}" \
  --frame-stride "${FRAME_STRIDE}" \
  --min-deformed-gap "${MIN_DEFORMED_GAP}" \
  --neutral-ratio "${NEUTRAL_RATIO}" \
  --validation-ratio "${VALIDATION_RATIO}" \
  --validation-unit "${VALIDATION_UNIT}" \
  --output-resolution "${OUTPUT_RESOLUTION}" \
  --resize-mode "${RESIZE_MODE}" \
  --image-format "${IMAGE_FORMAT}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --overwrite \
  "$@"
