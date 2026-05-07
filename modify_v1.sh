#!/usr/bin/env bash
set -euo pipefail

PY_SCRIPT="/mnt/shared-storage-user/zhengyuting/SIP_exp/2_modify_image/v1.py"

QA_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp/original_data"
IMAGE_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp/original_image"
OUTPUT_IMAGE_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp/modified_image"
CLASSIFICATION_PATH="/mnt/shared-storage-user/zhengyuting/SIP_exp/classification_result.json"
TEMP_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp/2_modify_image/temp_v1"
CLASS_LABEL="Chart"
MODEL="gpt-5.2"
MAX_WORKERS="8"

python "${PY_SCRIPT}" \
  --batch \
  --qa-root "${QA_ROOT}" \
  --image-root "${IMAGE_ROOT}" \
  --output-image-root "${OUTPUT_IMAGE_ROOT}" \
  --classification-path "${CLASSIFICATION_PATH}" \
  --temp-root "${TEMP_ROOT}" \
  --class-label "${CLASS_LABEL}" \
  --model "${MODEL}" \
  --max-workers "${MAX_WORKERS}"
