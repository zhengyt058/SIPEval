#!/usr/bin/env bash
set -euo pipefail

unset https_proxy http_prpxy
source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)

QA_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/qa_for_generate"
SCRIPT_PATH="/mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/generate_api.py"
OUTPUT_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/generate"
WORKPLACE_PATH="/mnt/shared-storage-user/zhengyuting/SIP_exp"
IMAGE_ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp"
ACTION_LIST="T2T4, T2T4V3, T2T4V2"
MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"
API_BASE_URL="${API_BASE_URL:}"
API_KEY=""
RUN_TAG="${RUN_TAG:-}"
MAX_WORKERS="${MAX_WORKERS:-12}"

DATASET_LIST="$(python - <<'PY'
from pathlib import Path
root = Path("/mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/qa_for_generate")
names = []
for p in sorted(root.glob("*_original.json")):
    stem = p.stem
    if stem.endswith("_original"):
        names.append(stem[:-len("_original")])
print(",".join(names))
PY
)"

if [[ -z "${DATASET_LIST}" ]]; then
  echo "[ERROR] No *_original.json found under ${QA_ROOT}"
  exit 1
fi

python "${SCRIPT_PATH}" \
  --dataset "${DATASET_LIST}" \
  --action "${ACTION_LIST}" \
  --run_tag "${RUN_TAG}" \
  --model_path "${MODEL_NAME}" \
  --output_root "${OUTPUT_ROOT}" \
  --workplace_path "${WORKPLACE_PATH}" \
  --qa_root "${QA_ROOT}" \
  --image_root "${IMAGE_ROOT}" \
  --api_base_url "${API_BASE_URL}" \
  --api_key "${API_KEY}" \
  --max_workers "${MAX_WORKERS}"
