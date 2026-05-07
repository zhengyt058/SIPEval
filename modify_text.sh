#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/mnt/shared-storage-user/zhengyuting/SIP_exp/2_modify_text"
INPUT_PATH="/mnt/shared-storage-user/zhengyuting/SIP_exp/original_data"
OUTPUT_PATH="/mnt/shared-storage-user/zhengyuting/SIP_exp/modified_data"
MODE="${1:-t2}"          # t1 / t2 / t3 / t4
MODEL=gpt-5.2

if [[ "${MODE}" == "t1" ]]; then
  PY_SCRIPT="${SCRIPT_DIR}/t1.py"
  ACTION="T1"
elif [[ "${MODE}" == "t2" ]]; then
  PY_SCRIPT="${SCRIPT_DIR}/t2.py"
  ACTION="T2"
elif [[ "${MODE}" == "t3" ]]; then
  PY_SCRIPT="${SCRIPT_DIR}/t3.py"
  ACTION="T3"
elif [[ "${MODE}" == "t4" ]]; then
  PY_SCRIPT="${SCRIPT_DIR}/t4.py"
  ACTION="T4"
else
  echo "Unsupported mode: ${MODE}. Use t1 / t2 / t3 / t4."
  exit 1
fi

python "${PY_SCRIPT}" \
  --dataset "MMMU" \
  --subject "math","physics","chemistry" \
  --action "${ACTION}" \
  --model "${MODEL}" \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_workers "10"
