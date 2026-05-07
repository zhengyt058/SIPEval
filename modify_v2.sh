#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/shared-storage-user/zhengyuting/SIP_exp"
SCRIPT_DIR="${ROOT}/2_modify_image"

QA_ROOT="${ROOT}/original_data"
IMAGE_ROOT="${ROOT}/original_image"
OUTPUT_ROOT="${ROOT}/modified_image"
CLASSIFICATION="${ROOT}/classification_result.json"
TEMP_DIR="${SCRIPT_DIR}/temp_v2"

STEP1="${SCRIPT_DIR}/v2_1.py"
STEP2="${SCRIPT_DIR}/v2_2.py"
STEP3="${SCRIPT_DIR}/v2_3.py"

STEP1_WORKERS="${STEP1_WORKERS:-8}"
STEP2_WORKERS="${STEP2_WORKERS:-1}"
STEP2_GPU_IDS="${STEP2_GPU_IDS:-0,1,2,3}"
STEP3_WORKERS="${STEP3_WORKERS:-8}"
LIMIT="${LIMIT:-0}"
MAX_RETRY_TIMES="${MAX_RETRY_TIMES:-3}"

mkdir -p "${TEMP_DIR}" "${OUTPUT_ROOT}"

round=1
retry_input=""
while [[ "${round}" -le "${MAX_RETRY_TIMES}" ]]; do
  echo "================ Round ${round}: Step-1 ================"
  STEP1_OUT="${TEMP_DIR}/step1_records_round${round}.jsonl"
  STEP1_SUMMARY="${TEMP_DIR}/step1_summary_round${round}.json"
  STEP1_ARGS=(
    --qa-root "${QA_ROOT}"
    --image-root "${IMAGE_ROOT}"
    --classification "${CLASSIFICATION}"
    --temp-dir "${TEMP_DIR}"
    --workers "${STEP1_WORKERS}"
    --limit "${LIMIT}"
    --round-idx "${round}"
    --output-path "${STEP1_OUT}"
    --summary-path "${STEP1_SUMMARY}"
  )
  if [[ -n "${retry_input}" ]]; then
    STEP1_ARGS+=(--input-records "${retry_input}")
  fi
  python3 "${STEP1}" "${STEP1_ARGS[@]}"

  echo
  STEP2_OUT="${TEMP_DIR}/step2_records_round${round}.jsonl"
  STEP2_SUMMARY="${TEMP_DIR}/step2_summary_round${round}.json"
  STEP2_RUN_SH="${TEMP_DIR}/run_step2_round${round}.sh"
  cat > "${STEP2_RUN_SH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

python3 "${STEP2}" \\
  --temp-dir "${TEMP_DIR}" \\
  --image-root "${IMAGE_ROOT}" \\
  --output-root "${OUTPUT_ROOT}" \\
  --workers "${STEP2_WORKERS}" \\
  --gpu-ids "${STEP2_GPU_IDS}" \\
  --limit "${LIMIT}" \\
  --input-path "${STEP1_OUT}" \\
  --output-path "${STEP2_OUT}" \\
  --summary-path "${STEP2_SUMMARY}"
EOF
  chmod +x "${STEP2_RUN_SH}"

  echo "================ Round ${round}: Step-2 (GPU terminal) ================"
  echo "Please run this script in a GPU terminal:"
  echo "  ${STEP2_RUN_SH}"
  echo
  read -r -p "After Step-2 finishes in GPU terminal, press Enter here to continue Step-3..."
  if [[ ! -f "${STEP2_OUT}" ]]; then
    echo "ERROR: ${STEP2_OUT} not found. Step-2 may not have finished."
    exit 1
  fi

  echo
  echo "================ Round ${round}: Step-3 ================"
  STEP3_OUT="${TEMP_DIR}/step3_records_round${round}.jsonl"
  STEP3_SUMMARY="${TEMP_DIR}/step3_summary_round${round}.json"
  RETRY_OUT="${TEMP_DIR}/retry_records_round${round}.jsonl"
  FINAL_FAIL_OUT="${TEMP_DIR}/final_failed_records_round${round}.jsonl"
  python3 "${STEP3}" \
    --temp-dir "${TEMP_DIR}" \
    --workers "${STEP3_WORKERS}" \
    --limit "${LIMIT}" \
    --max-retry-times "${MAX_RETRY_TIMES}" \
    --input-path "${STEP2_OUT}" \
    --output-path "${STEP3_OUT}" \
    --summary-path "${STEP3_SUMMARY}" \
    --retry-output-path "${RETRY_OUT}" \
    --final-failed-output-path "${FINAL_FAIL_OUT}"

  if [[ ! -s "${RETRY_OUT}" ]]; then
    echo "No verify_fail samples left after round ${round}. Stop retry."
    break
  fi

  if [[ "${round}" -ge "${MAX_RETRY_TIMES}" ]]; then
    echo "Reached MAX_RETRY_TIMES=${MAX_RETRY_TIMES}."
    break
  fi

  retry_input="${RETRY_OUT}"
  round=$((round + 1))
  echo
done

echo
echo "Done."
echo "round outputs are under: ${TEMP_DIR}"
echo "example files:"
echo "  ${TEMP_DIR}/step1_records_round1.jsonl"
echo "  ${TEMP_DIR}/step2_records_round1.jsonl"
echo "  ${TEMP_DIR}/step3_records_round1.jsonl"
echo "  ${TEMP_DIR}/retry_records_round1.jsonl"
