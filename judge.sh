#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <model1> [model2 ...]"
    echo "Example: bash $0 gpt-5.4 glm-4.5v qwen2.5-vl-72b"
    exit 1
fi

MODELS=("$@")

# export CUDA_VISIBLE_DEVICES=0
# export AUTO_SPLIT=1
export LMUData=/mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/custom_data
export RESULTS_DIR=/mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/results
# export USE_VLLM=1
export VLLM_WORKER_MULTIPROC_METHOD=fork

unset https_proxy http_prpxy
source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)

python /mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/form_custom_data.py \
    --qa_root /mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/qa_for_generate \
    --output_root /mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/custom_data \
    --image_root /mnt/shared-storage-user/zhengyuting/SIP_exp

python /mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/form_judge_results.py \
    --generate_root /mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/generate \
    --results_root /mnt/shared-storage-user/zhengyuting/SIP_exp/3_inference/results

data_list=()
for file in "$LMUData"/*.tsv; do
    data=$(basename "$file" .tsv)
    data_list+=("$data")
done

echo "Processing data: ${data_list[@]}"
cd /mnt/shared-storage-user/zhengyuting/SIP/MDK12/MDK12EvalHub
for MODEL in "${MODELS[@]}"; do
    echo "Running judge for model: $MODEL"
    python run.py --data "${data_list[@]}" --model "$MODEL" --verbose --reuse --work-dir "$RESULTS_DIR" --judge gpt-4o-mini
done
cd /mnt/shared-storage-user/zhengyuting/SIP_exp


# old batch2: grok-4-fast-non-reasoning InternVL3_5-8B InternVL3_5-14B InternVL3_5-38B Kimi-VL-A3B-Instruct qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b