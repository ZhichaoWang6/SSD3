#!/bin/bash
# Inference with Kangaroo speculative decoding
# Usage: bash scripts/inference_speculative.sh

exp_name=mmduet2_speculative
dataset=ego
output_dir=outputs/${exp_name}/${dataset}
mkdir -p $output_dir

# Set these paths
MMDUET2_CKPT=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
ADAPTER_PATH=./adapter_checkpoints/best/best_accept  # path to trained adapter

cd "$(dirname "$0")/.."

python -u inference.py \
    --llm_pretrained $MMDUET2_CKPT \
    --test_fname ./data/annotations/${dataset}-frame_input_format.json \
    --output_fname ${output_dir}/pred.jsonl \
    --use_speculative_decoding true \
    --compare_with_baseline true \
    --adapter_path $ADAPTER_PATH \
    --exit_layer 2 \
    --speculative_threshold 0.6 \
    --speculative_steps 4 \
    > ${output_dir}/pred.log 2>&1