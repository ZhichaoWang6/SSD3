#!/bin/bash
# Standard inference (baseline, no speculative decoding)

exp_name=mmduet2
dataset=ego
output_dir=outputs/${exp_name}/${dataset}
mkdir -p $output_dir

MMDUET2_CKPT=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt

cd "$(dirname "$0")/.."

python -u inference.py \
    --llm_pretrained $MMDUET2_CKPT \
    --test_fname ./data/annotations/${dataset}-frame_input_format.json \
    --output_fname ${output_dir}/pred.jsonl \
    > ${output_dir}/pred.log 2>&1
