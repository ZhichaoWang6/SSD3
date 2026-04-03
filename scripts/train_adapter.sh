#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_DIR=/data/wangzhichao/projects/SSD/SSD3/datasets/training_data
OUTPUT_DIR=./adapter_checkpoints/
EXIT_LAYER=2

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_adapter.py \
    --basepath $MODEL_PATH \
    --datadir $DATA_DIR \
    --outdir $OUTPUT_DIR \
    --exit_layer $EXIT_LAYER \
    --num_adapter_layers 1 \
    --lr 1e-4 \
    --bs 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 20 \
    --num_warmup_steps 2000 \
    --max_len 2048 \
    --grad_clip 0.5 \
    --save_freq 1