#!/bin/bash
# Step 1: Generate training data for the Kangaroo adapter
# This collects hidden states from the full model on MMDuet2 multimodal data.

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_PATH=./data/annotations/ego-frame_input_format.json
OUTPUT_DIR=./datasets/training_data/
EXIT_LAYERS=2  # Comma-separated list of exit layers to save hidden states for

cd "$(dirname "$0")/.."

python generate_training_data.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --exit_layers $EXIT_LAYERS \
    --max_seq_len 4096
