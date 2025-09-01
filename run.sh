#!/bin/bash

# HD-PiSSA Training Script
export HF_ENDPOINT=https://hf-mirror.com

# Configuration
MODEL_PATH="./models/model_name"  # Path to your model
OUTPUT_PATH="./outputs/experiment_name"  # Output directory
DATA_PATH="your_dataset_name"  # Dataset name or path
TARGET_MODULES="q_proj o_proj k_proj v_proj gate_proj up_proj down_proj"  # Target modules for HD-PiSSA

# Run HD-PiSSA training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python hd_pissa.py >> output.log 2>&1 \
    --model_path "${MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --world_size 8 \
    --data_path "${DATA_PATH}" \
    --data_split "train" \
    --dataset_field "query response" \
    --target_modules "${TARGET_MODULES}" \
    --ranks_per_gpu 16 \
    --batch_size 2 \
    --accumulation_steps 64 \
    --num_epochs 1 \
    --max_length 512 \
    --alpha 16 \
    --lr 2e-5 \
    --schedule "cosine" \
    --warmup_ratio 0.03 \
    --dropout 0.0