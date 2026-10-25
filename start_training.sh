#!/bin/bash
# Wait for captioning to finish, stop vLLM, start training pipeline
set -e

MODEL_PATH="${OPENGIF_MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

echo "=== Step 1: Stop captioning and vLLM ==="
# Kill captioning process
pkill -f "batch_caption_vllm.py" 2>/dev/null || true
sleep 2

# Stop vLLM Docker to free GPU
docker stop vllm-qwen35 2>/dev/null || true
echo "vLLM stopped. Waiting for GPU memory to free..."
sleep 5

# Verify GPU is free
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo ""

echo "=== Step 2: Pre-compute T5 embeddings (GPU) ==="
.venv/bin/python train_gif_lora.py --step prepare --model-path "$MODEL_PATH" 2>&1 | tee train_prepare.log

echo ""
echo "=== Step 3: Start LoRA training ==="
.venv/bin/python train_gif_lora.py --step train --model-path "$MODEL_PATH" 2>&1 | tee train_lora.log
