#!/bin/bash
# 消融实验矩阵 — 当前 50K 训练完成后执行
# 每组 5K 步，10K 数据，快速对比
set -e

MODEL_PATH="${OPENGIF_MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
CACHE_V1="dataset/gif-grids/cache"      # 128×128 网格
CACHE_V2="dataset/gif-grids-v2/cache"   # 96×176 网格

echo "============================================"
echo "  消融实验矩阵"
echo "============================================"

# --- Exp A: 宽高比对比 (128×128 vs 96×176) ---
echo ""
echo "=== Exp A: 宽高比 128×128 vs 96×176 ==="

# A1: 128×128 (baseline, 已有 V3 结果)
echo "A1: 128×128 已有结果 (output/grid-v3-flow/final)"

# A2: 96×176 (新宽高比)
echo "A2: 96×176 训练中..."
.venv/bin/python train_grid_v3.py \
    --cache-dir "$CACHE_V2" \
    --max-steps 5000 \
    --lora-rank 32 \
    --output-dir output/ablation/aspect_96x176 \
    2>&1 | tee output/ablation/aspect_96x176.log

# --- Exp B: LoRA rank 对比 ---
echo ""
echo "=== Exp B: LoRA rank 32 vs 128 ==="

# B1: rank 32 (baseline, 已有结果)
echo "B1: rank 32 已有结果"

# B2: rank 128
echo "B2: rank 128 训练中..."
.venv/bin/python train_grid_v3.py \
    --cache-dir "$CACHE_V1" \
    --max-steps 5000 \
    --lora-rank 128 \
    --output-dir output/ablation/rank128 \
    2>&1 | tee output/ablation/rank128.log

# --- Exp C: Flow loss weight ---
echo ""
echo "=== Exp C: Flow loss α = 0.1 vs 0.3 vs 0.5 ==="

for alpha in 0.1 0.3; do
    echo "C: α=$alpha 训练中..."
    .venv/bin/python train_grid_v3.py \
        --cache-dir "$CACHE_V1" \
        --max-steps 5000 \
        --flow-weight $alpha \
        --output-dir "output/ablation/alpha_${alpha}" \
        2>&1 | tee "output/ablation/alpha_${alpha}.log"
done
echo "C: α=0.5 已有结果 (output/grid-v3-flow/final)"

# --- Exp D: 推理步数 (不需重训) ---
echo ""
echo "=== Exp D: 推理步数 4/8/15/30 ==="
.venv/bin/python ablation_steps.py \
    --lora-path output/grid-v3-100k/final \
    --output-dir output/ablation/steps

# --- 生成对比报告 ---
echo ""
echo "=== 生成对比样本 ==="
for dir in output/ablation/*/; do
    if [ -f "$dir/final/adapter_model.safetensors" ] || [ -f "$dir/final/adapter_config.json" ]; then
        name=$(basename $dir)
        echo "生成 $name 测试图..."
        .venv/bin/python generate_grid_gif.py \
            "A golden retriever puppy running through autumn leaves" \
            --lora-path "$dir/final" \
            --output "output/ablation/${name}_dog.gif" \
            --save-grid --steps 30 --seed 88 2>/dev/null
        .venv/bin/python generate_grid_gif.py \
            "Two astronauts floating in zero gravity inside a space station" \
            --lora-path "$dir/final" \
            --output "output/ablation/${name}_astro.gif" \
            --save-grid --steps 30 --seed 55 2>/dev/null
    fi
done

echo ""
echo "============================================"
echo "  消融实验完成！结果在 output/ablation/"
echo "============================================"
