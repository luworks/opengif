#!/bin/bash
# 训练完成后自动评估脚本
set -e

LORA_PATH="$1"
NAME="$2"

if [ -z "$LORA_PATH" ] || [ -z "$NAME" ]; then
    echo "Usage: ./post_training_eval.sh <lora_path> <name>"
    exit 1
fi

echo "============================================"
echo "  Post-Training Evaluation: $NAME"
echo "============================================"

# 1. 推理步数消融
echo ""
echo "=== 推理步数消融 (4/8/15/30) ==="
.venv/bin/python ablation_steps.py \
    --lora-path "$LORA_PATH" \
    --output-dir "output/eval_${NAME}/steps"

# 2. 生成 10 个标准测试样本
echo ""
echo "=== 标准测试样本 ==="
PROMPTS=(
    "A cute cartoon cat waving its paw, bright colors"
    "A golden retriever puppy running through autumn leaves"
    "Two astronauts floating in zero gravity inside a space station"
    "A butterfly flapping its wings against a blue sky"
    "A chef tossing pizza dough in a kitchen"
    "A person dancing hip hop in a studio, dynamic motion"
    "A dragon breathing fire while flying over a castle"
    "A surfer riding a massive wave, golden hour"
    "一只可爱的熊猫在竹林里吃竹子"
    "烟花在夜空中绽放，五彩缤纷"
)

mkdir -p "output/eval_${NAME}/samples"
for i in "${!PROMPTS[@]}"; do
    .venv/bin/python generate_grid_gif.py "${PROMPTS[$i]}" \
        --lora-path "$LORA_PATH" \
        --output "output/eval_${NAME}/samples/sample_${i}.gif" \
        --save-grid --steps 30 --seed 42 \
        2>&1 | grep Generation
done

echo ""
echo "============================================"
echo "  评估完成: output/eval_${NAME}/"
echo "============================================"
