"""
AI Judge 评估脚本：生成网格图后自动让 Claude 看图打分。

输出结构化评分供自动化流程使用。
人工调用：python judge_grid.py output/some_grid.png "the prompt used"
"""

import argparse
import json
from pathlib import Path


# 评分维度和标准
EVAL_DIMENSIONS = {
    "grid_structure": "网格结构完整性 (1-10): 是否有清晰的 8 帧 2×4 排列？帧边界清楚吗？",
    "frame_consistency": "帧间一致性 (1-10): 同一角色/物体在 8 帧中是否保持一致？有无 identity drift？",
    "motion_quality": "运动质量 (1-10): 帧间是否有明显的、连贯的运动变化？还是几乎静态？",
    "prompt_adherence": "Prompt 跟随度 (1-10): 生成内容是否匹配文字描述？",
    "visual_quality": "视觉质量 (1-10): 清晰度、色彩、构图、细节是否好？",
    "loop_potential": "循环潜力 (1-10): 首帧和末帧是否相似，能形成无缝循环？",
}


def print_eval_template(image_path, prompt):
    """打印评估模板，供 Claude 看图后填写。"""
    print(f"\n{'='*60}")
    print(f"AI JUDGE 评估")
    print(f"{'='*60}")
    print(f"图片: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"\n请对以下维度打分 (1-10):\n")
    for dim, desc in EVAL_DIMENSIONS.items():
        print(f"  {dim}: {desc}")
    print(f"\n总分 = 各维度平均值")
    print(f"{'='*60}")


def save_eval(image_path, prompt, scores, output_dir="output/evaluations"):
    """保存评估结果。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    result = {
        "image": str(image_path),
        "prompt": prompt,
        "scores": scores,
        "avg_score": sum(scores.values()) / len(scores),
    }

    out_path = out_dir / f"{stem}_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"评估结果保存到: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="网格图路径")
    parser.add_argument("prompt", help="生成时使用的 prompt")
    args = parser.parse_args()
    print_eval_template(args.image, args.prompt)
