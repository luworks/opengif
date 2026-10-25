import os
"""
完整评估 Pipeline: 静态指标 + LLM Judge (Qwen3.5) 三合一。

1. 静态指标: 循环质量、帧间一致性、运动幅度 (eval_gif.py)
2. LLM Judge: Qwen3.5-2B via vLLM 看图打分 (6个维度)
3. 综合报告: JSON + 人类可读摘要

Usage:
  python eval_pipeline.py --lora-path output/grid-v3-r128-a03/final --name "r128-a03-50k"
"""

import argparse
import json
import time
import base64
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import aiohttp
import asyncio

from eval_gif import evaluate_gif
from prepare_grid_data import grid_to_gif
from generate_grid_gif import generate_grid


# 标准测试 prompts
TEST_PROMPTS = [
    ("simple_ball", "A red ball bouncing on a white background"),
    ("simple_candle", "A flickering candle flame in the dark"),
    ("medium_cat", "A cute cartoon cat waving its paw, bright colors"),
    ("medium_dog", "A golden retriever puppy running through autumn leaves"),
    ("medium_butterfly", "A butterfly flapping its wings against a blue sky"),
    ("complex_astronaut", "Two astronauts floating in zero gravity, sci-fi blue lighting"),
    ("complex_chef", "A chef tossing pizza dough, flour dust swirling, warm kitchen"),
    ("complex_dragon", "A dragon breathing fire while flying over a medieval castle"),
    ("chinese_panda", "一只可爱的熊猫在竹林里吃竹子"),
    ("chinese_firework", "烟花在夜空中绽放，五彩缤纷"),
]

# LLM Judge prompt
JUDGE_PROMPT = """You are evaluating a generated GIF grid image. The image shows 8 animation frames arranged in a 2×4 grid (2 rows, 4 columns), reading left-to-right, top-to-bottom.

The text prompt used to generate this image was: "{prompt}"

Score each dimension from 1-10:
1. grid_structure: Are there 8 clear, distinct frames in a 2×4 grid?
2. frame_consistency: Is the same character/object consistent across frames?
3. motion_quality: Is there visible, coherent motion between frames?
4. prompt_adherence: Does the content match the text description?
5. visual_quality: Sharpness, color, composition, detail quality?
6. loop_potential: Could first and last frames form a seamless loop?

Respond ONLY with JSON, no other text:
{{"grid_structure": N, "frame_consistency": N, "motion_quality": N, "prompt_adherence": N, "visual_quality": N, "loop_potential": N}}"""


async def llm_judge_score(image_path, prompt, vllm_url="http://localhost:30000/v1/chat/completions"):
    """用 Qwen3.5-2B via vLLM 对网格图打分。"""
    # 读取图片为 base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": "Qwen/Qwen3.5-2B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": JUDGE_PROMPT.format(prompt=prompt)},
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(vllm_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                result = await resp.json()
                content = result["choices"][0]["message"]["content"]
                # 提取 JSON
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                scores = json.loads(content)
                return scores
    except Exception as e:
        print(f"  LLM Judge error: {e}")
        return None


def run_eval(lora_path, name, output_dir="output/eval", steps=30, use_llm=False):
    """运行完整评估 pipeline。"""
    from diffusers import WanPipeline
    from peft import PeftModel

    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pipeline...")
    model_path = os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    if lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)

    all_results = []

    for prompt_id, prompt in TEST_PROMPTS:
        print(f"\n--- {prompt_id}: {prompt[:50]}... ---")

        # 1. 生成
        t0 = time.time()
        grid = generate_grid(pipe, prompt, num_inference_steps=steps, seed=42)
        gen_time = time.time() - t0

        if isinstance(grid, np.ndarray):
            grid = grid.squeeze()
            if grid.dtype in (np.float32, np.float64):
                grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
            grid = Image.fromarray(grid)

        grid_path = out_dir / f"{prompt_id}_grid.png"
        gif_path = out_dir / f"{prompt_id}.gif"
        grid.save(grid_path)
        grid_to_gif(grid, output_path=str(gif_path))

        # 2. 静态指标
        static_metrics = evaluate_gif(str(gif_path))

        # 3. LLM Judge (如果 vLLM 可用)
        llm_scores = None
        if use_llm:
            llm_scores = asyncio.run(llm_judge_score(str(grid_path), prompt))

        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "gen_time": round(gen_time, 2),
            "static": {
                "loop_similarity": static_metrics.get("loop_similarity"),
                "temporal_consistency": static_metrics.get("temporal_consistency"),
                "motion_magnitude": static_metrics.get("motion_magnitude"),
            },
            "llm_judge": llm_scores,
        }
        all_results.append(result)

        loop = static_metrics.get("loop_similarity", 0)
        motion = static_metrics.get("motion_magnitude", 0)
        print(f"  {gen_time:.1f}s | loop={loop:.3f} | motion={motion:.1f}", end="")
        if llm_scores:
            avg = sum(llm_scores.values()) / len(llm_scores)
            print(f" | LLM avg={avg:.1f}", end="")
        print()

    # 汇总
    summary = {
        "name": name,
        "lora_path": str(lora_path),
        "num_prompts": len(all_results),
        "avg_gen_time": round(np.mean([r["gen_time"] for r in all_results]), 2),
        "avg_loop": round(np.mean([r["static"]["loop_similarity"] for r in all_results]), 4),
        "avg_motion": round(np.mean([r["static"]["motion_magnitude"] for r in all_results]), 1),
    }

    if any(r["llm_judge"] for r in all_results):
        llm_avgs = []
        for r in all_results:
            if r["llm_judge"]:
                llm_avgs.append(sum(r["llm_judge"].values()) / len(r["llm_judge"]))
        summary["avg_llm_score"] = round(np.mean(llm_avgs), 1)

    report = {"summary": summary, "details": all_results}
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT: {name}")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Report: {out_dir}/eval_report.json")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--use-llm", action="store_true", help="Use Qwen3.5 LLM judge (requires vLLM)")
    args = parser.parse_args()
    run_eval(args.lora_path, args.name, steps=args.steps, use_llm=args.use_llm)
