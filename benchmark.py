import os
"""
OpenGIF Benchmark: 自动评估 Text-to-GIF 模型质量。

50 个标准测试 prompt，自动生成 + 评估 + 打分。
每次训练后跑一次，追踪质量变化。

Usage:
  python benchmark.py --lora-path output/grid-lora-v2/final --name "v2-10k"
  python benchmark.py --compare v1 v2  # 对比两个版本
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ─── 50 个标准测试 prompt ────────────────────────────────────────────────────

BENCHMARK_PROMPTS = {
    # 简单 (10个)
    "simple": [
        "A red ball bouncing on a white background",
        "A spinning colorful pinwheel toy",
        "A flickering candle flame in the dark",
        "Ocean waves gently rolling onto a sandy beach",
        "A smiling sun with rays moving outward",
        "Rain drops falling on a window",
        "A clock pendulum swinging back and forth",
        "Colorful confetti falling from the sky",
        "A blinking neon sign that says OPEN",
        "Bubbles floating upward and popping",
    ],
    # 中等 (15个)
    "medium": [
        "A cute cartoon cat waving its paw, bright colors",
        "A golden retriever puppy running through autumn leaves",
        "A butterfly flapping its wings against a blue sky",
        "A chef tossing pizza dough in a kitchen",
        "A ballerina doing a graceful pirouette on stage",
        "A rocket launching into space with flames and smoke",
        "A fish swimming in a coral reef, underwater scene",
        "A snowman melting slowly in the spring sun",
        "A hummingbird hovering near a red flower",
        "A DJ mixing music at a turntable with colorful lights",
        "A paper airplane flying through an office",
        "A campfire crackling with sparks flying up",
        "A cat chasing a laser dot on the floor",
        "A sunrise timelapse over a mountain range",
        "Wind blowing through a field of sunflowers",
    ],
    # 复杂 (15个)
    "complex": [
        "Two astronauts floating in zero gravity inside a space station, one passing a glowing orb to the other, sci-fi blue lighting",
        "A person dancing hip hop in a studio, dynamic motion, multiple poses, dramatic shadows",
        "A wizard casting a spell with glowing runes swirling around their hands, dark fantasy style",
        "A time-lapse of a flower blooming from bud to full bloom, macro photography, soft natural light",
        "A samurai drawing a katana in slow motion, cherry blossom petals floating in the background",
        "A robot and a human shaking hands in a futuristic laboratory, warm and cool lighting contrast",
        "A street performer juggling fire torches at night, crowd watching in the background",
        "A dragon breathing fire while flying over a medieval castle, epic fantasy scene",
        "An ice skater performing a triple axel jump, sparkling ice particles, arena lighting",
        "A sushi chef preparing nigiri with precise hand movements, close-up shot, warm kitchen lighting",
        "A surfer riding a massive wave, aerial drone perspective, golden hour lighting",
        "A pianist's hands playing a complex piece, close-up of fingers on ivory keys, concert hall",
        "A phoenix rising from ashes with flames transforming into feathers, mythical scene",
        "A barista doing latte art, pouring milk into coffee, steam rising, cozy cafe atmosphere",
        "An astronaut planting a flag on Mars surface, red dusty landscape, Earth visible in the sky",
    ],
    # 中文 (10个)
    "chinese": [
        "一只可爱的熊猫在竹林里吃竹子",
        "一条金龙在云间飞舞，中国风",
        "樱花飘落在日式庭院中",
        "一个小女孩在放风筝，蓝天白云",
        "功夫大师在山顶练太极拳",
        "一碗热气腾腾的拉面，筷子夹起面条",
        "夜晚的东京街头霓虹灯闪烁",
        "一只蝴蝶从花丛中飞起",
        "海浪拍打灯塔的岩石",
        "烟花在夜空中绽放，五彩缤纷",
    ],
}


def run_benchmark(lora_path, name, output_dir="output/benchmarks", steps=30):
    """运行完整 benchmark，生成所有测试图 + 评估。"""
    from generate_grid_gif import generate_grid
    from eval_gif import evaluate_gif, load_gif_frames
    from prepare_grid_data import grid_to_gif
    from diffusers import WanPipeline
    from peft import PeftModel
    import torch

    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print(f"Loading pipeline...")
    model_path = os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    if lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        print(f"LoRA loaded from {lora_path}")

    results = {"name": name, "lora_path": str(lora_path), "prompts": {}}
    total_time = 0
    total_count = 0

    for category, prompts in BENCHMARK_PROMPTS.items():
        print(f"\n=== {category} ({len(prompts)} prompts) ===")
        cat_results = []

        for i, prompt in enumerate(prompts):
            prompt_id = f"{category}_{i:02d}"
            grid_path = out_dir / f"{prompt_id}_grid.png"
            gif_path = out_dir / f"{prompt_id}.gif"

            # Generate
            t0 = time.time()
            grid_image = generate_grid(pipe, prompt, num_inference_steps=steps, seed=42)
            gen_time = time.time() - t0
            total_time += gen_time
            total_count += 1

            # Convert numpy to PIL if needed
            if isinstance(grid_image, np.ndarray):
                grid_image = grid_image.squeeze()
                if grid_image.dtype in (np.float32, np.float64):
                    grid_image = (np.clip(grid_image, 0, 1) * 255).astype(np.uint8)
                grid_image = Image.fromarray(grid_image)

            grid_image.save(grid_path)
            grid_to_gif(grid_image, output_path=str(gif_path))

            # Evaluate
            try:
                eval_result = evaluate_gif(str(gif_path))
            except Exception as e:
                eval_result = {"error": str(e)}

            result = {
                "prompt": prompt,
                "gen_time": round(gen_time, 2),
                **eval_result,
            }
            cat_results.append(result)
            print(f"  [{prompt_id}] {gen_time:.1f}s | loop={eval_result.get('loop_similarity', 'N/A'):.3f} | "
                  f"motion={eval_result.get('motion_magnitude', 'N/A'):.1f}")

        results["prompts"][category] = cat_results

    # Aggregate
    all_results = [r for cat in results["prompts"].values() for r in cat]
    loops = [r["loop_similarity"] for r in all_results if "loop_similarity" in r]
    temps = [r["temporal_consistency"] for r in all_results if "temporal_consistency" in r]
    motions = [r["motion_magnitude"] for r in all_results if "motion_magnitude" in r]

    results["summary"] = {
        "total_prompts": total_count,
        "avg_gen_time": round(total_time / total_count, 2),
        "avg_loop_quality": round(np.mean(loops), 4) if loops else None,
        "avg_temporal_consistency": round(np.mean(temps), 4) if temps else None,
        "avg_motion_magnitude": round(np.mean(motions), 1) if motions else None,
        "looping_rate": round(sum(1 for l in loops if l > 0.85) / len(loops), 3) if loops else None,
    }

    # Save
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS: {name}")
    print(f"{'='*50}")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")
    print(f"  Results saved to: {out_dir}/benchmark_results.json")

    return results


def compare_benchmarks(names, benchmark_dir="output/benchmarks"):
    """对比多个 benchmark 结果。"""
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPARISON")
    print(f"{'='*60}")

    header = f"{'Metric':<30}"
    for name in names:
        header += f"| {name:<15}"
    print(header)
    print("-" * len(header))

    all_results = {}
    for name in names:
        result_path = Path(benchmark_dir) / name / "benchmark_results.json"
        if result_path.exists():
            with open(result_path) as f:
                all_results[name] = json.load(f)

    metrics = ["avg_gen_time", "avg_loop_quality", "avg_temporal_consistency",
               "avg_motion_magnitude", "looping_rate"]

    for metric in metrics:
        row = f"{metric:<30}"
        for name in names:
            val = all_results.get(name, {}).get("summary", {}).get(metric, "N/A")
            row += f"| {str(val):<15}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="OpenGIF Benchmark")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--compare", nargs="+", help="Compare benchmark results")
    args = parser.parse_args()

    if args.compare:
        compare_benchmarks(args.compare)
    else:
        run_benchmark(args.lora_path, args.name, steps=args.steps)


if __name__ == "__main__":
    main()
