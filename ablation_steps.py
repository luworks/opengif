import os
"""
消融实验: 推理步数对质量的影响 (不需要重训)。

在同一个训练好的模型上，测试 4/8/15/30 步的生成质量和速度。

Usage:
  python ablation_steps.py --lora-path output/grid-v3-flow/final
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import WanPipeline
from peft import PeftModel
from generate_grid_gif import generate_grid
from prepare_grid_data import grid_to_gif
from eval_gif import evaluate_gif

PROMPTS = [
    "A cute cartoon cat waving its paw, bright colors",
    "A golden retriever puppy running through autumn leaves",
    "Two astronauts floating in zero gravity inside a space station",
    "A butterfly flapping its wings against a blue sky",
    "A chef tossing pizza dough in a kitchen",
]

STEP_COUNTS = [4, 8, 15, 30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-dir", default="output/ablation_steps")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)

    results = {}
    for steps in STEP_COUNTS:
        print(f"\n=== {steps} steps ===")
        step_results = []

        for i, prompt in enumerate(PROMPTS):
            t0 = time.time()
            grid = generate_grid(pipe, prompt, num_inference_steps=steps, seed=42)
            gen_time = time.time() - t0

            if isinstance(grid, np.ndarray):
                grid = grid.squeeze()
                if grid.dtype in (np.float32, np.float64):
                    grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
                grid = Image.fromarray(grid)

            grid.save(out_dir / f"steps{steps}_p{i}_grid.png")
            grid_to_gif(grid, output_path=str(out_dir / f"steps{steps}_p{i}.gif"))

            metrics = evaluate_gif(str(out_dir / f"steps{steps}_p{i}.gif"))
            metrics["gen_time"] = round(gen_time, 2)
            step_results.append(metrics)
            print(f"  [{i}] {gen_time:.2f}s loop={metrics.get('loop_similarity',0):.3f}")

        avg_time = np.mean([r["gen_time"] for r in step_results])
        avg_loop = np.mean([r["loop_similarity"] for r in step_results])
        avg_motion = np.mean([r["motion_magnitude"] for r in step_results])

        results[steps] = {
            "avg_time": round(avg_time, 2),
            "avg_loop": round(avg_loop, 4),
            "avg_motion": round(avg_motion, 1),
        }
        print(f"  Summary: {avg_time:.2f}s | loop={avg_loop:.3f} | motion={avg_motion:.1f}")

    # Print comparison
    print(f"\n{'Steps':<8}{'Time':<10}{'Loop':<10}{'Motion':<10}")
    print("-" * 38)
    for steps in STEP_COUNTS:
        r = results[steps]
        print(f"{steps:<8}{r['avg_time']:<10}{r['avg_loop']:<10}{r['avg_motion']:<10}")

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
