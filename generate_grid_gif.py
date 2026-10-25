import os
"""
Generate GIF from text using grid approach with Wan2.1 weights.

Text → T5 encode → DiT denoise (grid image) → VAE decode → split grid → GIF

Usage:
  python generate_grid_gif.py "a cat waving"
  python generate_grid_gif.py "a cat waving" --lora-path output/grid-lora-mvp/final
  python generate_grid_gif.py "a cat waving" --steps 4  # after LCM distillation
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image
from diffusers import WanPipeline
from diffusers.utils import export_to_video

from prepare_grid_data import grid_to_gif


def generate_grid(pipe, prompt, height=512, width=512, num_inference_steps=30,
                  guidance_scale=6.0, seed=42, grid_mode="4x4"):
    """Generate a grid image using Wan pipeline (single-frame mode)."""
    generator = torch.Generator("cuda").manual_seed(seed)

    # 自动加网格指令——用户不需要知道这个
    if grid_mode == "4x4":
        GRID_PREFIX = "16-frame animation grid, 4 rows by 4 columns, temporal sequence left-to-right top-to-bottom. "
        height, width = 512, 512
    else:
        GRID_PREFIX = "8-frame animation grid, 2 rows by 4 columns, temporal sequence left-to-right top-to-bottom. "
        height, width = 256, 512
    full_prompt = GRID_PREFIX + prompt

    # Generate as single-frame "video"
    output = pipe(
        prompt=full_prompt,
        negative_prompt="low quality, blurry, distorted, static, no motion",
        num_frames=1,       # Single frame = grid image
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Extract the generated image
    frames = output.frames[0]
    if isinstance(frames, list):
        frame = frames[0]
    else:
        frame = frames

    # Handle numpy arrays with extra dimensions
    if isinstance(frame, np.ndarray):
        frame = frame.squeeze()
        if frame.ndim == 3 and frame.shape[-1] == 3:
            # (H, W, 3) — correct
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(frame)
        elif frame.ndim == 3 and frame.shape[0] == 3:
            # (3, H, W) → (H, W, 3)
            frame = frame.transpose(1, 2, 0)
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(frame)

    if isinstance(frame, Image.Image):
        return frame

    return Image.fromarray(np.array(frame))


def main():
    parser = argparse.ArgumentParser(description="Generate GIF from text (grid approach)")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model-path", default=os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--output", default="output.gif")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-grid", action="store_true", help="Also save the grid image")
    args = parser.parse_args()

    print(f"Loading Wan pipeline...")
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...")
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)

    print(f"Generating grid for: {args.prompt[:80]}...")
    t0 = time.time()

    grid_image = generate_grid(
        pipe, args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    gen_time = time.time() - t0
    print(f"Generation time: {gen_time:.2f}s")

    # Ensure PIL Image
    if not isinstance(grid_image, Image.Image):
        grid_image = Image.fromarray(np.array(grid_image))

    if args.save_grid:
        grid_path = args.output.replace(".gif", "_grid.png")
        grid_image.save(grid_path)
        print(f"Grid saved to {grid_path}")

    # Split grid → GIF
    gif_path = grid_to_gif(grid_image, output_path=args.output)
    print(f"GIF saved to {gif_path}")


if __name__ == "__main__":
    main()
