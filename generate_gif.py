"""Generate GIFs using Wan2.1-T2V-1.3B with optional LoRA weights."""

import argparse
import torch
from pathlib import Path
from diffusers import WanPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Generate GIF from text")
    parser.add_argument("prompt", type=str, help="Text description of the GIF")
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=0.85)
    parser.add_argument("--output", default="output.gif")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=198)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading pipeline from {args.model_path}...")
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}...")
        pipe.load_lora_weights(args.lora_path)

    generator = torch.Generator("cuda").manual_seed(args.seed)

    print(f"Generating: {args.prompt[:80]}...")
    output = pipe(
        prompt=args.prompt,
        negative_prompt="low quality, blurry, distorted",
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    # Export as GIF
    out_path = Path(args.output)
    export_to_video(output.frames[0], str(out_path), fps=8)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
