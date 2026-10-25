"""Test VLM models for GIF recaptioning quality comparison."""

import os
import time
import glob
import torch
from PIL import Image

GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
SAMPLE_COUNT = 5
PROMPT = "Describe this animated GIF in detail. Include the subject, action, style, colors, and mood. Be concise but thorough in 2-3 sentences."


def extract_gif_frames(gif_path: str, max_frames: int = 8) -> list[Image.Image]:
    """Extract evenly-spaced frames from a GIF."""
    gif = Image.open(gif_path)
    total_frames = getattr(gif, "n_frames", 1)
    if total_frames <= max_frames:
        indices = range(total_frames)
    else:
        indices = [int(i * (total_frames - 1) / (max_frames - 1)) for i in range(max_frames)]

    frames = []
    for idx in indices:
        gif.seek(idx)
        frames.append(gif.convert("RGB").copy())
    gif.close()
    return frames


def get_sample_gifs(n: int = SAMPLE_COUNT) -> list[str]:
    """Get N sample GIF paths."""
    all_gifs = sorted(glob.glob(os.path.join(GIF_DIR, "*.gif")))
    step = max(1, len(all_gifs) // n)
    return [all_gifs[i * step] for i in range(n)]


def test_qwen3_vl_2b(gif_paths: list[str]) -> dict:
    """Test Qwen3-VL-2B-Instruct."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print("\n" + "=" * 60)
    print("Loading Qwen3-VL-2B-Instruct...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"VRAM usage: {vram_mb:.0f} MB")

    results = []
    total_time = 0
    for gif_path in gif_paths:
        frames = extract_gif_frames(gif_path, max_frames=8)
        frame_paths = []
        for i, frame in enumerate(frames):
            tmp_path = f"/tmp/gif_frame_{i}.jpg"
            frame.save(tmp_path)
            frame_paths.append(tmp_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_paths},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        elapsed = time.time() - t0
        total_time += elapsed

        trimmed = output_ids[0][inputs["input_ids"].shape[1] :]
        caption = processor.decode(trimmed, skip_special_tokens=True)

        fname = os.path.basename(gif_path)
        print(f"\n[{fname}] ({elapsed:.1f}s)")
        print(f"  Caption: {caption[:200]}")
        results.append({"file": fname, "caption": caption, "time": elapsed})

    del model, processor
    torch.cuda.empty_cache()

    return {
        "model": "Qwen3-VL-2B-Instruct",
        "vram_mb": vram_mb,
        "total_time": total_time,
        "avg_time": total_time / len(gif_paths),
        "results": results,
    }


def test_gemma4_e2b(gif_paths: list[str]) -> dict:
    """Test Gemma-4-E2B-it."""
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    print("\n" + "=" * 60)
    print("Loading Gemma-4-E2B-it...")
    model = AutoModelForMultimodalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"VRAM usage: {vram_mb:.0f} MB")

    results = []
    total_time = 0
    for gif_path in gif_paths:
        frames = extract_gif_frames(gif_path, max_frames=8)

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": frame} for frame in frames],
                    {"type": "text", "text": f"These are frames from an animated GIF. {PROMPT}"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        elapsed = time.time() - t0
        total_time += elapsed

        input_len = inputs["input_ids"].shape[-1]
        caption = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        fname = os.path.basename(gif_path)
        print(f"\n[{fname}] ({elapsed:.1f}s)")
        print(f"  Caption: {caption[:200]}")
        results.append({"file": fname, "caption": caption, "time": elapsed})

    del model, processor
    torch.cuda.empty_cache()

    return {
        "model": "Gemma-4-E2B-it",
        "vram_mb": vram_mb,
        "total_time": total_time,
        "avg_time": total_time / len(gif_paths),
        "results": results,
    }


if __name__ == "__main__":
    import json

    gif_paths = get_sample_gifs()
    print(f"Testing with {len(gif_paths)} sample GIFs")
    for p in gif_paths:
        print(f"  {os.path.basename(p)}")

    all_results = []

    # Test Qwen3-VL-2B
    try:
        r = test_qwen3_vl_2b(gif_paths)
        all_results.append(r)
        print(f"\n>> Qwen3-VL-2B: VRAM={r['vram_mb']:.0f}MB, Avg={r['avg_time']:.1f}s/gif")
    except Exception as e:
        print(f"\n>> Qwen3-VL-2B FAILED: {e}")
        import traceback; traceback.print_exc()

    # Test Gemma-4-E2B
    try:
        r = test_gemma4_e2b(gif_paths)
        all_results.append(r)
        print(f"\n>> Gemma-4-E2B: VRAM={r['vram_mb']:.0f}MB, Avg={r['avg_time']:.1f}s/gif")
    except Exception as e:
        print(f"\n>> Gemma-4-E2B FAILED: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"{r['model']:25s} | VRAM: {r['vram_mb']:6.0f}MB | Avg: {r['avg_time']:.1f}s/gif")

    with open('output/test_results.json', "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to test_results.json")
