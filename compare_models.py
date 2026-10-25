"""A/B test: Qwen3-VL-2B vs Gemma-4-E2B for GIF captioning quality.

Sends the same 10 GIFs to both vLLM servers and compares captions.
Qwen3-VL-2B: http://localhost:30000
Gemma-4-E2B: http://localhost:30001
"""

import asyncio
import base64
import glob
import io
import json
import os
import time

import aiohttp
from PIL import Image

GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
SAMPLE_COUNT = 10
MAX_FRAMES = 8

CAPTION_PROMPT = (
    "You are creating training data for a GIF generation model. "
    "Describe this animated sequence so precisely that an AI could recreate a very similar GIF from your description alone. "
    "Include: 1) Main subject and their appearance, 2) The specific motion/action frame by frame, "
    "3) Visual style (photorealistic, cartoon, 3D render, anime, etc.), 4) Color palette and lighting, "
    "5) Background/setting, 6) Camera movement if any. "
    "Be specific about temporal progression - what changes between the start and end. "
    "Write in one detailed paragraph, 3-5 sentences."
)

MODELS = {
    "Qwen3-VL-2B": {
        "url": "http://localhost:30000/v1/chat/completions",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
    },
    "Gemma-4-E2B": {
        "url": "http://localhost:30001/v1/chat/completions",
        "model": "google/gemma-4-E2B-it",
    },
}


def get_sample_gifs(n: int = SAMPLE_COUNT) -> list[str]:
    all_gifs = sorted(glob.glob(os.path.join(GIF_DIR, "*.gif")))
    step = max(1, len(all_gifs) // n)
    return [all_gifs[i * step] for i in range(n)]


def extract_frames_b64(gif_path: str) -> list[str]:
    gif = Image.open(gif_path)
    total = getattr(gif, "n_frames", 1)
    if total <= MAX_FRAMES:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]

    frames_b64 = []
    for idx in indices:
        gif.seek(idx)
        frame = gif.convert("RGB")
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=80)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    gif.close()
    return frames_b64


async def caption_gif(session, model_name, config, frames_b64):
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": CAPTION_PROMPT})

    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 384,
        "temperature": 0,
    }

    t0 = time.time()
    try:
        async with session.post(
            config["url"], json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"model": model_name, "error": f"HTTP {resp.status}: {text[:200]}", "time": 0}
            result = await resp.json()
            caption = result["choices"][0]["message"]["content"].strip()
            elapsed = time.time() - t0
            return {"model": model_name, "caption": caption, "time": elapsed}
    except Exception as e:
        return {"model": model_name, "error": str(e), "time": 0}


async def main():
    gif_paths = get_sample_gifs()
    print(f"Testing {len(gif_paths)} GIFs against {len(MODELS)} models\n")

    # Check which servers are available
    async with aiohttp.ClientSession() as session:
        available = {}
        for name, config in MODELS.items():
            try:
                base_url = config["url"].replace("/v1/chat/completions", "/health")
                async with session.get(base_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    available[name] = config
                    print(f"  [OK] {name} at {config['url']}")
            except Exception:
                print(f"  [SKIP] {name} not available at {config['url']}")

        if not available:
            print("No models available!")
            return

        results = []
        for i, gif_path in enumerate(gif_paths):
            fname = os.path.basename(gif_path)
            frames_b64 = extract_frames_b64(gif_path)
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(gif_paths)}] {fname} ({len(frames_b64)} frames)")

            # Send to all available models concurrently
            tasks = [
                caption_gif(session, name, config, frames_b64)
                for name, config in available.items()
            ]
            model_results = await asyncio.gather(*tasks)

            gif_result = {"filename": fname, "models": {}}
            for r in model_results:
                model = r["model"]
                if "error" in r:
                    print(f"  [{model}] ERROR: {r['error']}")
                else:
                    print(f"  [{model}] ({r['time']:.1f}s)")
                    print(f"    {r['caption'][:200]}...")
                gif_result["models"][model] = r
            results.append(gif_result)

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for name in available:
            times = [r["models"][name]["time"] for r in results if name in r["models"] and "error" not in r["models"][name]]
            if times:
                avg = sum(times) / len(times)
                errs = sum(1 for r in results if name in r["models"] and "error" in r["models"][name])
                print(f"  {name:20s} | Avg: {avg:.1f}s/gif | {1/avg:.2f} gifs/s | Errors: {errs}")

        with open('output/compare_results.json', "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\nResults saved to compare_results.json")


if __name__ == "__main__":
    asyncio.run(main())
