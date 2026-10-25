"""Evaluate generated GIFs: loop quality, temporal consistency, visual quality."""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_gif_frames(gif_path):
    """Load all frames from a GIF."""
    gif = Image.open(gif_path)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frames.append(np.array(gif.convert("RGB"), dtype=np.float32))
    return np.stack(frames)  # (T, H, W, 3)


def loop_quality(frames):
    """Measure seamless looping: cosine similarity between first and last frame."""
    first = frames[0].flatten()
    last = frames[-1].flatten()
    cos_sim = np.dot(first, last) / (np.linalg.norm(first) * np.linalg.norm(last) + 1e-8)
    return cos_sim


def temporal_consistency(frames):
    """Measure frame-to-frame consistency via mean pairwise similarity."""
    sims = []
    for i in range(len(frames) - 1):
        a = frames[i].flatten()
        b = frames[i + 1].flatten()
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sims.append(sim)
    return np.mean(sims), np.std(sims)


def motion_magnitude(frames):
    """Measure how much motion exists (mean pixel difference between frames)."""
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i] - frames[i + 1]).mean()
        diffs.append(diff)
    return np.mean(diffs)


def evaluate_gif(gif_path):
    """Run all metrics on a single GIF."""
    frames = load_gif_frames(gif_path)
    loop = loop_quality(frames)
    temp_mean, temp_std = temporal_consistency(frames)
    motion = motion_magnitude(frames)
    return {
        "path": str(gif_path),
        "num_frames": len(frames),
        "resolution": f"{frames.shape[2]}x{frames.shape[1]}",
        "loop_similarity": float(loop),
        "temporal_consistency": float(temp_mean),
        "temporal_std": float(temp_std),
        "motion_magnitude": float(motion),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GIF quality")
    parser.add_argument("path", help="GIF file or directory of GIFs")
    parser.add_argument("--limit", type=int, default=100, help="Max GIFs to evaluate")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        gifs = [path]
    else:
        gifs = sorted(path.glob("*.gif"))[:args.limit]

    print(f"Evaluating {len(gifs)} GIFs...")

    results = []
    for gif in gifs:
        try:
            r = evaluate_gif(gif)
            results.append(r)
        except Exception as e:
            print(f"  Error on {gif.name}: {e}")

    if not results:
        print("No GIFs evaluated.")
        return

    # Aggregate stats
    loops = [r["loop_similarity"] for r in results]
    temps = [r["temporal_consistency"] for r in results]
    motions = [r["motion_magnitude"] for r in results]

    print(f"\n=== Results ({len(results)} GIFs) ===")
    print(f"Loop quality:     {np.mean(loops):.4f} ± {np.std(loops):.4f} (1.0 = perfect loop)")
    print(f"Temporal consist: {np.mean(temps):.4f} ± {np.std(temps):.4f} (1.0 = static)")
    print(f"Motion magnitude: {np.mean(motions):.1f} ± {np.std(motions):.1f} (0 = static)")

    # Identify looping GIFs (similarity > 0.85)
    looping = sum(1 for l in loops if l > 0.85)
    print(f"Looping GIFs (>0.85): {looping}/{len(results)} ({looping/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
