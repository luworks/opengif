from opengif_config import database_dsn
"""
路线 B: 将 GIF 转换为网格图 (frame grid) 用于图像 DiT 训练。

核心思路:
  32帧 GIF → 采样 8 帧 → 排成 2×4 网格 → 一张 256×512 图片
  self-attention 天然让每帧看到其他所有帧，帧间一致性自动学习。

Usage:
  python prepare_grid_data.py                    # 准备全量数据
  python prepare_grid_data.py --limit 100        # 准备 100 个样本测试
  python prepare_grid_data.py --analyze          # 分析数据集统计
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import psycopg2


# Grid config
NUM_SAMPLE_FRAMES = 8    # 从 32 帧中采样 8 帧
GRID_ROWS = 2
GRID_COLS = 4            # 2×4 = 8 帧
FRAME_SIZE = 128          # 每帧 resize 到 128×128
# 最终网格图: 256×512 (2*128 × 4*128)


def gif_to_grid(gif_path, num_frames=NUM_SAMPLE_FRAMES, frame_size=FRAME_SIZE,
                rows=GRID_ROWS, cols=GRID_COLS):
    """将 GIF 转换为网格图。

    32帧 GIF → 等间距采样 8 帧 → 每帧 resize 到 128×128 → 排成 2×4 网格
    Returns: PIL Image (256×512)
    """
    gif = Image.open(gif_path)
    total_frames = getattr(gif, "n_frames", 1)

    # 等间距采样
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        gif.seek(min(idx, total_frames - 1))
        frame = gif.convert("RGB").resize((frame_size, frame_size), Image.LANCZOS)
        frames.append(np.array(frame))

    # 拼成网格: 从左到右，从上到下，时间顺序
    # Row 0: frame 0, 1, 2, 3
    # Row 1: frame 4, 5, 6, 7
    grid_h = rows * frame_size
    grid_w = cols * frame_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        r = i // cols
        c = i % cols
        y = r * frame_size
        x = c * frame_size
        grid[y:y + frame_size, x:x + frame_size] = frame

    return Image.fromarray(grid)


def grid_to_gif(grid_image, frame_size=FRAME_SIZE, rows=GRID_ROWS, cols=GRID_COLS,
                fps=8, output_path="output.gif"):
    """将网格图拆分回 GIF。推理时使用。"""
    grid = np.array(grid_image)
    frames = []

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        y = r * frame_size
        x = c * frame_size
        frame = Image.fromarray(grid[y:y + frame_size, x:x + frame_size])
        frames.append(frame)

    # 保存为 GIF
    duration = 1000 // fps  # ms per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return output_path


def prepare_dataset(output_dir, limit=None, source_dir="dataset/sora-gif-100k/gifs"):
    """从数据库读取 caption + GIF，生成网格图训练数据集。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(database_dsn())
    cur = conn.cursor()

    # 获取所有有 caption 的 GIF
    query = """
        SELECT g.filename, c.caption
        FROM gifs g
        JOIN captions c ON g.id = c.gif_id
        WHERE c.caption_length > 200
        ORDER BY c.caption_length DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()
    print(f"Preparing {len(rows)} grid images...")

    source = Path(source_dir)
    success = 0
    errors = 0

    for filename, caption in rows:
        gif_path = source / filename
        if not gif_path.exists():
            errors += 1
            continue

        stem = Path(filename).stem
        grid_path = output_dir / f"{stem}.png"
        txt_path = output_dir / f"{stem}.txt"

        # Skip if already done
        if grid_path.exists() and txt_path.exists():
            success += 1
            continue

        try:
            # Convert GIF to grid image
            grid = gif_to_grid(gif_path)
            grid.save(grid_path)

            # Write caption
            with open(txt_path, "w") as f:
                f.write(caption)

            success += 1
            if success % 1000 == 0:
                print(f"  Progress: {success}/{len(rows)}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {filename}: {e}")

    conn.close()
    print(f"Done: {success} grids saved, {errors} errors")
    print(f"Grid size: {GRID_ROWS * FRAME_SIZE}×{GRID_COLS * FRAME_SIZE} "
          f"({GRID_ROWS}×{GRID_COLS} grid of {FRAME_SIZE}×{FRAME_SIZE} frames)")
    return success


def analyze_grids(grid_dir):
    """分析网格图数据集的统计信息。"""
    grid_dir = Path(grid_dir)
    pngs = list(grid_dir.glob("*.png"))
    txts = list(grid_dir.glob("*.txt"))
    print(f"=== Grid Dataset Analysis ===")
    print(f"Grid images: {len(pngs)}")
    print(f"Captions: {len(txts)}")

    if pngs:
        img = Image.open(pngs[0])
        print(f"Grid size: {img.size} (W×H)")
        print(f"Per frame: {FRAME_SIZE}×{FRAME_SIZE}")
        print(f"Layout: {GRID_ROWS}×{GRID_COLS} = {GRID_ROWS * GRID_COLS} frames")

    if txts:
        lengths = []
        for txt in txts[:1000]:
            lengths.append(len(txt.read_text()))
        print(f"Caption length: avg={np.mean(lengths):.0f}, "
              f"min={np.min(lengths)}, max={np.max(lengths)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GIF grid dataset")
    parser.add_argument("--output-dir", default="dataset/gif-grids/train")
    parser.add_argument("--source-dir", default="dataset/sora-gif-100k/gifs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--analyze", action="store_true", help="Analyze existing dataset")
    parser.add_argument("--test-roundtrip", type=str, default=None,
                        help="Test GIF→grid→GIF roundtrip on a single file")
    args = parser.parse_args()

    if args.analyze:
        analyze_grids(args.output_dir)
    elif args.test_roundtrip:
        print(f"Testing roundtrip on {args.test_roundtrip}...")
        grid = gif_to_grid(args.test_roundtrip)
        grid.save("/tmp/test_grid.png")
        print(f"Grid saved to /tmp/test_grid.png ({grid.size})")
        grid_to_gif(grid, output_path="/tmp/test_roundtrip.gif")
        print(f"Roundtrip GIF saved to /tmp/test_roundtrip.gif")
    else:
        prepare_dataset(args.output_dir, limit=args.limit, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
