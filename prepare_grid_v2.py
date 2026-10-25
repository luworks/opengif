from opengif_config import database_dsn
"""
网格图 V2: 保持原始宽高比 (96×176/帧 → 384×352 网格)。

V1 问题: 128×128 正方形压缩了 198×360 竖屏 GIF，宽高比失真 82%。
V2 改进: 96×176 竖帧保持 ~11:20 比例，失真仅 1%。

| 版本 | 帧尺寸 | 网格 | Tokens | 失真 |
|------|--------|------|--------|------|
| V1   | 128×128 | 512×256 | 512 | 82% |
| V2   | 96×176  | 384×352 | 528 | 1%  |

Usage:
  python prepare_grid_v2.py                    # 全量
  python prepare_grid_v2.py --limit 100        # 测试
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import psycopg2


# V2 Grid config: 保持宽高比
NUM_SAMPLE_FRAMES = 8
GRID_ROWS = 2
GRID_COLS = 4
FRAME_W = 96      # 宽
FRAME_H = 176     # 高 (保持 ~11:20 比例)
# 网格: 384×352 (cols*96 × rows*176)

GRID_PREFIX = "8-frame animation grid, 2 rows by 4 columns, temporal sequence left-to-right top-to-bottom. "


def gif_to_grid_v2(gif_path, num_frames=NUM_SAMPLE_FRAMES, frame_w=FRAME_W, frame_h=FRAME_H,
                   rows=GRID_ROWS, cols=GRID_COLS):
    """将 GIF 转换为保持宽高比的网格图。"""
    gif = Image.open(gif_path)
    total_frames = getattr(gif, "n_frames", 1)
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        gif.seek(min(idx, total_frames - 1))
        frame = gif.convert("RGB").resize((frame_w, frame_h), Image.LANCZOS)
        frames.append(np.array(frame))

    grid_h = rows * frame_h
    grid_w = cols * frame_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        r = i // cols
        c = i % cols
        y = r * frame_h
        x = c * frame_w
        grid[y:y + frame_h, x:x + frame_w] = frame

    return Image.fromarray(grid)


def grid_to_gif_v2(grid_image, frame_w=FRAME_W, frame_h=FRAME_H,
                   rows=GRID_ROWS, cols=GRID_COLS, fps=8, output_path="output.gif"):
    """将 V2 网格图拆分回 GIF。"""
    grid = np.array(grid_image)
    frames = []
    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        y = r * frame_h
        x = c * frame_w
        frame = Image.fromarray(grid[y:y + frame_h, x:x + frame_w])
        frames.append(frame)

    duration = 1000 // fps
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    return output_path


def prepare_dataset(output_dir, limit=None, source_dir="dataset/sora-gif-100k/gifs"):
    """生成 V2 网格图数据集。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(database_dsn())
    cur = conn.cursor()

    query = """
        SELECT g.filename, c.caption
        FROM gifs g JOIN captions c ON g.id = c.gif_id
        WHERE c.caption_length > 200
        ORDER BY c.caption_length DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()
    print(f"Preparing {len(rows)} V2 grid images ({FRAME_W}×{FRAME_H}/frame, {GRID_COLS*FRAME_W}×{GRID_ROWS*FRAME_H} grid)...")

    source = Path(source_dir)
    success = errors = 0

    for filename, caption in rows:
        gif_path = source / filename
        if not gif_path.exists():
            errors += 1
            continue

        stem = Path(filename).stem
        grid_path = output_dir / f"{stem}.png"
        txt_path = output_dir / f"{stem}.txt"

        if grid_path.exists() and txt_path.exists():
            success += 1
            continue

        try:
            grid = gif_to_grid_v2(gif_path)
            grid.save(grid_path)
            with open(txt_path, "w") as f:
                f.write(GRID_PREFIX + caption)
            success += 1
            if success % 5000 == 0:
                print(f"  Progress: {success}/{len(rows)}")
        except Exception as e:
            errors += 1

    conn.close()
    print(f"Done: {success} grids, {errors} errors")
    print(f"Grid: {GRID_COLS*FRAME_W}×{GRID_ROWS*FRAME_H} ({GRID_ROWS}×{GRID_COLS} of {FRAME_W}×{FRAME_H})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dataset/gif-grids-v2/train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    prepare_dataset(args.output_dir, limit=args.limit)
