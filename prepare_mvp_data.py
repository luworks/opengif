from opengif_config import database_dsn
"""Prepare MVP training data: select 1K GIFs with captions for LoRA fine-tuning."""

import os
import shutil
import psycopg2
from pathlib import Path

# Config
MVP_SIZE = 1000
GIF_SOURCE = Path("dataset/sora-gif-100k/gifs")
OUTPUT_DIR = Path("dataset/mvp-1k")
TRAIN_DIR = OUTPUT_DIR / "train"



def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(database_dsn())
    cur = conn.cursor()

    # Select 1K captioned GIFs with longest captions (higher quality descriptions)
    cur.execute("""
        SELECT g.filename, c.caption, c.caption_length
        FROM gifs g
        JOIN captions c ON g.id = c.gif_id
        WHERE c.caption_length > 200
        ORDER BY c.caption_length DESC
        LIMIT %s
    """, (MVP_SIZE,))

    rows = cur.fetchall()
    print(f"Selected {len(rows)} GIFs for MVP training")

    copied = 0
    for filename, caption, cap_len in rows:
        src = GIF_SOURCE / filename
        if not src.exists():
            continue

        stem = Path(filename).stem
        dst_gif = TRAIN_DIR / filename
        dst_txt = TRAIN_DIR / f"{stem}.txt"

        # Copy GIF
        if not dst_gif.exists():
            shutil.copy2(src, dst_gif)

        # Write caption
        with open(dst_txt, "w") as f:
            f.write(caption)

        copied += 1

    print(f"Copied {copied} GIF+caption pairs to {TRAIN_DIR}")
    print(f"Average caption length: {sum(r[2] for r in rows) / len(rows):.0f} chars")

    conn.close()


if __name__ == "__main__":
    main()
