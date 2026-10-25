from opengif_config import database_dsn
"""Phase 2: Generate dual-modal embeddings for all captioned GIFs.

Uses Qwen3-VL-Embedding-2B to embed:
1. Caption text → caption_embedding (2048-dim)
2. GIF frames → visual_embedding (2048-dim)
3. Compute alignment_score = cosine(caption_emb, visual_emb)
"""

import glob
import logging
import os
import time

import psycopg2
import psycopg2.extras
import torch
from PIL import Image

from qwen3_vl_embedding import Qwen3VLEmbedder

# --- Config ---
GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
EMBED_MODEL_NAME = "Qwen3-VL-Embedding-2B"
TEXT_BATCH = 64    # text embeddings are lightweight
VISUAL_BATCH = 4   # visual embeddings with 8 frames each are heavy
MAX_FRAMES = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_embedding.log'),
    ],
)
log = logging.getLogger(__name__)


def extract_gif_frames(gif_path: str, max_frames: int = MAX_FRAMES) -> list[Image.Image]:
    gif = Image.open(gif_path)
    total = getattr(gif, "n_frames", 1)
    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
    frames = []
    for idx in indices:
        gif.seek(idx)
        frames.append(gif.convert("RGB").copy())
    gif.close()
    return frames


def get_pending_rows(conn) -> list[dict]:
    """Get captions that need embeddings."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT c.id, c.gif_id, c.caption, g.filename
            FROM captions c
            JOIN gifs g ON g.id = c.gif_id
            WHERE c.caption_model = 'Qwen3.5-2B'
              AND c.caption_embedding IS NULL
            ORDER BY c.id
        """)
        return cur.fetchall()


def main():
    log.info(f"=== Phase 2: Dual-Modal Embedding ({EMBED_MODEL_NAME}) ===")

    conn = psycopg2.connect(database_dsn())
    conn.autocommit = False

    pending = get_pending_rows(conn)
    log.info(f"Pending embeddings: {len(pending)}")

    if not pending:
        log.info("Nothing to do!")
        return

    log.info("Loading embedding model...")
    model = Qwen3VLEmbedder(model_name_or_path=f"Qwen/{EMBED_MODEL_NAME}")
    vram = torch.cuda.memory_allocated() / 1024**2
    log.info(f"Model loaded, VRAM: {vram:.0f} MB")

    t_start = time.time()
    processed = 0
    errors = 0
    DB_FLUSH = 100  # flush to DB every N items

    all_updates = []

    for i, row in enumerate(pending):
        try:
            # 1. Embed caption text (lightweight)
            caption_emb = model.process([{"text": row["caption"]}])  # [1, 2048]

            # 2. Embed GIF frames (heavier, one at a time)
            gif_path = os.path.join(GIF_DIR, row["filename"])
            if os.path.exists(gif_path):
                frames = extract_gif_frames(gif_path)
                visual_emb = model.process([{"video": frames}])  # [1, 2048]
            else:
                visual_emb = caption_emb  # fallback
                log.warning(f"GIF not found: {row['filename']}")

            # 3. Alignment score
            score = float((caption_emb[0] @ visual_emb[0]).item())

            all_updates.append((
                caption_emb[0].tolist(),
                visual_emb[0].tolist(),
                EMBED_MODEL_NAME,
                score,
                row["id"],
            ))
            processed += 1

        except Exception as e:
            log.error(f"Error {row['filename']}: {e}")
            errors += 1
            torch.cuda.empty_cache()
            continue

        # Flush to DB periodically
        if len(all_updates) >= DB_FLUSH:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """UPDATE captions SET
                        caption_embedding = %s::vector,
                        visual_embedding = %s::vector,
                        embedding_model = %s,
                        alignment_score = %s,
                        embedding_created_at = NOW()
                    WHERE id = %s""",
                    all_updates,
                )
            conn.commit()
            all_updates.clear()

            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(pending) - processed) / rate if rate > 0 else 0
            log.info(
                f"Progress: {processed}/{len(pending)} "
                f"({processed/len(pending)*100:.1f}%) | "
                f"{rate:.2f} items/s | ETA: {eta/60:.0f}min | "
                f"Last align: {score:.3f} | Errors: {errors}"
            )

    # Final flush
    if all_updates:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """UPDATE captions SET
                    caption_embedding = %s::vector,
                    visual_embedding = %s::vector,
                    embedding_model = %s,
                    alignment_score = %s,
                    embedding_created_at = NOW()
                WHERE id = %s""",
                all_updates,
            )
        conn.commit()

    elapsed = time.time() - t_start
    log.info(f"=== Done! {processed} embeddings in {elapsed/60:.1f} min ({processed/elapsed:.2f} items/s) ===")

    # Final stats
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) as total,
                   AVG(alignment_score) as avg_align,
                   MIN(alignment_score) as min_align,
                   MAX(alignment_score) as max_align,
                   PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY alignment_score) as p10_align
            FROM captions
            WHERE alignment_score IS NOT NULL
        """)
        stats = cur.fetchone()
        log.info(f"Alignment stats: avg={stats[1]:.3f}, min={stats[2]:.3f}, max={stats[3]:.3f}, p10={stats[4]:.3f}")

    conn.close()


if __name__ == "__main__":
    main()
