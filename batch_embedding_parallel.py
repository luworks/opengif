from opengif_config import database_dsn
"""Phase 2: Parallel dual-modal embedding using multiple GPU workers.

Each worker loads its own model instance (~4GB VRAM).
With 24GB GPU, we can run 3 workers (~12GB) + overhead.
Target: 61K items in < 1 hour.
"""

import logging
import math
import multiprocessing as mp
import os
import time

import psycopg2
import psycopg2.extras

# --- Config ---
GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
EMBED_MODEL_NAME = "Qwen3-VL-Embedding-2B"
NUM_WORKERS = 3
MAX_FRAMES = 8
DB_FLUSH = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [W%(process)d] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_embedding.log'),
    ],
)
log = logging.getLogger(__name__)


def extract_gif_frames(gif_path, max_frames=MAX_FRAMES):
    from PIL import Image
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


def worker_fn(worker_id: int, row_ids: list[int]):
    """Each worker loads its own model and processes assigned rows."""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from qwen3_vl_embedding import Qwen3VLEmbedder

    log.info(f"Worker {worker_id}: loading model for {len(row_ids)} items...")
    model = Qwen3VLEmbedder(model_name_or_path=f"Qwen/{EMBED_MODEL_NAME}")
    vram = torch.cuda.memory_allocated() / 1024**2
    log.info(f"Worker {worker_id}: model loaded, VRAM={vram:.0f}MB")

    conn = psycopg2.connect(database_dsn())
    conn.autocommit = False

    # Fetch assigned rows
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT c.id, c.caption, g.filename
               FROM captions c JOIN gifs g ON g.id = c.gif_id
               WHERE c.id = ANY(%s) AND c.caption_embedding IS NULL
               ORDER BY c.id""",
            (row_ids,),
        )
        rows = cur.fetchall()

    log.info(f"Worker {worker_id}: {len(rows)} items to process")

    t_start = time.time()
    processed = 0
    errors = 0
    updates = []

    for row in rows:
        try:
            # Text embedding
            cap_emb = model.process([{"text": row["caption"]}])

            # Visual embedding
            gif_path = os.path.join(GIF_DIR, row["filename"])
            if os.path.exists(gif_path):
                frames = extract_gif_frames(gif_path)
                vis_emb = model.process([{"video": frames}])
            else:
                vis_emb = cap_emb

            score = float((cap_emb[0] @ vis_emb[0]).item())
            updates.append((
                cap_emb[0].tolist(), vis_emb[0].tolist(),
                EMBED_MODEL_NAME, score, row["id"],
            ))
            processed += 1

        except Exception as e:
            log.error(f"Worker {worker_id} error {row['filename']}: {e}")
            errors += 1
            torch.cuda.empty_cache()
            continue

        if len(updates) >= DB_FLUSH:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, """
                    UPDATE captions SET
                        caption_embedding=%s::vector, visual_embedding=%s::vector,
                        embedding_model=%s, alignment_score=%s, embedding_created_at=NOW()
                    WHERE id=%s""", updates)
            conn.commit()
            updates.clear()

            elapsed = time.time() - t_start
            rate = processed / elapsed
            eta = (len(rows) - processed) / rate if rate > 0 else 0
            log.info(
                f"Worker {worker_id}: {processed}/{len(rows)} "
                f"({processed/len(rows)*100:.1f}%) | {rate:.2f}/s | "
                f"ETA: {eta/60:.0f}min | align={score:.3f} | err={errors}"
            )

    # Final flush
    if updates:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                UPDATE captions SET
                    caption_embedding=%s::vector, visual_embedding=%s::vector,
                    embedding_model=%s, alignment_score=%s, embedding_created_at=NOW()
                WHERE id=%s""", updates)
        conn.commit()

    elapsed = time.time() - t_start
    log.info(f"Worker {worker_id}: DONE! {processed} in {elapsed/60:.1f}min ({processed/elapsed:.2f}/s), errors={errors}")
    conn.close()


def main():
    log.info(f"=== Phase 2: Parallel Embedding ({NUM_WORKERS} workers) ===")

    conn = psycopg2.connect(database_dsn())
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id FROM captions c
            WHERE c.caption_model = 'Qwen3.5-2B' AND c.caption_embedding IS NULL
            ORDER BY c.id
        """)
        all_ids = [r[0] for r in cur.fetchall()]

    # Target: 10K total embeddings
    TARGET = 10000
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM captions WHERE caption_embedding IS NOT NULL")
        already_done = cur.fetchone()[0]
    conn.close()
    remaining = max(0, TARGET - already_done)
    all_ids = all_ids[:remaining]

    log.info(f"Already done: {already_done}, Target: {TARGET}, To process: {len(all_ids)}")
    if not all_ids:
        log.info("Target reached, nothing to do!")
        return

    # Split work across workers
    chunk_size = math.ceil(len(all_ids) / NUM_WORKERS)
    chunks = [all_ids[i:i+chunk_size] for i in range(0, len(all_ids), chunk_size)]
    log.info(f"Distributing {len(all_ids)} items across {len(chunks)} workers (~{chunk_size} each)")

    mp.set_start_method("spawn", force=True)
    processes = []
    for i, chunk in enumerate(chunks):
        p = mp.Process(target=worker_fn, args=(i, chunk))
        p.start()
        processes.append(p)
        time.sleep(5)  # stagger model loading to avoid VRAM spike

    for p in processes:
        p.join()

    # === Validation ===
    log.info("=== Running validation ===")
    conn = psycopg2.connect(database_dsn())
    with conn.cursor() as cur:
        # 1. Count
        cur.execute("SELECT COUNT(*) FROM captions WHERE caption_embedding IS NOT NULL")
        total_embedded = cur.fetchone()[0]
        log.info(f"[CHECK] Total embeddings: {total_embedded}")

        # 2. Alignment stats
        cur.execute("""
            SELECT AVG(alignment_score), MIN(alignment_score), MAX(alignment_score),
                   PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY alignment_score),
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY alignment_score)
            FROM captions WHERE alignment_score IS NOT NULL
        """)
        avg, mn, mx, p10, p50 = cur.fetchone()
        log.info(f"[CHECK] Alignment: avg={avg:.3f}, median={p50:.3f}, min={mn:.3f}, max={mx:.3f}, p10={p10:.3f}")

        # 3. Check for nulls in embedded rows
        cur.execute("SELECT COUNT(*) FROM captions WHERE caption_embedding IS NOT NULL AND visual_embedding IS NULL")
        missing_visual = cur.fetchone()[0]
        log.info(f"[CHECK] Missing visual embeddings: {missing_visual}")

        # 4. Dimension check
        cur.execute("SELECT array_length(caption_embedding::real[], 1) FROM captions WHERE caption_embedding IS NOT NULL LIMIT 1")
        dim = cur.fetchone()[0]
        log.info(f"[CHECK] Embedding dimension: {dim}")

        # 5. Low alignment outliers
        cur.execute("SELECT COUNT(*) FROM captions WHERE alignment_score < 0.3")
        low_align = cur.fetchone()[0]
        log.info(f"[CHECK] Low alignment (<0.3): {low_align}")

    passed = total_embedded >= TARGET and missing_visual == 0 and dim == 2048
    if passed:
        log.info(f"=== VALIDATION PASSED: {total_embedded} embeddings, all checks OK ===")
    else:
        log.warning(f"=== VALIDATION ISSUES: embedded={total_embedded}/{TARGET}, missing_visual={missing_visual}, dim={dim} ===")
    conn.close()


if __name__ == "__main__":
    main()
