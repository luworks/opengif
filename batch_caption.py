from opengif_config import database_dsn
"""Batch GIF recaptioning pipeline: Qwen3-VL-2B → PostgreSQL + pgvector."""

import base64
import glob
import logging
import os
import sys
import time

import psycopg2
import psycopg2.extras
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# --- Config ---
GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
MODEL_NAME = "Qwen3-VL-2B-Instruct"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim
BATCH_SIZE = 50  # DB insert batch size
MAX_FRAMES = 8
CAPTION_PROMPT = (
    "Describe this animated GIF in detail for a training dataset. "
    "Include the subject, action, visual style, colors, mood, and camera movement. "
    "Be specific and descriptive in 2-3 sentences. Do not start with 'This animated GIF'."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_caption.log'),
    ],
)
log = logging.getLogger(__name__)


def get_processed_filenames(conn) -> set[str]:
    """Get filenames already in the database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT g.filename FROM gifs g JOIN captions c ON c.gif_id = g.id"
        )
        return {row[0] for row in cur.fetchall()}


def extract_gif_frames(gif_path: str, max_frames: int = MAX_FRAMES) -> list[Image.Image]:
    gif = Image.open(gif_path)
    total = getattr(gif, "n_frames", 1)
    if total <= max_frames:
        indices = range(total)
    else:
        indices = [int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
    frames = []
    for idx in indices:
        gif.seek(idx)
        frames.append(gif.convert("RGB").copy())
    gif.close()
    return frames


def gif_metadata(gif_path: str) -> dict:
    """Extract GIF metadata."""
    gif = Image.open(gif_path)
    total = getattr(gif, "n_frames", 1)
    duration = getattr(gif, "info", {}).get("duration", 0) * total
    w, h = gif.size
    gif.close()
    size = os.path.getsize(gif_path)
    return {
        "width": w,
        "height": h,
        "frame_count": total,
        "duration_ms": duration,
        "file_size_bytes": size,
    }


def gif_to_base64(gif_path: str) -> str:
    with open(gif_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    log.info("=== Batch GIF Captioning Pipeline ===")

    # Connect to DB
    conn = psycopg2.connect(database_dsn())
    conn.autocommit = False
    log.info("Connected to PostgreSQL")

    # Get already processed files
    done = get_processed_filenames(conn)
    log.info(f"Already processed: {len(done)} GIFs")

    # Get all GIF paths
    all_gifs = sorted(glob.glob(os.path.join(GIF_DIR, "*.gif")))
    pending = [p for p in all_gifs if os.path.basename(p) not in done]
    log.info(f"Total GIFs: {len(all_gifs)}, Pending: {len(pending)}")

    if not pending:
        log.info("Nothing to do!")
        return

    # Load captioning model
    log.info(f"Loading {MODEL_NAME}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        f"Qwen/{MODEL_NAME}",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        f"Qwen/{MODEL_NAME}",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    vram = torch.cuda.memory_allocated() / 1024**2
    log.info(f"Caption model loaded, VRAM: {vram:.0f} MB")

    # Load embedding model
    log.info(f"Loading embedding model {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    vram2 = torch.cuda.memory_allocated() / 1024**2
    log.info(f"Embedding model loaded, VRAM: {vram2:.0f} MB")

    # Process
    t_start = time.time()
    gif_batch = []
    caption_batch = []
    processed = 0
    errors = 0

    for i, gif_path in enumerate(pending):
        fname = os.path.basename(gif_path)
        try:
            # Extract frames
            frames = extract_gif_frames(gif_path)
            frame_paths = []
            for j, frame in enumerate(frames):
                p = f"/tmp/gif_frame_{j}.jpg"
                frame.save(p)
                frame_paths.append(p)

            # Generate caption
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frame_paths, "fps": 2.0},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            trimmed = out[0][inputs["input_ids"].shape[1] :]
            caption = processor.decode(trimmed, skip_special_tokens=True).strip()

            # Get metadata + base64
            meta = gif_metadata(gif_path)
            b64 = gif_to_base64(gif_path)

            gif_batch.append((fname, b64, meta["width"], meta["height"],
                              meta["frame_count"], meta["duration_ms"],
                              meta["file_size_bytes"]))
            caption_batch.append((fname, caption))
            processed += 1

        except Exception as e:
            log.error(f"Error processing {fname}: {e}")
            errors += 1
            continue

        # Flush batch to DB
        if len(gif_batch) >= BATCH_SIZE:
            _flush_batch(conn, embedder, gif_batch, caption_batch)
            elapsed = time.time() - t_start
            rate = processed / elapsed
            eta = (len(pending) - processed) / rate if rate > 0 else 0
            log.info(
                f"Progress: {processed}/{len(pending)} "
                f"({processed/len(pending)*100:.1f}%) | "
                f"{rate:.1f} gifs/s | ETA: {eta/60:.0f}min | Errors: {errors}"
            )
            gif_batch.clear()
            caption_batch.clear()

    # Final flush
    if gif_batch:
        _flush_batch(conn, embedder, gif_batch, caption_batch)

    elapsed = time.time() - t_start
    log.info(f"=== Done! Processed {processed} GIFs in {elapsed/60:.1f} min ===")
    log.info(f"Errors: {errors}")

    conn.close()


def _flush_batch(conn, embedder, gif_batch, caption_batch):
    """Insert batch of gifs and captions into DB."""
    captions_text = [c[1] for c in caption_batch]
    embeddings = embedder.encode(captions_text, normalize_embeddings=True)

    with conn.cursor() as cur:
        # Insert gifs
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO gifs (filename, gif_base64, width, height, frame_count, duration_ms, file_size_bytes)
               VALUES %s ON CONFLICT (filename) DO NOTHING RETURNING id, filename""",
            gif_batch,
        )
        inserted = {row[1]: row[0] for row in cur.fetchall()}

        # Get IDs for already-existing ones
        existing_fnames = [g[0] for g in gif_batch if g[0] not in inserted]
        if existing_fnames:
            cur.execute(
                "SELECT id, filename FROM gifs WHERE filename = ANY(%s)",
                (existing_fnames,),
            )
            for row in cur.fetchall():
                inserted[row[1]] = row[0]

        # Insert captions with embeddings
        caption_rows = []
        for (fname, caption), emb in zip(caption_batch, embeddings):
            gif_id = inserted.get(fname)
            if gif_id:
                caption_rows.append(
                    (gif_id, MODEL_NAME, caption, emb.tolist())
                )

        if caption_rows:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO captions (gif_id, model_name, caption, embedding)
                   VALUES %s""",
                caption_rows,
                template="(%s, %s, %s, %s::vector)",
            )

    conn.commit()


if __name__ == "__main__":
    main()
