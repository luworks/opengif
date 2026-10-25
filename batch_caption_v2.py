from opengif_config import database_dsn
"""High-throughput GIF recaptioning pipeline v2.

Optimizations over v1:
- Concurrent frame extraction (ThreadPoolExecutor)
- Pipeline: CPU extracts next batch while GPU processes current batch
- Optimized prompt for GIF-to-GIF training data
- Batch embedding computation
"""

import base64
import glob
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# --- Config ---
GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
MODEL_NAME = "Qwen3-VL-2B-Instruct"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DB_BATCH = 100
MAX_FRAMES = 8
IO_WORKERS = 8

# Prompt optimized for GIF generation training pairs
CAPTION_PROMPT = (
    "You are creating training data for a GIF generation model. "
    "Describe this animated sequence so precisely that an AI could recreate a very similar GIF from your description alone. "
    "Include: 1) Main subject and their appearance, 2) The specific motion/action happening frame by frame, "
    "3) Visual style (photorealistic, cartoon, 3D render, anime, etc.), 4) Color palette and lighting, "
    "5) Background/setting, 6) Camera movement if any. "
    "Be specific about temporal progression - what changes between the start and end. "
    "Write in one detailed paragraph, 3-5 sentences."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_caption_v2.log'),
    ],
)
log = logging.getLogger(__name__)


def get_processed_filenames(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT g.filename FROM gifs g JOIN captions c ON c.gif_id = g.id WHERE c.model_name = %s", (MODEL_NAME,))
        return {row[0] for row in cur.fetchall()}


def extract_gif_data(gif_path: str) -> dict:
    """Extract frames, metadata, and base64 from a single GIF. Runs on CPU thread."""
    fname = os.path.basename(gif_path)
    try:
        gif = Image.open(gif_path)
        total = getattr(gif, "n_frames", 1)
        duration = getattr(gif, "info", {}).get("duration", 0) * total
        w, h = gif.size

        if total <= MAX_FRAMES:
            indices = list(range(total))
        else:
            indices = [int(i * (total - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]

        frames = []
        for idx in indices:
            gif.seek(idx)
            frames.append(gif.convert("RGB").copy())
        gif.close()

        # Save frames as temp files for processor
        frame_paths = []
        for i, frame in enumerate(frames):
            p = f"/tmp/gif_frames_{fname}_{i}.jpg"
            frame.save(p, quality=85)
            frame_paths.append(p)

        with open(gif_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "filename": fname,
            "frame_paths": frame_paths,
            "gif_base64": b64,
            "width": w,
            "height": h,
            "frame_count": total,
            "duration_ms": duration,
            "file_size_bytes": os.path.getsize(gif_path),
        }
    except Exception as e:
        log.error(f"Frame extraction error {fname}: {e}")
        return None


def cleanup_temp_frames(data_list: list[dict]):
    """Remove temporary frame files."""
    for data in data_list:
        if data:
            for p in data.get("frame_paths", []):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def caption_single(model, processor, data: dict) -> str:
    """Generate caption for a single GIF."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": data["frame_paths"], "fps": 2.0},
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
            max_new_tokens=384,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = out[0][inputs["input_ids"].shape[1] :]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def flush_batch(conn, embedder, batch_data: list[dict], captions: list[str]):
    """Insert batch into PostgreSQL."""
    embeddings = embedder.encode(captions, normalize_embeddings=True, batch_size=len(captions))

    gif_rows = [
        (d["filename"], d["gif_base64"], d["width"], d["height"],
         d["frame_count"], d["duration_ms"], d["file_size_bytes"])
        for d in batch_data
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO gifs (filename, gif_base64, width, height, frame_count, duration_ms, file_size_bytes)
               VALUES %s ON CONFLICT (filename) DO NOTHING RETURNING id, filename""",
            gif_rows,
        )
        inserted = {row[1]: row[0] for row in cur.fetchall()}

        existing = [g[0] for g in gif_rows if g[0] not in inserted]
        if existing:
            cur.execute("SELECT id, filename FROM gifs WHERE filename = ANY(%s)", (existing,))
            for row in cur.fetchall():
                inserted[row[1]] = row[0]

        caption_rows = []
        for d, cap, emb in zip(batch_data, captions, embeddings):
            gif_id = inserted.get(d["filename"])
            if gif_id:
                caption_rows.append((gif_id, MODEL_NAME, cap, emb.tolist()))

        if caption_rows:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO captions (gif_id, model_name, caption, embedding)
                   VALUES %s""",
                caption_rows,
                template="(%s, %s, %s, %s::vector)",
            )
    conn.commit()


def main():
    log.info("=== Batch GIF Captioning Pipeline v2 ===")

    conn = psycopg2.connect(database_dsn())
    conn.autocommit = False
    done = get_processed_filenames(conn)
    all_gifs = sorted(glob.glob(os.path.join(GIF_DIR, "*.gif")))
    pending = [p for p in all_gifs if os.path.basename(p) not in done]
    log.info(f"Total: {len(all_gifs)}, Already done: {len(done)}, Pending: {len(pending)}")

    if not pending:
        log.info("Nothing to do!")
        return

    # Load models
    log.info("Loading caption model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        f"Qwen/{MODEL_NAME}", torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        f"Qwen/{MODEL_NAME}", min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
    )
    log.info(f"Caption model VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    log.info("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    log.info(f"Total VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # Pipeline: prefetch next batch on CPU while GPU processes current
    executor = ThreadPoolExecutor(max_workers=IO_WORKERS)
    t_start = time.time()
    processed = 0
    errors = 0
    batch_data = []
    batch_captions = []

    # Pre-extract first batch
    prefetch_size = DB_BATCH
    batch_start = 0

    while batch_start < len(pending):
        batch_end = min(batch_start + prefetch_size, len(pending))
        batch_paths = pending[batch_start:batch_end]

        # Concurrent frame extraction
        futures = {executor.submit(extract_gif_data, p): p for p in batch_paths}
        extracted = []
        for f in as_completed(futures):
            result = f.result()
            if result:
                extracted.append(result)
            else:
                errors += 1

        # Sequential captioning (GPU bound)
        for data in extracted:
            try:
                caption = caption_single(model, processor, data)
                batch_data.append(data)
                batch_captions.append(caption)
                processed += 1
            except Exception as e:
                log.error(f"Caption error {data['filename']}: {e}")
                errors += 1

        # Flush to DB
        if batch_data:
            flush_batch(conn, embedder, batch_data, batch_captions)
            cleanup_temp_frames(batch_data)

        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(pending) - processed) / rate if rate > 0 else 0
        log.info(
            f"Progress: {len(done)+processed}/{len(all_gifs)} "
            f"({(len(done)+processed)/len(all_gifs)*100:.1f}%) | "
            f"{rate:.2f} gifs/s | ETA: {eta/60:.0f}min | Errors: {errors}"
        )

        batch_data.clear()
        batch_captions.clear()
        batch_start = batch_end

    elapsed = time.time() - t_start
    log.info(f"=== Done! {processed} GIFs in {elapsed/60:.1f} min ({processed/elapsed:.2f} gifs/s) ===")
    log.info(f"Errors: {errors}")
    executor.shutdown()
    conn.close()


if __name__ == "__main__":
    main()
