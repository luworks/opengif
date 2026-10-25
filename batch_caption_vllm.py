from opengif_config import database_dsn
"""High-throughput GIF recaptioning pipeline using vLLM Docker server.

Model: Qwen3.5-2B (native multimodal, early fusion)
Embedding: Phase 1 skips embedding (done in Phase 2 with Qwen3-VL-Embedding-2B)
"""

import asyncio
import base64
import glob
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import psycopg2
import psycopg2.extras
from PIL import Image

# --- Config ---
GIF_DIR = os.environ.get("OPENGIF_GIF_DIR", "dataset/gifs")
VLLM_URL = "http://localhost:30000/v1/chat/completions"
MODEL_NAME = "Qwen3.5-2B"
VLLM_MODEL = "Qwen/Qwen3.5-2B"
CONCURRENCY = 20
DB_BATCH = 100
MAX_FRAMES = 8
PROMPT_VERSION = "v2-gif-generation"
INFERENCE_ENGINE = "vllm-0.18.1-docker"

CAPTION_PROMPT = (
    "You are creating training data for a GIF generation model. "
    "Describe this animated sequence so precisely that an AI could recreate a very similar GIF from your description alone. "
    "Include: 1) Main subject and their appearance, 2) The specific motion/action frame by frame, "
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
        logging.FileHandler('batch_caption_vllm.log'),
    ],
)
log = logging.getLogger(__name__)


def get_processed_filenames(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT g.filename FROM gifs g JOIN captions c ON c.gif_id = g.id WHERE c.caption_model = %s",
            (MODEL_NAME,),
        )
        return {row[0] for row in cur.fetchall()}


def extract_gif_data(gif_path: str) -> dict | None:
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

        frame_b64s = []
        for idx in indices:
            gif.seek(idx)
            frame = gif.convert("RGB")
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=80)
            frame_b64s.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        gif.close()

        with open(gif_path, "rb") as f:
            file_b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "filename": fname,
            "frame_b64s": frame_b64s,
            "gif_base64": file_b64,
            "width": w,
            "height": h,
            "frame_count": total,
            "duration_ms": duration,
            "file_size_bytes": os.path.getsize(gif_path),
        }
    except Exception as e:
        log.error(f"Extract error {fname}: {e}")
        return None


async def caption_single(session: aiohttp.ClientSession, data: dict, sem: asyncio.Semaphore) -> tuple[dict, str, int] | None:
    """Send frames to vLLM, return (data, caption, inference_time_ms)."""
    async with sem:
        content = []
        for b64 in data["frame_b64s"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": CAPTION_PROMPT})

        payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 384,
            "temperature": 0,
        }

        try:
            t0 = time.time()
            async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.error(f"vLLM error {resp.status} for {data['filename']}: {text[:200]}")
                    return None
                result = await resp.json()
                caption = result["choices"][0]["message"]["content"].strip()
                inference_ms = int((time.time() - t0) * 1000)
                return (data, caption, inference_ms)
        except Exception as e:
            log.error(f"Request error {data['filename']}: {e}")
            return None


def flush_batch(conn, batch: list[tuple[dict, str, int]]):
    """Insert batch into PostgreSQL with full metadata."""
    gif_rows = [
        (d["filename"], d["gif_base64"], d["width"], d["height"],
         d["frame_count"], d["duration_ms"], d["file_size_bytes"])
        for d, _, _ in batch
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
        for d, cap, inf_ms in batch:
            gif_id = inserted.get(d["filename"])
            if gif_id:
                caption_rows.append((
                    gif_id, cap, MODEL_NAME, VLLM_MODEL,
                    PROMPT_VERSION, INFERENCE_ENGINE, inf_ms, len(cap),
                ))

        if caption_rows:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO captions
                   (gif_id, caption, caption_model, caption_model_id,
                    caption_prompt_version, inference_engine, inference_time_ms, caption_length)
                   VALUES %s
                   ON CONFLICT (gif_id, caption_model) DO NOTHING""",
                caption_rows,
            )
    conn.commit()


async def process_batch(session, sem, executor, gif_paths) -> list[tuple[dict, str, int]]:
    loop = asyncio.get_event_loop()
    extract_futures = [loop.run_in_executor(executor, extract_gif_data, p) for p in gif_paths]
    extracted = await asyncio.gather(*extract_futures)
    extracted = [d for d in extracted if d is not None]

    caption_tasks = [caption_single(session, data, sem) for data in extracted]
    results = await asyncio.gather(*caption_tasks)
    return [r for r in results if r is not None]


async def main():
    log.info(f"=== Batch GIF Captioning Pipeline ===")
    log.info(f"Model: {MODEL_NAME} ({VLLM_MODEL})")
    log.info(f"Prompt: {PROMPT_VERSION}")
    log.info(f"Engine: {INFERENCE_ENGINE}")

    conn = psycopg2.connect(database_dsn())
    conn.autocommit = False
    done = get_processed_filenames(conn)
    all_gifs = sorted(glob.glob(os.path.join(GIF_DIR, "*.gif")))
    pending = [p for p in all_gifs if os.path.basename(p) not in done]
    log.info(f"Total: {len(all_gifs)}, Done: {len(done)}, Pending: {len(pending)}")

    if not pending:
        log.info("Nothing to do!")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    executor = ThreadPoolExecutor(max_workers=8)

    t_start = time.time()
    processed = 0
    errors = 0

    async with aiohttp.ClientSession() as session:
        for chunk_start in range(0, len(pending), DB_BATCH):
            chunk = pending[chunk_start : chunk_start + DB_BATCH]
            results = await process_batch(session, sem, executor, chunk)

            if results:
                flush_batch(conn, results)
                processed += len(results)
            errors += len(chunk) - len(results)

            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(pending) - processed) / rate if rate > 0 else 0
            log.info(
                f"Progress: {processed}/{len(pending)} "
                f"({processed/len(pending)*100:.1f}%) | "
                f"{rate:.2f} gifs/s | ETA: {eta/60:.0f}min | Errors: {errors}"
            )

    elapsed = time.time() - t_start
    log.info(f"=== Done! {processed} GIFs in {elapsed/60:.1f} min ({processed/elapsed:.2f} gifs/s) ===")
    executor.shutdown()
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
