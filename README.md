# OpenGIF

Experimental Python tools for GIF captioning, temporal frame grids, embeddings, and video LoRA research. This is research code, not a hosted service or a pretrained model release. No speed, quality or training result is claimed by this release.

## Local setup

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

GPU model execution additionally needs compatible CUDA drivers, sufficient GPU memory, and model access under the model publisher's terms. Different research paths use different model backends; `configs/finetune_gif.yaml` is an advanced FastVideo example, whose backend must be installed separately.

## Start without a GPU or external services

Run the synthetic frame-grid smoke check using only NumPy, Pillow and psycopg2-binary:

```sh
uv run --no-project --with numpy --with pillow --with psycopg2-binary python examples/grid_smoke.py
```

The example creates an animated GIF in a temporary directory, converts it to a frame grid, and reconstructs the animation. It does not read private data, contact a model, or connect to a database.

## Optional research paths

- `prepare_grid_data.py`, `prepare_grid_v2.py`: temporal frame-grid conversion; complete dataset preparation also requires your own caption database.
- `generate_gif.py`, `generate_grid_gif.py`: model-backed generation. Despite the script name, the video generation entry point exports video; use an `.mp4` output path.
- `train_*`, `benchmark.py`, `eval_*`: experimental training and evaluation; supply authorized inputs and review arguments before running.
- `batch_caption*`, `batch_embedding*`: optional database-backed workflows.

Set `OPENGIF_DATABASE_URL` yourself for database workflows. Missing configuration fails before a connection is attempted. `OPENGIF_GIF_DIR` controls local source GIFs and `OPENGIF_MODEL_PATH` selects a compatible model identifier or local path. See `.env.example`; environment files are not loaded automatically. Never commit credentials, datasets, model weights or outputs.

No shared database or remote storage is included. No original media, model weights or training corpus is supplied. GPU inference and training have not been validated as part of the sanitized release preparation.

## License

Original project code is MIT licensed; see [LICENSE](LICENSE). The Qwen-derived embedding helper remains Apache-2.0; see [third-party notices](THIRD_PARTY_NOTICES.md). Model, dataset and third-party content rights are separate.
