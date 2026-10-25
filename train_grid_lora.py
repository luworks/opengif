import os
"""
路线 B: 用 Wan2.1 权重训练网格图 GIF 生成器。

核心思路:
  把 GIF 网格图当作「单帧视频」喂给 Wan2.1-T2V-1.3B。
  512×256 网格图 → VAE → 512 tokens → DiT → 生成网格图 → 拆回 GIF

  不需要改架构！Wan 的 DiT + VAE + T5 直接用。
  只需要在网格图数据上 LoRA 微调，让模型学会：
  1. 生成时间一致的多帧网格
  2. 帧间连贯的运动
  3. 首尾帧一致（循环）

Pipeline:
  Step 1: 预计算 T5 文本嵌入 (GPU, ~2 min for 1K samples)
  Step 2: 预计算 VAE 潜变量 (GPU, ~3 min for 1K samples)
  Step 3: LoRA 微调 DiT (GPU, 主训练循环)

Usage:
  python train_grid_lora.py --step prepare   # 预计算 embeddings
  python train_grid_lora.py --step train      # 开始训练
  python train_grid_lora.py --step all        # 全部
"""

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = dict(
    model_path=os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    data_dir="dataset/gif-grids/train",
    cache_dir="dataset/gif-grids/cache",
    output_dir="output/grid-lora-mvp",
    # Grid image dimensions (must be divisible by 8 for VAE)
    height=256,    # 2 rows × 128
    width=512,     # 4 cols × 128
    # LoRA
    lora_rank=32,
    lora_alpha=32,
    # Training
    learning_rate=1e-4,
    num_epochs=20,
    gradient_accumulation_steps=4,
    max_train_steps=5000,
    save_every=1000,
    # Loop loss: encourage first and last rows to be similar
    loop_loss_weight=0.2,
    seed=42,
    num_workers=4,
)


# ─── Step 1: Pre-compute T5 embeddings (GPU) ────────────────────────────────

def precompute_text_embeddings(config):
    """Pre-compute T5 text embeddings on GPU and save to disk."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    cache_dir = Path(config["cache_dir"]) / "text_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_dir"])
    txt_files = sorted(data_dir.glob("*.txt"))

    cached = list(cache_dir.glob("*.pt"))
    if len(cached) >= len(txt_files):
        logger.info(f"Text embeddings already cached ({len(cached)} files)")
        return

    to_encode = []
    for txt_file in txt_files:
        stem = txt_file.stem
        if not (cache_dir / f"{stem}.pt").exists():
            to_encode.append((stem, txt_file.read_text().strip()))

    logger.info(f"Pre-computing T5 embeddings for {len(to_encode)} captions on GPU...")

    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).cuda()
    text_encoder.eval()

    batch_size = 8
    for i in tqdm(range(0, len(to_encode), batch_size), desc="Encoding text (GPU)"):
        batch = to_encode[i:i + batch_size]
        stems = [b[0] for b in batch]
        captions = [b[1] for b in batch]

        with torch.no_grad():
            inputs = tokenizer(
                captions, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to("cuda")
            output = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            embeds = output.last_hidden_state.to(torch.float16).cpu()

        for j, stem in enumerate(stems):
            torch.save({"prompt_embeds": embeds[j]}, cache_dir / f"{stem}.pt")

    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Text embeddings saved to {cache_dir}")


# ─── Step 2: Pre-compute VAE latents (GPU) ───────────────────────────────────

def precompute_vae_latents(config):
    """Encode grid images through Wan VAE as single-frame 'videos'."""
    from diffusers import AutoencoderKLWan

    cache_dir = Path(config["cache_dir"]) / "vae_latents"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_dir"])
    png_files = sorted(data_dir.glob("*.png"))

    cached = list(cache_dir.glob("*.pt"))
    if len(cached) >= len(png_files):
        logger.info(f"VAE latents already cached ({len(cached)} files)")
        return

    logger.info(f"Pre-computing VAE latents for {len(png_files)} grid images...")

    vae = AutoencoderKLWan.from_pretrained(
        config["model_path"], subfolder="vae", torch_dtype=torch.float32,
    ).cuda()
    vae.eval()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).cuda()
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).cuda()

    for png_file in tqdm(png_files, desc="Encoding VAE latents"):
        stem = png_file.stem
        out_path = cache_dir / f"{stem}.pt"
        if out_path.exists():
            continue

        # Load grid image → normalize to [-1, 1]
        img = Image.open(png_file).convert("RGB")
        img = img.resize((config["width"], config["height"]), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)

        with torch.no_grad():
            # Treat as 1-frame video: (B, C, T=1, H, W)
            x = pixel_values.unsqueeze(0).unsqueeze(2).cuda()  # (1, 3, 1, H, W)
            latent_dist = vae.encode(x).latent_dist
            latents = latent_dist.sample()
            # Normalize
            latents = (latents - latents_mean) / latents_std
            torch.save(latents.squeeze(0).cpu().to(torch.float16), out_path)

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"VAE latents saved to {cache_dir}")


# ─── Step 3: Dataset ─────────────────────────────────────────────────────────

class CachedGridDataset(Dataset):
    """Dataset using pre-computed VAE latents and T5 embeddings."""

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        self.latent_dir = cache_dir / "vae_latents"
        self.text_dir = cache_dir / "text_embeddings"

        latent_stems = {p.stem for p in self.latent_dir.glob("*.pt")}
        text_stems = {p.stem for p in self.text_dir.glob("*.pt")}
        self.stems = sorted(latent_stems & text_stems)
        logger.info(f"Dataset: {len(self.stems)} samples")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        latents = torch.load(self.latent_dir / f"{stem}.pt", weights_only=True)
        text_data = torch.load(self.text_dir / f"{stem}.pt", weights_only=True)
        return {"latents": latents, "prompt_embeds": text_data["prompt_embeds"]}


# ─── Step 4: Training Loop ───────────────────────────────────────────────────

def train(config):
    """LoRA fine-tune Wan2.1 DiT on grid images with Flow Matching loss."""
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(config["seed"])
    device = torch.device("cuda")

    logger.info("Loading Wan2.1 transformer for grid training...")
    transformer = WanTransformer3DModel.from_pretrained(
        config["model_path"], subfolder="transformer", torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                         "ffn.net.0.proj", "ffn.net.2"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(device)
    transformer.enable_gradient_checkpointing()

    dataset = CachedGridDataset(config["cache_dir"])
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=1e-3)

    total_steps = min(config["max_train_steps"],
                      len(dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"])
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    loss_history = []

    logger.info(f"Training: {total_steps} steps, LR={config['learning_rate']}")
    logger.info(f"Grid: {config['width']}×{config['height']}, 512 tokens")
    logger.info(f"LoRA rank={config['lora_rank']}")

    transformer.train()
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)

            bsz = latents.shape[0]

            # Logit-normal timestep sampling (Wan paper)
            u = torch.randn(bsz, device=device)
            timesteps = torch.sigmoid(u)

            # Flow Matching (Wan Eq 1-2)
            noise = torch.randn_like(latents)
            t = timesteps.to(dtype=latents.dtype).reshape(-1, 1, 1, 1, 1)
            noisy_latents = t * noise + (1 - t) * latents
            target = noise - latents

            model_output = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps * 1000,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # Flow Matching loss
            loss_main = F.mse_loss(model_output, target)

            # Loop loss: first row (frames 0-3) should relate to last row (frames 4-7)
            # In latent space: top half vs bottom half
            _, _, _, lat_h, _ = latents.shape
            half_h = lat_h // 2
            loss_loop = F.mse_loss(latents[:, :, :, :half_h], latents[:, :, :, half_h:])
            loss = loss_main + config["loop_loss_weight"] * loss_loop

            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss_main.item()
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{epoch_loss / num_batches:.4f}",
                    "loop": f"{loss_loop.item():.4f}",
                    "step": global_step,
                })

                loss_history.append({
                    "step": global_step,
                    "loss": loss_main.item(),
                    "loop_loss": loss_loop.item(),
                })

                if global_step % config["save_every"] == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    transformer.save_pretrained(save_path)
                    with open(output_dir / "loss_history.json", "w") as f:
                        json.dump(loss_history, f)
                    logger.info(f"Checkpoint saved at step {global_step}")

                if global_step >= config["max_train_steps"]:
                    break

        if global_step >= config["max_train_steps"]:
            break
        if num_batches > 0:
            logger.info(f"Epoch {epoch+1}: avg loss = {epoch_loss / num_batches:.4f}")

    final_path = output_dir / "final"
    transformer.save_pretrained(final_path)
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training complete! Saved to {final_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Grid GIF-DiT with Wan2.1 weights")
    parser.add_argument("--step", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--cache-dir", default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_CONFIG["lora_rank"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_train_steps"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank,
        "learning_rate": args.lr,
        "max_train_steps": args.max_steps,
    })

    if args.step in ("prepare", "all"):
        precompute_text_embeddings(config)
        precompute_vae_latents(config)

    if args.step in ("train", "all"):
        train(config)


if __name__ == "__main__":
    main()
