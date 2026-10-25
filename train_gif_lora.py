"""
MVP LoRA fine-tuning of Wan2.1-T2V-1.3B for Text-to-GIF generation.

Steps:
1. Pre-compute T5 text embeddings (CPU) → disk cache
2. Pre-compute VAE latents (GPU) → disk cache
3. LoRA fine-tune DiT transformer with Flow Matching loss + loop loss

Hardware: Single RTX 4090 (24GB)
Expected VRAM: ~8-10GB during training
"""

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    data_dir="dataset/mvp-1k/train",
    cache_dir="dataset/mvp-1k/cache",
    output_dir="output/gif-lora-mvp",
    # GIF params (must be divisible by 16 for VAE 8x + patch 2x)
    num_frames=32,
    height=352,    # 360 → 352 (nearest multiple of 16)
    width=192,     # 198 → 192 (nearest multiple of 16)
    # LoRA params
    lora_rank=32,
    lora_alpha=32,
    # Training params
    learning_rate=1e-4,
    num_epochs=10,
    gradient_accumulation_steps=4,
    max_train_steps=5000,
    save_every=1000,
    # Loss
    loop_loss_weight=0.2,
    # System
    seed=42,
    num_workers=4,
)


# ─── Step 1: Pre-compute T5 embeddings ──────────────────────────────────────

def precompute_text_embeddings(config):
    """Pre-compute T5 text embeddings on GPU (batched) and save to disk.

    Loads UMT5-XXL on GPU (~11GB), encodes all captions in batches,
    then frees GPU memory for the next step.
    """
    from transformers import AutoTokenizer, UMT5EncoderModel

    cache_dir = Path(config["cache_dir"]) / "text_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_dir"])
    txt_files = sorted(data_dir.glob("*.txt"))
    cached = list(cache_dir.glob("*.pt"))
    if len(cached) >= len(txt_files):
        logger.info(f"Text embeddings already cached ({len(cached)} files)")
        return

    # Collect uncached captions
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
        batch = to_encode[i : i + batch_size]
        stems = [b[0] for b in batch]
        captions = [b[1] for b in batch]

        with torch.no_grad():
            inputs = tokenizer(
                captions,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            encoder_output = text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            embeds = encoder_output.last_hidden_state.to(torch.float16).cpu()
            masks = inputs["attention_mask"].cpu()

        for j, stem in enumerate(stems):
            torch.save({
                "prompt_embeds": embeds[j],
            }, cache_dir / f"{stem}.pt")

    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Text embeddings saved to {cache_dir}")


# ─── Step 2: Pre-compute VAE latents ────────────────────────────────────────

def load_gif_frames(gif_path, num_frames=32, height=352, width=192):
    """Load GIF and extract evenly-spaced frames as tensor.

    Returns: (T, C, H, W) tensor normalized to [-1, 1]
    """
    import numpy as np

    gif = Image.open(gif_path)
    total_frames = getattr(gif, "n_frames", 1)

    # Sample frames evenly
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        gif.seek(min(idx, total_frames - 1))
        frame = gif.convert("RGB").resize((width, height), Image.LANCZOS)
        arr = np.array(frame, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        frame_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
        frames.append(frame_tensor)

    return torch.stack(frames)  # (T, C, H, W)


def precompute_vae_latents(config):
    """Pre-compute VAE latents on GPU and save to disk."""
    from diffusers import AutoencoderKLWan

    cache_dir = Path(config["cache_dir"]) / "vae_latents"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["data_dir"])
    gif_files = sorted(data_dir.glob("*.gif"))

    cached = list(cache_dir.glob("*.pt"))
    if len(cached) >= len(gif_files):
        logger.info(f"VAE latents already cached ({len(cached)} files)")
        return

    logger.info(f"Pre-computing VAE latents for {len(gif_files)} GIFs on GPU...")

    vae = AutoencoderKLWan.from_pretrained(
        config["model_path"],
        subfolder="vae",
        torch_dtype=torch.float32,
    ).cuda()
    vae.eval()

    # Per-channel normalization constants
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).cuda()
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).cuda()

    for gif_file in tqdm(gif_files, desc="Encoding VAE latents"):
        stem = gif_file.stem
        out_path = cache_dir / f"{stem}.pt"
        if out_path.exists():
            continue

        frames = load_gif_frames(
            gif_file,
            num_frames=config["num_frames"],
            height=config["height"],
            width=config["width"],
        )

        with torch.no_grad():
            # VAE expects (B, C, T, H, W)
            pixel_values = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).cuda()  # (1, C, T, H, W)
            latent_dist = vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample()
            # Normalize: (raw - mean) / std (matches WanPipeline convention)
            latents = (latents - latents_mean) / latents_std
            torch.save(latents.squeeze(0).cpu().to(torch.float16), out_path)

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"VAE latents saved to {cache_dir}")


# ─── Step 3: Training dataset ───────────────────────────────────────────────

class CachedGIFDataset(Dataset):
    """Dataset using pre-computed VAE latents and T5 embeddings."""

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        self.latent_dir = cache_dir / "vae_latents"
        self.text_dir = cache_dir / "text_embeddings"

        # Find samples that have both latents and text embeddings
        latent_stems = {p.stem for p in self.latent_dir.glob("*.pt")}
        text_stems = {p.stem for p in self.text_dir.glob("*.pt")}
        self.stems = sorted(latent_stems & text_stems)
        logger.info(f"Dataset: {len(self.stems)} samples with cached latents + text")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        latents = torch.load(self.latent_dir / f"{stem}.pt", weights_only=True)
        text_data = torch.load(self.text_dir / f"{stem}.pt", weights_only=True)
        return {
            "latents": latents,
            "prompt_embeds": text_data["prompt_embeds"],
        }


# ─── Step 4: LoRA Training Loop ─────────────────────────────────────────────

def train(config):
    """Main LoRA training loop with Flow Matching loss."""
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(config["seed"])
    device = torch.device("cuda")

    # Load transformer
    logger.info("Loading Wan2.1-T2V-1.3B transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        config["model_path"],
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # Setup LoRA
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

    # Dataset
    dataset = CachedGIFDataset(config["cache_dir"])
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Optimizer
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=0.01)

    # Learning rate scheduler
    total_steps = min(config["max_train_steps"], len(dataloader) * config["num_epochs"])
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    loss_history = []

    logger.info(f"Starting training: {total_steps} steps, LR={config['learning_rate']}")
    logger.info(f"LoRA rank={config['lora_rank']}, alpha={config['lora_alpha']}")
    logger.info(f"Loop loss weight: {config['loop_loss_weight']}")

    transformer.train()
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        epoch_loop_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)

            # Sample timesteps from logit-normal distribution (Wan paper Sec 4.2.2)
            # This biases sampling toward middle timesteps for better training
            bsz = latents.shape[0]
            u = torch.randn(bsz, device=device)
            timesteps = torch.sigmoid(u)  # logit-normal → concentrated around 0.5

            # Flow Matching (Wan Eq 1-2): x_t = t*x_1 + (1-t)*x_0
            # where x_0 = clean latent, x_1 = noise
            noise = torch.randn_like(latents)
            t = timesteps.reshape(-1, 1, 1, 1, 1)
            noisy_latents = t * noise + (1 - t) * latents

            # Target velocity (Wan Eq 2): v_t = x_1 - x_0
            target = noise - latents

            # Forward pass — Wan expects timestep in [0, 1000]
            model_output = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps * 1000,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # Flow Matching MSE loss
            loss_main = F.mse_loss(model_output, target)

            # Loop loss: penalize first/last frame difference in latent space
            # latents shape: (B, C, T, H, W) — T is temporal dim
            loss_loop = F.mse_loss(latents[:, :, 0], latents[:, :, -1])
            loss = loss_main + config["loop_loss_weight"] * loss_loop

            # Gradient accumulation
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                epoch_loss += loss_main.item()
                epoch_loop_loss += loss_loop.item()
                num_batches += 1

                avg_loss = epoch_loss / num_batches
                avg_loop = epoch_loop_loss / num_batches
                current_lr = lr_scheduler.get_last_lr()[0]

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "loop": f"{avg_loop:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                })

                loss_history.append({
                    "step": global_step,
                    "loss": loss_main.item(),
                    "loop_loss": loss_loop.item(),
                    "lr": current_lr,
                })

                # Save checkpoint
                if global_step % config["save_every"] == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    transformer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint at step {global_step} to {save_path}")

                    # Save loss history
                    with open(output_dir / "loss_history.json", "w") as f:
                        json.dump(loss_history, f)

                if global_step >= config["max_train_steps"]:
                    break

        if global_step >= config["max_train_steps"]:
            break

        logger.info(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, loop: {avg_loop:.4f}")

    # Save final model
    final_path = output_dir / "final"
    transformer.save_pretrained(final_path)
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training complete! Final model saved to {final_path}")
    logger.info(f"Total steps: {global_step}, Final loss: {avg_loss:.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GIF-DiT LoRA")
    parser.add_argument("--step", choices=["prepare", "train", "all"], default="all",
                        help="Which step to run: prepare (cache embeddings), train, or all")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--cache-dir", default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_CONFIG["lora_rank"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_train_steps"])
    parser.add_argument("--loop-weight", type=float, default=DEFAULT_CONFIG["loop_loss_weight"])
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
        "num_epochs": args.epochs,
        "max_train_steps": args.max_steps,
        "loop_loss_weight": args.loop_weight,
    })

    if args.step in ("prepare", "all"):
        precompute_text_embeddings(config)
        precompute_vae_latents(config)

    if args.step in ("train", "all"):
        train(config)


if __name__ == "__main__":
    main()
