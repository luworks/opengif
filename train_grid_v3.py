import os
"""
Grid GIF-DiT V3: 集成 GRID 论文的 Flow Loss + α 渐增策略。

核心改进 (vs V2):
  1. Flow Loss 替换 Loop Loss — 约束所有相邻帧差异一致性
  2. α 渐增策略 — 先学内容 (α=0)，再学时间一致性 (α→0.5)
  3. 支持 Continue Training (全量微调) 模式

参考: GRID (arxiv 2412.10718), Wan (arxiv 2503.20314)
"""

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = dict(
    model_path=os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    data_dir="dataset/gif-grids/train",
    cache_dir="dataset/gif-grids/cache",
    output_dir="output/grid-v3-flow",
    # Grid
    height=256, width=512,
    grid_rows=2, grid_cols=4,
    # Training mode: "lora" or "full"
    train_mode="lora",
    lora_rank=32, lora_alpha=32,
    # Training
    learning_rate=1e-4,
    num_epochs=20,
    batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_train_steps=10000,
    save_every=2000,
    eval_every=2000,
    # Flow loss (GRID paper)
    flow_loss_max_weight=0.5,    # α 最终值
    flow_loss_warmup_steps=2000, # α 从 0 增到 max 的步数
    seed=42,
    num_workers=4,
)


# ─── Flow Loss (GRID 论文核心) ───────────────────────────────────────────────

def compute_flow_loss(noise, prediction, grid_rows, grid_cols, lat_h, lat_w):
    """GRID 论文的 Flow Loss: 约束相邻帧间变化方向一致。

    不是让帧本身相似，而是让帧间的"变化方向"在真实和预测之间一致。
    L_flow = E[|Δε - Δε_θ|²]

    Args:
        noise: 真实噪声 (B, C, 1, H, W)
        prediction: 模型预测 (B, C, 1, H, W)
        grid_rows, grid_cols: 网格布局
        lat_h, lat_w: latent 空间的高宽
    """
    cell_h = lat_h // grid_rows
    cell_w = lat_w // grid_cols

    loss = 0.0
    count = 0

    for idx in range(1, grid_rows * grid_cols):
        prev_idx = idx - 1

        # 当前帧和前一帧的位置
        curr_r, curr_c = idx // grid_cols, idx % grid_cols
        prev_r, prev_c = prev_idx // grid_cols, prev_idx % grid_cols

        # 提取 cells
        curr_noise = noise[:, :, :, curr_r*cell_h:(curr_r+1)*cell_h, curr_c*cell_w:(curr_c+1)*cell_w]
        prev_noise = noise[:, :, :, prev_r*cell_h:(prev_r+1)*cell_h, prev_c*cell_w:(prev_c+1)*cell_w]
        curr_pred = prediction[:, :, :, curr_r*cell_h:(curr_r+1)*cell_h, curr_c*cell_w:(curr_c+1)*cell_w]
        prev_pred = prediction[:, :, :, prev_r*cell_h:(prev_r+1)*cell_h, prev_c*cell_w:(prev_c+1)*cell_w]

        # 真实差异 vs 预测差异
        delta_noise = curr_noise - prev_noise
        delta_pred = curr_pred - prev_pred

        loss += F.mse_loss(delta_pred, delta_noise)
        count += 1

    return loss / count


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CachedGridDataset(Dataset):
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


# ─── Training ────────────────────────────────────────────────────────────────

def train(config):
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(config["seed"])
    device = torch.device("cuda")

    logger.info(f"Loading Wan2.1 transformer ({config['train_mode']} mode)...")
    transformer = WanTransformer3DModel.from_pretrained(
        config["model_path"], subfolder="transformer", torch_dtype=torch.bfloat16,
    )

    if config["train_mode"] == "lora":
        lora_config = LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0",
                             "ffn.net.0.proj", "ffn.net.2"],
            lora_dropout=0.0,
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
    else:
        logger.info(f"Full fine-tune: {sum(p.numel() for p in transformer.parameters())/1e6:.0f}M params")

    transformer.to(device)
    if config["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing: ON (saves memory, ~20% slower)")
    else:
        logger.info("Gradient checkpointing: OFF (faster, more VRAM)")

    dataset = CachedGridDataset(config["cache_dir"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                            num_workers=config["num_workers"], pin_memory=True,
                            drop_last=True)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]

    if config["train_mode"] == "full":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable_params, lr=config["learning_rate"], weight_decay=0.0)
        logger.info("Using 8-bit AdamW (full fine-tune mode)")
    else:
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

    # Grid dimensions in latent space
    lat_h = config["height"] // 8
    lat_w = config["width"] // 8

    global_step = 0
    loss_history = []

    logger.info(f"Training: {total_steps} steps, mode={config['train_mode']}")
    logger.info(f"Flow loss: α 0→{config['flow_loss_max_weight']} over {config['flow_loss_warmup_steps']} steps")

    transformer.train()
    for epoch in range(config["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_base, epoch_flow, num_batches = 0.0, 0.0, 0

        for batch_idx, batch in enumerate(pbar):
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
            bsz = latents.shape[0]

            # Logit-normal timestep (Wan paper)
            u = torch.randn(bsz, device=device)
            timesteps = torch.sigmoid(u)

            # Flow Matching
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

            # Base loss (standard flow matching)
            loss_base = F.mse_loss(model_output, target)

            # Flow loss (GRID paper) with α warmup
            alpha = min(global_step / max(config["flow_loss_warmup_steps"], 1), 1.0) * config["flow_loss_max_weight"]
            loss_flow = compute_flow_loss(
                target, model_output,
                config["grid_rows"], config["grid_cols"], lat_h, lat_w,
            )
            loss = loss_base + alpha * loss_flow

            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_base += loss_base.item()
                epoch_flow += loss_flow.item()
                num_batches += 1

                pbar.set_postfix({
                    "base": f"{epoch_base/num_batches:.4f}",
                    "flow": f"{epoch_flow/num_batches:.4f}",
                    "α": f"{alpha:.3f}",
                    "step": global_step,
                })

                loss_history.append({
                    "step": global_step,
                    "base_loss": loss_base.item(),
                    "flow_loss": loss_flow.item(),
                    "alpha": alpha,
                })

                if global_step % config["save_every"] == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    transformer.save_pretrained(save_path)
                    with open(output_dir / "loss_history.json", "w") as f:
                        json.dump(loss_history, f)
                    logger.info(f"Checkpoint at step {global_step}, α={alpha:.3f}")

                if global_step >= config["max_train_steps"]:
                    break

        if global_step >= config["max_train_steps"]:
            break
        if num_batches > 0:
            logger.info(f"Epoch {epoch+1}: base={epoch_base/num_batches:.4f} flow={epoch_flow/num_batches:.4f}")

    final_path = output_dir / "final"
    transformer.save_pretrained(final_path)
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training complete! Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid GIF-DiT V3 with Flow Loss")
    parser.add_argument("--train-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--cache-dir", default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--flow-weight", type=float, default=0.5)
    parser.add_argument("--flow-warmup", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--no-gc", action="store_true", help="Disable gradient checkpointing (faster, more VRAM)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "train_mode": args.train_mode,
        "model_path": args.model_path,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank,
        "learning_rate": args.lr,
        "max_train_steps": args.max_steps,
        "flow_loss_max_weight": args.flow_weight,
        "flow_loss_warmup_steps": args.flow_warmup,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_checkpointing": not args.no_gc,
        "height": args.height,
        "width": args.width,
    })
    train(config)


if __name__ == "__main__":
    main()
