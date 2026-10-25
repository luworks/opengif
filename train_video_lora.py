import os
"""
方案 B: Wan 视频直出 LoRA 训练。

用 diffusers 预计算的 T5+VAE 缓存训练 LoRA。
推理用 FastVideo (3步 11秒)。

用法:
  python train_video_lora.py --max-steps 5000  # 快速验证
  python train_video_lora.py --max-steps 50000 --lora-rank 128  # 全量
"""

import argparse, gc, json, logging, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from diffusers import WanTransformer3DModel
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = dict(
    model_path=os.environ.get("OPENGIF_MODEL_PATH", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    cache_dir="dataset/fastvideo-train/cache",
    output_dir="output/video-lora-gif",
    lora_rank=128,
    learning_rate=1e-4,
    batch_size=1,
    gradient_accumulation_steps=1,
    max_train_steps=5000,
    save_every=1000,
    flow_loss_weight=0.3,
    flow_loss_warmup=1000,
    seed=42,
    num_workers=4,
)


class CachedVideoDataset(Dataset):
    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        latent_dir = cache_dir / "vae_latents"
        text_dir = cache_dir / "text_embeddings"
        lat_stems = {p.stem for p in latent_dir.glob("*.pt")}
        txt_stems = {p.stem for p in text_dir.glob("*.pt")}
        self.stems = sorted(lat_stems & txt_stems)
        self.latent_dir = latent_dir
        self.text_dir = text_dir
        logger.info(f"Dataset: {len(self.stems)} video samples")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        latents = torch.load(self.latent_dir / f"{stem}.pt", weights_only=True)
        text = torch.load(self.text_dir / f"{stem}.pt", weights_only=True)
        return {"latents": latents, "prompt_embeds": text["prompt_embeds"]}


def train(config):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda")

    logger.info("Loading Wan2.1 DiT...")
    transformer = WanTransformer3DModel.from_pretrained(
        config["model_path"], subfolder="transformer", torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_rank"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(device)
    transformer.enable_gradient_checkpointing()
    logger.info("Gradient checkpointing: ON")

    dataset = CachedVideoDataset(config["cache_dir"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                            num_workers=config["num_workers"], pin_memory=True)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=1e-3)

    total_steps = min(config["max_train_steps"], len(dataloader) * 20)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    loss_history = []

    logger.info(f"Training: {total_steps} steps, rank={config['lora_rank']}")

    transformer.train()
    for epoch in range(100):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)

            # Logit-normal timestep
            bsz = latents.shape[0]
            u = torch.randn(bsz, device=device)
            timesteps = torch.sigmoid(u)

            # Flow Matching
            noise = torch.randn_like(latents)
            t = timesteps.to(dtype=latents.dtype).reshape(-1, 1, 1, 1, 1)
            noisy = t * noise + (1 - t) * latents
            target = noise - latents

            output = transformer(
                hidden_states=noisy,
                timestep=timesteps * 1000,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            loss = F.mse_loss(output, target)

            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (global_step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            loss_history.append({"step": global_step, "loss": loss.item()})
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            if global_step % config["save_every"] == 0:
                transformer.save_pretrained(output_dir / f"checkpoint-{global_step}")
                with open(output_dir / "loss_history.json", "w") as f:
                    json.dump(loss_history, f)
                logger.info(f"Checkpoint {global_step}")

            if global_step >= config["max_train_steps"]:
                break
        if global_step >= config["max_train_steps"]:
            break

    transformer.save_pretrained(output_dir / "final")
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)
    logger.info(f"Done! {output_dir}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="output/video-lora-gif")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "max_train_steps": args.max_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
    })
    train(config)
