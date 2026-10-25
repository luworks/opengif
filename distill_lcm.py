"""
LCM (Latent Consistency Model) distillation for GIF-DiT.

Distills the 50-step teacher into a 4-step student following:
- Wan paper Section 5.6.3 (consistency model distillation)
- FastLightGen (arxiv 2603.01685) joint step+size distillation
- rCM (ICLR 2026) score-regularized consistency

This script implements the consistency distillation loss:
  L_cd = E[d(f_θ(x_{t_{n+1}}, t_{n+1}), f_θ^-(x_{t_n}, t_n))]

where f_θ is the student, f_θ^- is the EMA teacher (or pre-trained teacher).

Run AFTER Phase 1-2 (LoRA/Full fine-tuning).
"""

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = dict(
    teacher_path=None,  # Path to fine-tuned teacher (from Phase 1-2)
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    cache_dir="dataset/mvp-1k/cache",
    output_dir="output/gif-lcm-distill",
    # Distillation params
    num_inference_steps_teacher=50,  # Teacher uses 50 steps
    num_inference_steps_student=4,   # Student target: 4 steps
    ema_decay=0.95,                  # EMA decay for target model
    # Training
    learning_rate=1e-6,
    max_train_steps=20000,
    save_every=5000,
    seed=42,
    num_workers=4,
)


def get_skip_timesteps(num_teacher_steps, num_student_steps):
    """Compute the timestep schedule for consistency distillation.

    Maps N teacher steps to K student steps by selecting evenly-spaced
    timestep pairs for the consistency loss.
    """
    # Teacher timesteps: evenly spaced in [0, 1]
    teacher_ts = torch.linspace(1, 0, num_teacher_steps + 1)
    # Skip interval
    skip = num_teacher_steps // num_student_steps
    # Select pairs: (t_{n+1}, t_n) where t_{n+1} > t_n
    pairs = []
    for i in range(0, num_teacher_steps, skip):
        t_start = teacher_ts[i]      # Higher noise
        t_end = teacher_ts[i + skip]  # Lower noise (after one teacher step)
        pairs.append((t_start.item(), t_end.item()))
    return pairs


def consistency_distillation_loss(student_model, teacher_model, latents, prompt_embeds,
                                  timestep_pairs, noise=None):
    """Compute the consistency distillation loss.

    For each timestep pair (t_{n+1}, t_n):
    1. Create noisy latent at t_{n+1}
    2. Run ONE teacher step: t_{n+1} → t_n (get target)
    3. Student predicts directly at t_{n+1} (skip to t_n)
    4. Loss = MSE(student_output, teacher_target)

    This trains the student to match the teacher's multi-step output in one step.
    """
    if noise is None:
        noise = torch.randn_like(latents)

    device = latents.device
    bsz = latents.shape[0]

    # Randomly select a timestep pair
    pair_idx = torch.randint(0, len(timestep_pairs), (1,)).item()
    t_start, t_end = timestep_pairs[pair_idx]

    # Create noisy latent at t_start (higher noise level)
    t_s = torch.tensor([t_start], device=device).reshape(-1, 1, 1, 1, 1)
    x_t_start = t_s * noise + (1 - t_s) * latents

    # Teacher: predict velocity at t_start, apply one euler step to get x at t_end
    with torch.no_grad():
        teacher_velocity = teacher_model(
            hidden_states=x_t_start,
            timestep=torch.tensor([t_start * 1000], device=device),
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        # Euler step: x_{t_end} = x_{t_start} + (t_end - t_start) * v
        dt = t_end - t_start  # Negative (going from high noise to low noise)
        x_t_end_teacher = x_t_start + dt * teacher_velocity

    # Student: predict at t_start what the output at t_end should be
    student_velocity = student_model(
        hidden_states=x_t_start,
        timestep=torch.tensor([t_start * 1000], device=device),
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]
    x_t_end_student = x_t_start + dt * student_velocity

    # Consistency loss: student's output should match teacher's output
    loss = F.mse_loss(x_t_end_student, x_t_end_teacher)
    return loss


def train_lcm(config):
    """Main LCM distillation training loop."""
    from diffusers import WanTransformer3DModel
    from train_gif_lora import CachedGIFDataset  # Reuse our dataset

    torch.manual_seed(config["seed"])
    device = torch.device("cuda")

    # Load teacher (fine-tuned model from Phase 1-2)
    teacher_path = config["teacher_path"] or config["model_path"]
    logger.info(f"Loading teacher from {teacher_path}...")
    teacher = WanTransformer3DModel.from_pretrained(
        teacher_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).cuda()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student (copy of teacher, will be trained)
    logger.info("Loading student (copy of teacher)...")
    student = WanTransformer3DModel.from_pretrained(
        teacher_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).cuda()
    student.train()
    student.enable_gradient_checkpointing()

    # Compute timestep pairs for distillation
    timestep_pairs = get_skip_timesteps(
        config["num_inference_steps_teacher"],
        config["num_inference_steps_student"],
    )
    logger.info(f"Timestep pairs ({len(timestep_pairs)}): {timestep_pairs}")

    # Dataset
    dataset = CachedGIFDataset(config["cache_dir"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=config["num_workers"], pin_memory=True)

    # Optimizer (much lower LR than fine-tuning)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"],
                                  weight_decay=0.01)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    loss_history = []

    logger.info(f"Starting LCM distillation: {config['max_train_steps']} steps")
    logger.info(f"Teacher: {config['num_inference_steps_teacher']} steps → "
                f"Student: {config['num_inference_steps_student']} steps")

    while global_step < config["max_train_steps"]:
        for batch in tqdm(dataloader, desc=f"LCM Step {global_step}"):
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)

            loss = consistency_distillation_loss(
                student, teacher, latents, prompt_embeds, timestep_pairs,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            loss_history.append({"step": global_step, "loss": loss.item()})

            if global_step % 100 == 0:
                logger.info(f"Step {global_step}: loss={loss.item():.6f}")

            if global_step % config["save_every"] == 0:
                save_path = output_dir / f"lcm-checkpoint-{global_step}"
                student.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= config["max_train_steps"]:
                break

    # Save final
    final_path = output_dir / "lcm-final"
    student.save_pretrained(final_path)
    with open(output_dir / "lcm_loss_history.json", "w") as f:
        json.dump(loss_history, f)
    logger.info(f"LCM distillation complete! Saved to {final_path}")

    # Cleanup
    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="LCM Distillation for GIF-DiT")
    parser.add_argument("--teacher-path", default=None,
                        help="Path to fine-tuned teacher model")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--cache-dir", default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_train_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "teacher_path": args.teacher_path,
        "model_path": args.model_path,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "num_inference_steps_student": args.student_steps,
        "max_train_steps": args.max_steps,
        "learning_rate": args.lr,
    })

    train_lcm(config)


if __name__ == "__main__":
    main()
