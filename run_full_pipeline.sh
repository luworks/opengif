#!/bin/bash
# 全量训练 Pipeline: T5完成后 → VAE → 训练 → Benchmark
set -e

MODEL_PATH="${OPENGIF_MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
CACHE_DIR="dataset/gif-grids/cache"

echo "=== Phase 1: VAE 预计算 100K ==="
.venv/bin/python -c "
import os, gc, torch, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKLWan

model_path = '$MODEL_PATH'
grid_dir = Path('dataset/gif-grids/train')
cache_dir = Path('$CACHE_DIR/vae_latents')
cache_dir.mkdir(parents=True, exist_ok=True)
text_cache = Path('$CACHE_DIR/text_embeddings')

existing = {p.stem for p in cache_dir.glob('*.pt')}
t5_stems = {p.stem for p in text_cache.glob('*.pt')}
to_encode = sorted(t5_stems - existing)
print(f'VAE 编码: {len(to_encode)} (已有: {len(existing)})')

if not to_encode:
    print('全部已缓存!')
    exit()

vae = AutoencoderKLWan.from_pretrained(model_path, subfolder='vae', torch_dtype=torch.float32).cuda()
vae.eval()
latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).cuda()
latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).cuda()

for stem in tqdm(to_encode, desc='VAE encoding'):
    png_path = grid_dir / f'{stem}.png'
    if not png_path.exists(): continue
    out_path = cache_dir / f'{stem}.pt'
    img = Image.open(png_path).convert('RGB').resize((512, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    pixel_values = torch.from_numpy(arr).permute(2, 0, 1)
    with torch.no_grad():
        x = pixel_values.unsqueeze(0).unsqueeze(2).cuda()
        latent_dist = vae.encode(x).latent_dist
        latents = latent_dist.sample()
        latents = (latents - latents_mean) / latents_std
        torch.save(latents.squeeze(0).cpu().to(torch.float16), out_path)

del vae; gc.collect(); torch.cuda.empty_cache()
print(f'VAE 完成! 总: {len(list(cache_dir.glob(\"*.pt\")))}')
"

echo ""
echo "=== Phase 2: V3 全量训练 (Flow Loss) ==="
.venv/bin/python train_grid_v3.py \
    --max-steps 50000 \
    --flow-weight 0.5 \
    --flow-warmup 5000 \
    --lr 1e-4 \
    --output-dir output/grid-v3-100k \
    2>&1 | tee train_v3_100k.log

echo ""
echo "=== Phase 3: Benchmark ==="
.venv/bin/python benchmark.py \
    --lora-path output/grid-v3-100k/final \
    --name v3-100k \
    --steps 30

echo ""
echo "=== Phase 4: 推理步数消融 ==="
.venv/bin/python ablation_steps.py \
    --lora-path output/grid-v3-100k/final

echo "全部完成!"
