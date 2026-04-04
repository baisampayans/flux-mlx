#!/usr/bin/env python3
"""
Benchmark: MLX Flux2 Klein transformer vs PyTorch MPS
=====================================================

Loads the real Flux2 Klein weights into both MLX and PyTorch,
runs the transformer with identical inputs, and compares:
  1. Numerical accuracy (MLX vs PyTorch)
  2. Performance (latency per forward pass)
"""

import time
import sys
import numpy as np
import torch

import mlx.core as mx
import mlx.nn as mlx_nn

sys.path.insert(0, "/Users/baisampayansaha/Desktop/AI/mlx-media-gen")
from mlx_media_gen.models.flux2_dit import (
    Flux2Config, Flux2Transformer, flux2_pos_embed, get_timestep_embedding,
)

MODEL_DIR = "/Users/baisampayansaha/Desktop/AI/mlx-media-gen/models/FLUX2-klein-9B"
DEVICE = "mps"
BATCH = 1
IMG_SEQ = 4096   # 64×64 latent grid for 1024×1024
TXT_SEQ = 512
IN_CH = 128
JOINT_DIM = 7680

# ── Step 1: Load weights from safetensors ──────────────────────
print("Loading transformer weights...")
from safetensors import safe_open

weights_path = f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors"
t0 = time.time()

# Load into a flat dict
pt_state = {}
with safe_open(weights_path, framework="pt") as f:
    for key in f.keys():
        pt_state[key] = f.get_tensor(key)

print(f"  Loaded {len(pt_state)} tensors in {time.time()-t0:.1f}s")

# ── Step 2: Convert weights to MLX format ──────────────────────
print("\nConverting weights to MLX...")

def convert_key(key: str) -> str:
    """Map diffusers key to MLX key.

    Diffusers uses nn.ModuleList with `.N.` indexing and `to_out.0` for the first linear.
    MLX uses Python lists, which map the same way with tree_unflatten.
    """
    # to_out is a ModuleList in diffusers but a list in MLX
    # diffusers: attn.to_out.0.weight -> MLX: attn.to_out.0.weight (same!)
    return key


mlx_weights = {}
for key, tensor in pt_state.items():
    mlx_key = convert_key(key)
    # Convert to numpy then to MLX
    arr = tensor.to(torch.float32).cpu().numpy()
    mlx_weights[mlx_key] = mx.array(arr)

print(f"  Converted {len(mlx_weights)} tensors")

# ── Step 3: Build MLX model and load weights ──────────────────
print("\nBuilding MLX Flux2 transformer...")

config = Flux2Config(
    in_channels=128,
    num_layers=5,
    num_single_layers=20,
    num_attention_heads=24,
    attention_head_dim=128,
    joint_attention_dim=7680,
    mlp_ratio=3.0,
    axes_dims_rope=(32, 32, 32, 32),
    rope_theta=2000,
    eps=1e-6,
    guidance_embeds=False,
)

mlx_model = Flux2Transformer(config)

# Load weights using tree_unflatten
import mlx.utils
weight_list = list(mlx_weights.items())
mlx_model.load_weights(weight_list)
mx.eval(mlx_model.parameters())

param_count = sum(p.size for _, p in mlx.utils.tree_flatten(mlx_model.parameters()))
print(f"  MLX model: {param_count/1e9:.2f}B params loaded")

# ── Step 4: Build PyTorch model ────────────────────────────────
print("\nLoading PyTorch Flux2 Klein...")
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
pt_transformer = pipe.transformer.eval().to(torch.bfloat16).to(DEVICE)

# ── Step 5: Create identical inputs ───────────────────────────
print("\nPreparing inputs...")
np.random.seed(42)

# Random inputs
x_np = np.random.randn(BATCH, IMG_SEQ, IN_CH).astype(np.float32)
ctx_np = np.random.randn(BATCH, TXT_SEQ, JOINT_DIM).astype(np.float32)
t_np = np.array([0.5], dtype=np.float32)

# Position IDs
img_ids_np = np.zeros((IMG_SEQ, 4), dtype=np.float32)
h = np.repeat(np.arange(64), 64).astype(np.float32)
w = np.tile(np.arange(64), 64).astype(np.float32)
img_ids_np[:, 1] = h
img_ids_np[:, 2] = w

txt_ids_np = np.zeros((TXT_SEQ, 4), dtype=np.float32)
txt_ids_np[:, 3] = np.arange(TXT_SEQ).astype(np.float32)

# MLX inputs
x_mx = mx.array(x_np).astype(mx.bfloat16)
ctx_mx = mx.array(ctx_np).astype(mx.bfloat16)
t_mx = mx.array(t_np).astype(mx.bfloat16)
img_ids_mx = mx.array(img_ids_np).astype(mx.bfloat16)
txt_ids_mx = mx.array(txt_ids_np).astype(mx.bfloat16)

# PyTorch inputs
x_pt = torch.from_numpy(x_np).to(torch.bfloat16).to(DEVICE)
ctx_pt = torch.from_numpy(ctx_np).to(torch.bfloat16).to(DEVICE)
t_pt = torch.tensor(t_np, dtype=torch.bfloat16, device=DEVICE)
img_ids_pt = torch.from_numpy(img_ids_np).to(torch.bfloat16).to(DEVICE)
txt_ids_pt = torch.from_numpy(txt_ids_np).to(torch.bfloat16).to(DEVICE)

# ── Step 6: Numerical comparison ──────────────────────────────
print(f"\n{'='*70}")
print("NUMERICAL ACCURACY (MLX vs PyTorch)")
print(f"{'='*70}")

# MLX forward
mlx_model_bf16 = mlx_model
# Cast model to bf16
bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(mlx_model.parameters())]
mlx_model.load_weights(bf16_params)
mx.eval(mlx_model.parameters())

mlx_out = mlx_model(x_mx, ctx_mx, t_mx, img_ids_mx, txt_ids_mx)
mx.eval(mlx_out)
mlx_out_np = np.array(mlx_out.astype(mx.float32))

# PyTorch forward
with torch.no_grad():
    pt_out = pt_transformer(
        hidden_states=x_pt,
        encoder_hidden_states=ctx_pt,
        timestep=t_pt,  # forward() expects [0,1], multiplies by 1000 internally
        img_ids=img_ids_pt[None],  # add batch dim
        txt_ids=txt_ids_pt[None],
        return_dict=False,
    )[0]
    torch.mps.synchronize()
    pt_out_np = pt_out.cpu().float().numpy()

# Note: diffusers normalizes timestep differently — it expects raw timestep
# and multiplies by 1000 internally. We pass 0.5 and it becomes 500.
# Our MLX code also multiplies by 1000, so passing 0.5 → 500. Same.

diff = np.abs(mlx_out_np - pt_out_np)
ref_mag = np.abs(pt_out_np).mean()
print(f"  Output shape: MLX={mlx_out_np.shape}, PT={pt_out_np.shape}")
print(f"  Max diff:     {diff.max():.4f}")
print(f"  Mean diff:    {diff.mean():.6f}")
print(f"  Rel error:    {diff.mean()/max(ref_mag, 1e-10)*100:.4f}%")

signal = (pt_out_np**2).mean()
noise = (diff**2).mean()
psnr = 10 * np.log10(signal / max(noise, 1e-10))
print(f"  PSNR:         {psnr:.1f} dB  {'✓' if psnr > 20 else '✗'}")

# Cosine similarity
mlx_flat = mlx_out_np.flatten()
pt_flat = pt_out_np.flatten()
cosine = np.dot(mlx_flat, pt_flat) / (np.linalg.norm(mlx_flat) * np.linalg.norm(pt_flat))
print(f"  Cosine sim:   {cosine:.6f}  {'✓' if cosine > 0.99 else '✗'}")

# ── Step 7: Performance benchmark ─────────────────────────────
print(f"\n{'='*70}")
print("PERFORMANCE BENCHMARK")
print(f"{'='*70}")

# Warmup MLX
for _ in range(2):
    out = mlx_model(x_mx, ctx_mx, t_mx, img_ids_mx, txt_ids_mx)
    mx.eval(out)

N = 5
mlx_times = []
for i in range(N):
    t0 = time.time()
    out = mlx_model(x_mx, ctx_mx, t_mx, img_ids_mx, txt_ids_mx)
    mx.eval(out)
    t1 = time.time()
    mlx_times.append(t1 - t0)
    print(f"  MLX run {i+1}: {(t1-t0)*1000:.1f} ms")

mlx_avg = np.mean(mlx_times) * 1000
mlx_min = min(mlx_times) * 1000

# Warmup PyTorch
with torch.no_grad():
    for _ in range(2):
        _ = pt_transformer(x_pt, ctx_pt, t_pt, img_ids_pt[None], txt_ids_pt[None], return_dict=False)
        torch.mps.synchronize()

pt_times = []
with torch.no_grad():
    for i in range(N):
        torch.mps.synchronize()
        t0 = time.time()
        _ = pt_transformer(x_pt, ctx_pt, t_pt, img_ids_pt[None], txt_ids_pt[None], return_dict=False)
        torch.mps.synchronize()
        t1 = time.time()
        pt_times.append(t1 - t0)
        print(f"  PyTorch run {i+1}: {(t1-t0)*1000:.1f} ms")

pt_avg = np.mean(pt_times) * 1000
pt_min = min(pt_times) * 1000

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  MLX bf16:     {mlx_min:.1f} ms  (min of {N})")
print(f"  PyTorch bf16: {pt_min:.1f} ms  (min of {N})")
print(f"  Speedup:      {pt_min/mlx_min:.2f}x")
print(f"\n  2-step end-to-end projection (+ 2.3s text enc + 0.5s VAE):")
mlx_e2e = mlx_min * 2 / 1000 + 2.8
pt_e2e = pt_min * 2 / 1000 + 2.8
print(f"    MLX:     {mlx_e2e:.1f}s")
print(f"    PyTorch: {pt_e2e:.1f}s")
