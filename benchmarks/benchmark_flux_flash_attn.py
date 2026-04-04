#!/usr/bin/env python3
"""
Benchmark: MLX Flux2 Klein with FlashAttention-Metal v18
=========================================================

Draw Things achieves 43-120% speedup via custom Metal FlashAttention.
Our v18 kernel hits 19.12 TFLOPS on M3 Ultra.
This benchmark tests the REAL impact on Flux2 Klein end-to-end.
"""

import time
import sys
import os
import numpy as np
import torch

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.utils

sys.path.insert(0, "/Users/baisampayansaha/Desktop/AI/mlx-media-gen")

MODEL_DIR = "/Users/baisampayansaha/Desktop/AI/mlx-media-gen/models/FLUX2-klein-9B"
BATCH = 1
IMG_SEQ = 4096
TXT_SEQ = 512

# ── Load weights ───────────────────────────────────────────────
print("Loading Flux2 Klein weights...")
from safetensors import safe_open

weights_path = f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors"
weights = {}
with safe_open(weights_path, framework="pt") as f:
    for key in f.keys():
        weights[key] = mx.array(f.get_tensor(key).to(torch.float32).numpy())

# ── Prepare inputs ─────────────────────────────────────────────
np.random.seed(42)
x = mx.array(np.random.randn(BATCH, IMG_SEQ, 128).astype(np.float32)).astype(mx.bfloat16)
ctx = mx.array(np.random.randn(BATCH, TXT_SEQ, 7680).astype(np.float32)).astype(mx.bfloat16)
t = mx.array([0.5]).astype(mx.bfloat16)

img_ids_np = np.zeros((IMG_SEQ, 4), dtype=np.float32)
img_ids_np[:, 1] = np.repeat(np.arange(64), 64)
img_ids_np[:, 2] = np.tile(np.arange(64), 64)
img_ids = mx.array(img_ids_np).astype(mx.bfloat16)

txt_ids_np = np.zeros((TXT_SEQ, 4), dtype=np.float32)
txt_ids_np[:, 3] = np.arange(TXT_SEQ)
txt_ids = mx.array(txt_ids_np).astype(mx.bfloat16)


def build_and_benchmark(label, use_custom_attn=False, N=5):
    """Build model, load weights, benchmark."""
    from mlx_media_gen.models.flux2_dit import (
        Flux2Config, Flux2Transformer, set_flux_attention_backend
    )

    # Set attention backend BEFORE building model
    if use_custom_attn:
        set_flux_attention_backend("custom")
    else:
        set_flux_attention_backend("mlx")

    config = Flux2Config()
    model = Flux2Transformer(config)
    model.load_weights(list(weights.items()))
    bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(model.parameters())]
    model.load_weights(bf16_params)
    mx.eval(model.parameters())

    # Warmup
    for _ in range(2):
        out = model(x, ctx, t, img_ids, txt_ids)
        mx.eval(out)

    # Benchmark
    times = []
    for i in range(N):
        t0 = time.time()
        out = model(x, ctx, t, img_ids, txt_ids)
        mx.eval(out)
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  {label} run {i+1}: {(t1-t0)*1000:.1f} ms")

    min_t = min(times) * 1000
    avg_t = np.mean(times) * 1000
    print(f"  {label}: min={min_t:.1f}ms, avg={avg_t:.1f}ms")
    return min_t, out


# ── Benchmark 1: MLX SDPA (baseline) ──────────────────────────
print(f"\n{'='*70}")
print("MODE 1: MLX with mx.fast.scaled_dot_product_attention")
print(f"{'='*70}")
sdpa_time, ref_out = build_and_benchmark("MLX SDPA", use_custom_attn=False)

# ── Benchmark 2: Custom FlashAttention-Metal v18 ──────────────
print(f"\n{'='*70}")
print("MODE 2: MLX with FlashAttention-Metal v18 (19.12 TFLOPS)")
print(f"{'='*70}")
flash_time, flash_out = build_and_benchmark("Flash v18", use_custom_attn=True)

# ── Numerical comparison ──────────────────────────────────────
print(f"\n{'='*70}")
print("NUMERICAL QUALITY (FlashAttention vs SDPA)")
print(f"{'='*70}")
ref_np = np.array(ref_out.astype(mx.float32))
flash_np = np.array(flash_out.astype(mx.float32))
diff = np.abs(ref_np - flash_np)
signal = (ref_np**2).mean()
noise = (diff**2).mean()
psnr = 10 * np.log10(signal / max(noise, 1e-10))
cosine = np.dot(ref_np.flatten(), flash_np.flatten()) / (
    np.linalg.norm(ref_np.flatten()) * np.linalg.norm(flash_np.flatten()))
print(f"  Max diff:   {diff.max():.4f}")
print(f"  Mean diff:  {diff.mean():.6f}")
print(f"  PSNR:       {psnr:.1f} dB  {'✓' if psnr > 20 else '✗'}")
print(f"  Cosine sim: {cosine:.6f}  {'✓' if cosine > 0.99 else '✗'}")

# ── Summary ────────────────────────────────────────────────────
PT_MPS = 2402
PT_SDPA_FIX = 1380

print(f"\n{'='*70}")
print("SUMMARY — Per transformer forward pass")
print(f"{'='*70}")
print(f"  PyTorch MPS bf16 (stock):     {PT_MPS:.0f} ms")
print(f"  PyTorch MPS bf16 (SDPA fix):  ~{PT_SDPA_FIX:.0f} ms")
print(f"  MLX + SDPA:                   {sdpa_time:.0f} ms  ({PT_MPS/sdpa_time:.2f}x vs stock PT)")
print(f"  MLX + FlashAttn v18:          {flash_time:.0f} ms  ({PT_MPS/flash_time:.2f}x vs stock PT)")
print(f"  FlashAttn speedup over SDPA:  {sdpa_time/flash_time:.2f}x")

print(f"\n  2-step end-to-end projections:")
for label, t_ms in [
    ("PyTorch stock", PT_MPS),
    ("PyTorch SDPA fix", PT_SDPA_FIX),
    ("MLX + SDPA", sdpa_time),
    ("MLX + FlashAttn v18", flash_time),
]:
    e2e = t_ms * 2 / 1000 + 2.5
    print(f"    {label:25s} {e2e:.1f}s")
