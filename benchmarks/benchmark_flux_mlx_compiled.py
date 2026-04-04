#!/usr/bin/env python3
"""
Benchmark: MLX Flux2 Klein with mx.compile vs PyTorch MPS
==========================================================

The key insight: PyTorch MPS has 66% dispatch overhead (34% compute utilization).
CoreML eliminates this with graph fusion → 2.17x speedup.
mx.compile should achieve similar fusion, WITH bf16 support.
"""

import time
import sys
import numpy as np

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.utils

sys.path.insert(0, "/Users/baisampayansaha/Desktop/AI/mlx-media-gen")
from mlx_media_gen.models.flux2_dit import Flux2Config, Flux2Transformer

MODEL_DIR = "/Users/baisampayansaha/Desktop/AI/mlx-media-gen/models/FLUX2-klein-9B"
BATCH = 1
IMG_SEQ = 4096
TXT_SEQ = 512

# ── Load model ─────────────────────────────────────────────────
print("Loading MLX Flux2 Klein transformer...")
from safetensors import safe_open

config = Flux2Config()
model = Flux2Transformer(config)

weights_path = f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors"
weights = {}
import torch
with safe_open(weights_path, framework="pt") as f:
    for key in f.keys():
        weights[key] = mx.array(f.get_tensor(key).to(torch.float32).numpy())

model.load_weights(list(weights.items()))

# Cast to bf16
bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(model.parameters())]
model.load_weights(bf16_params)
mx.eval(model.parameters())

param_count = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
print(f"  {param_count/1e9:.2f}B params loaded (bf16)")

# ── Prepare inputs ─────────────────────────────────────────────
np.random.seed(42)
x = mx.array(np.random.randn(BATCH, IMG_SEQ, 128).astype(np.float32)).astype(mx.bfloat16)
ctx = mx.array(np.random.randn(BATCH, TXT_SEQ, 7680).astype(np.float32)).astype(mx.bfloat16)
t = mx.array([0.5]).astype(mx.bfloat16)

img_ids = mx.zeros((IMG_SEQ, 4), dtype=mx.bfloat16)
h = mx.repeat(mx.arange(64), 64).astype(mx.bfloat16)
w = mx.tile(mx.arange(64), 64).astype(mx.bfloat16)
img_ids = img_ids.at[:, 1].add(h[:, None].squeeze(-1))
# Build img_ids properly
img_ids_np = np.zeros((IMG_SEQ, 4), dtype=np.float32)
img_ids_np[:, 1] = np.repeat(np.arange(64), 64)
img_ids_np[:, 2] = np.tile(np.arange(64), 64)
img_ids = mx.array(img_ids_np).astype(mx.bfloat16)

txt_ids_np = np.zeros((TXT_SEQ, 4), dtype=np.float32)
txt_ids_np[:, 3] = np.arange(TXT_SEQ)
txt_ids = mx.array(txt_ids_np).astype(mx.bfloat16)

# ── Benchmark 1: Eager mode (no compile) ──────────────────────
print(f"\n{'='*70}")
print("MODE 1: MLX Eager (no compile)")
print(f"{'='*70}")

# Warmup
for _ in range(2):
    out = model(x, ctx, t, img_ids, txt_ids)
    mx.eval(out)

N = 5
eager_times = []
for i in range(N):
    t0 = time.time()
    out = model(x, ctx, t, img_ids, txt_ids)
    mx.eval(out)
    t1 = time.time()
    eager_times.append(t1 - t0)
    print(f"  Run {i+1}: {(t1-t0)*1000:.1f} ms")

eager_min = min(eager_times) * 1000
print(f"  Min: {eager_min:.1f} ms")

# ── Benchmark 2: mx.compile ───────────────────────────────────
print(f"\n{'='*70}")
print("MODE 2: MLX Compiled (mx.compile)")
print(f"{'='*70}")

# Compile the forward pass
compiled_forward = mx.compile(model.__call__)

# Warmup (first call triggers compilation)
print("  Compiling...")
for _ in range(3):
    out = compiled_forward(x, ctx, t, img_ids, txt_ids)
    mx.eval(out)

compiled_times = []
for i in range(N):
    t0 = time.time()
    out = compiled_forward(x, ctx, t, img_ids, txt_ids)
    mx.eval(out)
    t1 = time.time()
    compiled_times.append(t1 - t0)
    print(f"  Run {i+1}: {(t1-t0)*1000:.1f} ms")

compiled_min = min(compiled_times) * 1000
print(f"  Min: {compiled_min:.1f} ms")

# ── Benchmark 3: Try compiling individual blocks ──────────────
print(f"\n{'='*70}")
print("MODE 3: Per-block compile analysis")
print(f"{'='*70}")

# Time a single-stream block (80% of compute)
single_block = model.single_transformer_blocks[0]

# Prepare block inputs
if True:
    temb = model.time_guidance_embed(t * 1000, None)
    smod = model.single_stream_modulation(temb)

    from mlx_media_gen.models.flux2_dit import flux2_pos_embed
    img_cos, img_sin = flux2_pos_embed(img_ids, config.axes_dims_rope, config.rope_theta)
    txt_cos, txt_sin = flux2_pos_embed(txt_ids, config.axes_dims_rope, config.rope_theta)
    rope_cos = mx.concatenate([txt_cos, img_cos], axis=0)
    rope_sin = mx.concatenate([txt_sin, img_sin], axis=0)

    # Full sequence (text + image concatenated for single-stream)
    block_x = mx.random.normal((BATCH, TXT_SEQ + IMG_SEQ, config.inner_dim)).astype(mx.bfloat16)
    mx.eval(smod, rope_cos, rope_sin, block_x)

# Eager single block
for _ in range(3):
    out = single_block(block_x, smod, rope_cos, rope_sin)
    mx.eval(out)

block_eager_times = []
for i in range(10):
    t0 = time.time()
    out = single_block(block_x, smod, rope_cos, rope_sin)
    mx.eval(out)
    block_eager_times.append(time.time() - t0)

block_eager_min = min(block_eager_times) * 1000

# Compiled single block
compiled_block = mx.compile(single_block.__call__)
for _ in range(3):
    out = compiled_block(block_x, smod, rope_cos, rope_sin)
    mx.eval(out)

block_compiled_times = []
for i in range(10):
    t0 = time.time()
    out = compiled_block(block_x, smod, rope_cos, rope_sin)
    mx.eval(out)
    block_compiled_times.append(time.time() - t0)

block_compiled_min = min(block_compiled_times) * 1000

print(f"  Single-stream block (eager):    {block_eager_min:.1f} ms")
print(f"  Single-stream block (compiled): {block_compiled_min:.1f} ms")
print(f"  Block compile speedup:          {block_eager_min/block_compiled_min:.2f}x")

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

# PyTorch reference from previous benchmark
PT_MIN = 2402  # ms, from benchmark_flux_mlx.py
PT_SDPA_FIX = 1380  # ms estimated (5.56s - 2.8s overhead) / 2 steps

print(f"\n  Per transformer forward pass:")
print(f"    PyTorch MPS bf16 (no fix):    {PT_MIN:.0f} ms  (baseline)")
print(f"    PyTorch MPS bf16 (SDPA fix):  ~{PT_SDPA_FIX:.0f} ms  (estimated)")
print(f"    MLX eager:                    {eager_min:.0f} ms  ({PT_MIN/eager_min:.2f}x vs unpatched PT)")
print(f"    MLX compiled:                 {compiled_min:.0f} ms  ({PT_MIN/compiled_min:.2f}x vs unpatched PT)")

print(f"\n  2-step end-to-end (+ 2.3s text + 0.5s VAE):")
pt_e2e = PT_MIN * 2 / 1000 + 2.8
pt_fix_e2e = PT_SDPA_FIX * 2 / 1000 + 2.8
mlx_eager_e2e = eager_min * 2 / 1000 + 2.8
mlx_compiled_e2e = compiled_min * 2 / 1000 + 2.8
print(f"    PyTorch (no fix):    {pt_e2e:.1f}s")
print(f"    PyTorch (SDPA fix):  ~{pt_fix_e2e:.1f}s")
print(f"    MLX eager:           {mlx_eager_e2e:.1f}s")
print(f"    MLX compiled:        {mlx_compiled_e2e:.1f}s")

print(f"\n  Full MLX pipeline (projected, text+VAE also in MLX):")
mlx_full = compiled_min * 2 / 1000 + 1.5 + 0.3  # MLX text encoder + VAE faster
print(f"    MLX full pipeline:   ~{mlx_full:.1f}s")
print(f"    vs PyTorch SDPA fix: {pt_fix_e2e/mlx_full:.2f}x speedup")
