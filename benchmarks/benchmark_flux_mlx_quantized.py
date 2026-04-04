#!/usr/bin/env python3
"""
Benchmark: MLX Flux2 Klein with quantization (4-bit, 8-bit)
============================================================

Key hypothesis: MLX's native nn.quantize() uses fused dequant+matmul Metal kernels
that are fundamentally faster than bf16 matmul, unlike PyTorch MPS where
INT4/INT8/FP16 all performed the same (~80-84s/step on Wan 14B).
"""

import time
import sys
import numpy as np

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.utils
import torch

sys.path.insert(0, "/Users/baisampayansaha/Desktop/AI/mlx-media-gen")
from mlx_media_gen.models.flux2_dit import Flux2Config, Flux2Transformer

MODEL_DIR = "/Users/baisampayansaha/Desktop/AI/mlx-media-gen/models/FLUX2-klein-9B"

# ── Load model ─────────────────────────────────────────────────
print("Loading MLX Flux2 Klein transformer...")
from safetensors import safe_open

config = Flux2Config()

weights_path = f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors"
weights = {}
with safe_open(weights_path, framework="pt") as f:
    for key in f.keys():
        weights[key] = mx.array(f.get_tensor(key).to(torch.float32).numpy())

# ── Prepare inputs ─────────────────────────────────────────────
BATCH = 1
IMG_SEQ = 4096
TXT_SEQ = 512

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


def benchmark_model(model, label, N=5, ref_output=None):
    """Run benchmark and optionally compare to reference output."""
    # Warmup
    for _ in range(2):
        out = model(x, ctx, t, img_ids, txt_ids)
        mx.eval(out)

    times = []
    for i in range(N):
        t0 = time.time()
        out = model(x, ctx, t, img_ids, txt_ids)
        mx.eval(out)
        t1 = time.time()
        times.append(t1 - t0)

    min_t = min(times) * 1000
    avg_t = np.mean(times) * 1000
    print(f"  {label}: min={min_t:.1f}ms, avg={avg_t:.1f}ms")

    # Memory
    param_bytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(model.parameters()))
    print(f"    Model size: {param_bytes/1e9:.2f} GB")

    # Numerical quality vs reference
    if ref_output is not None:
        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref_output.astype(mx.float32))
        diff = np.abs(out_np - ref_np)
        signal = (ref_np ** 2).mean()
        noise = (diff ** 2).mean()
        psnr = 10 * np.log10(signal / max(noise, 1e-10))
        cosine = np.dot(out_np.flatten(), ref_np.flatten()) / (
            np.linalg.norm(out_np.flatten()) * np.linalg.norm(ref_np.flatten())
        )
        print(f"    PSNR: {psnr:.1f} dB, Cosine: {cosine:.6f}")

    return min_t, out


# ── bf16 baseline ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("BENCHMARK: bf16 vs INT8 vs INT4 quantization")
print(f"{'='*70}\n")

# Build bf16 model
model_bf16 = Flux2Transformer(config)
model_bf16.load_weights(list(weights.items()))
bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(model_bf16.parameters())]
model_bf16.load_weights(bf16_params)
mx.eval(model_bf16.parameters())

bf16_time, ref_output = benchmark_model(model_bf16, "bf16 (baseline)")

# ── INT8 quantization ─────────────────────────────────────────
print()
model_int8 = Flux2Transformer(config)
model_int8.load_weights(list(weights.items()))
# Quantize linear layers to 8-bit
mlx_nn.quantize(model_int8, bits=8)
mx.eval(model_int8.parameters())

int8_time, _ = benchmark_model(model_int8, "INT8 quantized", ref_output=ref_output)

# ── INT4 quantization ─────────────────────────────────────────
print()
model_int4 = Flux2Transformer(config)
model_int4.load_weights(list(weights.items()))
mlx_nn.quantize(model_int4, bits=4)
mx.eval(model_int4.parameters())

int4_time, _ = benchmark_model(model_int4, "INT4 quantized", ref_output=ref_output)

# ── INT4 + mx.compile ─────────────────────────────────────────
print()
model_int4_c = Flux2Transformer(config)
model_int4_c.load_weights(list(weights.items()))
mlx_nn.quantize(model_int4_c, bits=4)
mx.eval(model_int4_c.parameters())

compiled_forward = mx.compile(model_int4_c.__call__)
# Warmup compiled
for _ in range(3):
    out = compiled_forward(x, ctx, t, img_ids, txt_ids)
    mx.eval(out)

times = []
for i in range(5):
    t0 = time.time()
    out = compiled_forward(x, ctx, t, img_ids, txt_ids)
    mx.eval(out)
    times.append(time.time() - t0)

int4c_time = min(times) * 1000
param_bytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(model_int4_c.parameters()))
out_np = np.array(out.astype(mx.float32))
ref_np = np.array(ref_output.astype(mx.float32))
diff = np.abs(out_np - ref_np)
psnr = 10 * np.log10((ref_np**2).mean() / max((diff**2).mean(), 1e-10))
cosine = np.dot(out_np.flatten(), ref_np.flatten()) / (
    np.linalg.norm(out_np.flatten()) * np.linalg.norm(ref_np.flatten()))
print(f"  INT4 + mx.compile: min={int4c_time:.1f}ms, avg={np.mean(times)*1000:.1f}ms")
print(f"    Model size: {param_bytes/1e9:.2f} GB")
print(f"    PSNR: {psnr:.1f} dB, Cosine: {cosine:.6f}")

# ── Summary ────────────────────────────────────────────────────
PT_MPS = 2402  # from our benchmark
PT_SDPA_FIX = 1380  # estimated

print(f"\n{'='*70}")
print("SUMMARY — Per transformer forward pass")
print(f"{'='*70}")
print(f"  PyTorch MPS bf16 (no fix):    {PT_MPS:.0f} ms  (baseline)")
print(f"  PyTorch MPS bf16 (SDPA fix):  ~{PT_SDPA_FIX:.0f} ms")
print(f"  MLX bf16:                     {bf16_time:.0f} ms  ({PT_MPS/bf16_time:.2f}x vs PT)")
print(f"  MLX INT8:                     {int8_time:.0f} ms  ({PT_MPS/int8_time:.2f}x vs PT)")
print(f"  MLX INT4:                     {int4_time:.0f} ms  ({PT_MPS/int4_time:.2f}x vs PT)")
print(f"  MLX INT4 + compile:           {int4c_time:.0f} ms  ({PT_MPS/int4c_time:.2f}x vs PT)")

print(f"\n  2-step end-to-end (+ ~2s text + 0.5s VAE in MLX):")
for label, t_ms in [("PyTorch (SDPA fix)", PT_SDPA_FIX),
                     ("MLX bf16", bf16_time),
                     ("MLX INT8", int8_time),
                     ("MLX INT4", int4_time),
                     ("MLX INT4+compile", int4c_time)]:
    e2e = t_ms * 2 / 1000 + 2.5
    print(f"    {label:25s} {e2e:.1f}s")
