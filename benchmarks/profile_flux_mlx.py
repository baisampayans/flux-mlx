#!/usr/bin/env python3
"""
Profile: Where is time being spent in MLX Flux2 Klein?
======================================================

Break down:
  1. Per-component timing (embeddings, modulation, blocks, output)
  2. Per-operation within a single block (norm, attn, FFN, modulation)
  3. Compare our implementation vs mflux on same weights
  4. Check if transpose/reshape overhead is significant
"""

import time
import sys
import numpy as np
import torch

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.utils

sys.path.insert(0, "/Users/baisampayansaha/Desktop/AI/mlx-media-gen")
from mlx_media_gen.models.flux2_dit import (
    Flux2Config, Flux2Transformer, flux2_pos_embed,
    Flux2Modulation, apply_rotary_emb, _attention,
    set_flux_attention_backend,
)

set_flux_attention_backend("mlx")

MODEL_DIR = "/Users/baisampayansaha/Desktop/AI/mlx-media-gen/models/FLUX2-klein-9B"

# ── Load model ─────────────────────────────────────────────────
print("Loading model...")
from safetensors import safe_open

config = Flux2Config()
model = Flux2Transformer(config)

weights = {}
with safe_open(f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors", framework="pt") as f:
    for key in f.keys():
        weights[key] = mx.array(f.get_tensor(key).to(torch.float32).numpy())
model.load_weights(list(weights.items()))
bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(model.parameters())]
model.load_weights(bf16_params)
mx.eval(model.parameters())

# ── Inputs ─────────────────────────────────────────────────────
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

# ── Warmup ─────────────────────────────────────────────────────
out = model(x, ctx, t, img_ids, txt_ids)
mx.eval(out)

# ── Profile 1: Component-level timing ─────────────────────────
print(f"\n{'='*70}")
print("PROFILE 1: Component-level timing")
print(f"{'='*70}")

import math

def timed(fn, label, reps=3):
    """Time a function, return min time in ms."""
    times = []
    for _ in range(reps):
        t0 = time.time()
        result = fn()
        mx.eval(result) if isinstance(result, mx.array) else [mx.eval(r) for r in result if isinstance(r, mx.array)]
        times.append(time.time() - t0)
    ms = min(times) * 1000
    print(f"  {label:40s} {ms:8.1f} ms")
    return ms, result

# Pre-compute what we need
t_scaled = (t * 1000).astype(mx.bfloat16)

# Timestep embedding
t_emb, temb = timed(lambda: model.time_guidance_embed(t_scaled, None).astype(mx.bfloat16), "Timestep embedding")

# Modulation
t_mod, _ = timed(lambda: (
    model.double_stream_modulation_img(temb),
    model.double_stream_modulation_txt(temb),
    model.single_stream_modulation(temb),
), "Modulation (all 3)")

dmod_img = model.double_stream_modulation_img(temb)
dmod_txt = model.double_stream_modulation_txt(temb)
smod = model.single_stream_modulation(temb)
mx.eval(dmod_img, dmod_txt, smod)

# Input projections
t_xemb, x_emb = timed(lambda: model.x_embedder(x), "x_embedder (128→3072)")
t_cemb, ctx_emb = timed(lambda: model.context_embedder(ctx), "context_embedder (7680→3072)")

# RoPE
t_rope, _ = timed(lambda: flux2_pos_embed(img_ids, config.axes_dims_rope, config.rope_theta), "RoPE (img + txt)")
img_cos, img_sin = flux2_pos_embed(img_ids, config.axes_dims_rope, config.rope_theta)
txt_cos, txt_sin = flux2_pos_embed(txt_ids, config.axes_dims_rope, config.rope_theta)
rope_cos = mx.concatenate([txt_cos, img_cos], axis=0)
rope_sin = mx.concatenate([txt_sin, img_sin], axis=0)
mx.eval(rope_cos, rope_sin)

# Double-stream blocks (one at a time)
x_cur = x_emb
ctx_cur = ctx_emb
mx.eval(x_cur, ctx_cur)

print(f"\n  --- Double-stream blocks (5) ---")
double_total = 0
for i, block in enumerate(model.transformer_blocks):
    t_block, (ctx_cur, x_cur) = timed(
        lambda b=block, x=x_cur, c=ctx_cur: b(x, c, dmod_img, dmod_txt, rope_cos, rope_sin),
        f"  double_block[{i}]"
    )
    double_total += t_block

# Concatenate
x_cur = mx.concatenate([ctx_cur, x_cur], axis=1)
mx.eval(x_cur)

print(f"\n  --- Single-stream blocks (20) ---")
single_total = 0
for i, block in enumerate(model.single_transformer_blocks):
    t_block, x_cur = timed(
        lambda b=block, x=x_cur: b(x, smod, rope_cos, rope_sin),
        f"  single_block[{i}]"
    )
    single_total += t_block

# Output
x_final = x_cur[:, TXT_SEQ:, :]
mx.eval(x_final)
t_out, _ = timed(lambda: model.proj_out(model.norm_out(x_final, temb)), "Output (norm + proj)")

print(f"\n  {'─'*50}")
total = t_emb + t_mod + t_xemb + t_cemb + t_rope + double_total + single_total + t_out
print(f"  {'TOTAL (sum of components)':40s} {total:8.1f} ms")
print(f"  {'Double blocks total':40s} {double_total:8.1f} ms ({double_total/total*100:.0f}%)")
print(f"  {'Single blocks total':40s} {single_total:8.1f} ms ({single_total/total*100:.0f}%)")
print(f"  {'Overhead (emb+mod+proj+rope)':40s} {total-double_total-single_total:8.1f} ms ({(total-double_total-single_total)/total*100:.0f}%)")

# ── Profile 2: Inside a single block ──────────────────────────
print(f"\n{'='*70}")
print("PROFILE 2: Inside a single-stream block")
print(f"{'='*70}")

block = model.single_transformer_blocks[0]
block_x = mx.random.normal((BATCH, TXT_SEQ + IMG_SEQ, config.inner_dim)).astype(mx.bfloat16)
mx.eval(block_x)

(mod_shift, mod_scale, mod_gate), = Flux2Modulation.split(smod, 1)
mx.eval(mod_shift, mod_scale, mod_gate)

# Norm + modulation
t_nm, norm_x = timed(
    lambda: (1 + mod_scale) * block.norm(block_x) + mod_shift,
    "Norm + modulation"
)

# Fused QKV+MLP projection
t_proj, proj = timed(
    lambda: block.attn.to_qkv_mlp_proj(norm_x),
    "Fused QKV+MLP projection"
)

qkv = proj[:, :, :block.attn.inner_dim * 3]
mlp_in = proj[:, :, block.attn.inner_dim * 3:]

# QK split + norm + RoPE
def qk_rope():
    q, k, v = mx.split(qkv, 3, axis=-1)
    B = q.shape[0]
    H, D = block.attn.heads, block.attn.head_dim
    q = q.reshape(B, -1, H, D)
    k = k.reshape(B, -1, H, D)
    v = v.reshape(B, -1, H, D)
    q = block.attn.norm_q(q)
    k = block.attn.norm_k(k)
    q = apply_rotary_emb(q, rope_cos, rope_sin).astype(mx.bfloat16)
    k = apply_rotary_emb(k, rope_cos, rope_sin).astype(mx.bfloat16)
    return q, k, v

t_qkr, (q, k, v) = timed(qk_rope, "QK norm + RoPE")

# Attention
def run_attn():
    qq = q.transpose(0, 2, 1, 3)
    kk = k.transpose(0, 2, 1, 3)
    vv = v.transpose(0, 2, 1, 3)
    return _attention(qq, kk, vv, 1.0 / math.sqrt(128))

t_attn, attn_out = timed(run_attn, "Attention (SDPA)")

# MLP
def run_mlp():
    x1, x2 = mx.split(mlp_in, 2, axis=-1)
    return mlx_nn.silu(x1) * x2

t_mlp, mlp_out = timed(run_mlp, "MLP (SwiGLU)")

# Output projection
def run_out():
    ao = attn_out.transpose(0, 2, 1, 3).reshape(BATCH, -1, block.attn.inner_dim)
    return block.attn.to_out(mx.concatenate([ao, mlp_out], axis=-1))

t_out2, _ = timed(run_out, "Output projection (concat + linear)")

# Residual
t_res, _ = timed(lambda: block_x + mod_gate * _, "Residual + gate")

block_total = t_nm + t_proj + t_qkr + t_attn + t_mlp + t_out2
print(f"\n  {'Block total':40s} {block_total:8.1f} ms")
print(f"  {'Attention':40s} {t_attn:8.1f} ms ({t_attn/block_total*100:.0f}%)")
print(f"  {'Projections (QKV+MLP + out)':40s} {t_proj+t_out2:8.1f} ms ({(t_proj+t_out2)/block_total*100:.0f}%)")
print(f"  {'QK norm + RoPE':40s} {t_qkr:8.1f} ms ({t_qkr/block_total*100:.0f}%)")
print(f"  {'MLP (SwiGLU)':40s} {t_mlp:8.1f} ms ({t_mlp/block_total*100:.0f}%)")
print(f"  {'Norm + modulation':40s} {t_nm:8.1f} ms ({t_nm/block_total*100:.0f}%)")

# ── Profile 3: Theoretical minimum ────────────────────────────
print(f"\n{'='*70}")
print("PROFILE 3: Theoretical analysis")
print(f"{'='*70}")

# Single-stream block FLOPs
dim = 3072
mlp_dim = int(dim * 3.0)  # 9216
seq = TXT_SEQ + IMG_SEQ  # 4608

# QKV+MLP fused: (4608, 3072) × (3072, 3*3072+2*9216) = (3072, 27648)
qkv_flops = 2 * seq * dim * (3 * dim + 2 * mlp_dim)
# Attention: 2 × seq² × D × H (QK^T) + 2 × seq² × D × H (attn@V)
attn_flops = 2 * 2 * seq * seq * dim
# Output: (4608, 3072+9216) × (12288, 3072)
out_flops = 2 * seq * (dim + mlp_dim) * dim

block_flops = qkv_flops + attn_flops + out_flops
total_flops = block_flops * 25  # 20 single + ~5 double (rough)

m3_ultra_peak = 27e12  # 27 TFLOPS bf16
theoretical_min = total_flops / m3_ultra_peak * 1000  # ms

print(f"  Per single-stream block: {block_flops/1e9:.1f} GFLOPS")
print(f"    QKV+MLP proj: {qkv_flops/1e9:.1f} GFLOPS ({qkv_flops/block_flops*100:.0f}%)")
print(f"    Attention:     {attn_flops/1e9:.1f} GFLOPS ({attn_flops/block_flops*100:.0f}%)")
print(f"    Output proj:   {out_flops/1e9:.1f} GFLOPS ({out_flops/block_flops*100:.0f}%)")
print(f"  Total (25 blocks): {total_flops/1e12:.1f} TFLOPS")
print(f"  M3 Ultra peak (bf16): {m3_ultra_peak/1e12:.0f} TFLOPS")
print(f"  Theoretical minimum: {theoretical_min:.0f} ms")
print(f"  Our measured: ~1575 ms")
print(f"  Compute utilization: {theoretical_min/1575*100:.0f}%")
print(f"  Overhead: {(1575-theoretical_min)/1575*100:.0f}%")
