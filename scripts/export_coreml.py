#!/usr/bin/env python3
"""
Export Flux2 Klein transformer block to CoreML → .mtlpackage for Metal 4 ML pipeline.

Strategy:
  1. Wrap a single transformer block with pre-computed RoPE/modulation
  2. Export to CoreML mlprogram (bf16 if supported, else fp16)
  3. Convert via metal-package-builder to .mtlpackage
  4. Benchmark Metal 4 dispatch vs MLX compute

We start with ONE single-stream block (the hot path — 80% of compute)
to validate before attempting the full transformer.
"""

import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = "models/FLUX2-klein-9B"
OUTPUT_DIR = "/tmp/flux_metal4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH = 1
SEQ = 4608    # 512 text + 4096 image
HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128

print("Loading Flux2 Klein pipeline...")
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
transformer = pipe.transformer.eval()

# ── Step 1: Prepare a traceable single-stream block ────────────
# The block takes: hidden_states, temb_mod, rope_cos, rope_sin
# We need to flatten all inputs for torch.jit.trace

real_block = transformer.single_transformer_blocks[0].eval()

class TraceableSingleBlock(nn.Module):
    """Wraps Flux2SingleTransformerBlock with flat inputs for tracing."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, temb_mod, rope_cos, rope_sin):
        image_rotary_emb = (rope_cos, rope_sin)
        return self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=temb_mod,
            image_rotary_emb=image_rotary_emb,
        )

traceable = TraceableSingleBlock(real_block).eval()

# Pre-compute RoPE
with torch.no_grad():
    img_ids = torch.zeros(4096, 4)
    img_ids[:, 1] = torch.repeat_interleave(torch.arange(64), 64).float()
    img_ids[:, 2] = torch.tile(torch.arange(64), (64,)).float()
    txt_ids = torch.zeros(512, 4)
    txt_ids[:, 3] = torch.arange(512).float()

    image_rotary = transformer.pos_embed(img_ids)
    text_rotary = transformer.pos_embed(txt_ids)
    rope_cos = torch.cat([text_rotary[0], image_rotary[0]], dim=0)
    rope_sin = torch.cat([text_rotary[1], image_rotary[1]], dim=0)

# Test forward
x = torch.randn(BATCH, SEQ, HIDDEN)
temb_mod = torch.randn(BATCH, HIDDEN * 3)  # single-stream: 1 set of (shift, scale, gate)

with torch.no_grad():
    out = traceable(x, temb_mod, rope_cos, rope_sin)
    print(f"Block output: {out.shape}")

# ── Step 2: Trace for CoreML ──────────────────────────────────
print("\nTracing for CoreML...")
traced = torch.jit.trace(traceable, (x, temb_mod, rope_cos, rope_sin))

# ── Step 3: Convert to CoreML ─────────────────────────────────
print("Converting to CoreML mlprogram...")
import coremltools as ct

t0 = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=(BATCH, SEQ, HIDDEN)),
        ct.TensorType(name="temb_mod", shape=(BATCH, HIDDEN * 3)),
        ct.TensorType(name="rope_cos", shape=(SEQ, HEAD_DIM)),
        ct.TensorType(name="rope_sin", shape=(SEQ, HEAD_DIM)),
    ],
    outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS26,
)
print(f"  Conversion: {time.time()-t0:.1f}s")

mlpackage_path = os.path.join(OUTPUT_DIR, "flux_single_block.mlpackage")
mlmodel.save(mlpackage_path)
print(f"  Saved to {mlpackage_path}")

# ── Step 4: Convert to .mtlpackage ────────────────────────────
print("\nConverting to .mtlpackage via metal-package-builder...")
mtlpackage_path = os.path.join(OUTPUT_DIR, "flux_single_block.mtlpackage")
metal_pb = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal-package-builder"

import subprocess
result = subprocess.run(
    [metal_pb, mlpackage_path, "-o", mtlpackage_path],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"  metal-package-builder FAILED:")
    print(f"  stdout: {result.stdout}")
    print(f"  stderr: {result.stderr}")
else:
    print(f"  Saved .mtlpackage to {mtlpackage_path}")

# ── Step 5: Benchmark CoreML (as proxy for Metal 4 ML pipeline) ─
print(f"\n{'='*60}")
print("Benchmarking CoreML GPU vs PyTorch MPS")
print(f"{'='*60}")

# CoreML benchmark
loaded = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

x_np = np.random.randn(BATCH, SEQ, HIDDEN).astype(np.float16)
mod_np = np.random.randn(BATCH, HIDDEN * 3).astype(np.float16)
cos_np = rope_cos.numpy().astype(np.float16)
sin_np = rope_sin.numpy().astype(np.float16)

cml_inputs = {
    "hidden_states": x_np,
    "temb_mod": mod_np,
    "rope_cos": cos_np,
    "rope_sin": sin_np,
}

# Warmup
for _ in range(5):
    _ = loaded.predict(cml_inputs)

N = 20
cml_times = []
for i in range(N):
    t0 = time.time()
    _ = loaded.predict(cml_inputs)
    cml_times.append(time.time() - t0)

cml_min = min(cml_times) * 1000
cml_avg = np.mean(cml_times) * 1000

# PyTorch MPS bf16 benchmark
block_mps = real_block.to(torch.bfloat16).to("mps")
x_mps = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.bfloat16, device="mps")
mod_mps = torch.randn(BATCH, HIDDEN * 3, dtype=torch.bfloat16, device="mps")
cos_mps = rope_cos.to(torch.bfloat16).to("mps")
sin_mps = rope_sin.to(torch.bfloat16).to("mps")

for _ in range(5):
    with torch.no_grad():
        _ = block_mps(x_mps, None, mod_mps, (cos_mps, sin_mps))
    torch.mps.synchronize()

pt_times = []
for i in range(N):
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = block_mps(x_mps, None, mod_mps, (cos_mps, sin_mps))
    torch.mps.synchronize()
    pt_times.append(time.time() - t0)

pt_min = min(pt_times) * 1000
pt_avg = np.mean(pt_times) * 1000

# MLX benchmark (what we currently use)
import mlx.core as mx
import mlx.utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flux_mlx.models.transformer import Flux2Config, Flux2SingleTransformerBlock, flux2_pos_embed, Flux2Modulation, set_flux_attention_backend
set_flux_attention_backend("mlx")

config = Flux2Config()
mlx_block = Flux2SingleTransformerBlock(config)

# Load weights for block 0
from safetensors import safe_open
weights = {}
with safe_open(f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors", framework="pt") as f:
    prefix = "single_transformer_blocks.0."
    for key in f.keys():
        if key.startswith(prefix):
            short_key = key[len(prefix):]
            weights[short_key] = mx.array(f.get_tensor(key).to(torch.float32).numpy()).astype(mx.bfloat16)

mlx_block.load_weights(list(weights.items()))
mx.eval(mlx_block.parameters())

x_mx = mx.random.normal((BATCH, SEQ, HIDDEN)).astype(mx.bfloat16)
mod_mx = mx.random.normal((BATCH, HIDDEN * 3)).astype(mx.bfloat16)
img_ids_mx = mx.zeros((4096, 4)).astype(mx.bfloat16)
img_ids_np = np.zeros((4096, 4), dtype=np.float32)
img_ids_np[:, 1] = np.repeat(np.arange(64), 64)
img_ids_np[:, 2] = np.tile(np.arange(64), 64)
img_ids_mx = mx.array(img_ids_np).astype(mx.bfloat16)
txt_ids_np = np.zeros((512, 4), dtype=np.float32)
txt_ids_np[:, 3] = np.arange(512)
txt_ids_mx = mx.array(txt_ids_np).astype(mx.bfloat16)
img_cos, img_sin = flux2_pos_embed(img_ids_mx, config.axes_dims_rope, config.rope_theta)
txt_cos, txt_sin = flux2_pos_embed(txt_ids_mx, config.axes_dims_rope, config.rope_theta)
r_cos = mx.concatenate([txt_cos, img_cos], axis=0)
r_sin = mx.concatenate([txt_sin, img_sin], axis=0)
mx.eval(r_cos, r_sin, x_mx, mod_mx)

for _ in range(5):
    out = mlx_block(x_mx, mod_mx, r_cos, r_sin)
    mx.eval(out)

mlx_times = []
for i in range(N):
    t0 = time.time()
    out = mlx_block(x_mx, mod_mx, r_cos, r_sin)
    mx.eval(out)
    mlx_times.append(time.time() - t0)

mlx_min = min(mlx_times) * 1000
mlx_avg = np.mean(mlx_times) * 1000

# Results
print(f"\n  Per single-stream block ({N} runs):")
print(f"    PyTorch MPS bf16:  {pt_min:.1f} ms (avg {pt_avg:.1f})")
print(f"    MLX bf16:          {mlx_min:.1f} ms (avg {mlx_avg:.1f})")
print(f"    CoreML GPU fp16:   {cml_min:.1f} ms (avg {cml_avg:.1f})")
print(f"    CoreML speedup:    {mlx_min/cml_min:.2f}x over MLX")

print(f"\n  25-block projection:")
print(f"    MLX:               {mlx_min*25/1000:.2f}s")
print(f"    CoreML:            {cml_min*25/1000:.2f}s")
print(f"    Savings:           {(mlx_min-cml_min)*25/1000:.2f}s")

print(f"\n  Full pipeline projection (text 0.2s + DiT + VAE 0.5s):")
mlx_total = 0.2 + mlx_min * 25 / 1000 + 0.5
cml_total = 0.2 + cml_min * 25 / 1000 + 0.5
print(f"    MLX Layer 1:       {mlx_total:.1f}s")
print(f"    Metal 4 Layer 3:   {cml_total:.1f}s")
