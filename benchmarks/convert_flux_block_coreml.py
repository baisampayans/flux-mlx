#!/usr/bin/env python3
"""
Convert a real Flux2 Klein single-stream block to CoreML and benchmark.
======================================================================

The single-stream blocks are 80% of compute (20 blocks vs 5 double-stream).
We wrap one block to flatten tuple/dict inputs for torch.jit.trace.
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

MODEL_DIR = "/Users/baisampayansaha/Desktop/AI/mlx-media-gen/models/FLUX2-klein-9B"
DEVICE = "mps"
BATCH = 1
TOTAL_SEQ = 4608  # 512 text + 4096 image (concatenated for single-stream)
HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128
OUTPUT_DIR = "/tmp/flux_coreml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load real model ────────────────────────────────────────────
print("Loading Flux2 Klein pipeline...")
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
transformer = pipe.transformer.eval()

# Get a real single-stream block (block 0 of 20)
real_block = transformer.single_transformer_blocks[0].eval()

print(f"Single block params: {sum(p.numel() for p in real_block.parameters())/1e6:.1f}M")

# ── Prepare real inputs ────────────────────────────────────────
# Pre-compute RoPE embeddings (these are just cos/sin tensors)
img_ids = torch.zeros(4096, 4)
h_coords = torch.arange(64).repeat_interleave(64).float()
w_coords = torch.arange(64).repeat(64).float()
img_ids[:, 1] = h_coords
img_ids[:, 2] = w_coords

txt_ids = torch.zeros(512, 4)
txt_ids[:, 3] = torch.arange(512).float()

with torch.no_grad():
    image_rotary_emb = transformer.pos_embed(img_ids)
    text_rotary_emb = transformer.pos_embed(txt_ids)
    rope_cos = torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0)  # (4608, 128)
    rope_sin = torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0)  # (4608, 128)

print(f"RoPE cos: {rope_cos.shape}, sin: {rope_sin.shape}")

# ── Wrap block for tracing ─────────────────────────────────────
# torch.jit.trace needs flat tensor inputs, no tuples/dicts/None
class TracableSingleBlock(nn.Module):
    """Wraps Flux2SingleTransformerBlock with flat tensor inputs for CoreML."""
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

traceable = TracableSingleBlock(real_block).eval()

# Test forward pass
x = torch.randn(BATCH, TOTAL_SEQ, HIDDEN)
# Modulation: single-stream needs (shift, scale, gate) = 3 * HIDDEN = 9216
temb_mod = torch.randn(BATCH, 9216)

with torch.no_grad():
    test_out = traceable(x, temb_mod, rope_cos, rope_sin)
    print(f"Block output: {test_out.shape}")

# ── Convert to CoreML ──────────────────────────────────────────
print(f"\n{'='*70}")
print("Converting single-stream block to CoreML fp16...")
print(f"{'='*70}")

traced = torch.jit.trace(traceable, (x, temb_mod, rope_cos, rope_sin))

t0 = time.time()
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="hidden_states", shape=(BATCH, TOTAL_SEQ, HIDDEN)),
        ct.TensorType(name="temb_mod", shape=(BATCH, 9216)),
        ct.TensorType(name="rope_cos", shape=(TOTAL_SEQ, HEAD_DIM)),
        ct.TensorType(name="rope_sin", shape=(TOTAL_SEQ, HEAD_DIM)),
    ],
    outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
)
convert_time = time.time() - t0
print(f"  Conversion: {convert_time:.1f}s")

model_path = os.path.join(OUTPUT_DIR, "flux_single_block.mlpackage")
mlmodel.save(model_path)
print(f"  Saved to {model_path}")

# ── Benchmark CoreML ───────────────────────────────────────────
print(f"\n{'='*70}")
print("Benchmarking CoreML fp16 vs PyTorch MPS bf16/fp16")
print(f"{'='*70}")

# Load CoreML model
loaded = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

# Prepare inputs
x_np = np.random.randn(BATCH, TOTAL_SEQ, HIDDEN).astype(np.float16)
temb_np = np.random.randn(BATCH, 9216).astype(np.float16)
rope_cos_np = rope_cos.numpy().astype(np.float16)
rope_sin_np = rope_sin.numpy().astype(np.float16)

cml_inputs = {
    "hidden_states": x_np,
    "temb_mod": temb_np,
    "rope_cos": rope_cos_np,
    "rope_sin": rope_sin_np,
}

# Warmup CoreML
print("  Warming up CoreML...")
for _ in range(5):
    _ = loaded.predict(cml_inputs)

N = 20
cml_times = []
for i in range(N):
    t0 = time.time()
    _ = loaded.predict(cml_inputs)
    t1 = time.time()
    cml_times.append(t1 - t0)

cml_avg = np.mean(cml_times) * 1000
cml_min = min(cml_times) * 1000
print(f"  CoreML fp16 ({N} runs): avg={cml_avg:.1f}ms, min={cml_min:.1f}ms")

# Benchmark PyTorch bf16 single block
block_bf16 = real_block.to(torch.bfloat16).to(DEVICE)
x_bf16 = torch.randn(BATCH, TOTAL_SEQ, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
temb_bf16 = torch.randn(BATCH, 9216, dtype=torch.bfloat16, device=DEVICE)
rope_cos_mps = rope_cos.to(torch.bfloat16).to(DEVICE)
rope_sin_mps = rope_sin.to(torch.bfloat16).to(DEVICE)

for _ in range(5):
    with torch.no_grad():
        _ = block_bf16(x_bf16, None, temb_bf16, (rope_cos_mps, rope_sin_mps))
    torch.mps.synchronize()

pt_times = []
for i in range(N):
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = block_bf16(x_bf16, None, temb_bf16, (rope_cos_mps, rope_sin_mps))
    torch.mps.synchronize()
    t1 = time.time()
    pt_times.append(t1 - t0)

pt_avg = np.mean(pt_times) * 1000
pt_min = min(pt_times) * 1000
print(f"  PyTorch bf16 ({N} runs): avg={pt_avg:.1f}ms, min={pt_min:.1f}ms")

# Benchmark hybrid bridge (bf16→fp16 CoreML→bf16)
print("\n  Benchmarking hybrid bridge (bf16→fp16→CoreML→fp16→bf16)...")
bridge_times = []
for i in range(N):
    torch.mps.synchronize()
    t0 = time.time()

    # bf16 → fp16 → numpy
    x_fp16_np = x_bf16.to(torch.float16).cpu().numpy()
    temb_fp16_np = temb_bf16.to(torch.float16).cpu().numpy()

    # CoreML inference
    result = loaded.predict({
        "hidden_states": x_fp16_np,
        "temb_mod": temb_fp16_np,
        "rope_cos": rope_cos_np,
        "rope_sin": rope_sin_np,
    })

    # numpy → fp16 → bf16 → MPS
    out_bf16 = torch.from_numpy(result["output"]).to(torch.bfloat16).to(DEVICE)

    torch.mps.synchronize()
    t1 = time.time()
    bridge_times.append(t1 - t0)

bridge_avg = np.mean(bridge_times) * 1000
bridge_min = min(bridge_times) * 1000
print(f"  Hybrid bridge ({N} runs): avg={bridge_avg:.1f}ms, min={bridge_min:.1f}ms")

# ── Numerical quality ──────────────────────────────────────────
print(f"\n{'='*70}")
print("NUMERICAL QUALITY (single block)")
print(f"{'='*70}")

# Reference: PyTorch bf16
x_test = torch.randn(BATCH, TOTAL_SEQ, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
temb_test = torch.randn(BATCH, 9216, dtype=torch.bfloat16, device=DEVICE)

with torch.no_grad():
    ref = block_bf16(x_test, None, temb_test, (rope_cos_mps, rope_sin_mps))
    ref_cpu = ref.cpu().float()

# CoreML
x_fp16_np = x_test.to(torch.float16).cpu().numpy()
temb_fp16_np = temb_test.to(torch.float16).cpu().numpy()
cml_result = loaded.predict({
    "hidden_states": x_fp16_np,
    "temb_mod": temb_fp16_np,
    "rope_cos": rope_cos_np,
    "rope_sin": rope_sin_np,
})
cml_out = torch.from_numpy(cml_result["output"]).float()

diff = (ref_cpu - cml_out).abs()
signal = ref_cpu.pow(2).mean()
noise = diff.pow(2).mean()
psnr = 10 * torch.log10(signal / max(noise, 1e-10))
cosine = torch.nn.functional.cosine_similarity(
    ref_cpu.reshape(1, -1), cml_out.reshape(1, -1)
).item()

print(f"  Max diff:    {diff.max():.6f}")
print(f"  Mean diff:   {diff.mean():.6f}")
print(f"  Rel error:   {diff.mean()/ref_cpu.abs().mean()*100:.4f}%")
print(f"  PSNR:        {psnr:.1f} dB  {'✓ excellent' if psnr > 30 else '✓ good' if psnr > 20 else '✗'}")
print(f"  Cosine sim:  {cosine:.6f}  {'✓' if cosine > 0.999 else ''}")

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY (per single-stream block)")
print(f"{'='*70}")
print(f"  PyTorch bf16:   {pt_min:.1f} ms  (baseline)")
print(f"  CoreML fp16:    {cml_min:.1f} ms  ({pt_min/cml_min:.2f}x)")
print(f"  Hybrid bridge:  {bridge_min:.1f} ms  ({pt_min/bridge_min:.2f}x)")
print(f"\n  Full transformer projection (5 double + 20 single blocks):")
print(f"    PyTorch bf16:   {pt_min*25:.0f} ms")
print(f"    CoreML fp16:    {cml_min*25:.0f} ms")
print(f"    Hybrid bridge:  {bridge_min*25:.0f} ms")
print(f"\n  2-step end-to-end (+ 2.3s text enc + 0.5s VAE):")
pt_e2e = pt_min * 25 * 2 / 1000 + 2.8
cml_e2e = cml_min * 25 * 2 / 1000 + 2.8
bridge_e2e = bridge_min * 25 * 2 / 1000 + 2.8
print(f"    PyTorch bf16:   {pt_e2e:.1f}s")
print(f"    CoreML fp16:    {cml_e2e:.1f}s")
print(f"    Hybrid bridge:  {bridge_e2e:.1f}s")
