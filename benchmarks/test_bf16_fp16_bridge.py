#!/usr/bin/env python3
"""
Test: bf16↔fp16 precision bridge for Flux2 Klein
=================================================
Key question: If we run each transformer block in fp16 and convert back to bf16
at block boundaries, does the per-block error stay bounded?

Compare:
  1. All-bf16 (reference): 25 blocks in bf16
  2. All-fp16 (no bridge): 25 blocks in fp16 — error accumulates
  3. Hybrid bridge: each block in fp16, convert back to bf16 between blocks
"""

import torch
import torch.nn as nn
import time

# Flux2 Klein dimensions
BATCH = 1
SEQ = 4608  # 512 text + 4096 image tokens
HIDDEN = 3072  # Klein hidden dim
HEADS = 24
HEAD_DIM = 128
FFN_DIM = HIDDEN * 4  # 12288
N_DOUBLE = 5
N_SINGLE = 20
N_BLOCKS = N_DOUBLE + N_SINGLE
DEVICE = 'mps'

class FluxBlock(nn.Module):
    """Simplified Flux transformer block (covers both single/double stream numerics)."""
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.wq = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=True)
        self.wk = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=True)
        self.wv = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=True)
        self.wo = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=True)
        self.norm2 = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.w_gate = nn.Linear(HIDDEN, FFN_DIM, bias=True)
        self.w_up = nn.Linear(HIDDEN, FFN_DIM, bias=True)
        self.w_down = nn.Linear(FFN_DIM, HIDDEN, bias=True)

    def forward(self, x):
        h = self.norm1(x)
        B, S, _ = h.shape
        q = self.wq(h).view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.wk(h).view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        v = self.wv(h).view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, HIDDEN)
        x = x + self.wo(attn)
        h = self.norm2(x)
        gate = torch.nn.functional.silu(self.w_gate(h))
        x = x + self.w_down(gate * self.w_up(h))
        return x

print(f"Flux2 Klein config: B={BATCH}, S={SEQ}, H={HIDDEN}, heads={HEADS}")
print(f"Blocks: {N_DOUBLE} double + {N_SINGLE} single = {N_BLOCKS} total")
print(f"Device: {DEVICE}")

# Create blocks with SHARED weights (same block repeated — tests accumulation)
print("\nCreating block...")
block = FluxBlock().eval()

# Create versions in both precisions with SAME weights
block_bf16 = FluxBlock().eval()
block_bf16.load_state_dict(block.state_dict())
block_bf16 = block_bf16.to(torch.bfloat16).to(DEVICE)

block_fp16 = FluxBlock().eval()
block_fp16.load_state_dict(block.state_dict())
block_fp16 = block_fp16.to(torch.float16).to(DEVICE)

# Input tensor
x_init = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.float32)
x_bf16 = x_init.to(torch.bfloat16).to(DEVICE)

print(f"\n{'='*70}")
print(f"Running {N_BLOCKS} blocks in 3 modes...")
print(f"{'='*70}")

# ── Mode 1: All bf16 (reference) ───────────────────────────────
x_ref = x_bf16.clone()
with torch.no_grad():
    for i in range(N_BLOCKS):
        x_ref = block_bf16(x_ref)
    torch.mps.synchronize()
ref_out = x_ref.cpu().float()

# ── Mode 2: All fp16 (no bridge — accumulates error) ──────────
x_fp16_accum = x_init.to(torch.float16).to(DEVICE)
with torch.no_grad():
    for i in range(N_BLOCKS):
        x_fp16_accum = block_fp16(x_fp16_accum)
    torch.mps.synchronize()
no_bridge_out = x_fp16_accum.cpu().float()

# ── Mode 3: Hybrid bridge (fp16 compute, bf16 at boundaries) ──
x_bridge = x_bf16.clone()
per_block_diffs = []
with torch.no_grad():
    for i in range(N_BLOCKS):
        # bf16 → fp16
        x_fp16 = x_bridge.to(torch.float16)
        # Run block in fp16
        x_fp16 = block_fp16(x_fp16)
        # fp16 → bf16 (the "bridge" correction)
        x_bridge = x_fp16.to(torch.bfloat16)

        # Also compute bf16 reference for this block to measure per-block diff
        # (rerun in bf16 from same input for comparison)
    torch.mps.synchronize()
bridge_out = x_bridge.cpu().float()

# ── Mode 4: Track per-block divergence ─────────────────────────
print(f"\n{'='*70}")
print(f"Per-block divergence (bf16 vs fp16-with-bridge)")
print(f"{'='*70}")

x_ref_track = x_bf16.clone()
x_bridge_track = x_bf16.clone()
with torch.no_grad():
    for i in range(N_BLOCKS):
        # bf16 path
        x_ref_track = block_bf16(x_ref_track)

        # bridge path
        x_fp16 = x_bridge_track.to(torch.float16)
        x_fp16 = block_fp16(x_fp16)
        x_bridge_track = x_fp16.to(torch.bfloat16)

        torch.mps.synchronize()

        diff = (x_ref_track.cpu().float() - x_bridge_track.cpu().float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        ref_mag = x_ref_track.cpu().float().abs().mean().item()
        rel_err = mean_diff / max(ref_mag, 1e-10) * 100

        label = "double" if i < N_DOUBLE else "single"
        print(f"  Block {i:2d} ({label:6s}): max={max_diff:.4f}, mean={mean_diff:.6f}, "
              f"rel={rel_err:.4f}%, |ref|={ref_mag:.3f}")

# ── Final comparison ───────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL OUTPUT COMPARISON (after {N_BLOCKS} blocks)")
print(f"{'='*70}")

for name, out in [("All-fp16 (no bridge)", no_bridge_out),
                   ("Hybrid bridge (fp16+bf16)", bridge_out)]:
    diff = (ref_out - out).abs()
    signal = ref_out.pow(2).mean()
    noise = diff.pow(2).mean()
    psnr = 10 * torch.log10(signal / max(noise, 1e-10))
    cosine = torch.nn.functional.cosine_similarity(
        ref_out.reshape(1, -1), out.reshape(1, -1)
    ).item()

    print(f"\n  {name}:")
    print(f"    Max diff:     {diff.max():.4f}")
    print(f"    Mean diff:    {diff.mean():.6f}")
    print(f"    Rel error:    {diff.mean()/ref_out.abs().mean()*100:.4f}%")
    print(f"    PSNR:         {psnr:.1f} dB  {'(>30=excellent, >20=good)' if psnr > 20 else '(BAD)'}")
    print(f"    Cosine sim:   {cosine:.6f}  {'(>0.999=excellent)' if cosine > 0.999 else ''}")

# ── Check for fp16 overflow risk ───────────────────────────────
print(f"\n{'='*70}")
print(f"OVERFLOW CHECK (fp16 max = 65504)")
print(f"{'='*70}")
x_check = x_bf16.clone()
with torch.no_grad():
    for i in range(N_BLOCKS):
        h = block_bf16.norm1(x_check)
        q = block_bf16.wq(h)
        # Check attention logits scale
        B, S, _ = h.shape
        q_r = q.view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        k = block_bf16.wk(h).view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
        logits = torch.matmul(q_r, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)

        max_logit = logits.abs().max().item()
        max_act = x_check.abs().max().item()
        print(f"  Block {i:2d}: max_activation={max_act:.1f}, max_attn_logit={max_logit:.2f}"
              f"  {'⚠️ OVERFLOW RISK' if max_act > 60000 or max_logit > 60000 else '✓'}")

        x_check = block_bf16(x_check)
        torch.mps.synchronize()
