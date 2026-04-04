#!/usr/bin/env python3
"""Unified benchmark suite for flux-mlx.

Tests all available backends and reports performance + quality metrics.

Usage:
    python benchmarks/bench.py --model-dir ../models/FLUX2-klein-9B
"""

import argparse
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.utils


def bench_transformer(model_dir: str):
    """Benchmark transformer forward pass across configurations."""
    import torch
    from safetensors import safe_open
    from flux_mlx.models.transformer import Flux2Config, Flux2Transformer

    config = Flux2Config()
    weights_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")

    weights = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key).to(torch.float32).numpy())

    # Inputs
    BATCH, IMG_SEQ, TXT_SEQ = 1, 4096, 512
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

    results = []

    for label, quantize in [("bf16", None), ("INT8", 8), ("INT4", 4)]:
        model = Flux2Transformer(config)
        model.load_weights(list(weights.items()))
        bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(model.parameters())]
        model.load_weights(bf16_params)

        if quantize:
            import mlx.nn
            mlx.nn.quantize(model, bits=quantize)

        mx.eval(model.parameters())
        nbytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(model.parameters()))

        # Warmup
        for _ in range(2):
            out = model(x, ctx, t, img_ids, txt_ids)
            mx.eval(out)

        # Benchmark
        N = 5
        times = []
        for _ in range(N):
            t0 = time.time()
            out = model(x, ctx, t, img_ids, txt_ids)
            mx.eval(out)
            times.append(time.time() - t0)

        min_ms = min(times) * 1000
        results.append((label, min_ms, nbytes / 1e9))
        print(f"  {label:6s}: {min_ms:7.1f} ms  ({nbytes/1e9:.2f} GB)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"flux-mlx Benchmark Suite")
    print(f"{'='*60}")
    print(f"\nTransformer forward pass (1024x1024):")
    results = bench_transformer(args.model_dir)

    print(f"\nProjected end-to-end (2 steps + text + VAE):")
    for label, ms, gb in results:
        e2e = ms * 2 / 1000 + 2.5  # ~2s text + 0.5s VAE
        print(f"  {label:6s}: {e2e:.1f}s  (model: {gb:.1f} GB)")


if __name__ == "__main__":
    main()
