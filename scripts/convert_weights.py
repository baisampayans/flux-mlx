#!/usr/bin/env python3
"""Convert diffusers Flux2 Klein weights to flux-mlx format.

Converts safetensors weights to MLX-native .npz for faster loading.
Optionally pre-quantizes to INT4/INT8.

Usage:
    python scripts/convert_weights.py --model-dir models/FLUX2-klein-9B
    python scripts/convert_weights.py --model-dir models/FLUX2-klein-9B --quantize 4
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=None, help="Output path (default: model_dir/mlx/)")
    parser.add_argument("--quantize", type=int, default=None, choices=[4, 8])
    args = parser.parse_args()

    import torch
    import mlx.core as mx
    import mlx.nn
    import mlx.utils
    from safetensors import safe_open
    from flux_mlx.models.transformer import Flux2Config, Flux2Transformer

    output_dir = args.output or os.path.join(args.model_dir, "mlx")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading weights from {args.model_dir}...")
    t0 = time.time()

    weights_path = os.path.join(args.model_dir, "transformer", "diffusion_pytorch_model.safetensors")
    weights = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key).to(torch.float32).numpy())

    print(f"  Loaded {len(weights)} tensors in {time.time()-t0:.1f}s")

    # Build model and load weights
    config = Flux2Config()
    model = Flux2Transformer(config)
    model.load_weights(list(weights.items()))

    # Cast to bf16
    bf16_params = [(k, v.astype(mx.bfloat16)) for k, v in mlx.utils.tree_flatten(model.parameters())]
    model.load_weights(bf16_params)

    if args.quantize:
        print(f"Quantizing to INT{args.quantize}...")
        mlx.nn.quantize(model, bits=args.quantize)

    # Save
    params = dict(mlx.utils.tree_flatten(model.parameters()))
    output_path = os.path.join(output_dir, f"transformer_{'q' + str(args.quantize) if args.quantize else 'bf16'}.safetensors")
    mx.save_safetensors(output_path, params)

    total_bytes = sum(v.nbytes for v in params.values())
    print(f"  Saved {len(params)} tensors ({total_bytes/1e9:.2f} GB) to {output_path}")


if __name__ == "__main__":
    main()
