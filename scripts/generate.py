#!/usr/bin/env python3
"""Generate images with Flux2 Klein on Apple Silicon.

Usage:
    python scripts/generate.py --prompt "a cat on a windowsill"
    python scripts/generate.py --prompt "sunset over mountains" --steps 4 --size 512
    python scripts/generate.py --prompt "portrait" --quantize 8  # INT8 for 8GB Macs
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux_mlx import FluxPipeline
import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description="Flux2 Klein image generation on Apple Silicon")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--model-dir", type=str, default="models/FLUX2-klein-9B",
                        help="Path to model directory")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2, help="Denoising steps (2 for distilled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "mlx", "mpp", "metal4"])
    parser.add_argument("--quantize", type=int, default=None, choices=[4, 8],
                        help="Quantize model (4 or 8 bit)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="Model precision")
    args = parser.parse_args()

    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16

    pipe = FluxPipeline(
        model_dir=args.model_dir,
        backend=args.backend,
        dtype=dtype,
        quantize=args.quantize,
    )

    image = pipe.generate(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        seed=args.seed,
    )

    from PIL import Image
    img = Image.fromarray(image)
    img.save(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
