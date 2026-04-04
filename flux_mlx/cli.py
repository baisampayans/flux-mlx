#!/usr/bin/env python3
"""CLI for flux-mlx image generation."""

import argparse
import time
import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(
        description="Flux2 Klein image generation on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flux-mlx --prompt "a cat on a windowsill"
  flux-mlx --prompt "sunset over mountains" --size 768
  flux-mlx --prompt "portrait" --quantize 8 --seed 123
  flux-mlx --prompt "landscape" --steps 4 --output landscape.png
        """,
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="models/FLUX2-klein-9B")
    parser.add_argument("--size", type=int, default=1024, help="Image size (square)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--quantize", type=int, default=None, choices=[4, 8],
                        help="Quantize model weights (4 or 8 bit)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    height = args.height or args.size
    width = args.width or args.size
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16

    from flux_mlx import FluxPipeline

    pipe = FluxPipeline(
        model_dir=args.model_dir,
        dtype=dtype,
        quantize=args.quantize,
    )

    image = pipe.generate(
        prompt=args.prompt,
        height=height,
        width=width,
        num_steps=args.steps,
        seed=args.seed,
    )

    from PIL import Image
    Image.fromarray(image).save(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
