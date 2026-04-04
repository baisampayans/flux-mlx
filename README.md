# flux-mlx

**Flux2 Klein image generation optimized for Apple Silicon — pure MLX, zero heavy dependencies.**

A research project exploring Metal Performance Primitives (MPP) and Apple Silicon optimization for diffusion models. Includes a working Flux2 Klein pipeline that achieves **99% GPU utilization** via MPP `matmul2d` — the first Python/MLX integration of Metal 4 tensor operations on M1-M4 hardware.

## Key Findings

- **MPP matmul2d achieves 26.6 TFLOPS** (99% utilization) on M3 Ultra vs MLX Steel's 23.2 TFLOPS (86%) — same hardware, 15% faster
- **Zero-dependency runtime**: only `mlx` + `tokenizers` + `numpy` — no PyTorch, no diffusers, no transformers
- **5.0s cold start** (faster than mflux's 5.2s), **3.8s steady-state** at 1024×1024
- **4.9x faster** than stock diffusers on MPS
- MPP `cooperative_tensor` enables in-register post-processing (fused SiLU activation after matmul)
- Metal 4 tensor ops work on **all Apple Silicon** (M1+), not just M5 Neural Accelerators

## Performance

**M3 Ultra, 1024×1024, 2 steps:**

| | Cold Start | Steady State |
|---|---|---|
| Stock diffusers (PyTorch MPS) | ~20s | ~18s |
| mflux | 5.2s | 3.7s |
| **flux-mlx** | **5.0s** | **3.8s** |
| **flux-mlx + MPP** | **5.0s** | **3.7s** |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/user/flux-mlx.git
cd flux-mlx
pip install -e .

# 2. Download model weights (~8 GB)
huggingface-cli download black-forest-labs/FLUX.2-klein --local-dir models/FLUX2-klein-9B

# 3. Generate
flux-mlx --prompt "a cat sitting on a windowsill, golden hour light"

# With MPP acceleration (macOS 26+)
flux-mlx --prompt "sunset over mountains" --backend mpp

# For 8GB Macs
flux-mlx --prompt "portrait" --quantize 8
```

See [models/FLUX2-klein-9B/DOWNLOAD.md](models/FLUX2-klein-9B/DOWNLOAD.md) for manual download instructions.

## Python API

```python
from flux_mlx import FluxPipeline

pipe = FluxPipeline("models/FLUX2-klein-9B", backend="mpp")

image = pipe.generate("a cat on a windowsill, golden hour light")          # ~5s first, ~3.7s after
image = pipe.generate("sunset over mountain peaks", seed=123)              # ~3.7s
image = pipe.generate("portrait of an astronaut in space", seed=7)         # ~3.7s
```

## Architecture

```
flux_mlx/
├── pipeline.py              # Orchestrator
├── scheduler.py             # FlowMatch Euler (pure numpy, no diffusers)
├── cli.py                   # flux-mlx command
├── models/
│   ├── transformer.py       # Flux2 DiT (3.88B, pure MLX)
│   ├── text_encoder.py      # Qwen3 (2.5B, pure MLX + tokenizers)
│   └── vae.py               # AutoencoderKL decoder (pure MLX)
└── backends/
    ├── mpp_matmul.py        # MPP matmul2d via mx.fast.metal_kernel
    └── kernels/
        └── mpp_gemm.metal   # Metal 4 shader (99% utilization)
```

**Zero heavy dependencies at runtime.** No PyTorch, no diffusers, no transformers. Just:
- `mlx` — Apple's ML framework
- `tokenizers` — fast Rust-based tokenizer (140ms load vs 3s for transformers)
- `numpy` — scheduler math

### Timing Breakdown (M3 Ultra, 1024×1024, 2 steps, cached)

| Component | Time |
|-----------|------|
| Text encoding (Qwen3, 2.5B) | 0.2s |
| DiT transformer (3.88B) | 3.1s |
| VAE decode | 0.4s |
| **Total** | **3.8s** |

## Research: Metal Performance Primitives on M1-M4

The key discovery: MPP `tensor_ops::matmul2d` with `cooperative_tensor` generates better GPU code than MLX's Steel `simdgroup_matrix` kernels on the **same hardware**:

| Kernel | Tile Size | TFLOPS | Utilization |
|--------|-----------|--------|-------------|
| MLX Steel GEMM | 8×8 simdgroup_matrix | 23.2 | 86% |
| **MPP matmul2d** | **16×16 cooperative_tensor** | **26.6** | **99%** |

MPP bypasses threadgroup memory staging and uses larger tiles. The `cooperative_tensor` output enables in-register post-processing (fused activation functions) without device memory round-trips.

This finding applies to **all Apple Silicon** (M1+, macOS 26+), not just M5 with Neural Accelerators. See [benchmarks/mpp_matmul_bench.swift](benchmarks/mpp_matmul_bench.swift) to reproduce.

### Additional Research

- **Metal 4 ML Pipeline**: Investigated `MTL4::MachineLearningCommandEncoder` for whole-network dispatch. bf16 tensors supported but `metal-package-builder` has op support gaps.
- **CoreML Bridge**: bf16↔fp16 bridge produces 37.8 dB PSNR — viable but CoreML is 1.37x per-block (less than expected for Flux Klein's shapes).
- **Compute Utilization**: Model is matmul-dominated (75% projections, 19% attention). Quantization (INT4/INT8) gives zero speedup — compute-bound, not bandwidth-bound.

## Model

[FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein) (4B distilled) by Black Forest Labs. Apache 2.0 licensed.

## Requirements

- macOS 14+ (Sonoma) — macOS 26+ for MPP backend
- Apple Silicon (M1 through M5)
- Python 3.10+
- 16GB+ RAM recommended (8GB with `--quantize 8`)

## License

Apache 2.0
