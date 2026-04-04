"""Flux2 Klein Pipeline — text-to-image generation on Apple Silicon.

Orchestrates: text encoding → denoising loop → VAE decode → image.
Auto-selects the fastest execution backend for the current hardware.
"""

import time
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn
import mlx.utils

from flux_mlx.models.transformer import Flux2Config, Flux2Transformer
from flux_mlx.models.text_encoder import Qwen3TextEncoder
from flux_mlx.models.vae import Flux2VAEDecoder
from flux_mlx import scheduler


class FluxPipeline:
    """Flux2 Klein image generation pipeline.

    Args:
        model_dir: Path to FLUX2-klein-9B model directory
        backend: Execution backend ('auto', 'mlx', 'mpp', 'metal4')
        dtype: Model precision (mx.bfloat16 or mx.float16)
        quantize: Quantization bits (None, 4, or 8)
    """

    def __init__(
        self,
        model_dir: str,
        backend: str = "auto",
        dtype=mx.bfloat16,
        quantize: int = None,
    ):
        self.model_dir = Path(model_dir)
        self.dtype = dtype
        self.backend = backend

        # Enable MPP matmul if requested
        if backend == "mpp":
            self._enable_mpp()

        # Load transformer
        print(f"Loading Flux2 Klein from {model_dir}...")
        t0 = time.time()

        self.config = Flux2Config()
        self.transformer = Flux2Transformer(self.config)
        self._load_transformer_weights()

        if quantize:
            mlx.nn.quantize(self.transformer, bits=quantize)
            print(f"  Quantized to INT{quantize}")

        mx.eval(self.transformer.parameters())
        params = sum(p.size for _, p in mlx.utils.tree_flatten(self.transformer.parameters()))
        nbytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(self.transformer.parameters()))
        print(f"  Transformer: {params/1e9:.2f}B params, {nbytes/1e9:.1f} GB ({time.time()-t0:.1f}s)")

        # Text encoder and VAE (lazy-loaded)
        self.text_encoder = Qwen3TextEncoder(str(self.model_dir), dtype=dtype)
        self.vae = Flux2VAEDecoder(str(self.model_dir))

    def _enable_mpp(self):
        """Monkey-patch nn.Linear to use MPP matmul2d for large projections."""
        from flux_mlx.backends.mpp_matmul import mpp_matmul, mpp_available
        if not mpp_available():
            print("  Warning: MPP not available, falling back to MLX")
            return

        orig_call = mlx.nn.Linear.__call__
        min_flops = 1_000_000  # Only use MPP for large enough matmuls

        def mpp_linear(self, x):
            if (x.dtype == mx.bfloat16 and self.weight.dtype == mx.bfloat16
                    and x.ndim >= 2):
                shape = x.shape
                x2d = x.reshape(-1, shape[-1])
                M, K = x2d.shape
                N = self.weight.shape[0]
                if M * N * K > min_flops:
                    out = mpp_matmul(x2d, self.weight)
                    if "bias" in self and self.bias is not None:
                        out = out + self.bias
                    return out.reshape(*shape[:-1], N)
            return orig_call(self, x)

        mlx.nn.Linear.__call__ = mpp_linear
        self._mpp_orig_linear = orig_call
        print("  MPP matmul enabled (Metal Performance Primitives)")

    def _load_transformer_weights(self):
        """Load transformer weights from safetensors — zero-copy via mx.load."""
        weights_path = self.model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        weights = mx.load(str(weights_path))
        # mx.load reads bf16 natively from safetensors — no conversion needed
        self.transformer.load_weights(list(weights.items()))

    def _prepare_latents(self, height: int, width: int, seed: int) -> mx.array:
        """Generate random initial latents matching diffusers' prepare_latents.

        Creates noise in spatial format (B, C*4, H/2, W/2) then packs to (B, seq, C*4).
        C = in_channels // 4 = 32, so C*4 = 128 (matching x_embedder).
        """
        vae_scale = 8
        h = 2 * (height // (vae_scale * 2))  # e.g. 1024 → 128
        w = 2 * (width // (vae_scale * 2))

        num_latent_channels = self.config.in_channels // 4  # 128 // 4 = 32
        noise_channels = num_latent_channels * 4  # 128

        mx.random.seed(seed)
        # Noise in spatial format
        noise = mx.random.normal((1, noise_channels, h // 2, w // 2)).astype(self.dtype)
        # Pack: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = noise.shape
        latents = noise.reshape(B, C, H * W).transpose(0, 2, 1)  # (1, seq, 128)
        return latents

    def _prepare_ids(self, height: int, width: int, txt_len: int):
        """Build position IDs matching diffusers' _prepare_latent_ids and _prepare_text_ids.

        Image IDs: (T, H, W, L) with T=0, H=[0..h-1], W=[0..w-1], L=0
        Text IDs: (T, H, W, L) with T=0, H=0, W=0, L=[0..txt_len-1]
        """
        vae_scale = 8
        h = 2 * (height // (vae_scale * 2))  # spatial height before packing
        w = 2 * (width // (vae_scale * 2))
        # Position IDs are built from the SPATIAL latent before packing
        # _prepare_latent_ids takes (B,C,H/2,W/2) and uses H/2, W/2 dims
        ph = h // 2  # packed height
        pw = w // 2  # packed width

        # Image IDs: cartesian_prod(t=[0], h=[0..ph-1], w=[0..pw-1], l=[0])
        img_ids = np.zeros((ph * pw, 4), dtype=np.float32)
        # T=0 (col 0), H (col 1), W (col 2), L=0 (col 3)
        img_ids[:, 1] = np.repeat(np.arange(ph), pw)
        img_ids[:, 2] = np.tile(np.arange(pw), ph)
        img_ids = mx.array(img_ids).astype(self.dtype)

        # Text IDs: (T=0, H=0, W=0, L=[0..txt_len-1])
        txt_ids = np.zeros((txt_len, 4), dtype=np.float32)
        txt_ids[:, 3] = np.arange(txt_len)
        txt_ids = mx.array(txt_ids).astype(self.dtype)

        return img_ids, txt_ids

    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 2,
        seed: int = 42,
        guidance_scale: float = 1.0,
    ) -> np.ndarray:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            height: Image height (must be divisible by 16)
            width: Image width (must be divisible by 16)
            num_steps: Denoising steps (2 for Klein distilled)
            seed: Random seed for reproducibility
            guidance_scale: CFG scale (1.0 = no CFG for Klein)

        Returns:
            image: (H, W, 3) numpy array in uint8
        """
        assert height % 16 == 0 and width % 16 == 0, "Height and width must be divisible by 16"
        img_seq_len = (height // 16) * (width // 16)
        txt_seq_len = 512

        print(f"\nGenerating {width}x{height}, {num_steps} steps, seed={seed}")

        # 1. Text encoding
        t0 = time.time()
        prompt_embeds = self.text_encoder.encode(prompt, max_length=txt_seq_len)
        mx.eval(prompt_embeds)
        t_text = time.time() - t0
        print(f"  Text encoding: {t_text:.1f}s")

        # 2. Prepare latents and position IDs
        latents = self._prepare_latents(height, width, seed)
        img_ids, txt_ids = self._prepare_ids(height, width, txt_seq_len)

        # 3. Compute timestep schedule
        sigmas, mu = scheduler.get_sigmas(num_steps, img_seq_len)
        print(f"  Scheduler: mu={mu:.3f}, sigmas={sigmas}")

        # 4. Denoising loop
        t0 = time.time()
        for i in range(num_steps):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])

            # Transformer expects sigma (in [0,1]), multiplies by 1000 internally
            t_input = mx.array([sigma]).astype(self.dtype)

            # Transformer forward
            noise_pred = self.transformer(
                latents, prompt_embeds, t_input, img_ids, txt_ids
            )
            mx.eval(noise_pred)

            # Euler step
            latents = scheduler.step(noise_pred, sigma, sigma_next, latents)
            mx.eval(latents)

            t_step = time.time() - t0
            print(f"  Step {i+1}/{num_steps}: {t_step:.1f}s")

        t_dit = time.time() - t0
        print(f"  DiT total: {t_dit:.1f}s")

        # 5. VAE decode
        t0 = time.time()
        image = self.vae.decode(latents, height, width)
        t_vae = time.time() - t0
        print(f"  VAE decode: {t_vae:.1f}s")

        print(f"  Total: {t_text + t_dit + t_vae:.1f}s")

        return image  # (H, W, 3) uint8
