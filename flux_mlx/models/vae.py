"""AutoencoderKL VAE Encoder + Decoder for Flux2 Klein — pure MLX.

Decoder: conv_in → mid_block → 4 up_blocks → conv_out  (latents → image)
Encoder: conv_in → 4 down_blocks → mid_block → conv_out (image → latents)
"""

import time
import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn


# ── Building blocks ───────────────────────────────────────────

class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x):
        # x: NCHW → NHWC for MLX conv
        h = x.transpose(0, 2, 3, 1)
        residual = h

        h = self.norm1(h.astype(mx.float32)).astype(x.dtype)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h.astype(mx.float32)).astype(x.dtype)
        h = nn.silu(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return (h + residual).transpose(0, 3, 1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x):
        h = x.transpose(0, 2, 3, 1)  # NCHW → NHWC
        B, H, W, C = h.shape

        normed = self.group_norm(h.astype(mx.float32)).astype(x.dtype)
        q = self.to_q(normed).reshape(B, H * W, 1, C).transpose(0, 2, 1, 3)
        k = self.to_k(normed).reshape(B, H * W, 1, C).transpose(0, 2, 1, 3)
        v = self.to_v(normed).reshape(B, H * W, 1, C).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(C)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, H, W, C)
        attn = self.to_out(attn)

        return (h + attn).transpose(0, 3, 1, 2)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        x = mx.repeat(x, 2, axis=2)
        x = mx.repeat(x, 2, axis=3)
        x = x.transpose(0, 2, 3, 1)
        x = self.conv(x)
        return x.transpose(0, 3, 1, 2)


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def __call__(self, x):
        # Asymmetric padding (pad right and bottom by 1) to match PyTorch stride=2 padding=1
        x = x.transpose(0, 2, 3, 1)  # NCHW → NHWC
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])  # pad H and W
        x = self.conv(x)
        return x.transpose(0, 3, 1, 2)


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_layers: int = 3,
                 groups: int = 32, eps: float = 1e-6, add_upsample: bool = True):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, groups, eps)
            for i in range(n_layers)
        ]
        self.upsamplers = [Upsample2D(out_ch)] if add_upsample else []

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for up in self.upsamplers:
            x = up(x)
        return x


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_layers: int = 2,
                 groups: int = 32, eps: float = 1e-6, add_downsample: bool = True):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, groups, eps)
            for i in range(n_layers)
        ]
        self.downsamplers = [Downsample2D(out_ch)] if add_downsample else []

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for down in self.downsamplers:
            x = down(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels, groups, eps),
            ResnetBlock2D(channels, channels, groups, eps),
        ]
        self.attentions = [AttentionBlock(channels, groups, eps)]


# ── Decoder ───────────────────────────────────────────────────

class Decoder(nn.Module):
    """Flux2 VAE Decoder: (B, 32, H, W) → (B, 3, 8H, 8W)."""
    def __init__(self):
        super().__init__()
        block_channels = [128, 256, 512, 512]
        groups = 32
        eps = 1e-6

        self.conv_in = nn.Conv2d(32, 512, 3, padding=1)
        self.mid_block = MidBlock(512, groups, eps)

        reversed_ch = list(reversed(block_channels))
        self.up_blocks = []
        for i, out_ch in enumerate(reversed_ch):
            in_ch = out_ch if i == 0 else reversed_ch[i - 1]
            self.up_blocks.append(UpDecoderBlock2D(
                in_ch, out_ch, n_layers=3, groups=groups, eps=eps,
                add_upsample=(i < len(reversed_ch) - 1),
            ))

        self.conv_norm_out = nn.GroupNorm(groups, 128, eps=eps, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1)
        x = self.conv_in(x)
        x = x.transpose(0, 3, 1, 2)

        x = self.mid_block.resnets[0](x)
        x = self.mid_block.attentions[0](x)
        x = self.mid_block.resnets[1](x)

        for block in self.up_blocks:
            x = block(x)

        x = x.transpose(0, 2, 3, 1)
        x = self.conv_norm_out(x.astype(mx.float32)).astype(x.dtype)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x.transpose(0, 3, 1, 2)


# ── Encoder ───────────────────────────────────────────────────

class Encoder(nn.Module):
    """Flux2 VAE Encoder: (B, 3, H, W) → (B, 64, H/8, W/8).

    4 down blocks: 128→128 (↓2), 128→256 (↓2), 256→512 (↓2), 512→512 (no ↓)
    Total spatial reduction: 8x
    Output: 64 channels (32 mean + 32 logvar for diagonal Gaussian)
    """
    def __init__(self):
        super().__init__()
        block_channels = [128, 256, 512, 512]
        groups = 32
        eps = 1e-6

        # Input: 3 → 128
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)

        # Down blocks
        self.down_blocks = []
        prev_ch = 128
        for i, out_ch in enumerate(block_channels):
            self.down_blocks.append(DownEncoderBlock2D(
                prev_ch, out_ch, n_layers=2, groups=groups, eps=eps,
                add_downsample=(i < len(block_channels) - 1),  # No downsample on last block
            ))
            prev_ch = out_ch

        # Mid block
        self.mid_block = MidBlock(512, groups, eps)

        # Output
        self.conv_norm_out = nn.GroupNorm(groups, 512, eps=eps, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(512, 64, 3, padding=1)

    def __call__(self, x):
        # Input conv
        x = x.transpose(0, 2, 3, 1)  # NCHW → NHWC
        x = self.conv_in(x)
        x = x.transpose(0, 3, 1, 2)  # NHWC → NCHW

        # Down blocks
        for block in self.down_blocks:
            x = block(x)

        # Mid block
        x = self.mid_block.resnets[0](x)
        x = self.mid_block.attentions[0](x)
        x = self.mid_block.resnets[1](x)

        # Output
        x = x.transpose(0, 2, 3, 1)
        x = self.conv_norm_out(x.astype(mx.float32)).astype(x.dtype)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x.transpose(0, 3, 1, 2)  # (B, 64, H/8, W/8)


# ── Full VAE Decoder ─────────────────────────────────────────

class Flux2VAEDecoder:
    """Complete VAE decoder with unpatchify and BatchNorm denormalization."""

    def __init__(self, model_dir: str, dtype=mx.bfloat16):
        self.model_dir = model_dir
        self.dtype = dtype
        self._decoder = None
        self._post_quant_conv = None
        self._bn_mean = None
        self._bn_var = None
        self._bn_eps = 1e-4

    def _load(self):
        if self._decoder is not None:
            return
        t0 = time.time()

        self._decoder = Decoder()
        self._post_quant_conv = nn.Conv2d(32, 32, 1)

        weights = mx.load(f"{self.model_dir}/vae/diffusion_pytorch_model.safetensors")

        fixed = {}
        for key, val in weights.items():
            if "num_batches_tracked" in key:
                continue
            if "encoder" in key:
                continue

            if "weight" in key and val.ndim == 4:
                val = val.transpose(0, 2, 3, 1)

            mlx_key = key.replace("to_out.0.", "to_out.")

            if key == "bn.running_mean":
                self._bn_mean = val
                continue
            elif key == "bn.running_var":
                self._bn_var = val
                continue

            fixed[mlx_key] = val

        decoder_weights = {k.replace("decoder.", ""): v for k, v in fixed.items() if k.startswith("decoder.")}
        pqc_weights = {k.replace("post_quant_conv.", ""): v for k, v in fixed.items() if k.startswith("post_quant_conv.")}

        self._decoder.load_weights(list(decoder_weights.items()))
        self._post_quant_conv.load_weights(list(pqc_weights.items()))

        mx.eval(self._decoder.parameters())
        mx.eval(self._post_quant_conv.parameters())

        print(f"  VAE decoder loaded in {time.time()-t0:.1f}s")

    def decode(self, latents: mx.array, height: int, width: int,
               img_ids: mx.array = None) -> np.ndarray:
        self._load()

        vae_scale = 8
        h = 2 * (height // (vae_scale * 2))
        w = 2 * (width // (vae_scale * 2))
        ph, pw = h // 2, w // 2

        B = latents.shape[0]
        spatial = latents.reshape(B, ph, pw, 128).transpose(0, 3, 1, 2)

        bn_mean = self._bn_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self._bn_var.reshape(1, -1, 1, 1) + self._bn_eps)
        spatial = spatial * bn_std + bn_mean

        spatial = spatial.reshape(B, 32, 4, ph, pw)
        spatial = spatial.reshape(B, 32, 2, 2, ph, pw)
        spatial = spatial.transpose(0, 1, 4, 2, 5, 3)
        spatial = spatial.reshape(B, 32, ph * 2, pw * 2)

        spatial = spatial.transpose(0, 2, 3, 1)
        spatial = self._post_quant_conv(spatial)
        spatial = spatial.transpose(0, 3, 1, 2)

        decoded = self._decoder(spatial)
        mx.eval(decoded)

        img = decoded[0].transpose(1, 2, 0)
        img = mx.clip(img, -1, 1)
        img = ((img + 1) / 2 * 255).astype(mx.uint8)
        return np.array(img)


# ── Full VAE Encoder ─────────────────────────────────────────

class Flux2VAEEncoder:
    """Complete VAE encoder with patchify and BatchNorm normalization.

    Image (H, W, 3) uint8 → latents (1, seq, 128) matching pipeline format.
    """

    def __init__(self, model_dir: str, dtype=mx.bfloat16):
        self.model_dir = model_dir
        self.dtype = dtype
        self._encoder = None
        self._quant_conv = None
        self._bn_mean = None
        self._bn_var = None
        self._bn_eps = 1e-4

    def _load(self):
        if self._encoder is not None:
            return
        t0 = time.time()

        self._encoder = Encoder()
        self._quant_conv = nn.Conv2d(64, 64, 1)

        weights = mx.load(f"{self.model_dir}/vae/diffusion_pytorch_model.safetensors")

        fixed = {}
        for key, val in weights.items():
            if "num_batches_tracked" in key:
                continue
            if "decoder" in key:
                continue
            if "post_quant_conv" in key:
                continue

            if "weight" in key and val.ndim == 4:
                val = val.transpose(0, 2, 3, 1)

            mlx_key = key.replace("to_out.0.", "to_out.")

            if key == "bn.running_mean":
                self._bn_mean = val
                continue
            elif key == "bn.running_var":
                self._bn_var = val
                continue

            fixed[mlx_key] = val

        encoder_weights = {k.replace("encoder.", ""): v for k, v in fixed.items() if k.startswith("encoder.")}
        qc_weights = {k.replace("quant_conv.", ""): v for k, v in fixed.items() if k.startswith("quant_conv.")}

        self._encoder.load_weights(list(encoder_weights.items()))
        self._quant_conv.load_weights(list(qc_weights.items()))

        mx.eval(self._encoder.parameters())
        mx.eval(self._quant_conv.parameters())

        print(f"  VAE encoder loaded in {time.time()-t0:.1f}s")

    def encode(self, image: np.ndarray, height: int, width: int) -> mx.array:
        """Encode an image to latents.

        Args:
            image: (H, W, 3) uint8 numpy array
            height: Target latent height (image height, must be divisible by 16)
            width: Target latent width (image width, must be divisible by 16)

        Returns:
            latents: (1, seq_len, 128) matching pipeline's latent format
        """
        self._load()

        # Preprocess: uint8 → float [-1, 1], NHWC → NCHW
        img = mx.array(image).astype(self.dtype) / 127.5 - 1.0
        img = img[None]  # (1, H, W, 3)
        img = img.transpose(0, 3, 1, 2)  # (1, 3, H, W)

        # Encode → (1, 64, H/8, W/8)
        encoded = self._encoder(img)
        mx.eval(encoded)

        # Quant conv
        encoded = encoded.transpose(0, 2, 3, 1)
        encoded = self._quant_conv(encoded)
        encoded = encoded.transpose(0, 3, 1, 2)

        # Split mean and logvar, sample from diagonal Gaussian
        mean, logvar = mx.split(encoded, 2, axis=1)  # Each (1, 32, H/8, W/8)
        logvar = mx.clip(logvar, -30.0, 20.0)
        # Use mean directly (no sampling noise for deterministic encoding)
        latents_spatial = mean  # (1, 32, H/8, W/8)

        # Patchify FIRST: (B, 32, h, w) → (B, 128, ph, pw)
        # Reverse of decoder's unpatchify
        B = latents_spatial.shape[0]
        vae_scale = 8
        h = 2 * (height // (vae_scale * 2))
        w = 2 * (width // (vae_scale * 2))
        ph, pw = h // 2, w // 2

        # (B, 32, h, w) → (B, 32, ph, 2, pw, 2)
        spatial = latents_spatial.reshape(B, 32, ph, 2, pw, 2)
        # → (B, 32, 2, 2, ph, pw)
        spatial = spatial.transpose(0, 1, 3, 5, 2, 4)
        # → (B, 128, ph, pw)
        spatial = spatial.reshape(B, 128, ph, pw)

        # THEN BatchNorm normalization (inverse of decoder's denormalization)
        # Now spatial is (B, 128, ph, pw) — matches BN stats shape (128,)
        bn_mean = self._bn_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self._bn_var.reshape(1, -1, 1, 1) + self._bn_eps)
        spatial = (spatial - bn_mean) / bn_std

        # → (B, ph*pw, 128) = (B, seq, 128)
        latents = spatial.reshape(B, 128, ph * pw).transpose(0, 2, 1)

        mx.eval(latents)
        return latents
