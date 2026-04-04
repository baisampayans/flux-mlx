"""
Flux2 Klein Transformer — Pure MLX Implementation
==================================================

Flux2 Klein (4B distilled) architecture:
  - 5 double-stream blocks (joint text+image attention)
  - 20 single-stream blocks (parallel attention+MLP)
  - 3072 hidden dim (24 heads × 128 head_dim)
  - SwiGLU FFN with mlp_ratio=3.0
  - Multi-axis RoPE (4×32 dims, theta=2000)
  - No guidance embeddings (distilled)

Weight key mapping matches diffusers Flux2Transformer2DModel exactly.
"""

import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Custom Metal FlashAttention kernel (v18, 19.12 TFLOPS on M3 Ultra)
# Set MLX_CUSTOM_FLASH_ATTN=1 or call set_flux_attention_backend("custom")
_USE_CUSTOM_ATTN = os.environ.get("MLX_CUSTOM_FLASH_ATTN", "0") == "1"

_CUSTOM_ATTN_AVAILABLE = False
_custom_flash_attention = None


def set_flux_attention_backend(backend: str = "auto"):
    """Switch attention: 'auto', 'custom' (Metal FlashAttention v18), or 'mlx'."""
    global _USE_CUSTOM_ATTN
    if backend == "custom":
        if not _CUSTOM_ATTN_AVAILABLE:
            raise RuntimeError("Custom FlashAttention kernel not available. Build with: pip install -e .")
        _USE_CUSTOM_ATTN = True
    elif backend == "mlx":
        _USE_CUSTOM_ATTN = False
    elif backend == "auto":
        _USE_CUSTOM_ATTN = _CUSTOM_ATTN_AVAILABLE
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def _attention(q, k, v, scale):
    """Dispatch to custom FlashAttention-Metal or MLX SDPA."""
    if _USE_CUSTOM_ATTN and _CUSTOM_ATTN_AVAILABLE:
        return _custom_flash_attention(q, k, v, scale)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


@dataclass
class Flux2Config:
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 5           # double-stream blocks
    num_single_layers: int = 20   # single-stream blocks
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    joint_attention_dim: int = 7680
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple = (32, 32, 32, 32)
    rope_theta: int = 2000
    eps: float = 1e-6
    timestep_guidance_channels: int = 256
    guidance_embeds: bool = False  # Klein is distilled

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


# ── RoPE ──────────────────────────────────────────────────────

def get_1d_rotary_pos_embed(dim: int, pos: mx.array, theta: float = 2000.0):
    """Compute cos/sin RoPE embeddings for one axis.

    Args:
        dim: Embedding dimension (e.g. 32)
        pos: Position indices [S]
        theta: Base frequency

    Returns:
        (cos, sin) each of shape [S, dim] with repeat_interleave pattern
    """
    half = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half).astype(mx.float32) / half))  # [D/2]
    angles = mx.outer(pos.astype(mx.float32), freqs)  # [S, D/2]
    # repeat_interleave(2): [a,b,c] -> [a,a,b,b,c,c]
    cos = mx.repeat(mx.cos(angles), 2, axis=-1)  # [S, D]
    sin = mx.repeat(mx.sin(angles), 2, axis=-1)  # [S, D]
    return cos, sin


def flux2_pos_embed(ids: mx.array, axes_dim: tuple = (32, 32, 32, 32), theta: int = 2000):
    """Compute multi-axis RoPE from position IDs.

    Args:
        ids: [S, 4] position coordinates
        axes_dim: dimension per axis (32,32,32,32) = 128 total

    Returns:
        (cos, sin) each [S, 128]
    """
    cos_parts, sin_parts = [], []
    pos = ids.astype(mx.float32)
    for i, d in enumerate(axes_dim):
        c, s = get_1d_rotary_pos_embed(d, pos[:, i], theta=theta)
        cos_parts.append(c)
        sin_parts.append(s)
    return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array):
    """Apply RoPE to query or key tensor.

    Args:
        x: [B, S, H, D] (sequence_dim=1 for Flux2)
        cos: [S, D]
        sin: [S, D]

    Returns:
        Rotated tensor [B, S, H, D]
    """
    # Reshape cos/sin for broadcasting: [1, S, 1, D]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Split into pairs and rotate: [-x_imag, x_real]
    *shape, d = x.shape
    x_pairs = x.reshape(*shape, d // 2, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    # Stack [-imag, real] and flatten back
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(*shape, d)

    # Compute in float32 for precision, cast back to input dtype
    out = x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin
    return out.astype(x.dtype)


# ── Timestep Embedding ────────────────────────────────────────

def get_timestep_embedding(timesteps: mx.array, dim: int = 256):
    """Sinusoidal timestep embedding (flip_sin_to_cos=True, downscale_freq_shift=0)."""
    half = dim // 2
    exponent = -math.log(10000) * mx.arange(half).astype(mx.float32) / half
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)  # flip: cos first
    return emb


class TimestepEmbedding(nn.Module):
    """Linear → SiLU → Linear."""
    def __init__(self, in_channels: int, time_embed_dim: int, bias: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, config: Flux2Config):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(
            config.timestep_guidance_channels, config.inner_dim, bias=False
        )
        self.guidance_embedder = None
        if config.guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                config.timestep_guidance_channels, config.inner_dim, bias=False
            )

    def __call__(self, timestep: mx.array, guidance: mx.array = None) -> mx.array:
        t_proj = get_timestep_embedding(timestep, 256)
        temb = self.timestep_embedder(t_proj)
        if guidance is not None and self.guidance_embedder is not None:
            g_proj = get_timestep_embedding(guidance, 256)
            temb = temb + self.guidance_embedder(g_proj)
        return temb


# ── Modulation ────────────────────────────────────────────────

class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=bias)

    def __call__(self, temb: mx.array) -> mx.array:
        return self.linear(nn.silu(temb))

    @staticmethod
    def split(mod: mx.array, mod_param_sets: int):
        """Split modulation into (shift, scale, gate) tuples."""
        if mod.ndim == 2:
            mod = mod[:, None, :]  # [B, 1, D*3*sets]
        chunks = mx.split(mod, 3 * mod_param_sets, axis=-1)
        return tuple(chunks[3 * i: 3 * (i + 1)] for i in range(mod_param_sets))


# ── Attention ─────────────────────────────────────────────────

class Flux2Attention(nn.Module):
    """Double-stream joint attention (text + image)."""
    def __init__(self, config: Flux2Config):
        super().__init__()
        dim = config.inner_dim
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim

        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = dim

        # Image projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=config.eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=config.eps)
        self.to_out = [nn.Linear(dim, dim, bias=False)]

        # Text (encoder) projections
        self.add_q_proj = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=config.eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=config.eps)
        self.to_add_out = nn.Linear(dim, dim, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ):
        B = hidden_states.shape[0]
        H, D = self.heads, self.head_dim
        S_enc = encoder_hidden_states.shape[1]

        # Project image stream
        q = self.to_q(hidden_states).reshape(B, -1, H, D)
        k = self.to_k(hidden_states).reshape(B, -1, H, D)
        v = self.to_v(hidden_states).reshape(B, -1, H, D)

        # Project text stream
        eq = self.add_q_proj(encoder_hidden_states).reshape(B, -1, H, D)
        ek = self.add_k_proj(encoder_hidden_states).reshape(B, -1, H, D)
        ev = self.add_v_proj(encoder_hidden_states).reshape(B, -1, H, D)

        # QK norm (per head)
        q = self.norm_q(q)
        k = self.norm_k(k)
        eq = self.norm_added_q(eq)
        ek = self.norm_added_k(ek)

        # Concatenate text + image (text first)
        q = mx.concatenate([eq, q], axis=1)  # [B, S_enc+S_img, H, D]
        k = mx.concatenate([ek, k], axis=1)
        v = mx.concatenate([ev, v], axis=1)

        # Apply RoPE
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Scaled dot-product attention [B, H, S, D]
        dtype = hidden_states.dtype
        q = q.transpose(0, 2, 1, 3).astype(dtype)
        k = k.transpose(0, 2, 1, 3).astype(dtype)
        v = v.transpose(0, 2, 1, 3).astype(dtype)

        scale = 1.0 / math.sqrt(D)
        out = _attention(q, k, v, scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.inner_dim)

        # Split back and project
        enc_out = self.to_add_out(out[:, :S_enc, :])
        img_out = self.to_out[0](out[:, S_enc:, :])

        return img_out, enc_out


class Flux2ParallelSelfAttention(nn.Module):
    """Single-stream parallel attention + MLP (ViT-22B style)."""
    def __init__(self, config: Flux2Config):
        super().__init__()
        dim = config.inner_dim
        heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        mlp_hidden_dim = int(dim * config.mlp_ratio)

        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = dim
        self.mlp_hidden_dim = mlp_hidden_dim

        # Fused QKV + MLP-in projection
        self.to_qkv_mlp_proj = nn.Linear(dim, dim * 3 + mlp_hidden_dim * 2, bias=False)

        # QK norm
        self.norm_q = nn.RMSNorm(head_dim, eps=config.eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=config.eps)

        # Fused attention-out + MLP-out projection
        self.to_out = nn.Linear(dim + mlp_hidden_dim, dim, bias=False)

    def __call__(self, hidden_states: mx.array, rope_cos: mx.array, rope_sin: mx.array):
        B = hidden_states.shape[0]
        H, D = self.heads, self.head_dim

        # Fused projection
        proj = self.to_qkv_mlp_proj(hidden_states)
        qkv = proj[:, :, :self.inner_dim * 3]
        mlp_in = proj[:, :, self.inner_dim * 3:]

        # Attention path
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, -1, H, D)
        k = k.reshape(B, -1, H, D)
        v = v.reshape(B, -1, H, D)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        dtype = hidden_states.dtype
        q = q.transpose(0, 2, 1, 3).astype(dtype)
        k = k.transpose(0, 2, 1, 3).astype(dtype)
        v = v.transpose(0, 2, 1, 3).astype(dtype)

        scale = 1.0 / math.sqrt(D)
        attn_out = _attention(q, k, v, scale)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, -1, self.inner_dim)

        # MLP path (SwiGLU)
        x1, x2 = mx.split(mlp_in, 2, axis=-1)
        mlp_out = nn.silu(x1) * x2

        # Concatenate and output
        out = mx.concatenate([attn_out, mlp_out], axis=-1)
        return self.to_out(out)


# ── Feedforward ───────────────────────────────────────────────

class Flux2FeedForward(nn.Module):
    """SwiGLU feedforward: Linear(dim→2*inner) → SiLU gate → Linear(inner→dim)."""
    def __init__(self, dim: int, mlp_ratio: float = 3.0):
        super().__init__()
        inner = int(dim * mlp_ratio)
        self.linear_in = nn.Linear(dim, inner * 2, bias=False)
        self.linear_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_in(x)
        x1, x2 = mx.split(x, 2, axis=-1)
        return self.linear_out(nn.silu(x1) * x2)


# ── Transformer Blocks ────────────────────────────────────────

class Flux2TransformerBlock(nn.Module):
    """Double-stream block: separate image + text attention and FFN."""
    def __init__(self, config: Flux2Config):
        super().__init__()
        dim = config.inner_dim
        self.norm1 = nn.LayerNorm(dim, affine=False, eps=config.eps)
        self.norm1_context = nn.LayerNorm(dim, affine=False, eps=config.eps)
        self.attn = Flux2Attention(config)
        self.norm2 = nn.LayerNorm(dim, affine=False, eps=config.eps)
        self.ff = Flux2FeedForward(dim, config.mlp_ratio)
        self.norm2_context = nn.LayerNorm(dim, affine=False, eps=config.eps)
        self.ff_context = Flux2FeedForward(dim, config.mlp_ratio)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb_mod_img: mx.array,
        temb_mod_txt: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ):
        # Split modulation params
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = \
            Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = \
            Flux2Modulation.split(temb_mod_txt, 2)

        # Image stream: norm + modulate
        norm_x = self.norm1(hidden_states)
        norm_x = (1 + scale_msa) * norm_x + shift_msa

        # Text stream: norm + modulate
        norm_ctx = self.norm1_context(encoder_hidden_states)
        norm_ctx = (1 + c_scale_msa) * norm_ctx + c_shift_msa

        # Joint attention
        attn_img, attn_txt = self.attn(norm_x, norm_ctx, rope_cos, rope_sin)

        # Image residual + FFN
        hidden_states = hidden_states + gate_msa * attn_img
        norm_x = self.norm2(hidden_states)
        norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_x)

        # Text residual + FFN
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * attn_txt
        norm_ctx = self.norm2_context(encoder_hidden_states)
        norm_ctx = norm_ctx * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_ctx)

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    """Single-stream block: parallel attention + MLP."""
    def __init__(self, config: Flux2Config):
        super().__init__()
        self.norm = nn.LayerNorm(config.inner_dim, affine=False, eps=config.eps)
        self.attn = Flux2ParallelSelfAttention(config)

    def __call__(
        self,
        hidden_states: mx.array,
        temb_mod: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ):
        (shift, scale, gate), = Flux2Modulation.split(temb_mod, 1)

        norm_x = self.norm(hidden_states)
        norm_x = (1 + scale) * norm_x + shift

        attn_out = self.attn(norm_x, rope_cos, rope_sin)
        return hidden_states + gate * attn_out


# ── AdaLayerNormContinuous (output norm) ──────────────────────

class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False, eps=eps)
        self.linear = nn.Linear(cond_dim, dim * 2, bias=False)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(nn.silu(conditioning))
        scale, shift = mx.split(emb, 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


# ── Main Transformer Model ───────────────────────────────────

class Flux2Transformer(nn.Module):
    """Flux2 Klein Transformer — pure MLX."""
    def __init__(self, config: Flux2Config):
        super().__init__()
        self.config = config
        dim = config.inner_dim

        # Positional embedding (computed, no params)
        # Timestep embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(config)

        # Modulation
        self.double_stream_modulation_img = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(dim, mod_param_sets=1, bias=False)

        # Input projections
        self.x_embedder = nn.Linear(config.in_channels, dim, bias=False)
        self.context_embedder = nn.Linear(config.joint_attention_dim, dim, bias=False)

        # Double-stream blocks
        self.transformer_blocks = [Flux2TransformerBlock(config) for _ in range(config.num_layers)]

        # Single-stream blocks
        self.single_transformer_blocks = [Flux2SingleTransformerBlock(config) for _ in range(config.num_single_layers)]

        # Output
        self.norm_out = AdaLayerNormContinuous(dim, dim, eps=config.eps)
        self.proj_out = nn.Linear(dim, config.in_channels, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,       # [B, img_seq, 128]
        encoder_hidden_states: mx.array,  # [B, txt_seq, 7680]
        timestep: mx.array,            # [B] in [0, 1]
        img_ids: mx.array,             # [img_seq, 4]
        txt_ids: mx.array,             # [txt_seq, 4]
        guidance: mx.array = None,
    ) -> mx.array:
        config = self.config
        num_txt = encoder_hidden_states.shape[1]

        # 1. Timestep embedding
        t_scaled = timestep * 1000
        g_scaled = guidance * 1000 if guidance is not None else None
        temb = self.time_guidance_embed(t_scaled, g_scaled)
        temb = temb.astype(hidden_states.dtype)  # float32 → bf16

        # 2. Modulation
        dmod_img = self.double_stream_modulation_img(temb)
        dmod_txt = self.double_stream_modulation_txt(temb)
        smod = self.single_stream_modulation(temb)

        # 3. Input projections
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 4. RoPE
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        img_cos, img_sin = flux2_pos_embed(img_ids, config.axes_dims_rope, config.rope_theta)
        txt_cos, txt_sin = flux2_pos_embed(txt_ids, config.axes_dims_rope, config.rope_theta)
        rope_cos = mx.concatenate([txt_cos, img_cos], axis=0)
        rope_sin = mx.concatenate([txt_sin, img_sin], axis=0)

        # 5. Double-stream blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states,
                dmod_img, dmod_txt, rope_cos, rope_sin,
            )

        # 6. Concatenate for single-stream
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 7. Single-stream blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, smod, rope_cos, rope_sin)

        # 8. Remove text tokens
        hidden_states = hidden_states[:, num_txt:, :]

        # 9. Output
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
