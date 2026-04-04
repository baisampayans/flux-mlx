"""Qwen3 Text Encoder for Flux2 Klein — Pure MLX.

Qwen3 (2.5B): 36 layers, hidden=2560, 32 heads / 8 KV heads, GQA, SwiGLU.
Extracts hidden states from layers 9, 18, 27, stacks to (B, seq, 7680).
"""

import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn


# ── Model Components ──────────────────────────────────────────

class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return self.weight * mx.fast.rms_norm(x, self.weight, self.eps)


class Qwen3RotaryEmbedding:
    """Standard RoPE with theta=1e6."""
    def __init__(self, dim: int = 128, theta: float = 1e6):
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int):
        positions = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(positions, self.inv_freq)  # (seq, dim/2)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        return cos, sin


def apply_rotary_emb(x, cos, sin):
    """Apply RoPE. x: (B, H, S, D), cos/sin: (S, D/2)."""
    *_, S, D = x.shape
    cos = cos[:S]  # trim to seq len
    sin = sin[:S]
    # Reshape for broadcasting: (1, 1, S, D/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    # Split pairs and rotate
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return (x * mx.concatenate([cos, cos], axis=-1) +
            rotated * mx.concatenate([sin, sin], axis=-1))


class Qwen3Attention(nn.Module):
    """GQA attention: 32 query heads, 8 KV heads."""
    def __init__(self, hidden: int = 2560, n_heads: int = 32,
                 n_kv_heads: int = 8, head_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # 4

        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def __call__(self, x, mask, cos, sin):
        B, S, _ = x.shape
        H, KVH, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, KVH, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, KVH, D).transpose(0, 2, 1, 3)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Repeat KV for GQA
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        # SDPA
        scale = 1.0 / math.sqrt(D)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU FFN: gate_proj + up_proj → silu gate → down_proj."""
    def __init__(self, hidden: int = 2560, intermediate: int = 9728):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden: int = 2560, eps: float = 1e-6):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.self_attn = Qwen3Attention(hidden=hidden, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.mlp = Qwen3MLP(hidden=hidden)

    def __call__(self, x, mask, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))


# ── Main Encoder ──────────────────────────────────────────────

class Qwen3TextEncoderMLX(nn.Module):
    """Pure MLX Qwen3 encoder for Flux2 Klein.

    36 layers, extracts hidden states at layers 9, 18, 27.
    """

    def __init__(self, hidden: int = 2560, n_layers: int = 36,
                 vocab_size: int = 151936, eps: float = 1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = [Qwen3DecoderLayer(hidden, eps) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.rope = Qwen3RotaryEmbedding(dim=128, theta=1e6)
        self.hidden = hidden

    def __call__(self, input_ids, attention_mask=None,
                 extract_layers=(9, 18, 27)):
        """Forward pass with multi-layer extraction.

        Args:
            input_ids: (B, S) int32
            attention_mask: (B, S) with 1=attend, 0=pad
            extract_layers: which layer outputs to extract

        Returns:
            prompt_embeds: (B, S, 7680) — stacked hidden states
        """
        B, S = input_ids.shape

        # Embedding
        h = self.embed_tokens(input_ids)

        # Build causal + padding mask
        # Causal: lower triangular
        causal = mx.triu(mx.full((S, S), -1e9), k=1)  # upper tri = -inf
        causal = causal[None, None, :, :]  # (1, 1, S, S)

        if attention_mask is not None:
            # Padding mask: 0 → -inf, 1 → 0
            pad_mask = (1 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e9
            mask = causal + pad_mask
        else:
            mask = causal

        # RoPE
        cos, sin = self.rope(S)

        # Run layers, collecting hidden states
        collected = {}
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cos, sin)
            if i + 1 in extract_layers:  # layer output = index i+1 in hidden_states list
                collected[i + 1] = h

        # Final norm (not used for extraction, but applied for completeness)
        # Note: diffusers uses hidden_states BEFORE final norm for layers 9,18,27

        # Stack extracted layers: (B, 3, S, 2560) → (B, S, 7680)
        stacked = mx.stack([collected[k] for k in extract_layers], axis=1)
        B, C, S, D = stacked.shape
        prompt_embeds = stacked.transpose(0, 2, 1, 3).reshape(B, S, C * D)

        return prompt_embeds


# ── Tokenizer + Weight Loading ────────────────────────────────

class Qwen3TextEncoder:
    """Complete text encoder: tokenizer + MLX model + weight loading."""

    def __init__(self, model_dir: str, dtype=mx.bfloat16):
        self.model_dir = model_dir
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import time
        t0 = time.time()

        # Fast tokenizer — loads in ~140ms vs 3s for AutoTokenizer
        from tokenizers import Tokenizer as FastTokenizer
        self._fast_tokenizer = FastTokenizer.from_file(
            f"{self.model_dir}/tokenizer/tokenizer.json"
        )
        self._pad_token_id = 151643  # <|endoftext|>

        # Build model
        self._model = Qwen3TextEncoderMLX()

        # Load weights from safetensors
        self._load_weights()
        print(f"  Qwen3 text encoder loaded in {time.time()-t0:.1f}s")

    def _load_weights(self):
        """Load HuggingFace weights into MLX model — zero-copy via mx.load."""
        import os
        import glob
        import json

        enc_dir = f"{self.model_dir}/text_encoder"
        index_file = f"{enc_dir}/model.safetensors.index.json"

        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
        else:
            files = sorted(os.path.basename(f)
                          for f in glob.glob(f"{enc_dir}/*.safetensors"))

        # mx.load reads bf16 natively — zero conversion
        all_weights = {}
        for fname in files:
            shard = mx.load(f"{enc_dir}/{fname}")
            for key, val in shard.items():
                mlx_key = key.replace("model.", "") if key.startswith("model.") else key
                all_weights[mlx_key] = val

        self._model.load_weights(list(all_weights.items()))
        mx.eval(self._model.parameters())

    def encode(self, prompt: str, max_length: int = 512) -> mx.array:
        """Encode prompt to conditioning embeddings (B, seq, 7680)."""
        self._load()

        # Qwen3 chat template (enable_thinking=False)
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Fast tokenizer — no transformers dependency
        encoded = self._fast_tokenizer.encode(formatted)
        ids = list(encoded.ids)[:max_length]
        n_real = len(ids)
        ids = ids + [self._pad_token_id] * (max_length - n_real)
        attn = [1] * n_real + [0] * (max_length - n_real)

        input_ids = mx.array([ids], dtype=mx.int32)
        attention_mask = mx.array([attn], dtype=mx.int32)

        prompt_embeds = self._model(input_ids, attention_mask)
        return prompt_embeds.astype(self.dtype)
