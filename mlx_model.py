"""MLX backend for the tiny diffusion transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
from mlx.core import fast
import mlx.nn as nn


@dataclass
class DiffusionConfig:
    sequence_len: int = 256
    vocab_size: int = 128
    mask_token_id: int = 0
    n_layer: int = 10
    n_head: int = 12
    n_embd: int = 768
    diffusion_steps: int = 128
    context_len: int = 64


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Functional RMSNorm with no learnable parameters."""
    return fast.rms_norm(x, None, eps)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    head_dim = x.shape[-1] // 2
    x1 = x[..., :head_dim]
    x2 = x[..., head_dim:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


class BidirectionalAttention(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def __call__(self, x: mx.array, cos_sin: tuple[mx.array, mx.array]) -> mx.array:
        b, t, _ = x.shape
        q = self.c_q(x).reshape(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(b, t, self.n_head, self.head_dim)
        v = self.c_v(x).reshape(b, t, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = rms_norm(apply_rotary_emb(q, cos, sin))
        k = rms_norm(apply_rotary_emb(k, cos, sin))
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        scale = 1.0 / math.sqrt(self.head_dim)
        y = fast.scaled_dot_product_attention(q, k, v, scale=scale)
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(b, t, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = mx.maximum(x, 0.0)
        x = x * x
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.attn = BidirectionalAttention(config)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, cos_sin: tuple[mx.array, mx.array]) -> mx.array:
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.diffusion_steps, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim
        )

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: float = 10000.0
    ) -> tuple[mx.array, mx.array]:
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        exponent = -channel_range / head_dim
        inv_freq = mx.power(mx.array(base, dtype=mx.float32), exponent)
        steps = mx.arange(seq_len, dtype=mx.float32)
        freqs = steps[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs)[None, :, None, :]
        sin = mx.sin(freqs)[None, :, None, :]
        return cos, sin

    def init_weights(self) -> None:
        mx.eval(self.parameters())
        self.output_head.weight = mx.zeros_like(self.output_head.weight)
        for block in self.blocks:
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim
        )
        mx.eval(self.cos, self.sin)

    def get_device(self) -> str:
        return str(mx.default_device())

    def __call__(self, x_t: mx.array, t: mx.array) -> mx.array:
        b, seq_len = x_t.shape
        x = rms_norm(self.token_emb(x_t) + self.time_emb(t)[:, None, :])
        cos = self.cos[:, :seq_len]
        sin = self.sin[:, :seq_len]
        for block in self.blocks:
            x = block(x, (cos, sin))
        x = rms_norm(x)
        return self.output_head(x)

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        mask_schedule: "MaskedDiffusionSchedule",
        num_steps: int | None = None,
        temperature: float = 1.0,
        context_tokens: mx.array | None = None,
    ) -> mx.array:
        if num_steps is None:
            num_steps = self.config.diffusion_steps
        x = mx.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=mx.int32,
        )
        if context_tokens is not None:
            context_len = context_tokens.shape[1]
            x = mx.concatenate(
                [context_tokens, x[:, context_len:]], axis=1
            )
        for t in range(num_steps - 1, -1, -1):
            t_batch = mx.full((batch_size,), t, dtype=mx.int32)
            x_masked = mask_schedule.add_masks(x, t_batch)
            logits = self(x_masked, t_batch)
            logits = logits / temperature
            sampled = mx.random.categorical(logits, axis=-1).astype(mx.int32)
            mask = mx.equal(x_masked, self.config.mask_token_id)
            x = mx.where(mask, sampled, x)
        mx.eval(x)
        return x


class MaskedDiffusionSchedule:
    def __init__(self, num_timesteps: int, mask_token_id: int, context_len: int = 0):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.context_len = context_len
        self.mask_probs = mx.linspace(1.0 / num_timesteps, 1.0, num_timesteps)

    def add_masks(
        self,
        x_0: mx.array,
        t: mx.array,
        *,
        return_mask: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        b, seq_len = x_0.shape
        mask_prob = mx.take(self.mask_probs, t)
        noise = mx.random.uniform(shape=(b, seq_len))
        mask = noise < mask_prob[:, None]
        if self.context_len > 0:
            context_guard = mx.arange(seq_len) >= self.context_len
            mask = mask & context_guard[None, :]
        mask_token = mx.full(1, self.mask_token_id, dtype=x_0.dtype)
        x_t = mx.where(mask, mask_token, x_0)
        if return_mask:
            return x_t, mask
        return x_t

    def get_mask_prob(self, t: int) -> float:
        return float(self.mask_probs[t].item())


def encode_text(text: str) -> mx.array:
    tokens = [min(ord(c), 127) for c in text]
    return mx.array(tokens, dtype=mx.int32)


def decode_tokens(tokens: mx.array) -> str:
    mx.eval(tokens)
    flat = tokens.tolist()
    return "".join(chr(int(t)) for t in flat)
