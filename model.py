# Heavily modified version of nanochat gpt.py to do diffusion
# https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py

"""
Simple Character-Level Discrete Diffusion Transformer
Notable features:
- Bidirectional attention (no causal masking)
- Time step conditioning for diffusion
- Rotary embeddings for positional encoding
- QK norm
- relu^2 activation in MLP
- RMSNorm with no learnable params
- No bias in linear layers
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    sequence_len: int = 256
    vocab_size: int = 128  # Full ASCII (0-127), where 0 is reserved for mask
    mask_token_id: int = 0  # NUL character used as [MASK] token
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    diffusion_steps: int = 128
    context_len: int = 64  # Number of prefix tokens that are never masked

    @property
    def total_vocab_size(self):
        """Total vocabulary size (128 tokens including mask at 0)"""
        return self.vocab_size

    def __str__(self):
        return (
            f"Training Configuration:\n"
            f"  sequence_len: {self.sequence_len}\n"
            f"  vocab_size: {self.vocab_size}\n"
            f"  mask_token_id: {self.mask_token_id}\n"
            f"  n_layer: {self.n_layer}\n"
            f"  n_head: {self.n_head}\n"
            f"  n_embd: {self.n_embd}\n"
            f"  diffusion_steps: {self.diffusion_steps}\n"
            f"  context_len: {self.context_len}\n"
        )


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, T, H, D) -> (B, H, T, D)

        # Bidirectional attention - no causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Re-assemble the heads and project back
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and time embeddings (include mask token in vocab)
        self.token_emb = nn.Embedding(config.total_vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.diffusion_steps, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Output head to predict denoised tokens
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out output head weights
        torch.nn.init.zeros_(self.output_head.weight)
        # Zero out c_proj weights in all blocks
        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # Init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims
        return cos, sin

    def get_device(self):
        return self.token_emb.weight.device

    def forward(self, x_t, t):
        """
        Forward pass for diffusion model
        Args:
            x_t: Noisy tokens at timestep t, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            logits: Predicted token logits, shape (B, T, vocab_size)
        """
        B, T = x_t.size()

        # Get embeddings
        x = self.token_emb(x_t)  # (B, T, n_embd)
        t_emb = self.time_emb(t)  # (B, n_embd)

        # Add time embedding to all positions
        x = x + t_emb.unsqueeze(1)  # broadcast time embedding across sequence
        x = norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)

        # Predict denoised tokens
        logits = self.output_head(x)  # (B, T, vocab_size)
        return logits

    @torch.inference_mode()
    def sample(
        self,
        batch_size,
        seq_len,
        mask_schedule,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
    ):
        """
        Generate samples using masked diffusion process
        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            mask_schedule: MaskedDiffusionSchedule instance
            num_steps: Number of denoising steps (defaults to diffusion_steps)
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = self.config.diffusion_steps

        # Start from all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context tokens provided, set them in the first context_len positions
        if context_tokens is not None:
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens.to(device)

        # Denoise step by step
        for t in reversed(range(num_steps)):
            # Apply masking for this timestep
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_masked = mask_schedule.add_masks(x, t_batch)

            # Predict clean tokens given masked input
            logits = self.forward(x_masked, t_batch)

            # Sample from predicted distribution
            probs = F.softmax(logits / temperature, dim=-1)
            x_new = torch.multinomial(
                probs.view(-1, self.config.vocab_size), num_samples=1
            ).view(batch_size, seq_len)

            # Only update positions that were masked
            mask = x_masked == self.config.mask_token_id
            x = torch.where(mask, x_new, x)

        return x


class MaskedDiffusionSchedule:
    """
    Masked diffusion schedule for discrete diffusion.
    At each timestep, we have a probability of masking a token with [MASK].
    """

    def __init__(self, num_timesteps, mask_token_id, context_len=0):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.context_len = context_len

        # Linear schedule: probability of masking increases linearly
        self.mask_probs = torch.linspace(1.0 / num_timesteps, 1.0, num_timesteps)

    def add_masks(self, x_0, t):
        """
        Add masks to tokens x_0 at timestep
        Args:
            x_0: Clean tokens, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            x_t: Masked tokens at timestep t
        """
        B, T = x_0.shape
        device = x_0.device

        # Get masking probability for each sample (index on CPU, then move to device)
        mask_prob = self.mask_probs[t.cpu()].to(device)  # (B,)

        # Create mask: which tokens to replace with [MASK]
        mask = torch.rand(B, T, device=device) < mask_prob.unsqueeze(1)  # (B, T)

        # Never mask the first context_len tokens
        if self.context_len > 0:
            mask[:, :self.context_len] = False

        # Replace masked positions with mask token
        x_t = torch.where(mask, self.mask_token_id, x_0)

        return x_t

    def get_mask_prob(self, t):
        """Get the masking probability for timestep t"""
        return self.mask_probs[t].item()


def encode_text(text):
    """Convert text to vocab indices using direct ASCII mapping"""
    tokens = torch.tensor([min(ord(c), 127) for c in text], dtype=torch.long)
    return tokens


def decode_tokens(tokens):
    """Convert vocab indices to text using direct ASCII mapping"""
    text = "".join([chr(int(t)) for t in tokens])
    return text
