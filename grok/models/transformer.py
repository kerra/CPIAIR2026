from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.tasks import EQ, VOCAB
from .common import RMSNorm


# ======================================================================
# Transformer (RoPE) — from transformer_modular_4tasks_colab.py
# ======================================================================

@dataclass
class ToyTransformerConfig:
    d_model: int = 384
    n_heads: int = 6
    d_head: int = 64
    n_layers: int = 2
    ffn_mult: float = 4.0
    vocab_size: int = VOCAB
    max_seq_len: int = 4
    dropout: float = 0.2
    tie_embeddings: bool = False
    norm_eps: float = 1e-6
    use_pos_emb: bool = False

    def validate(self) -> None:
        assert self.d_model == self.n_heads * self.d_head
        assert self.max_seq_len > 0 and self.vocab_size > 0
        assert self.n_layers > 0 and self.ffn_mult > 0

    @property
    def ffn_dim(self) -> int:
        return int(self.d_model * self.ffn_mult)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len, device=inv_freq.device, dtype=torch.float32)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0):
        needed = offset + seq_len
        if self.cos_cached.numel() == 0 or needed > self.cos_cached.size(0):
            new_len = max(needed, 2 * max(1, self.cos_cached.size(0)))
            self._build_cache(new_len, device=device, dtype=torch.float32)
        cos = self.cos_cached[offset:offset + seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[offset:offset + seq_len].to(device=device, dtype=dtype)
        return cos, sin


# Rotates the halves of the last dimension, the (-x2, x1) step of RoPE.
# Input: x - tensor [..., d] with even d
# Output: tensor of the same shape
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Applies rotary position embeddings to the query and key tensors.
# Input: q, k - [B, h, T, dh] tensors; cos, sin - the RoPE cache for these positions
# Output: tuple (q_rope, k_rope) of the same shapes
def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ToyTransformerConfig):
        super().__init__()
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        self.h, self.dh = h, dh
        self.W_q = nn.Linear(d, h * dh, bias=False)
        self.W_k = nn.Linear(d, h * dh, bias=False)
        self.W_v = nn.Linear(d, h * dh, bias=False)
        self.W_o = nn.Linear(h * dh, d, bias=False)
        self.rope = RotaryEmbedding(dh, max_seq_len=cfg.max_seq_len)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = self.rope(T, device=x.device, dtype=x.dtype)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, self.h * self.dh)
        return self.W_o(y)

    # forward in eval mode that also returns the sub-layer activations the deep manifold preset reads:
    # q / k / v before RoPE, q_rope / k_rope, the attention read-out before W_o (attn_raw_y) and the
    # sub-layer output.
    # Input: x - [B, T, d] residual-stream tensor
    # Output: dict of tensors, heads flattened to [B, T, h*dh] and outputs [B, T, d]
    @torch.no_grad()
    def forward_with_internals(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = self.rope(T, device=x.device, dtype=x.dtype)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.h * self.dh)
        out = self.W_o(y)
        flat = lambda t: t.transpose(1, 2).contiguous().view(B, T, -1)   # noqa: E731
        return out, {"q": flat(q), "k": flat(k), "v": flat(v), "q_rope": flat(q_rope), "k_rope": flat(k_rope),
                     "attn_raw_y": y, "attn_out": out}


class TransformerSwiGLUFFN(nn.Module):
    def __init__(self, cfg: ToyTransformerConfig):
        super().__init__()
        d, f = cfg.d_model, cfg.ffn_dim
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))

    @torch.no_grad()
    def forward_with_internals(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        mid = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(self.dropout(mid))
        return out, {"ffn_mid": mid, "ffn_out": out}


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ToyTransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = TransformerSwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ToyRoPETransformer(nn.Module):
    def __init__(self, cfg: ToyTransformerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.use_pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, input_ids: torch.Tensor):
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(input_ids.size(1), device=input_ids.device)
            x = x + self.pos_emb(pos)[None, :, :]
        return self.drop(x)

    def forward_hidden(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class TransformerArithmeticModel(nn.Module):
    def __init__(self, cfg: ToyTransformerConfig, out_p: int):
        super().__init__()
        self.cfg = cfg
        self.backbone = ToyRoPETransformer(cfg)
        self.answer_head = nn.Linear(cfg.d_model, out_p, bias=False)

    # Packs a pair of operands into the four-token sequence a, op, b, EQ.
    # Input: a, b - long tensors [B] of operands; op_token - int task token
    # Output: long tensor [B, 4] of token ids
    def make_input_ids(self, a, op_token: int, b):
        B = a.size(0)
        op = torch.full((B,), op_token, device=a.device, dtype=torch.long)
        eq = torch.full((B,), EQ, device=a.device, dtype=torch.long)
        return torch.stack([a, op, b, eq], dim=1)

    def forward(self, a, op_token: int, b):
        h = self.backbone.forward_hidden(self.make_input_ids(a, op_token, b))
        return self.answer_head(h[:, -1, :])
