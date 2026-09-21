from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.tasks import EQ
from .common import RMSNorm


# ======================================================================
# Kimi-KDA — from kimi_modular_4tasks_colab.py
# ======================================================================

@dataclass
class ToyKimiConfig:
    d_model: int = 512
    n_heads: int = 8
    d_k: int = 64
    d_v: int = 64
    n_blocks: int = 2
    kda_per_block: int = 3
    ffn_mult: float = 3.0
    vocab_size: int = 50257
    max_seq_len: int = 1024
    chunk_size: int = 64
    conv_kernel: int = 4
    gate_rank: int = 64
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-6

    @property
    def ffn_dim(self) -> int:
        return int(self.d_model * self.ffn_mult)

    def validate(self) -> None:
        assert self.d_model > 0 and self.n_heads > 0
        assert self.n_blocks > 0 and self.kda_per_block > 0
        assert self.ffn_dim > 0 and self.chunk_size > 0
        assert self.conv_kernel >= 1 and self.gate_rank > 0


class ShortConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0,
                              groups=channels, bias=False)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.pad_len, 0))
        return self.conv(xt).transpose(1, 2)


# Reshapes a sequence into chunks and moves the head axis in front of them.
# Input: x - [B, T, H, D] tensor; chunk_size - int, must divide T
# Output: tensor [B, H, T/chunk_size, chunk_size, D]
def _to_chunks(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H, D = x.shape
    N = T // chunk_size
    return x.reshape(B, N, chunk_size, H, D).permute(0, 3, 1, 2, 4).contiguous()


# The chunked gated delta rule of Kimi Linear: within each chunk the delta-rule updates are solved
# in closed form, and the carried state is decayed and passed between chunks. The sequence is
# padded up to a whole number of chunks and trimmed again on the way out.
# Input: q, k, v - [B, T, H, d] projections; log_alpha - per-channel log decay; beta - update gate;
# chunk_size - chunk length; initial_state - carried state or None; scale_q - scale q by 1/sqrt(dk)
# Output: tuple (out - [B, T, H, dv] read-out, S - the final carried state)
def chunk_kda(q, k, v, log_alpha, beta, chunk_size: int,
              initial_state=None, scale_q: bool = True):
    orig_dtype = q.dtype
    B, T_orig, H, dk = q.shape
    dv = v.shape[-1]
    C = chunk_size

    pad = (C - T_orig % C) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        log_alpha = F.pad(log_alpha, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, 0, 0, pad), value=0.0)

    T = T_orig + pad
    N_chunks = T // C

    q_c = _to_chunks(q, C)
    k_c = _to_chunks(k, C)
    v_c = _to_chunks(v, C)
    g = _to_chunks(log_alpha, C).float()
    beta_c = _to_chunks(beta, C).float().squeeze(-1)

    if scale_q:
        q_c = q_c * (dk ** -0.5)

    gc = g.cumsum(dim=3)

    mask_lower = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)
    mask_causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=0)
    zero = torch.tensor(0.0, device=q.device, dtype=torch.float32)

    g_exp = gc.exp()
    neg_g_exp = (-gc).clamp(max=80.0).exp()

    kp = k_c * g_exp.to(k_c.dtype)
    km = k_c * neg_g_exp.to(k_c.dtype)
    Akk = torch.matmul(kp, km.transpose(-1, -2)).float()
    Akk = torch.where(mask_lower, Akk, zero)

    M = -(Akk * beta_c.unsqueeze(-1))
    for i in range(1, C):
        M[..., i, :i] = M[..., i, :i].clone() + (
            M[..., i, :, None].clone() * M[..., :, :i].clone()
        ).sum(dim=-2)
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    M = M + eye
    M = M * beta_c.unsqueeze(-2)

    W = torch.matmul(M.to(k_c.dtype), kp).float()
    U = torch.matmul(M.to(v_c.dtype), v_c).float()

    if initial_state is None:
        S = torch.zeros(B, H, dk, dv, device=q.device, dtype=torch.float32)
    else:
        S = initial_state.float()

    outputs = []
    qp = q_c * g_exp.to(q_c.dtype)

    for n in range(N_chunks):
        qpn = qp[:, :, n]
        kmn = km[:, :, n]
        kn = k_c[:, :, n].float()
        gn = gc[:, :, n]
        Wn = W[:, :, n]
        Un = U[:, :, n]

        Aqk = torch.matmul(qpn, kmn.transpose(-1, -2)).float()
        Aqk = torch.where(mask_causal, Aqk, zero)

        pseudo_v = Un - torch.einsum("bhck,bhkv->bhcv", Wn, S)
        inter = torch.einsum("bhck,bhkv->bhcv", qpn.float(), S)
        intra = torch.matmul(Aqk, pseudo_v)
        outputs.append(inter + intra)

        g_last = gn[:, :, -1:]
        S = S * g_last.squeeze(2).exp().unsqueeze(-1)
        k_decayed = (g_last - gn).exp() * kn
        S = S + torch.einsum("bhck,bhcv->bhkv", k_decayed, pseudo_v)

    out = torch.stack(outputs, dim=2)
    out = out.reshape(B, H, T, dv).permute(0, 2, 1, 3).contiguous()
    if pad > 0:
        out = out[:, :T_orig]
    return out.to(orig_dtype), S.to(orig_dtype)


class KDALayer(nn.Module):
    def __init__(self, cfg: ToyKimiConfig):
        super().__init__()
        cfg.validate()
        d, h, dk, dv = cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v
        self.cfg = cfg
        self.h, self.dk, self.dv = h, dk, dv

        self.W_q = nn.Linear(d, h * dk, bias=False)
        self.W_k = nn.Linear(d, h * dk, bias=False)
        self.W_v = nn.Linear(d, h * dv, bias=False)

        self.conv_q = ShortConv(h * dk, cfg.conv_kernel)
        self.conv_k = ShortConv(h * dk, cfg.conv_kernel)
        self.conv_v = ShortConv(h * dv, cfg.conv_kernel)

        self.W_alpha_down = nn.Linear(d, cfg.gate_rank, bias=False)
        self.W_alpha_up = nn.Linear(cfg.gate_rank, h * dk, bias=False)
        self.W_beta = nn.Linear(d, h, bias=True)

        self.W_g_down = nn.Linear(d, cfg.gate_rank, bias=False)
        self.W_g_up = nn.Linear(cfg.gate_rank, h * dv, bias=False)

        self.head_norm = RMSNorm(dv, eps=cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)
        self.W_o = nn.Linear(h * dv, d, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        h, dk, dv = self.h, self.dk, self.dv

        q_conv = self.conv_q.forward_full(self.W_q(x))
        k_conv = self.conv_k.forward_full(self.W_k(x))
        v_conv = self.conv_v.forward_full(self.W_v(x))

        q = F.silu(q_conv).reshape(B, T, h, dk)
        k = F.silu(k_conv).reshape(B, T, h, dk)
        v = F.silu(v_conv).reshape(B, T, h, dv)
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k = F.normalize(k, p=2, dim=-1, eps=1e-8)

        log_alpha = F.logsigmoid(self.W_alpha_up(self.W_alpha_down(x)))
        log_alpha = log_alpha.reshape(B, T, h, dk).clamp(min=-10.0)
        beta = torch.sigmoid(self.W_beta(x)).reshape(B, T, h, 1)

        o, _ = chunk_kda(q, k, v, log_alpha, beta, self.cfg.chunk_size)

        o = self.head_norm(o)
        gate = torch.sigmoid(self.W_g_up(self.W_g_down(x))).reshape(B, T, h, dv)
        o = gate * o
        o = self.dropout(o)
        return self.W_o(o.reshape(B, T, h * dv))

    # forward in eval mode that also returns the KDA internals behind the legacy "deep KDA" panels.
    # Input: x - [B, T, d] residual-stream tensor
    # Output: dict - q / k / v after conv and silu (q, k also L2-normalised), raw_o (delta-rule
    # read-out), o_norm (head RMSNorm), gate, gated_o and the sub-layer output finished_y, with the
    # heads flattened to [B, T, h*d]
    @torch.no_grad()
    def forward_with_internals(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape
        h, dk, dv = self.h, self.dk, self.dv
        q = F.silu(self.conv_q.forward_full(self.W_q(x))).reshape(B, T, h, dk)
        k = F.silu(self.conv_k.forward_full(self.W_k(x))).reshape(B, T, h, dk)
        v = F.silu(self.conv_v.forward_full(self.W_v(x))).reshape(B, T, h, dv)
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        k = F.normalize(k, p=2, dim=-1, eps=1e-8)
        log_alpha = F.logsigmoid(self.W_alpha_up(self.W_alpha_down(x))).reshape(B, T, h, dk).clamp(min=-10.0)
        beta = torch.sigmoid(self.W_beta(x)).reshape(B, T, h, 1)
        raw_o, _ = chunk_kda(q, k, v, log_alpha, beta, self.cfg.chunk_size)
        o_norm = self.head_norm(raw_o)
        gate = torch.sigmoid(self.W_g_up(self.W_g_down(x))).reshape(B, T, h, dv)
        gated = gate * o_norm
        out = self.W_o(self.dropout(gated).reshape(B, T, h * dv))
        fl = lambda t: t.reshape(B, T, -1)   # noqa: E731
        return out, {"q": fl(q), "k": fl(k), "v": fl(v), "raw_o": fl(raw_o), "o_norm": fl(o_norm),
                     "gate": fl(gate), "gated_o": fl(gated), "finished_y": out}


class NoPEMHALayer(nn.Module):
    def __init__(self, cfg: ToyKimiConfig):
        super().__init__()
        d, h, dk, dv = cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_v
        self.h, self.dk, self.dv = h, dk, dv
        self.W_q = nn.Linear(d, h * dk, bias=False)
        self.W_k = nn.Linear(d, h * dk, bias=False)
        self.W_v = nn.Linear(d, h * dv, bias=False)
        self.W_o = nn.Linear(h * dv, d, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.W_q(x).reshape(B, T, self.h, self.dk).transpose(1, 2)
        k = self.W_k(x).reshape(B, T, self.h, self.dk).transpose(1, 2)
        v = self.W_v(x).reshape(B, T, self.h, self.dv).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        return self.W_o(attn_out.transpose(1, 2).reshape(B, T, self.h * self.dv))


class KimiSwiGLUFFN(nn.Module):
    def __init__(self, cfg: ToyKimiConfig):
        super().__init__()
        d, ffn = cfg.d_model, cfg.ffn_dim
        self.W_gate = nn.Linear(d, ffn, bias=False)
        self.W_up = nn.Linear(d, ffn, bias=False)
        self.W_down = nn.Linear(ffn, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.W_gate(x)) * self.W_up(x)
        x = self.dropout(x)
        return self.W_down(x)

    @torch.no_grad()
    def forward_with_internals(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        mid = F.silu(self.W_gate(x)) * self.W_up(x)
        out = self.W_down(self.dropout(mid))
        return out, {"ffn_mid": mid, "ffn_out": out}


# One macro block: [KDA + FFN] x kda_per_block, followed by [NoPE attention + FFN].
class KimiLinearBlock(nn.Module):
    def __init__(self, cfg: ToyKimiConfig):
        super().__init__()
        self.kda_layers = nn.ModuleList([KDALayer(cfg) for _ in range(cfg.kda_per_block)])
        self.kda_norms = nn.ModuleList(
            [RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(cfg.kda_per_block)])
        self.kda_ffns = nn.ModuleList([KimiSwiGLUFFN(cfg) for _ in range(cfg.kda_per_block)])
        self.kda_ffn_norms = nn.ModuleList(
            [RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(cfg.kda_per_block)])

        self.full_attn = NoPEMHALayer(cfg)
        self.full_attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.full_ffn = KimiSwiGLUFFN(cfg)
        self.full_ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for kda, norm, ffn, ffn_norm in zip(
                self.kda_layers, self.kda_norms, self.kda_ffns, self.kda_ffn_norms):
            x = x + kda(norm(x))
            x = x + ffn(ffn_norm(x))
        x = x + self.full_attn(self.full_attn_norm(x))
        x = x + self.full_ffn(self.full_ffn_norm(x))
        return x


class ToyKimiLinear(nn.Module):
    def __init__(self, cfg: ToyKimiConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([KimiLinearBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class KimiArithmeticModel(nn.Module):
    def __init__(self, cfg: ToyKimiConfig, out_p: int, use_pos_emb: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_pos_emb = use_pos_emb
        self.backbone = ToyKimiLinear(cfg)
        self.answer_head = nn.Linear(cfg.d_model, out_p, bias=False)
        if use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        else:
            self.pos_emb = None

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.backbone.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(input_ids.size(1), device=input_ids.device)
            x = x + self.pos_emb(pos)[None, :, :]
        return self.backbone.drop(x)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(input_ids)
        for block in self.backbone.blocks:
            x = block(x)
        return self.backbone.final_norm(x)

    # Packs a pair of operands into the four-token sequence a, op, b, EQ.
    # Input: a, b - long tensors [B] of operands; op_token - int task token
    # Output: long tensor [B, 4] of token ids
    def make_input_ids(self, a, op_token: int, b):
        B = a.size(0)
        op = torch.full((B,), op_token, device=a.device, dtype=torch.long)
        eq = torch.full((B,), EQ, device=a.device, dtype=torch.long)
        return torch.stack([a, op, b, eq], dim=1)

    def forward(self, a, op_token: int, b):
        h = self.forward_hidden(self.make_input_ids(a, op_token, b))
        return self.answer_head(h[:, -1, :])
