from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.tasks import EQ, P, VOCAB
from .common import RMSNorm


# ======================================================================
# Mamba-2 — from mamba_modular_4tasks_colab.py
# ======================================================================

@dataclass
class ToyMamba2Config:
    d_model: int = 384
    d_head: int = 64
    d_state: int = 64
    expand: int = 2
    n_layers: int = 2
    n_groups: int = 1
    conv_dim: int = 4
    block_len: int = 8
    vocab_size: int = VOCAB
    max_seq_len: int = 4
    dropout: float = 0.2
    tie_embeddings: bool = False
    norm_eps: float = 1e-6
    use_pos_emb: bool = False

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.d_head

    def validate(self) -> None:
        assert self.d_inner % self.d_head == 0
        assert self.n_heads % self.n_groups == 0


# Segment-sum matrix of a decay sequence: entry (i, j) is the cumulative decay from j to i, and
# -inf above the diagonal so that exponentiating it gives a causal mask.
# Input: x - tensor [..., T] of per-step log decays
# Output: tensor [..., T, T]
def segsum(x: torch.Tensor) -> torch.Tensor:
    T = x.size(-1)
    cumsum = x.cumsum(dim=-1)
    diff = cumsum[..., :, None] - cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return diff.masked_fill(~mask, -float("inf"))


# The SSD (state-space duality) scan of Mamba-2: a chunked linear recurrence, computed as a
# within-chunk quadratic term plus a cross-chunk state that is scanned over chunks. The sequence is
# padded up to a whole number of chunks and trimmed again on the way out.
# Input: X - [B, T, H, P] inputs; A - [B, T, H] log decays; B, C - [B, T, H, N] state projections;
# block_len - chunk length; initial_states - carried state or None
# Output: tuple (Y - [B, T, H, P] outputs, final_state - [B, H, P, N])
def ssd(X, A, B, C, block_len: int = 8, initial_states=None):
    batch, T_orig, H, P_head = X.shape
    N = B.shape[-1]
    Q = block_len

    pad = (Q - T_orig % Q) % Q
    if pad > 0:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    T = T_orig + pad
    n_chunks = T // Q

    X = X.reshape(batch, n_chunks, Q, H, P_head)
    A = A.reshape(batch, n_chunks, Q, H)
    B = B.reshape(batch, n_chunks, Q, H, N)
    C = C.reshape(batch, n_chunks, Q, H, N)

    A = A.permute(0, 3, 1, 2).contiguous()
    A_cumsum = A.cumsum(dim=-1)

    L = segsum(A).exp()
    G = torch.einsum("bclhn,bcshn->bhcls", C, B)
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", L * G, X)

    decay_states = (A_cumsum[..., -1:] - A_cumsum).exp()
    chunk_states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    if initial_states is None:
        initial_states = chunk_states.new_zeros(batch, 1, H, P_head, N)

    states_with_initial = torch.cat([initial_states, chunk_states], dim=1)
    A_chunk_total = F.pad(A_cumsum[..., -1], (1, 0))
    decay_chunk = segsum(A_chunk_total).exp()
    scanned_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_with_initial)

    states = scanned_states[:, :-1]
    final_state = scanned_states[:, -1]

    state_decay_out = A_cumsum.exp()
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(batch, T, H, P_head)
    if pad > 0:
        Y = Y[:, :T_orig]
    return Y, final_state


class Mamba2Block(nn.Module):
    def __init__(self, cfg: ToyMamba2Config):
        super().__init__()
        cfg.validate()
        d = cfg.d_model
        E = cfg.d_inner
        H = cfg.n_heads
        N = cfg.d_state
        P_head = cfg.d_head
        G = cfg.n_groups

        self.H, self.P, self.N, self.G, self.E = H, P_head, N, G, E
        self.block_len = cfg.block_len

        conv_channels = E + 2 * G * N
        proj_dim = E + conv_channels + H
        self.in_proj = nn.Linear(d, proj_dim, bias=False)
        self.conv1d = nn.Conv1d(conv_channels, conv_channels, kernel_size=cfg.conv_dim,
                                padding=cfg.conv_dim - 1, groups=conv_channels, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        dt_init = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001))
                            + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt_init + torch.log(-torch.expm1(-dt_init)))
        self.D = nn.Parameter(torch.ones(H))

        self.norm = RMSNorm(E, eps=cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out_proj = nn.Linear(E, d, bias=False)

        self.proj_splits = [E, conv_channels, H]
        self.conv_splits = [E, G * N, G * N]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, T, _ = u.shape
        H, P_head, N, G, E = self.H, self.P, self.N, self.G, self.E

        z, raw_xBC, dt = self.in_proj(u).split(self.proj_splits, dim=-1)
        xBC = self.conv1d(raw_xBC.transpose(1, 2))[..., :T].transpose(1, 2)
        xBC = F.silu(xBC)
        x, b_flat, c_flat = xBC.split(self.conv_splits, dim=-1)
        x = x.reshape(batch, T, H, P_head)
        b = b_flat.reshape(batch, T, G, N)
        c = c_flat.reshape(batch, T, G, N)
        if G != H:
            repeats = H // G
            b = b.repeat_interleave(repeats, dim=2)
            c = c.repeat_interleave(repeats, dim=2)

        A = -torch.exp(self.A_log.float()).to(dt.dtype)
        dt_pos = F.softplus(dt + self.dt_bias.to(dt.dtype))
        a = dt_pos * A.view(1, 1, H)
        y_ssm, _ = ssd(x, a, b, c, self.block_len)

        y_skip = y_ssm + self.D.to(y_ssm.dtype).view(1, 1, H, 1) * x
        y_flat = y_skip.reshape(batch, T, E)
        y_gated = y_flat * F.silu(z)
        y_norm = self.norm(y_gated)
        return self.out_proj(self.dropout(y_norm))

    # Same computation as forward (eval mode assumed), also returning the block's internal streams at
    # every position - the locations the legacy Mamba manifold figures were drawn at.
    # Input: x - [B, T, d] residual-stream tensor
    # Output: dict - x_stream (post-conv input to the SSD), ssm_y (raw SSD output), skip_y (+D skip),
    # gated_y (x * silu(z)), norm_y (RMSNorm) and out (out_proj)
    @torch.no_grad()
    def forward_with_internals(self, u: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, T, _ = u.shape
        H, P_head, N, G, E = self.H, self.P, self.N, self.G, self.E
        z, raw_xBC, dt = self.in_proj(u).split(self.proj_splits, dim=-1)
        xBC = self.conv1d(raw_xBC.transpose(1, 2))[..., :T].transpose(1, 2)
        xBC = F.silu(xBC)
        x, b_flat, c_flat = xBC.split(self.conv_splits, dim=-1)
        x = x.reshape(batch, T, H, P_head)
        b = b_flat.reshape(batch, T, G, N)
        c = c_flat.reshape(batch, T, G, N)
        if G != H:
            repeats = H // G
            b = b.repeat_interleave(repeats, dim=2)
            c = c.repeat_interleave(repeats, dim=2)
        A = -torch.exp(self.A_log.float()).to(dt.dtype)
        dt_pos = F.softplus(dt + self.dt_bias.to(dt.dtype))
        a = dt_pos * A.view(1, 1, H)
        y_ssm, _ = ssd(x, a, b, c, self.block_len)
        y_skip = y_ssm + self.D.to(y_ssm.dtype).view(1, 1, H, 1) * x
        y_flat = y_skip.reshape(batch, T, E)
        y_gated = y_flat * F.silu(z)
        y_norm = self.norm(y_gated)
        out = self.out_proj(self.dropout(y_norm))
        return out, {"x_stream": x.reshape(batch, T, E), "ssm_y": y_ssm.reshape(batch, T, E),
                     "skip_y": y_flat, "gated_y": y_gated, "norm_y": y_norm, "out": out}


class ToyMamba2Backbone(nn.Module):
    def __init__(self, cfg: ToyMamba2Config):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.use_pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"norm": RMSNorm(cfg.d_model, cfg.norm_eps), "mamba": Mamba2Block(cfg)})
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(input_ids.size(1), device=input_ids.device
                               ).unsqueeze(0).expand_as(input_ids)
            x = x + self.pos_emb(pos)
        return self.drop(x)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer["mamba"](layer["norm"](x))
        return self.final_norm(x)


class MambaArithmeticModel(nn.Module):
    def __init__(self, cfg: ToyMamba2Config, out_p: int = P):
        super().__init__()
        self.cfg = cfg
        self.backbone = ToyMamba2Backbone(cfg)
        self.answer_head = nn.Linear(cfg.d_model, out_p, bias=False)
        nn.init.normal_(self.answer_head.weight, std=0.02)

    # Packs a pair of operands into the four-token sequence a, op, b, EQ.
    # Input: a, b - long tensors [B] of operands; op_token - int task token
    # Output: long tensor [B, 4] of token ids
    def tokens(self, a, op_token: int, b):
        B = a.size(0)
        op = torch.full((B,), op_token, device=a.device, dtype=torch.long)
        eq = torch.full((B,), EQ, device=a.device, dtype=torch.long)
        return torch.stack([a, op, b, eq], dim=1)

    def forward(self, a, op_token: int, b):
        h = self.backbone.forward_hidden(self.tokens(a, op_token, b))
        return self.answer_head(h[:, -1, :])
