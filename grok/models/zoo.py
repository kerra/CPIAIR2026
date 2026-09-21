from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core.tasks import ARCHS, P, TASK_OP, VOCAB, rule_label
from .common import PARAM_TARGET, PARAM_TOL, count_params, device
from .kimi import KimiArithmeticModel, ToyKimiConfig
from .mamba import MambaArithmeticModel, ToyMamba2Config
from .transformer import ToyTransformerConfig, TransformerArithmeticModel

# legacy per-script defaults that matter for comparability
ARCH_INFO = {
    "transformer": {"use_pos_emb": False, "dropout": 0.2},
    "mamba":       {"use_pos_emb": False, "dropout": 0.2},
    "kimi":        {"use_pos_emb": False, "dropout": 0.2},
}


# The matrix (99 runs) is "matched_2M": no learned positional embeddings anywhere (RoPE in the transformer,
# none in mamba / kimi). "matched_2M_pos" is the SAME configuration plus a learned 4 x d_model positional
# embedding in every architecture (+1024 / +1536 / +768 params, inside the ±10 % budget). It is an ablation:
# train it into its own --base-out (e.g. ./exceptions_runs_pos), never into ./exceptions_runs.
VARIANTS = {"matched_2M": {"use_pos_emb": None},        # None >> per-arch ARCH_INFO default (False)
            "matched_2M_pos": {"use_pos_emb": True}}


# Seed-deterministic construction of one of the three architectures. There is ONE size,
# matched_2M, in two variants (see VARIANTS).
# Input: arch - "transformer" | "mamba" | "kimi"; seed - int, seeds torch; variant - key of VARIANTS
# Output: nn.Module already moved to the active device
def make_model(arch: str, seed: int, variant: str = "matched_2M") -> nn.Module:
    assert variant in VARIANTS, f"unknown variant {variant!r}; sanctioned: {list(VARIANTS)}"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pos_override = VARIANTS[variant]["use_pos_emb"]
    use_pos = ARCH_INFO[arch]["use_pos_emb"] if pos_override is None else bool(pos_override)

    if arch == "transformer":
        cfg = ToyTransformerConfig(
            d_model=256, n_heads=4, d_head=64, n_layers=2, ffn_mult=3.0,
            vocab_size=VOCAB, max_seq_len=4, dropout=0.2,
            tie_embeddings=False, use_pos_emb=use_pos)
        model = TransformerArithmeticModel(cfg, out_p=P)
    elif arch == "mamba":
        cfg = ToyMamba2Config(
            d_model=384, d_head=64, d_state=64, expand=2, n_layers=2,
            n_groups=1, conv_dim=4, block_len=8, vocab_size=VOCAB,
            max_seq_len=4, dropout=0.2, tie_embeddings=False,
            use_pos_emb=use_pos)
        model = MambaArithmeticModel(cfg, out_p=P)
    elif arch == "kimi":
        cfg = ToyKimiConfig(
            d_model=192, n_heads=3, d_k=64, d_v=64, n_blocks=1, kda_per_block=3,
            ffn_mult=2.5, vocab_size=VOCAB, max_seq_len=4, chunk_size=4,
            conv_kernel=4, gate_rank=64, dropout=0.2, tie_embeddings=False)
        model = KimiArithmeticModel(cfg, out_p=P, use_pos_emb=use_pos)
    else:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")

    return model.to(device)


# lean per-arch capture for the mechanistic analysis: final_eq (the EQ-position state after the final norm)

# Lean per-architecture capture for the mechanistic analysis: the EQ-position state after the
# final norm. The caller must have set model.eval() first, since dropout is 0.2 on every arch.
# Input: arch - architecture name; model - nn.Module; a, b - long tensors of operands;
# op_token - int task token
# Output: dict - {"final_eq": np.ndarray [B, d]}
@torch.no_grad()
def eq_states(arch: str, model: nn.Module, a: torch.Tensor, op_token: int,
              b: torch.Tensor) -> dict:
    if arch == "transformer":
        ids = model.make_input_ids(a, op_token, b)
        x = model.backbone.embed(ids)
        for layer in model.backbone.layers:
            x = layer(x)
        final = model.backbone.final_norm(x)[:, -1, :]
    elif arch == "mamba":
        ids = model.tokens(a, op_token, b)
        x = model.backbone.embed(ids)
        for layer in model.backbone.layers:
            x = x + layer["mamba"](layer["norm"](x))
        final = model.backbone.final_norm(x)[:, -1, :]
    elif arch == "kimi":
        ids = model.make_input_ids(a, op_token, b)
        x = model._embed(ids)
        for block in model.backbone.blocks:
            x = block(x)
        final = model.backbone.final_norm(x)[:, -1, :]
    else:
        raise ValueError(arch)
    return {"final_eq": final.detach().float().cpu().numpy()}


# torch adapters for evaluation

# Runs a model over numpy (a, b) arrays in batches and brings the logits back to numpy.
# Input: model - nn.Module; a_np, b_np - operand arrays; op_token - int task token;
# device - torch device; batch_size - items per forward pass
# Output: np.ndarray [n_items, P] - the logits
def batched_logits(model, a_np, op_token: int, b_np, device,
                   batch_size: int = 4096) -> np.ndarray:
    a = torch.as_tensor(np.asarray(a_np), dtype=torch.long, device=device)
    b = torch.as_tensor(np.asarray(b_np), dtype=torch.long, device=device)
    outs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(a), batch_size):
            outs.append(model(a[s:s + batch_size], op_token,
                              b[s:s + batch_size]).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


# Model predictions on the complete task table, as the offline cognitive fits consume them.
# Input: model - nn.Module; task - task name; op_token - int task token; P - modulus;
# device - torch device; batch_size - items per forward pass
# Output: dict - a, b, pred (argmax) and rule (ground truth) over the full table
def full_table_predictions(model, task: str, op_token: int, P: int, device,
                           batch_size: int = 4096) -> dict:
    aa, bb = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    if task == "div":
        keep = bb != 0
        aa, bb = aa[keep], bb[keep]
    logits = batched_logits(model, aa, op_token, bb, device, batch_size)
    return {"a": aa, "b": bb, "pred": logits.argmax(axis=1),
            "rule": rule_label(task, aa, bb, P)}


# Builds all three architectures, checks their forward pass and EQ capture, and prints whether
# each one sits inside the 2.0M +- 10 % budget.
# Input: none
# Output: None - prints one row per architecture
def param_report() -> None:
    rows = []
    for arch in ARCHS:
        m = make_model(arch, seed=0)
        n = count_params(m)
        rows.append((arch, n))
        a = torch.randint(0, P, (8,), device=device)
        b = torch.randint(0, P, (8,), device=device)
        logits = m(a, TASK_OP["add"], b)
        assert logits.shape == (8, P), (arch, logits.shape)
        m.eval()
        st = eq_states(arch, m, a, TASK_OP["add"], b)
        assert st["final_eq"].shape[0] == 8
        del m
    print(f"{'arch':<12} {'params':>10}   in 2.0M ± 10%?")
    for arch, n in rows:
        ok = PARAM_TARGET * (1 - PARAM_TOL) <= n <= PARAM_TARGET * (1 + PARAM_TOL)
        print(f"{arch:<12} {n:>10,}   {'OK' if ok else '*** OUT OF BAND ***'}")


if __name__ == "__main__":
    param_report()
