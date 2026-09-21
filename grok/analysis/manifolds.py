from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.fourier import best_mode_corr, fourier_concentration, is_degenerate, linear_corr
from ..core.tasks import P, TASK_OP, full_table
from ..models.common import device

BATCH = {"transformer": 2048, "kimi": 2048, "mamba": 512}   # mamba's SSD einsums are memory-hungry
# forward-pass capture (manual walk mirroring arch_zoo.eq_states)

POS_NAMES = {0: "pos0_a", 1: "pos1_op", 2: "pos2_b"}   # positions kept besides EQ when --positions is on


# the three location families of the paper's manifold summary (regexes over location names)
PRESETS = {
    "final": r"^(emb_a|emb_b|input_eq|attn\d_out_eq|L\d_after_attn_eq|ffn\d_out_eq|L\d_eq|ssm\d_out_eq|"
             r"kda\d_out_eq|K\d_eq|attn_out_eq|Lfull_eq|final_eq)$",
    "positions": r"^(input|L\d|Lfull|final)_(pos0_a|pos1_op|pos2_b|eq)$",
    "deep": r"^(t0_\w+|m0_\w+|kda0_(q|k|v|raw_o|o_norm|gate|gated_o)_eq|kf0_ffn_mid_eq|"
            r"attn0_out_eq|L0_after_attn_eq|ffn0_out_eq|ssm0_out_eq|kda0_out_eq|K0_eq)$",   # first sublayer
}


# Walks the forward pass by hand (mirroring models.zoo.eq_states) and captures the residual stream
# and sublayer outputs at the EQ position, plus the token embeddings. deep also captures the
# inside of every block; positions also captures the residual stream at the a, op and b positions,
# which answers where along the sequence the answer becomes decodable.
# Input: arch - architecture name; model - nn.Module; a, b - operand tensors; op - task token;
# deep - capture block internals; positions - capture the non-EQ positions
# Output: dict {location: np.ndarray [n_items, d]}
@torch.no_grad()
def capture_locations(arch: str, model, a, op: int, b, deep: bool = False,
                      positions: bool = False) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}

    def put(name, v):
        out.setdefault(name, []).append(v.detach().float().cpu().numpy())

    def put_pos(prefix, x):
        if positions:
            for pos, nm in POS_NAMES.items():
                put(f"{prefix}_{nm}", x[:, pos])

    bs = BATCH[arch]
    for s in range(0, len(a), bs):
        aa, bb = a[s:s + bs], b[s:s + bs]
        if arch == "transformer":
            x = model.backbone.embed(model.make_input_ids(aa, op, bb))
            put("emb_a", x[:, 0]); put("emb_b", x[:, 2]); put("input_eq", x[:, -1]); put_pos("input", x)
            for i, layer in enumerate(model.backbone.layers):
                xn = layer.attn_norm(x)
                if deep:
                    put(f"t{i}_attn_norm_eq", xn[:, -1])
                    d, it = layer.attn.forward_with_internals(xn)
                    for k in ("q", "k", "v", "q_rope", "k_rope", "attn_raw_y"):
                        put(f"t{i}_{k}_eq", it[k][:, -1])
                else:
                    d = layer.attn(xn)
                put(f"attn{i}_out_eq", d[:, -1]); x = x + d; put(f"L{i}_after_attn_eq", x[:, -1])
                fn = layer.ffn_norm(x)
                if deep:
                    put(f"t{i}_ffn_norm_eq", fn[:, -1])
                    d, it = layer.ffn.forward_with_internals(fn); put(f"t{i}_ffn_mid_eq", it["ffn_mid"][:, -1])
                else:
                    d = layer.ffn(fn)
                put(f"ffn{i}_out_eq", d[:, -1]); x = x + d
                put(f"L{i}_eq", x[:, -1]); put_pos(f"L{i}", x)
            xf = model.backbone.final_norm(x)
            put("final_eq", xf[:, -1]); put_pos("final", xf)
        elif arch == "mamba":
            x = model.backbone.embed(model.tokens(aa, op, bb))
            put("emb_a", x[:, 0]); put("emb_b", x[:, 2]); put("input_eq", x[:, -1]); put_pos("input", x)
            for i, layer in enumerate(model.backbone.layers):
                if deep:
                    d, internals = layer["mamba"].forward_with_internals(layer["norm"](x))
                    for k in ("x_stream", "ssm_y", "skip_y", "gated_y", "norm_y"):
                        put(f"m{i}_{k}_eq", internals[k][:, -1])
                        if positions:
                            put(f"m{i}_{k}_pos2_b", internals[k][:, 2])
                else:
                    d = layer["mamba"](layer["norm"](x))
                put(f"ssm{i}_out_eq", d[:, -1]); x = x + d
                put(f"L{i}_eq", x[:, -1]); put_pos(f"L{i}", x)
            xf = model.backbone.final_norm(x)
            put("final_eq", xf[:, -1]); put_pos("final", xf)
        elif arch == "kimi":
            x = model._embed(model.make_input_ids(aa, op, bb))
            put("emb_a", x[:, 0]); put("emb_b", x[:, 2]); put("input_eq", x[:, -1]); put_pos("input", x)
            for block in model.backbone.blocks:
                for j, (kda, norm, ffn, ffn_norm) in enumerate(zip(
                        block.kda_layers, block.kda_norms, block.kda_ffns, block.kda_ffn_norms)):
                    xn = norm(x)
                    if deep:
                        d, it = kda.forward_with_internals(xn)
                        for k in ("q", "k", "v", "raw_o", "o_norm", "gate", "gated_o"):
                            put(f"kda{j}_{k}_eq", it[k][:, -1])
                    else:
                        d = kda(xn)
                    put(f"kda{j}_out_eq", d[:, -1]); x = x + d
                    fn = ffn_norm(x)
                    if deep:
                        d, it = ffn.forward_with_internals(fn); put(f"kf{j}_ffn_mid_eq", it["ffn_mid"][:, -1])
                    else:
                        d = ffn(fn)
                    x = x + d; put(f"K{j}_eq", x[:, -1]); put_pos(f"K{j}", x)
                d = block.full_attn(block.full_attn_norm(x)); put("attn_out_eq", d[:, -1]); x = x + d
                x = x + block.full_ffn(block.full_ffn_norm(x)); put("Lfull_eq", x[:, -1]); put_pos("Lfull", x)
            xf = model.backbone.final_norm(x)
            put("final_eq", xf[:, -1]); put_pos("final", xf)
        else:
            raise ValueError(arch)
    return {k: np.concatenate(v, 0) for k, v in out.items()}


# manifold metrics

# Mean activation per class.
# Input: H - [n_items, d] activations; c - class label per item; n_cls - number of classes
# Output: np.ndarray [n_cls, d]; rows of empty classes stay zero
def per_class_mean(H: np.ndarray, c: np.ndarray, n_cls: int) -> np.ndarray:
    M = np.zeros((n_cls, H.shape[1]), dtype=np.float64)
    for cls in range(n_cls):
        idx = c == cls
        if idx.any():
            M[cls] = H[idx].mean(0)
    return M


# The geometry metrics of one manifold panel: Fourier concentration (over all modes and over
# modes >= 3), the dominant modes, linear correlation and the best-mode circle correlation. The
# marker column picks the instrument the task is judged by - linear correlation for max, Fourier
# concentration otherwise.
# Input: rows - [n_classes, d] centred class means; task - task name
# Output: dict of metrics; all NaN or -1 when the panel is degenerate
def task_metrics(rows: np.ndarray, task: str) -> dict:
    if is_degenerate(rows):
        return {"fourier_conc": np.nan, "top_mode": -1, "top_modes": "", "fourier_conc_hi": np.nan,
                "top_mode_hi": -1, "top_modes_hi": "",
                "linear_corr": np.nan, "best_mode": -1, "best_mode_corr": np.nan, "marker": np.nan}
    conc, modes = fourier_concentration(rows)
    conc_hi, modes_hi = fourier_concentration(rows, min_mode=3)
    top = modes[0] if modes else -1
    top_hi = modes_hi[0] if modes_hi else -1
    m = {"fourier_conc": conc, "top_mode": top, "top_modes": "|".join(map(str, modes)),
         "fourier_conc_hi": conc_hi, "top_mode_hi": top_hi,
         "top_modes_hi": "|".join(map(str, modes_hi)),
         "linear_corr": linear_corr(rows)}
    m["best_mode"], m["best_mode_corr"] = best_mode_corr(rows)
    m["marker"] = m["linear_corr"] if task == "max" else m["fourier_conc"]
    return m


# Class means at every captured location over the full task table. The token-embedding panels
# (emb_a, emb_b) are grouped by the token's own value - the number line as the model embeds it -
# and every EQ-position panel by the answer class.
# Input: arch - architecture; model - nn.Module; task - task name; deep, positions - see
# capture_locations; locs - regex keeping only some locations, or None
# Output: dict {location: np.ndarray [P, d]}
def task_class_means(arch: str, model, task: str, deep: bool = False, positions: bool = False,
                     locs: str | None = None) -> dict[str, np.ndarray]:
    a, b, c = full_table(task)
    ta, tb = torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
    H = capture_locations(arch, model, ta, TASK_OP[task], tb, deep=deep, positions=positions)
    if locs:
        H = {k: v for k, v in H.items() if re.search(locs, k)}
    grp = {loc: (a if loc == "emb_a" else b if loc == "emb_b" else c) for loc in H}
    return {loc: per_class_mean(H[loc], grp[loc], P) for loc in H}


# Merges new rows into an existing location_metrics.csv, which several presets and calls share:
# rows with the same (arch, seed, step, task, location) are replaced and everything else is kept.
# Input: csv_path - Path of the table; df - the new rows
# Output: pd.DataFrame - the merged table, also written back to csv_path
def merge_location_metrics(csv_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        key = ["arch", "seed", "step", "task", "location"]
        new_keys = set(map(tuple, df[key].astype(str).values))
        old = old[~old[key].astype(str).apply(tuple, axis=1).isin(new_keys)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df
