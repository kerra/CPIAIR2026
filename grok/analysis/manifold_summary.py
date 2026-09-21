from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.tasks import ARCHS, TASKS

# the per-task instrument: add's natural-order F is contaminated by max's ordinal
# solution >> use the concentration among modes >= 3; div is read in the dlog basis; max is ordinal.
METRIC = {"add": "fourier_conc_hi", "div": "fourier_conc", "max": "linear_corr"}
METRIC_LABEL = {"add": "F, modes ≥ 3", "div": "F, dlog basis", "max": "linear corr"}
MODE_COL = {"add": "top_mode_hi", "div": "top_mode", "max": None}
# forward order of the comparable EQ-position locations (the 'final' figure family)
FINAL_ORDER = {
    "transformer": ["emb_a", "emb_b", "input_eq", "attn0_out_eq", "L0_after_attn_eq", "ffn0_out_eq", "L0_eq",
                    "attn1_out_eq", "L1_after_attn_eq", "ffn1_out_eq", "L1_eq", "final_eq"],
    "mamba": ["emb_a", "emb_b", "input_eq", "ssm0_out_eq", "L0_eq", "ssm1_out_eq", "L1_eq", "final_eq"],
    "kimi": ["emb_a", "emb_b", "input_eq", "kda0_out_eq", "K0_eq", "kda1_out_eq", "K1_eq", "kda2_out_eq", "K2_eq",
             "attn_out_eq", "Lfull_eq", "final_eq"],
}
LAYER0 = re.compile(r"^(t0_\w+|m0_\w+|kda0_\w+|kf0_\w+|attn0_out_eq|L0_after_attn_eq|ffn0_out_eq|ssm0_out_eq|K0_eq|L0_eq)$")


# Loads location_metrics.csv at one checkpoint and adds the task-specific value and dominant mode
# columns, so that every task is read with its own instrument.
# Input: path - Path of the table; step - which checkpoint, "final" by default
# Output: pd.DataFrame with the extra value and mode columns
def load_location_metrics(path: Path, step: str = "final") -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.step.astype(str) == step].copy()
    d["value"] = [getattr(row, METRIC[row.task]) for row in d.itertuples()]
    d["mode"] = [(int(getattr(row, MODE_COL[row.task])) if MODE_COL[row.task] else -1) for row in d.itertuples()]
    return d


# Averages each (arch, task, location) over seeds. A location that is constant across items in ANY
# seed is treated as constant by construction - the EQ token before any mixing, for instance - so
# float noise cannot turn one seed into fake structure.
# Input: d - the table from load_location_metrics
# Output: pd.DataFrame - mean and sd of the metric, the dominant mode per seed, and the circle
# correlation, one row per (arch, task, location)
def seed_mean(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, task, loc), g in d.groupby(["arch", "task", "location"]):
        g = g.sort_values("seed")
        # a location that is constant across items in ANY seed (NaN metrics) is constant by construction
        # (e.g. the EQ token before any mixing); float noise must not turn one seed into fake structure
        const = g.value.isna().any()
        rows.append({"arch": arch, "task": task, "location": loc,
                     "value_mean": np.nan if const else g.value.mean(),
                     "value_sd": np.nan if const else g.value.std(ddof=0), "n_seeds": len(g),
                     "modes": " / ".join(str(m) if m >= 0 else "–" for m in g["mode"]),
                     "best_mode_corr_mean": g.best_mode_corr.mean()})
    return pd.DataFrame(rows)


# The heatmap of one architecture: tasks against the comparable EQ locations in forward order.
# Input: sm - the seed-mean table; arch - architecture name
# Output: tuple (tasks, locations in forward order, np.ndarray [tasks x locations] of values)
def heatmap_grid(sm: pd.DataFrame, arch: str):
    g = sm[sm.arch == arch]
    locs = [l for l in FINAL_ORDER[arch] if l in set(g.location)]
    tasks = [t for t in TASKS if t in set(g.task)]
    grid = np.full((len(tasks), len(locs)), np.nan)
    for i, t in enumerate(tasks):
        for j, loc in enumerate(locs):
            v = g[(g.task == t) & (g.location == loc)].value_mean
            if len(v):
                grid[i, j] = float(v.iloc[0])
    return tasks, locs, grid


# Per (arch, task), the location where the task's structure is strongest among the comparable EQ
# locations of the heatmap, together with the best location of ANY preset (block internals, token
# positions) as best_anywhere.
# Input: sm - the seed-mean table
# Output: pd.DataFrame - one row per (arch, task) with the metric, modes per seed and circle corr
def best_locations(sm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for t in TASKS:
            g = sm[(sm.arch == arch) & (sm.task == t)]
            if g.empty:
                continue
            eq = g[g.location.isin(FINAL_ORDER[arch])]
            r = (eq if len(eq) else g).sort_values("value_mean", ascending=False).iloc[0]
            anyr = g.sort_values("value_mean", ascending=False).iloc[0]
            rows.append({"arch": arch, "task": t, "metric": METRIC_LABEL[t], "location": r.location,
                         "value": f"{r.value_mean:.2f} ± {r.value_sd:.2f}", "modes_per_seed": r.modes,
                         "circle_corr": round(float(r.best_mode_corr_mean), 2),
                         "best_anywhere": f"{anyr.location} ({anyr.value_mean:.2f}; modes {anyr.modes})"})
    return pd.DataFrame(rows)


# The positional-embedding ablation over the layer-0 locations: the task metric without versus
# with learned positional embeddings, plus a flag on the best layer-0 location under each condition.
# Input: sm_nopos, sm_pos - the seed-mean tables of the two variants
# Output: pd.DataFrame - one row per (arch, task, location) with both values, their delta and best
def pos_ablation(sm_nopos: pd.DataFrame, sm_pos: pd.DataFrame) -> pd.DataFrame:
    key = ["arch", "task", "location"]
    a = sm_nopos[sm_nopos.location.str.match(LAYER0)][key + ["value_mean", "modes"]]
    b = sm_pos[sm_pos.location.str.match(LAYER0)][key + ["value_mean", "modes"]]
    m = a.merge(b, on=key, suffixes=("_nopos", "_pos"), how="outer")
    m["delta"] = m.value_mean_pos - m.value_mean_nopos
    best_np = m.loc[m.groupby(["arch", "task"]).value_mean_nopos.idxmax().dropna(), key].assign(b="no-pos")
    best_p = m.loc[m.groupby(["arch", "task"]).value_mean_pos.idxmax().dropna(), key].assign(b="pos")
    flags = pd.concat([best_np, best_p]).groupby(key).b.apply(lambda s: "best " + " & ".join(sorted(s))).reset_index()
    m = m.merge(flags.rename(columns={"b": "best"}), on=key, how="left").fillna({"best": ""})
    m["arch"] = pd.Categorical(m.arch, ARCHS)
    m["task"] = pd.Categorical(m.task, TASKS)
    return m.sort_values(["arch", "task", "value_mean_nopos"], ascending=[True, True, False]).reset_index(drop=True)


# The order in which the tasks of each joint run grokked, one line per run.
# Input: runs_table - Path of runs_table.csv
# Output: list[str] like "arch seed: task(grok) >> task(grok) >> ..."; empty when the file is missing
def grok_order_lines(runs_table: Path) -> list[str]:
    if not Path(runs_table).exists():
        return []
    t = pd.read_csv(runs_table)
    t = t[t.joint] if "joint" in t.columns else t
    out = []
    for (arch, seed), g in t.groupby(["arch", "seed"]):
        g2 = g.sort_values("grok_step", na_position="last")
        out.append(f"{arch:<12} s{seed}: " + " → ".join(
            f"{r.task}({int(r.grok_step) if pd.notna(r.grok_step) else 'cap'})" for r in g2.itertuples()))
    return out


# Renders a frame as a Markdown table, blanking non-finite floats.
# Input: df - the frame; floatfmt - format string for floats
# Output: str - the Markdown table
def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: (floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)  # noqa: E731
                     else ("" if (isinstance(v, float) and not np.isfinite(v)) else str(v)))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
