from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core.tasks import ARCHS
from ..runs.records import Run


# Per-item learning time of one run: for each test item, the first step after which it is never
# predicted wrongly again. Items still wrong at the final dump count as never learned.
# Input: run - Run; task - task name; P - modulus; grok - the run's grok step, or None
# Output: pd.DataFrame - one row per test item with a, b, t_item, t_item_rel and run identity
def item_times(run: Run, task: str, P: int, grok: int | None) -> pd.DataFrame:
    files = sorted((run.dir / "preds" / task).glob("step_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    steps = np.array([int(p.stem.split("_")[1]) for p in files])
    z0 = np.load(files[0]); a = np.asarray(z0["a"], dtype=np.int64); b = np.asarray(z0["b"], dtype=np.int64)
    rule = np.asarray(z0["rule"], dtype=np.int64)
    split = np.load(run.dir / "data_split.npz")
    tr = set((split[f"{task}_train_a"].astype(int) * P + split[f"{task}_train_b"].astype(int)).tolist())
    is_test = np.array([(int(x) * P + int(y)) not in tr for x, y in zip(a, b)])
    correct = np.stack([np.asarray(np.load(f)["pred"], dtype=np.int64) == rule for f in files])   # [S, N]
    last_err = np.full(len(a), -1)
    for i, s in enumerate(steps):
        last_err[~correct[i]] = i
    t_item = np.where(last_err + 1 < len(steps), steps[np.minimum(last_err + 1, len(steps) - 1)], np.nan)
    never = last_err == len(steps) - 1        # wrong at the final dump == never learned
    t_item[never] = np.nan
    df = pd.DataFrame({"a": a[is_test], "b": b[is_test], "t_item": t_item[is_test]})
    df["arch"], df["condition"], df["seed"], df["task"] = run.arch, run.condition, run.seed, task
    df["t_item_rel"] = df.t_item / grok if grok else np.nan
    return df


# H3b: is the ORDER in which individual items are learned shared across architectures? Reports
# between-seed reliability within each architecture, then the cross-architecture Spearman
# correlation of the seed-mean item times (the plan criterion is >= 0.4).
# Input: T - stacked item_times frames; runs - the Run records; groks - their grok steps;
# task, rho, wd - the condition; P - modulus
# Output: tuple (pd.DataFrame seed-mean item times [item key x arch], str report text)
def item_order_stats(T: pd.DataFrame, runs, groks, task: str, rho: float, wd: float, P: int):
    T = T.assign(key=T.a * P + T.b)
    L = [f"H3b item order — task {task}, rho {rho:g}, wd {wd:g}; {len(runs)} runs; "
         f"{T.groupby(['arch','seed']).key.nunique().min()} test items per run; never-learned items dropped pairwise"]
    L.append("grid resolution: dense 25 steps in [0,1500], 250 after; grok steps: " +
             ", ".join(f"{r.arch[:5]} s{r.seed} {g}" for r, g in zip(runs, groks)))
    wide = T.pivot_table(index="key", columns=["arch", "seed"], values="t_item")
    L.append("\n[within-arch reliability] Spearman between seeds of per-item grok times")
    for arch in ARCHS:
        cols = [c for c in wide.columns if c[0] == arch]
        rs = [sps.spearmanr(wide[c1], wide[c2], nan_policy="omit")[0] for c1, c2 in combinations(cols, 2)]
        if rs:
            L.append(f"  {arch:<12} " + ", ".join(f"{r:.2f}" for r in rs) + f"   (median {np.median(rs):.2f})")
    means = pd.DataFrame({arch: wide[[c for c in wide.columns if c[0] == arch]].mean(axis=1) for arch in ARCHS if any(c[0] == arch for c in wide.columns)})
    L.append("\n[cross-arch] Spearman of seed-mean per-item grok times (plan criterion >= 0.4)")
    for a1, a2 in combinations(means.columns, 2):
        r, p = sps.spearmanr(means[a1], means[a2], nan_policy="omit")
        L.append(f"  {a1:<12} vs {a2:<12} rho = {r:+.3f}  (p = {p:.1e}, n = {int((means[a1].notna() & means[a2].notna()).sum())})")
    return means, "\n".join(L)
