from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core.tasks import ARCHS

FOURIER = ("lowpass", "etop", "ebottom", "erand")


UNITS = ("neurons", "randunits")


# development: the same lesions at grok-relative checkpoints

TAG_ORDER = ["0.25g", "0.5g", "1g", "final"]


# Loads and concatenates every raw sweep table under base_out for one task. Partially written
# files are reported and skipped, and duplicate rows are resolved keeping the last.
# Input: base_out - Path of the run tree; task - task name
# Output: pd.DataFrame of raw sweep rows; exits when nothing is found
def load_lesion_tables(base_out: Path, task: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(base_out / "paper" / "lesions" / "tables" / "lesions_table*.csv"))
                   + glob.glob(str(base_out / "paper" / "lesions" / "lesions_table*.csv")))
    if not files:
        raise SystemExit("no lesion tables found")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:   # partially written table
            print(f"  skipping {Path(f).name}: {e.__class__.__name__}")
            continue
        if len(df):
            frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.task == task].copy()
    if d.empty:
        raise SystemExit(f"no rows for task {task}")
    if "step_req" not in d.columns:
        d["step_req"] = "final"
    d["step_req"] = d["step_req"].fillna("final").astype(str)
    key = ["arch", "condition", "seed", "step", "task", "lesion", "side", "param", "dose", "control_seed"]
    d = d.drop_duplicates(subset=key, keep="last")
    print(f"loaded {len(files)} tables -> task {task}: {len(d)} rows, {d.groupby(['arch','condition','seed']).ngroups} runs")
    return d


# The unlesioned baseline of each run and checkpoint, renamed so it can be merged back on.
# Input: d - raw sweep rows
# Output: pd.DataFrame - one row per (run, checkpoint) with test0, exc0 and ov0
def baseline_of(d: pd.DataFrame) -> pd.DataFrame:
    b = d[d.lesion == "none"][["arch", "condition", "seed", "step", "step_req", "test_acc", "acc_exc", "overreg_rate"]]
    return b.rename(columns={"test_acc": "test0", "acc_exc": "exc0", "overreg_rate": "ov0"})


# Adds the damage columns: how far the rule and the exception route fell relative to the
# unlesioned baseline, the overregularisation the lesion induced, and the dissociation index
# SI = d_exc - d_test, which is positive when the lesion hurts exceptions more than the rule.
# Input: d - raw sweep rows
# Output: pd.DataFrame - the same rows plus d_test, d_exc, d_ov and SI
def with_deltas(d: pd.DataFrame) -> pd.DataFrame:
    b = baseline_of(d)
    x = d.merge(b, on=["arch", "condition", "seed", "step", "step_req"], how="left")
    x["d_test"] = x.test0 - x.test_acc          # drop of the rule
    x["d_exc"] = x.exc0 - x.acc_exc             # drop of the exception route
    x["d_ov"] = x.overreg_rate - x.ov0          # induced overregularization
    x["SI"] = x.d_exc - x.d_test                # dissociation index (+ = exceptions hurt more)
    return x


# Averages the random-control draws per (run, lesion, side, dose) so that every lesion has exactly
# one row per dose.
# Input: x - sweep rows with deltas
# Output: pd.DataFrame - numeric columns averaged, best_model taken as the mode
def mean_over_controls(x: pd.DataFrame) -> pd.DataFrame:
    g = ["arch", "condition", "seed", "step", "step_req", "task", "rho", "wd", "lesion", "side", "dose"]
    num = [c for c in x.columns if x[c].dtype.kind in "fi" and c not in g]
    out = x.groupby(g, as_index=False)[num].mean()
    bm = x.groupby(g)["best_model"].agg(lambda s: s.mode().iloc[0]).reset_index()
    return out.merge(bm, on=g)


# summary table + stats

# The dose at which a curve first crosses below a threshold, linearly interpolated - the dose that
# destroys a route that was PRESENT without the lesion.
# Input: doses, vals - the dose-response curve; thr - threshold; decreasing - False reverses the dose
# axis, as lowpass needs (accuracy falls as f_cut decreases)
# Output: float - the crossing dose, NaN when the route is already absent at dose 0 or never breaks
def d50(doses, vals, thr=0.5, decreasing=True):
    o = np.argsort(doses); xs, ys = np.asarray(doses)[o], np.asarray(vals)[o]
    if not decreasing:
        xs, ys = xs[::-1], ys[::-1]
    if ys[0] < thr:
        return np.nan
    for i in range(1, len(xs)):
        if ys[i - 1] >= thr > ys[i]:
            f = (ys[i - 1] - thr) / (ys[i - 1] - ys[i] + 1e-12)
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return np.nan


# One row per (run, lesion) summarising its whole dose-response: D50 of the rule and of the
# exception route, the peak overregularisation, and the dissociation index at the reference doses.
# Input: x - sweep rows with deltas; side - which Fourier side to summarise
# Output: pd.DataFrame - one row per run per lesion
def summary_table(x: pd.DataFrame, side: str) -> pd.DataFrame:
    rows = []
    for (arch, cond, seed, rho, wd, les, sd), g in x.groupby(["arch", "condition", "seed", "rho", "wd", "lesion", "side"]):
        if les == "none" or (les in FOURIER and sd != side) or (les in UNITS and sd != "units"):
            continue
        g = g.sort_values("dose")
        dec = les != "lowpass"                       # lowpass: accuracy falls as f_cut DEcreases
        row = {"arch": arch, "condition": cond, "seed": seed, "rho": rho, "wd": wd, "lesion": les, "side": sd,
               "D50_rule": d50(g.dose, g.test_acc, 0.5, dec),
               "D50_exc": d50(g.dose, g.acc_exc, 0.5, dec) if rho > 0 else np.nan,
               "max_overreg": float(g.overreg_rate.max()) if rho > 0 else np.nan,
               "overreg_at_rule_intact": float(g[g.test_acc >= 0.9].overreg_rate.max()) if rho > 0 and (g.test_acc >= 0.9).any() else np.nan}
        for q in (0.25, 0.5):
            gg = g[np.isclose(g.dose, q)] if les in ("etop", "ebottom", "erand") else pd.DataFrame()
            if len(gg):
                row[f"SI_q{q}"] = float(gg.SI.iloc[0]); row[f"d_test_q{q}"] = float(gg.d_test.iloc[0]); row[f"d_exc_q{q}"] = float(gg.d_exc.iloc[0])
        for k in (64, 256):
            gg = g[g.param == k] if les in UNITS else pd.DataFrame()
            if len(gg):
                row[f"SI_k{k}"] = float(gg.SI.iloc[0]); row[f"d_test_k{k}"] = float(gg.d_test.iloc[0]); row[f"d_exc_k{k}"] = float(gg.d_exc.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


# The specificity test: ebottom paired against erand (random modes at the same energy fraction)
# per (run, dose), over rho > 0 runs and interior doses only.
# Input: x - sweep rows with deltas; side - Fourier side; arch - architecture
# Output: pd.DataFrame - condition, seed, dose and the four paired damage columns
def paired_bottom_vs_random(x: pd.DataFrame, side: str, arch: str) -> pd.DataFrame:
    a = x[(x.lesion == "ebottom") & (x.side == side) & (x.arch == arch) & (x.rho > 0)]
    b = x[(x.lesion == "erand") & (x.side == side) & (x.arch == arch) & (x.rho > 0)]
    if a.empty or b.empty:
        return pd.DataFrame(columns=["condition", "seed", "dose", "d_test_bot", "d_exc_bot", "d_test_rnd", "d_exc_rnd"])
    key = ["condition", "seed", "dose"]
    m = a[key + ["d_test", "d_exc"]].merge(b[key + ["d_test", "d_exc"]], on=key, suffixes=("_bot", "_rnd"))
    return m[(m.dose > 0) & (m.dose < 1)]


# The full lesion statistics report: the dose-response of removing the most energetic modes, whether
# exceptions break before the rule when the least energetic modes go, the paired specificity tests
# against random modes and random units, and the cognitive path under band-limiting.
# Input: x - sweep rows with deltas; summ - the summary table; side - Fourier side
# Output: str - the report text
def stats_text(x: pd.DataFrame, summ: pd.DataFrame, side: str) -> str:
    L = [f"Lesion sweep statistics (Fourier side = {side}; per-run values, seeds x rho x WD pooled within arch)"]
    L.append("runs: " + ", ".join(f"{a} {x[x.arch == a].groupby(['condition','seed']).ngroups}" for a in ARCHS))

    def pooled_spearman(les, ycol, sd):
        g = x[(x.lesion == les) & (x.side == sd) & (x.rho > 0 if ycol == "acc_exc" else True)]
        if len(g) < 5:
            return None
        return sps.spearmanr(g.dose, g[ycol])

    L.append("\n[C_R] etop (remove the most energetic modes): dose-response of the RULE (test acc) and of the exceptions")
    for arch in ARCHS:
        g = x[(x.lesion == "etop") & (x.side == side) & (x.arch == arch)]
        if len(g) < 5:
            continue
        r1 = sps.spearmanr(g.dose, g.test_acc); ge = g[g.rho > 0]
        r2 = sps.spearmanr(ge.dose, ge.acc_exc) if len(ge) > 4 else None
        s = summ[(summ.lesion == "etop") & (summ.arch == arch)]
        L.append(f"  {arch:<12} Spearman(dose, test_acc) = {r1[0]:+.2f}" + (f", (dose, acc_exc) = {r2[0]:+.2f}" if r2 else "") +
                 f"; median D50 rule = {s.D50_rule.median():.2f}, D50 exc = {s.D50_exc.median():.2f} (energy fraction)")
    L.append("\n[C_E] ebottom (remove the LEAST energetic modes): do exceptions break before the rule?")
    for arch in ARCHS:
        s = summ[(summ.lesion == "ebottom") & (summ.arch == arch) & (summ.rho > 0)]
        if s.empty:
            continue
        for q in (0.25, 0.5):
            if f"SI_q{q}" in s:
                v = s[f"SI_q{q}"].dropna()
                L.append(f"  {arch:<12} q={q}: SI median = {v.median():+.2f}, SI >= 0.3 in {int((v >= 0.3).sum())}/{len(v)} runs; "
                         f"drop test = {s[f'd_test_q{q}'].median():.2f}, drop exc = {s[f'd_exc_q{q}'].median():.2f}")
        L.append(f"  {arch:<12} median D50 exc = {s.D50_exc.median():.2f} vs D50 rule = {s.D50_rule.median():.2f}; "
                 f"max overreg with rule intact (test >= .9): median {s.overreg_at_rule_intact.median():.2f}, "
                 f">= 0.2 in {int((s.overreg_at_rule_intact >= 0.2).sum())}/{len(s)} runs")
    L.append("\n[specificity] ebottom vs erand (random modes, same energy fraction), paired over runs")
    for arch in ARCHS:
        m = paired_bottom_vs_random(x, side, arch)
        if len(m) < 5:
            continue
        w1 = sps.wilcoxon(m.d_test_bot, m.d_test_rnd, alternative="less")
        w2 = sps.wilcoxon(m.d_exc_bot, m.d_exc_rnd, alternative="greater")
        L.append(f"  {arch:<12} rule damage: bottom < random in {int((m.d_test_bot < m.d_test_rnd).sum())}/{len(m)} (p={w1.pvalue:.1e}); "
                 f"exception damage: bottom > random in {int((m.d_exc_bot > m.d_exc_rnd).sum())}/{len(m)} (p={w2.pvalue:.1e})")
    L.append("\n[units] gradient-ranked units vs random units (same k), paired over runs, rho > 0")
    for arch in ARCHS:
        a = x[(x.lesion == "neurons") & (x.arch == arch) & (x.rho > 0)]
        b = x[(x.lesion == "randunits") & (x.arch == arch) & (x.rho > 0)]
        if a.empty or b.empty:
            continue
        key = ["condition", "seed", "param"]
        m = a[key + ["d_test", "d_exc", "d_ov"]].merge(b[key + ["d_test", "d_exc", "d_ov"]], on=key, suffixes=("_sel", "_rnd"))
        for k in (64, 256):
            mm = m[m.param == k]
            if len(mm) < 3:
                continue
            L.append(f"  {arch:<12} k={k}: drop exc sel {mm.d_exc_sel.median():.2f} vs rnd {mm.d_exc_rnd.median():.2f}; "
                     f"drop test sel {mm.d_test_sel.median():.2f} vs rnd {mm.d_test_rnd.median():.2f}; "
                     f"induced overreg sel {mm.d_ov_sel.median():+.2f} vs rnd {mm.d_ov_rnd.median():+.2f}; "
                     f"sel > rnd on exc in {int((mm.d_exc_sel > mm.d_exc_rnd).sum())}/{len(mm)}")
    L.append("\n[low-pass] best cognitive model along f_cut (74 -> 0) and GCM kernel width")
    lp = x[(x.lesion == "lowpass") & (x.side == side)]
    for arch in ARCHS:
        g = lp[lp.arch == arch]
        if g.empty:
            continue
        paths, n_gcm_between, n_runs = [], 0, 0
        for _, r in g.groupby(["condition", "seed"]):
            r = r.sort_values("dose", ascending=False)
            seq = []
            for m in r.best_model:
                fam = "gcm" if m.startswith("gcm") else m
                if not seq or seq[-1] != fam:
                    seq.append(fam)
            paths.append(">".join(seq)); n_runs += 1
            if "gcm" in seq and seq[0] in ("rule", "rulex"):
                n_gcm_between += 1
        vc = pd.Series(paths).value_counts()
        gl = g[(g.dose > 0) & (g.test_acc < 0.9)]
        rl = sps.spearmanr(gl.dose, gl.gcm_circular_lambda) if len(gl) > 4 else None
        L.append(f"  {arch:<12} rule -> gcm -> ... in {n_gcm_between}/{n_runs} runs; paths: " + "; ".join(f"{k} x{v}" for k, v in vc.items()))
        if rl:
            L.append(f"  {'':<12} Spearman(f_cut, GCM lambda | test acc < .9) = {rl[0]:+.2f} (kernel sharpens with bandwidth if > 0)")
    return "\n".join(L)
