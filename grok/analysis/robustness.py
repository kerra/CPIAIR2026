from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core.tasks import ARCHS
from ..runs.io import read_csv_or_empty
from .stats import fit_logistic

GROK_THRS = (0.90, 0.95, 0.99)
EXC_THRS = (0.90, 0.95, 0.99)


# The three route variables of one run - grok time, exception-memorisation time and peak
# overregularisation - recomputed under alternative threshold definitions.
# Input: mdf - the run's metrics frame; task - task name; grok_thr - test-accuracy threshold;
# exc_thr - exception-accuracy threshold, applied to a 5-point rolling median
# Output: dict - t_grok, t_exc, peak_overreg and their ratio
def route_variables(mdf: pd.DataFrame, task: str, grok_thr: float, exc_thr: float) -> dict:
    steps = mdf["step"].values.astype(float)
    test = mdf[f"test_{task}"].values.astype(float)
    hit = np.where(test > grok_thr)[0]
    t_grok = float(steps[hit[0]]) if len(hit) else np.nan
    ae = pd.Series(mdf["acc_exc"].values.astype(float)).rolling(5, center=True, min_periods=1).median().values
    hit2 = np.where(ae >= exc_thr)[0]
    t_exc = float(steps[hit2[0]]) if len(hit2) else np.nan
    ov = mdf["overreg_rate"].values.astype(float)
    return {"t_grok": t_grok, "t_exc": t_exc, "peak_overreg": float(np.nanmax(ov)),
            "ratio": (t_exc / t_grok if np.isfinite(t_exc) and np.isfinite(t_grok) and t_grok > 0 else np.nan)}


# One row per single-task rho > 0 run that has a defined ratio, with cell = arch x task x rho x wd
# as the unit of independent variation. With censored, runs whose exceptions never reached the
# memorisation threshold before the stop rule ended them enter with t_exc set to the last step - a
# lower bound on the ratio - instead of being dropped.
# Input: runs - the Run records; grok_thr, exc_thr - threshold definitions; censored - keep the
# runs that never memorised
# Output: pd.DataFrame with the route variables, the cell label and x = log10(ratio)
def route_table(runs, grok_thr: float = 0.95, exc_thr: float = 0.95, censored: bool = False) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run.is_joint or run.rho <= 0:
            continue
        mdf = read_csv_or_empty(run.dir / "analysis_offline" / "metrics.csv")
        if mdf.empty or "acc_exc" not in mdf.columns:
            continue
        v = route_variables(mdf, run.task, grok_thr, exc_thr)
        if censored and np.isfinite(v["t_grok"]) and not np.isfinite(v["t_exc"]):
            v["t_exc"] = float(mdf["step"].values[-1]); v["ratio"] = v["t_exc"] / v["t_grok"]; v["censored"] = True
        else:
            v["censored"] = False
        rows.append({"arch": run.arch, "task": run.task, "rho": run.rho, "wd": run.wd, "seed": run.seed,
                     "cell": f"{run.arch}|{run.task}|{run.rho:g}|{run.wd:g}", **v})
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d.ratio)].copy()
    d["x"] = np.log10(d.ratio)
    return d.reset_index(drop=True)


# The C2 fit on one route table: the Spearman correlation and the logistic R^2, with and without
# per-architecture intercepts, so that delta_r2_arch says what architecture adds.
# Input: d - a route table
# Output: dict - n, spearman, the two R^2 values, their difference and the fitted parameters
def logistic_summary(d: pd.DataFrame) -> dict:
    y, x = d.peak_overreg.values.astype(float), d.x.values.astype(float)
    r2_t, _, p = fit_logistic(x, y)
    r2_a, _, _ = fit_logistic(x, y, groups=list(d.arch))
    return {"n": int(len(d)), "spearman": float(sps.spearmanr(x, y)[0]), "r2_logistic": float(r2_t),
            "r2_logistic_arch": float(r2_a), "delta_r2_arch": float(r2_a - r2_t), "a": float(p[0]), "b": float(p[1])}


# The C2 fit recomputed across the whole grid of grok and exception thresholds.
# Input: runs - the Run records
# Output: pd.DataFrame - one row per threshold pair with at least 10 runs
def sensitivity(runs) -> pd.DataFrame:
    rows = []
    for g in GROK_THRS:
        for e in EXC_THRS:
            d = route_table(runs, g, e)
            if len(d) >= 10:
                rows.append({"grok_thr": g, "exc_thr": e, **logistic_summary(d)})
    return pd.DataFrame(rows)


# Percentile confidence intervals for the C2 statistics, resampling CELLS with replacement -
# the unit of independent variation, since the three seeds inside a cell are not independent.
# Input: d - a route table; n_boot - bootstrap draws; seed - RNG seed
# Output: pd.DataFrame - one row per statistic with its 95 % interval
def cluster_bootstrap(d: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = d.cell.unique()
    by_cell = {c: d[d.cell == c] for c in cells}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(cells, size=len(cells), replace=True)
        db = pd.concat([by_cell[c] for c in pick], ignore_index=True)
        if db.arch.nunique() >= 2 and db.x.std() > 0:
            boots.append(logistic_summary(db))
        rng.integers(0, len(d), len(d))          # keeps the draw sequence of the stored statistics
    df = pd.DataFrame(boots)
    rows = []
    for col in ("spearman", "r2_logistic", "delta_r2_arch", "b", "a"):
        lo, hi = np.percentile(df[col], [2.5, 97.5])
        rows.append({"resampled": "cells", "statistic": col, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": len(df)})
    return pd.DataFrame(rows)


# Out-of-sample test of the C2 relation: fit the logistic curve on two architectures and
# predict the third.
# Input: d - a route table
# Output: pd.DataFrame - one row per held-out architecture with R^2, RMSE and mean absolute error
def leave_one_arch_out(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        tr, te = d[d.arch != arch], d[d.arch == arch]
        if len(te) < 3:
            continue
        _, _, p = fit_logistic(tr.x.values, tr.peak_overreg.values)
        pred = 1.0 / (1.0 + np.exp(-(p[0] + p[1] * te.x.values)))
        yte = te.peak_overreg.values
        ss_res = float(np.sum((yte - pred) ** 2)); ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        rows.append({"held_out_arch": arch, "n_train": len(tr), "n_test": len(te),
                     "r2_out_of_sample": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rmse": float(np.sqrt(ss_res / len(te))), "mean_abs_err": float(np.mean(np.abs(yte - pred)))})
    return pd.DataFrame(rows)


# Splits the route-order relation into its between-cell part (cell medians) and its within-cell
# part (cell-demeaned residuals), which says whether the relation survives inside a condition.
# Input: d - a route table
# Output: dict - the between-cell correlation and R^2, the pooled within-cell correlation, and
# how many cells show a positive within-cell relation
def between_within(d: pd.DataFrame) -> dict:
    cm = d.groupby("cell").agg(x=("x", "median"), y=("peak_overreg", "median"), arch=("arch", "first")).reset_index()
    r2_b, _, _ = fit_logistic(cm.x.values, cm.y.values)
    dm = d.copy()
    dm["x_w"] = dm.x - dm.groupby("cell").x.transform("mean")
    dm["y_w"] = dm.peak_overreg - dm.groupby("cell").peak_overreg.transform("mean")
    ok = (dm.groupby("cell").x.transform("std") > 0)
    within = sps.spearmanr(dm.x_w[ok], dm.y_w[ok])
    signs = []
    for _, g in d.groupby("cell"):
        if len(g) >= 3 and g.x.std() > 0 and g.peak_overreg.std() > 0:
            signs.append(sps.spearmanr(g.x, g.peak_overreg)[0])
    signs = np.array(signs)
    return {"n_cells": int(len(cm)), "spearman_between": float(sps.spearmanr(cm.x, cm.y)[0]), "r2_logistic_between": float(r2_b),
            "spearman_within_pooled": float(within[0]), "p_within": float(within[1]), "n_within": int(ok.sum()),
            "cells_with_positive_within_rho": int((signs > 0).sum()), "cells_tested": int(len(signs))}


# Random-intercept model on the logit of the peak, logit(y) ~ x + (1 | cell), and the same model
# with architecture as a fixed effect, compared by a likelihood-ratio test.
# Input: d - a route table
# Output: dict of fitted effects, or None when statsmodels is not installed
def mixed_model(d: pd.DataFrame) -> dict | None:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return None
    dd = d.copy()
    yc = np.clip(dd.peak_overreg.values, 1e-3, 1 - 1e-3)
    dd["logit_y"] = np.log(yc / (1 - yc))
    m0 = smf.mixedlm("logit_y ~ x", dd, groups=dd["cell"]).fit(reml=False)
    m1 = smf.mixedlm("logit_y ~ x + C(arch)", dd, groups=dd["cell"]).fit(reml=False)
    lr = 2 * (m1.llf - m0.llf)
    return {"slope": float(m0.params["x"]), "slope_se": float(m0.bse["x"]),
            "slope_ci": tuple(float(v) for v in m0.conf_int().loc["x"]),
            "intercept": float(m0.params["Intercept"]), "random_intercept_var": float(m0.cov_re.iloc[0, 0]),
            "residual_var": float(m0.scale), "llf": float(m0.llf),
            "arch_lr_stat": float(lr), "arch_lr_p": float(sps.chi2.sf(lr, 2)),
            "arch_effects": {k: float(v) for k, v in m1.params.items() if k.startswith("C(arch)")}}


# C1 under alternative grok thresholds: where the cognitive switch falls relative to the grok
# step, on the single-task add runs.
# Input: runs - the Run records; tab - the runs table, which supplies the switch times
# Output: pd.DataFrame - per (arch, threshold), the median ratio and the share switching before grok
def c1_sensitivity(runs, tab: pd.DataFrame) -> pd.DataFrame:
    sw = tab[(~tab.joint) & (tab.task == "add")][["arch", "condition", "seed", "t_switch_rule_or_rulex"]]
    rows = []
    for run in runs:
        if run.is_joint or run.task != "add":
            continue
        mdf = read_csv_or_empty(run.dir / "analysis_offline" / "metrics.csv")
        if mdf.empty:
            continue
        r = sw[(sw.condition == run.condition) & (sw.seed == run.seed)]
        if r.empty or not np.isfinite(r.t_switch_rule_or_rulex.iloc[0]):
            continue
        t_sw = float(r.t_switch_rule_or_rulex.iloc[0])
        steps = mdf.step.values.astype(float); test = mdf.test_add.values.astype(float)
        for g in GROK_THRS:
            hit = np.where(test > g)[0]
            if len(hit):
                rows.append({"arch": run.arch, "seed": run.seed, "condition": run.condition, "grok_thr": g,
                             "t_grok": float(steps[hit[0]]), "t_switch": t_sw, "ratio": t_sw / float(steps[hit[0]])})
    d = pd.DataFrame(rows)
    return d.groupby(["arch", "grok_thr"]).agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                                              share_switch_before_grok=("ratio", lambda s: float((s < 1).mean()))).reset_index()


# Median lead of behaviour over mechanism (t_beh / t_mech) per architecture on the single-task add
# runs, with a cluster bootstrap over cells for the interval.
# Input: tab - the runs table; n_boot - bootstrap draws; seed - RNG seed
# Output: pd.DataFrame - one row per architecture with the median, its 95 % interval and the
# share of runs in which mechanism leads
def lead_bootstrap(tab: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = tab[(~tab.joint) & (tab.task == "add")].dropna(subset=["t_half_mech", "t_half_test_acc"]).copy()
    s["lead"] = s.t_half_test_acc / s.t_half_mech
    s["cell"] = s.condition
    rows = []
    for arch, g in s.groupby("arch"):
        cells = g.cell.unique(); by = {c: g[g.cell == c].lead.values for c in cells}
        meds = []
        for _ in range(n_boot):
            pick = rng.choice(cells, size=len(cells), replace=True)
            meds.append(np.median(np.concatenate([by[c] for c in pick])))
        lo, hi = np.percentile(meds, [2.5, 97.5])
        rows.append({"arch": arch, "n": int(len(g)), "median_lead": float(g.lead.median()), "ci_lo": float(lo), "ci_hi": float(hi),
                     "share_mech_leads": float((g.lead > 1).mean())})
    return pd.DataFrame(rows)


# The full C1 / C2 robustness report: the primary fit, the threshold sensitivity grid, the cluster
# bootstrap, leave-one-architecture-out prediction, the between- versus within-cell split, the mixed
# model, and the censored-data check.
# Input: runs - the Run records; tab - the runs table
# Output: tuple (str report text, pd.DataFrame the sensitivity grid)
def report(runs, tab: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    d = route_table(runs)
    L = ["Robustness of the C2 route-order relation (peak overregularisation vs log10(t_exc / t_grok)); single-task rho > 0 runs"]
    base = logistic_summary(d)
    L.append(f"primary definition (grok thr 0.95, exception-memorisation thr 0.95): n = {base['n']}, Spearman = {base['spearman']:.3f}, "
             f"logistic R2 = {base['r2_logistic']:.3f}, + arch intercepts {base['r2_logistic_arch']:.3f} (delta {base['delta_r2_arch']:.3f}), "
             f"a = {base['a']:.2f}, b = {base['b']:.2f}")
    dc = route_table(runs, censored=True)
    cens = logistic_summary(dc)
    L.append(f"censored runs included (t_exc := last step for the {int(dc.censored.sum())} runs stopped before their exceptions reached 0.95: "
             f"{', '.join(dc[dc.censored].apply(lambda r: f'{r.arch} {r.task} rho{r.rho:g} wd{r.wd:g} s{r.seed}', axis=1))}): "
             f"n = {cens['n']}, Spearman = {cens['spearman']:.3f}, logistic R2 = {cens['r2_logistic']:.3f}, delta R2 arch = {cens['delta_r2_arch']:.3f}")
    sens = sensitivity(runs)
    L.append("\n[threshold sensitivity] rows = grok threshold x exception threshold")
    L.append(sens.round(3).to_string(index=False))
    boot = cluster_bootstrap(d)
    L.append("\n[cluster bootstrap, 2000 resamples] 95% percentile CIs; cells = arch x task x rho x wd (the independent units)")
    L.append(boot.round(3).to_string(index=False))
    loo = leave_one_arch_out(d)
    L.append("\n[leave-one-architecture-out] logistic fitted on two architectures, evaluated on the third")
    L.append(loo.round(3).to_string(index=False))
    bw = between_within(d)
    L.append("\n[between vs within cells]")
    for k, v in bw.items():
        L.append(f"  {k} = {v:.3f}" if isinstance(v, float) else f"  {k} = {v}")
    mm = mixed_model(d)
    if mm is None:
        L.append("\n[mixed model] statsmodels not available -- skipped")
    else:
        L.append("\n[mixed model] logit(peak) ~ log10 ratio + (1 | cell), ML")
        L.append(f"  slope = {mm['slope']:.3f} (SE {mm['slope_se']:.3f}, 95% CI {mm['slope_ci'][0]:.3f}..{mm['slope_ci'][1]:.3f}); intercept {mm['intercept']:.3f}")
        L.append(f"  random-intercept variance = {mm['random_intercept_var']:.3f}, residual variance = {mm['residual_var']:.3f}")
        L.append(f"  adding architecture as a fixed effect: LR = {mm['arch_lr_stat']:.2f}, p = {mm['arch_lr_p']:.3f}; effects {mm['arch_effects']}")
    c1 = c1_sensitivity(runs, tab)
    L.append("\n[C1 sensitivity] cognitive switch / grok step under alternative grok thresholds (single-task add)")
    L.append(c1.round(3).to_string(index=False))
    lb = lead_bootstrap(tab)
    L.append("\n[C1 hidden progress] median t_beh / t_mech (behavioural half-rise over mechanistic half-rise), cluster bootstrap over cells")
    L.append(lb.round(3).to_string(index=False))
    return "\n".join(L), sens


# uncertainty of the headline statistics (cluster bootstrap over cells = arch x task x rho x wd)

# Cluster bootstrap of an arbitrary statistic, resampling cells with replacement.
# Input: df - the rows; stat - callable frame >> float; n_boot - draws; seed - RNG seed;
# cell - the column that identifies a cell
# Output: tuple (float ci_lo, float ci_hi) - the 95 % percentile interval
def _cell_bootstrap(df: pd.DataFrame, stat, n_boot: int = 2000, seed: int = 0, cell: str = "condition") -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cells = df[cell].unique()
    by = {c: df[df[cell] == c] for c in cells}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cells, size=len(cells), replace=True)
        v = stat(pd.concat([by[c] for c in pick], ignore_index=True))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# 95 % percentile intervals for every statistic quoted in the text, resampling cells (the
# independent units, three seeds each) with replacement: the pooled hidden-progress correlations,
# the C2 Spearman, the per-architecture median switch and route-order ratios, and the
# per-architecture lesion medians.
# Input: tab - the runs table; lesion_summary - the lesion summary table; n_boot - draws
# Output: pd.DataFrame - one row per statistic with estimate, interval and n
def headline_intervals(tab: pd.DataFrame, lesion_summary: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    s = tab[~tab.joint].copy()
    rows = []

    def add(name, arch, df, stat, n):
        est = stat(df)
        lo, hi = _cell_bootstrap(df, stat, n_boot)
        rows.append({"statistic": name, "arch": arch, "estimate": float(est), "ci_lo": lo, "ci_hi": hi, "n": int(n)})

    h = s.dropna(subset=["t_half_mech_use", "t_half_test_acc"]); h = h[(h.t_half_mech_use > 0) & (h.t_half_test_acc > 0)]
    add("Spearman(t_mech, behavioural half-rise)", "all", h, lambda d: sps.spearmanr(d.t_half_mech_use, d.t_half_test_acc)[0], len(h))
    h2 = s.dropna(subset=["t_half_mech_use", "t_switch_rule_or_rulex"]); h2 = h2[(h2.t_half_mech_use > 0) & (h2.t_switch_rule_or_rulex > 0)]
    add("Spearman(t_mech, cognitive switch)", "all", h2, lambda d: sps.spearmanr(d.t_half_mech_use, d.t_switch_rule_or_rulex)[0], len(h2))
    r = s[s.rho > 0].dropna(subset=["peak_overreg", "mem_to_grok_ratio"])
    add("Spearman(peak overreg, log10 t_exc/t_grok)", "all", r, lambda d: sps.spearmanr(np.log10(d.mem_to_grok_ratio), d.peak_overreg)[0], len(r))
    a = s[s.task == "add"].dropna(subset=["switch_rel_grok"])
    for arch, g in a.groupby("arch"):
        add("median t_switch/t_grok (add)", arch, g, lambda d: d.switch_rel_grok.median(), len(g))
    hp = s[s.task == "add"].dropna(subset=["t_half_mech_use", "t_half_test_acc"])
    for arch, g in hp.groupby("arch"):
        add("median lead t_beh/t_mech (add)", arch, g, lambda d: (d.t_half_test_acc / d.t_half_mech_use).median(), len(g))
    for arch, g in r[r.task == "add"].groupby("arch"):
        add("median t_exc/t_grok (add)", arch, g, lambda d: d.mem_to_grok_ratio.median(), len(g))
    L = lesion_summary[(lesion_summary.lesion == "ebottom") & (lesion_summary.rho > 0)]
    for arch, g in L.groupby("arch"):
        add("median D50 exceptions (ebottom, add)", arch, g, lambda d: d.D50_exc.median(), len(g))
        add("median D50 rule (ebottom, add)", arch, g, lambda d: d.D50_rule.median(), len(g))
        add("median SI at q = 0.5 (ebottom, add)", arch, g, lambda d: d["SI_q0.5"].median(), len(g))
    return pd.DataFrame(rows)
