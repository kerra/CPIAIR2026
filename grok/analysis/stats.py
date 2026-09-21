from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import least_squares


# Least-squares fit of a logistic curve y = 1/(1 + exp(-(a + b x))). With groups, the slope is
# shared and each group gets its own intercept.
# Input: x, y - the data; groups - group label per point, or None for a single curve
# Output: tuple (float R^2, residuals, fitted parameters)
def fit_logistic(x, y, groups=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if groups is None:
        def resid(p):
            return y - 1.0 / (1.0 + np.exp(-(p[0] + p[1] * x)))
        p0 = np.array([0.0, 1.0])
    else:
        cats = sorted(set(groups)); gi = np.array([cats.index(g) for g in groups])

        def resid(p):
            return y - 1.0 / (1.0 + np.exp(-(p[1 + gi] + p[0] * x)))
        p0 = np.array([1.0] + [0.0] * len(cats))
    sol = least_squares(resid, p0, max_nfev=20000)
    r = resid(sol.x)
    r2 = 1.0 - float(np.sum(r ** 2) / np.sum((y - y.mean()) ** 2))
    return r2, r, sol.x


# R^2 of an ordinary least-squares fit of y on X, with an intercept added.
# Input: X - [n, k] design matrix; y - response vector
# Output: tuple (float R^2, residuals)
def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ coef
    return 1.0 - float(np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)), res


# C2: does architecture explain anything about overregularisation beyond the timing of the two
# routes? Correlates peak overregularisation with exception accuracy at grok and with the
# memorisation-to-grok ratio, then checks whether architecture dummies add any R^2.
# Input: d - the single-task rho > 0 rows of the runs table; out - Path of the output directory
# Output: None - writes order_vs_overreg_stats.txt and prints the same text
def order_vs_overreg_stats(d: pd.DataFrame, out: Path):
    lines = []
    y = d.peak_overreg.values.astype(float)
    x1 = d.acc_exc_at_grok.values.astype(float)
    lines.append(f"n runs = {len(d)}")
    lines.append(f"[left]  Spearman(peak_overreg, acc_exc_at_grok)  = {d[['peak_overreg','acc_exc_at_grok']].corr('spearman').iloc[0,1]:.3f}")
    lines.append(f"        Pearson                                  = {d[['peak_overreg','acc_exc_at_grok']].corr().iloc[0,1]:.3f}")
    dev = y - (1.0 - x1)
    lines.append("        deviation from identity y = 1 - acc_exc_at_grok, mean +- sd by arch:")
    for arch, g in d.assign(dev=dev).groupby("arch"):
        lines.append(f"            {arch:<12} {g.dev.mean():+.3f} +- {g.dev.std():.3f}")
    dr = d.dropna(subset=["mem_to_grok_ratio"])
    if len(dr) >= 6:
        yr = dr.peak_overreg.values.astype(float)
        x2 = np.log10(dr.mem_to_grok_ratio.values.astype(float))
        lines.append(f"[right] Spearman(peak_overreg, log t_mem/t_grok) = {pd.Series(yr).corr(pd.Series(x2), method='spearman'):.3f}")
        r2_t, res_t = r2(x2[:, None], yr)
        A = pd.get_dummies(dr.arch).values.astype(float)[:, 1:]        # arch dummies
        r2_ta, _ = r2(np.column_stack([x2, A]), yr)
        lines.append(f"        R^2  y ~ log-ratio                 = {r2_t:.3f}")
        lines.append(f"        R^2  y ~ log-ratio + arch dummies  = {r2_ta:.3f}   (delta = {r2_ta - r2_t:+.3f})")
        lines.append("        residuals of the timing-only fit, mean +- sd by arch (0 = arch adds nothing):")
        for arch, g in dr.assign(res=res_t).groupby("arch"):
            lines.append(f"            {arch:<12} {g.res.mean():+.3f} +- {g.res.std():.3f}  (n={len(g)})")
        # the relation is sigmoid in log-ratio; a linear fit leaves signed residuals at the
        # extremes by construction, so the arch test is repeated with a logistic curve
        r2_l, res_l, p_l = fit_logistic(x2, yr)
        r2_la, _, p_la = fit_logistic(x2, yr, groups=dr.arch.values)
        lines.append(f"[right, logistic] y = 1/(1+exp(-(a + b*log10 ratio))): a={p_l[0]:+.2f} b={p_l[1]:+.2f}")
        lines.append(f"        R^2  logistic, timing only          = {r2_l:.3f}")
        lines.append(f"        R^2  logistic + per-arch intercepts = {r2_la:.3f}   (delta = {r2_la - r2_l:+.3f})")
        lines.append("        residuals of the logistic timing-only fit, mean +- sd by arch:")
        for arch, g in dr.assign(res=res_l).groupby("arch"):
            lines.append(f"            {arch:<12} {g.res.mean():+.3f} +- {g.res.std():.3f}  (n={len(g)})")
        verdict = ("architecture adds nothing beyond route timing"
                   if (r2_la - r2_l) < 0.05 and all(abs(g.res.mean()) < 0.05 for _, g in dr.assign(res=res_l).groupby("arch"))
                   else "architecture still explains variance beyond route timing (logistic)")
        lines.append(f"        H0-new verdict (delta R^2 < .05 and |mean arch residual| < .05): {verdict}")
    txt = "\n".join(lines)
    (out / "order_vs_overreg_stats.txt").write_text(txt + "\n")
    print("\n== order vs overreg ==\n" + txt)


# H1 pooled statistics (mechanism vs behaviour)

# The pre-registered H1 on the pooled single-task rows: t_mech is the guarded mechanistic half-rise
# and t_beh is either the cognitive switch or the behavioural half-rise. Reports the Spearman
# correlation, a sign test of "mechanism leads", and |t_mech - t_beh| relative to the
# memorisation-to-generalisation window W.
# Input: tab - the runs table; out - Path of the output directory
# Output: None - writes hidden_progress_stats.txt and prints the same text
def hidden_progress_stats(tab: pd.DataFrame, out: Path):
    d = tab[~tab.joint].copy()
    if d.empty or "t_half_mech_use" not in d.columns:
        return
    lines = ["H1 (pooled over single-task runs; n=3 seeds per cell makes within-cell rho meaningless)"]
    for mech_col, mech_name in (("t_half_mech_use", "first structure (raw half-rise, plan definition)"),
                                ("t_half_mech_finalmode", "final dominant mode only (robustness)")):
      for beh_col, beh_name in (("t_half_test_acc", "behavioural half-rise (test acc)"),
                                ("t_switch_rule_or_rulex", "cognitive switch to rule/rulex")):
        dd = d.dropna(subset=[mech_col, beh_col]).copy()
        dd["tm"] = dd[mech_col].astype(float)
        dd = dd[(dd.tm > 0) & (dd[beh_col] > 0)]
        if len(dd) < 5:
            continue
        rho, pval = sps.spearmanr(dd.tm, dd[beh_col])
        lead = dd.tm < dd[beh_col]
        bt = sps.binomtest(int(lead.sum()), len(dd), 0.5)
        ratio = dd[beh_col] / dd.tm
        lines.append(f"\n[t_mech = {mech_name}; t_beh = {beh_name}]  n = {len(dd)}")
        lines.append(f"  Spearman(t_mech, t_beh) = {rho:.3f}  (p = {pval:.1e}); plan criterion rho >= 0.5")
        lines.append(f"  mechanism leads in {int(lead.sum())}/{len(dd)} runs  (two-sided sign test p = {bt.pvalue:.1e})")
        lines.append(f"  median t_beh / t_mech = {ratio.median():.2f}  (IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f})")
        for (arch, task), g in dd.groupby(["arch", "task"]):
            r = g[beh_col] / g.tm
            lines.append(f"    {arch:<12}{task:<4} n={len(g):2d}  leads {int((g.tm < g[beh_col]).sum())}/{len(g)}"
                         f"  median t_beh/t_mech = {r.median():.2f}")
        if "W_mem_to_gen_reg" in dd.columns:
            w = dd.dropna(subset=["W_mem_to_gen_reg"])
            w = w[w["W_mem_to_gen_reg"] > 0]
            if len(w):
                rel = (w.tm - w[beh_col]).abs() / w["W_mem_to_gen_reg"]
                lines.append(f"  |t_mech - t_beh| / W [W on regular train items]: median = {rel.median():.2f}, share <= 0.25 = {(rel <= 0.25).mean():.2f}"
                             f"  (plan: median <= 0.25; n with W>0 = {len(w)} of {len(dd)})")
    if "mech_mode_stable" in d.columns:
        unstable = d[d.mech_mode_stable == False]  # noqa: E712
        lines.append(f"\nrows where the dominant mode at the half-rise differs from the final one: {len(unstable)} of {int(d.mech_mode_stable.notna().sum())}"
                     " (add: pre-grok low-f ring -> high modes = frequency switch; div: Legendre 74 -> ring)")
        for r in unstable.itertuples():
            lines.append(f"    {r.condition} s{r.seed} {r.task}: first-structure {r.t_half_mech:.0f} | final-mode {r.t_half_mech_finalmode:.0f} (mode {r.mech_final_mode:g})")
    txt = "\n".join(lines)
    (out / "hidden_progress_stats.txt").write_text(txt + "\n")
    print("\n== hidden progress / H1 ==\n" + txt)
