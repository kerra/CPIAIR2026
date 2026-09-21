from __future__ import annotations

import numpy as np
import pandas as pd


# timing primitives

# First sustained crossing of the baseline-to-peak midpoint on the rising limb of a curve,
# interpolated in log-step. start_step restricts BOTH the baseline and the search.
# Input: steps, vals - the curve; start_step - earliest step considered, or None;
# min_rise - smallest rise that counts as a rise at all
# Output: tuple (float t_half, float t_peak); t_half is NaN when the curve never really rises
def half_rise(steps, vals, start_step=None, min_rise=0.05):
    s = np.asarray(steps, float)
    v = np.asarray(vals, float)
    m = np.isfinite(s) & np.isfinite(v)
    if start_step is not None:
        m &= s >= start_step
    s, v = s[m], v[m]
    if len(v) < 3:
        return np.nan, np.nan
    o = np.argsort(s)
    s, v = s[o], v[o]
    i_peak = int(np.argmax(v))
    peak, t_peak = float(v[i_peak]), float(s[i_peak])
    baseline = float(np.min(v[:3]))
    if peak - baseline < min_rise:
        return np.nan, t_peak
    thr = baseline + 0.5 * (peak - baseline)
    above = v >= thr
    for i in range(i_peak + 1):
        if above[i] and (i == i_peak or above[i + 1]):
            if i == 0:
                return float(s[0]), t_peak
            x0, x1 = np.log10(max(s[i - 1], 1.0)), np.log10(max(s[i], 1.0))
            y0, y1 = v[i - 1], v[i]
            if y1 == y0:
                return float(s[i]), t_peak
            frac = float(np.clip((thr - y0) / (y1 - y0), 0.0, 1.0))
            return float(10 ** (x0 + frac * (x1 - x0))), t_peak
    return np.nan, t_peak


# The cognitive switch: the first sustained checkpoint whose best model is rule or rulex.
# Input: cdf - cognitive trajectory frame with step and best_model columns
# Output: float - the step, or NaN when the switch never happens
def switch_step(cdf: pd.DataFrame):
    if cdf.empty or "best_model" not in cdf.columns:
        return np.nan
    cdf = cdf.sort_values("step")
    steps = cdf["step"].values
    ok = cdf["best_model"].isin(["rule", "rulex"]).values
    for i in range(len(ok)):
        if ok[i] and (i == len(ok) - 1 or ok[i + 1]):
            return float(steps[i])
    return np.nan


# The first step at which a curve reaches a threshold, with no sustain requirement.
# Input: steps, vals - the curve; thr - threshold
# Output: float - the step, or NaN when it is never reached
def first_step_at_least(steps, vals, thr):
    s = np.asarray(steps, float)
    v = np.asarray(vals, float)
    idx = np.where(np.isfinite(v) & (v >= thr))[0]
    return float(s[idx[0]]) if len(idx) else np.nan


# The pre-registered event definition: the first step at which vals >= thr holds for n_consec
# consecutive evaluations (t_mem uses train >= .99, t_gen uses test >= .95).
# Input: steps, vals - the curve; thr - threshold; n_consec - evaluations that must hold
# Output: float - the step, or NaN when the threshold is never sustained
def first_sustained(steps, vals, thr, n_consec: int = 3):
    s = np.asarray(steps, float)
    v = np.asarray(vals, float)
    ok = np.isfinite(v) & (v >= thr)
    for i in range(len(ok) - n_consec + 1):
        if ok[i:i + n_consec].all():
            return float(s[i])
    return np.nan


# The collapsed sequence of winning models over checkpoints, with the gcm_* variants pooled into
# one "gcm" family - for example "uniform>gcm>rulex".
# Input: cdf - cognitive trajectory frame with step and best_model columns
# Output: str - the regimes joined by ">", empty when there is no trajectory
def regime_sequence(cdf: pd.DataFrame) -> str:
    if cdf.empty or "best_model" not in cdf.columns:
        return ""
    fam = ["gcm" if str(m).startswith("gcm") else str(m) for m in cdf.sort_values("step")["best_model"].values]
    seq = [fam[0]]
    for f in fam[1:]:
        if f != seq[-1]:
            seq.append(f)
    return ">".join(seq)
