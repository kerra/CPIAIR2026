from __future__ import annotations

import numpy as np
import pandas as pd

from .timing import first_step_at_least, first_sustained, half_rise, regime_sequence, switch_step

MECH_METRIC = {"add": "fourier_conc_final_eq",
               "div": "fourier_conc_final_eq",      # dlog basis upstream
               "max": "linear_corr_final_eq"}


# per-run derived rows

# One row per task of one run for runs_table.csv, built from the light offline outputs: grok step,
# behavioural and mechanistic half-rises, the pre-registered memorisation and generalisation events,
# the cognitive regime sequence, and - for single-task rho > 0 runs - the exception metrics.
# Input: R - the dict load_offline returns (run, summary, metrics, fourier, cognitive)
# Output: list[dict] - one row per task of the run
def derive_rows(R):
    run, summ, mdf = R["run"], R["summary"], R["metrics"]
    joint = len(run.tasks) > 1
    grok = summ.get("grok_step", {}) or {}
    cap = summ.get("cap")
    steps_run = summ.get("steps_run")
    steps = mdf["step"].values
    # window start for the joint-ADD mechanistic correction: after the max grok
    g_max = grok.get("max")
    add_window_start = (max(2 * g_max, 500) if (joint and g_max is not None) else None)

    rows = []
    for task in run.tasks:
        tcol = f"test_{task}"
        g = grok.get(task)
        t_half_test, _ = (half_rise(steps, mdf[tcol].values) if tcol in mdf else (np.nan, np.nan))
        trcol = f"train_{task}"
        t_gen_persist = first_sustained(steps, mdf[tcol].values, 0.95) if tcol in mdf else np.nan
        # t_mem on the REGULAR train items: acc_reg = (train_acc - rho * acc_exc) / (1 - rho), exact since
        # rho = n_exc / n_train; equals the plain train accuracy when the run has no exceptions
        if trcol in mdf and "acc_exc" in mdf.columns and run.rho > 0 and not joint:
            acc_reg = (mdf[trcol].values - run.rho * mdf["acc_exc"].values) / (1.0 - run.rho)
        else:
            acc_reg = mdf[trcol].values if trcol in mdf else None
        t_train_mem_reg = first_sustained(steps, acc_reg, 0.99) if acc_reg is not None else np.nan
        W_reg = (t_gen_persist - t_train_mem_reg
                 if np.isfinite(t_gen_persist) and np.isfinite(t_train_mem_reg) else np.nan)
        cdf = R["cognitive"].get(task, pd.DataFrame())
        fdf = R["fourier"].get(task, pd.DataFrame())
        mech_col = MECH_METRIC[task]
        t_half_mech = np.nan
        t_half_mech_corr = np.nan
        t_half_mech_fm, mech_mode_stable, final_mode = np.nan, np.nan, np.nan
        if not fdf.empty and mech_col in fdf.columns:
            t_half_mech, _ = half_rise(fdf["step"].values, fdf[mech_col].values)
            if task == "add" and add_window_start is not None:
                t_half_mech_corr, _ = half_rise(fdf["step"].values, fdf[mech_col].values, start_step=add_window_start)
            # The dominant mode often changes between the first structure and the final
            # checkpoint (add: single low-f ring before the grok, high modes after -- the
            # frequency switch; div: Legendre character (mode 74) before the ring). Both are
            # real mechanism, so the RAW half-rise (first structure) stays the primary t_mech;
            # the half-rise on the snapshots carrying the FINAL mode is kept as a robustness
            # column together with a stability flag.
            mode_col = "best_mode_final_eq"
            if task in ("add", "div") and mode_col in fdf.columns:
                f2 = fdf.sort_values("step")
                final_mode = int(f2[mode_col].iloc[-1])
                if np.isfinite(t_half_mech):
                    k = int(np.argmin(np.abs(f2["step"].values - t_half_mech)))
                    mech_mode_stable = bool(int(f2[mode_col].iloc[k]) == final_mode)
                sub = f2[f2[mode_col] == final_mode]
                if len(sub) >= 3:
                    t_half_mech_fm, _ = half_rise(sub["step"].values, sub[mech_col].values)
            else:
                mech_mode_stable = True
        t_sw = switch_step(cdf)
        row = {
            "condition": run.condition, "arch": run.arch, "task": task,
            "joint": joint, "rho": run.rho, "wd": run.wd, "seed": run.seed,
            "grok_step": g, "non_grokking": bool(summ.get("non_grokking", False)),
            "steps_run": steps_run, "cap": cap,
            "t_half_test_acc": t_half_test,
            "t_switch_rule_or_rulex": t_sw,
            "mech_metric": mech_col,
            "t_half_mech": t_half_mech,
            "t_half_mech_corrected": (t_half_mech_corr if (task == "add" and joint) else t_half_mech),
            "mech_window_start": add_window_start if task == "add" else np.nan,
            # pre-registered event definitions (3 consecutive evals), memorisation on the regular items
            "t_train_mem_reg": t_train_mem_reg, "t_gen_persist": t_gen_persist, "W_mem_to_gen_reg": W_reg,
            # mechanistic half-rise of the FINAL dominant mode (robustness) + stability flag
            "t_half_mech_finalmode": t_half_mech_fm, "mech_mode_stable": mech_mode_stable,
            "mech_final_mode": final_mode, "t_half_mech_use": t_half_mech,   # primary = first structure
            # cognitive regimes from the fixed fits
            "regime_seq": regime_sequence(cdf),
            "switch_rel_grok": (t_sw / g if (g is not None and np.isfinite(g) and g > 0
                                             and np.isfinite(t_sw)) else np.nan),
        }
        # exception metrics exist only for the exception task of single-task rho>0 runs
        if (not joint) and run.rho > 0 and "acc_exc" in mdf.columns:
            ae = mdf["acc_exc"].values.astype(float)
            ov = mdf["overreg_rate"].values.astype(float)
            i_pk = int(np.nanargmax(ov)) if np.isfinite(ov).any() else None
            # exception memorisation: 5-point rolling median of acc_exc reaches 0.95
            ae_s = pd.Series(ae).rolling(5, center=True, min_periods=1).median().values
            t_mem = first_step_at_least(steps, ae_s, 0.95)
            acc_exc_at_grok, mem_to_grok = np.nan, np.nan
            if g is not None and np.isfinite(g):
                k = int(np.argmin(np.abs(steps - g)))
                acc_exc_at_grok = float(ae_s[k])
                if np.isfinite(t_mem) and g > 0:
                    mem_to_grok = float(t_mem / g)
            row.update({
                "peak_overreg": float(ov[i_pk]) if i_pk is not None else np.nan,
                "t_peak_overreg": float(steps[i_pk]) if i_pk is not None else np.nan,
                "t_exc_memorized": t_mem,
                "acc_exc_at_grok": acc_exc_at_grok,
                "mem_to_grok_ratio": mem_to_grok,
                "final_acc_exc": float(ae[-1]), "final_overreg": float(ov[-1]),
            })
        rows.append(row)
    return rows


EXCEPTION_COLUMNS = ("peak_overreg", "t_peak_overreg", "t_exc_memorized", "acc_exc_at_grok", "mem_to_grok_ratio",
                     "final_acc_exc", "final_overreg")


# Iterates the single-task add runs that grokked and have a usable cognitive trajectory - the
# selector the regime figures are drawn from.
# Input: loaded - {(condition, seed): offline dict}; tab - the runs table
# Output: generator of (table row, cognitive frame sorted by step)
def single_add_runs(loaded, tab):
    rows = tab[(~tab.joint) & (tab.task == "add") & tab.grok_step.notna()]
    for r in rows.itertuples():
        R = loaded.get((r.condition, r.seed))
        if R is None:
            continue
        cdf = R["cognitive"].get("add", pd.DataFrame())
        if cdf.empty or "dbic_gcm_vs_uniform" not in cdf.columns:
            continue
        yield r, cdf.sort_values("step")


# console summaries

# Prints the console summaries of the whole matrix: grok order in the joint runs, the exception
# cost in grok steps, peak overregularisation per cell, and the cognitive regime counts.
# Input: tab - the runs table
# Output: None
def print_summaries(tab: pd.DataFrame):
    pd.set_option("display.width", 200)
    print("\n== Grok order (joint runs), tasks sorted by behavioral grok ==")
    j = tab[tab.joint]
    for (arch, seed), g in j.groupby(["arch", "seed"]):
        g2 = g.sort_values("grok_step", na_position="last")
        print(f"  {arch:<12} s{seed}: " + " -> ".join(
            f"{r.task}({int(r.grok_step) if pd.notna(r.grok_step) else 'cap'})" for r in g2.itertuples()))

    print("\n== Exception cost: median grok step (add, single-task), cap-runs counted separately ==")
    s = tab[(~tab.joint) & (tab.task == "add")]
    piv = (s.assign(g=s.grok_step.astype(float))
            .groupby(["arch", "wd", "rho"])
            .agg(median_grok=("g", "median"), n=("g", "size"), n_cap=("non_grokking", "sum"))
            .reset_index())
    print(piv.to_string(index=False))

    print("\n== Peak overregularisation per cell (mean over seeds) ==")
    e = tab[(~tab.joint) & (tab.rho > 0) & tab.peak_overreg.notna()]
    if len(e):
        piv = (e.groupby(["arch", "task", "rho", "wd"])
                .agg(peak_overreg=("peak_overreg", "mean"), n=("seed", "size"))
                .reset_index())
        print(piv.round(3).to_string(index=False))

    if "regime_seq" in tab.columns:
        print("\n== Cognitive regimes (fixed common-probe fits), single-task add: sequence counts per arch ==")
        s1 = tab[(~tab.joint) & (tab.task == "add") & (tab.regime_seq != "")]
        for arch, g in s1.groupby("arch"):
            vc = g.regime_seq.value_counts()
            print(f"  {arch:<12} " + "; ".join(f"{k} x{v}" for k, v in vc.items()))
        gg = s1.dropna(subset=["switch_rel_grok"])
        if len(gg):
            print("  median t_switch / t_grok by arch: " +
                  ", ".join(f"{a} {g.switch_rel_grok.median():.2f} (n={len(g)})" for a, g in gg.groupby("arch")))
