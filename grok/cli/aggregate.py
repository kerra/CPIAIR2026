"""Clean aggregation layer over FINISHED runs.

Reads ONLY light files (no torch, no snapshots, no preds):
    <run>/run_config.json, <run>/summary.json
    <run>/analysis_offline/metrics.csv
    <run>/analysis_offline/fourier_trajectory_<task>.csv
    <run>/analysis_offline/cognitive_trajectory_<task>.csv

Writes to <base_out>/paper/ (main-text figures + runs_table) and <base_out>/paper/supp/ (detail):
    runs_table.csv            one row per run x task: grok, cognitive switch, mechanistic half-rise
                              (first structure; final-mode variant + stability flag), t_mem / t_gen / W on
                              the regular items, regime sequence, peak overreg, t_exc, acc_exc at the grok,
                              route-order ratio, final acc_exc / overreg
    grok_order_clean.csv      B1: behavioral grok vs cognitive switch vs mech half-rise (joint-ADD half-rise
                              recomputed on the window AFTER the max grok: ordinal contamination of the
                              natural-order Fourier metric)
    regime_shares.png (+csv)  share of runs per model family (uniform / exemplar / rule) vs step/grok
    hidden_progress.png       single-task: mech half-rise vs behavioral half-rise
    hidden_progress_stats.txt pooled H1: Spearman(t_mech, t_beh), sign test of "mechanism leads",
                              |t_mech - t_beh| / W (plan rule)
    order_vs_overreg.png      peak overreg vs acc_exc at the grok (left) and vs t_exc / t_grok (right)
                              (+ order_vs_overreg_stats.txt: linear AND logistic fits, arch residuals, delta R^2)
    grok_order.png            B1 timelines per arch/seed
    supp/ucurve_rel_add_rho0.02_wd0.1.png   test acc / acc_exc / overreg vs grok-relative step (the
                                            representative cell)
    supp/grok_delay.png       single-task grok step vs rho, per arch, WD styles
    supp/cognitive_regimes.png  per-probe dBIC trajectories: exemplar vs uniform, rule family vs exemplar,
                              rulex vs rule, vs step/grok (single-task add)

Usage:
    python3 grok/aggpaper.py --base-out ./exceptions_runs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.stats import hidden_progress_stats
from ..analysis.table import EXCEPTION_COLUMNS, derive_rows, print_summaries
from ..plots.aggregate import (fig_cognitive_regimes, fig_grok_delay, fig_grok_order, fig_hidden_progress,
                               fig_order_vs_overreg, fig_regime_shares, fig_ucurve)
from ..runs.discover import find_runs
from ..runs.io import load_offline


# Entry point of the aggregation layer over finished runs. Reads only the light per-run files - no
# torch, no snapshots, no prediction dumps - and builds the runs table, the main-text figures and
# the statistics texts.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes <base_out>/paper/ and <base_out>/paper/supp/
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out = args.base_out / "paper"
    out.mkdir(exist_ok=True)
    supp = out / "supp"                 # supplementary detail; main-text figures stay at the top level
    supp.mkdir(exist_ok=True)

    runs = find_runs(args.base_out)
    print(f"found {len(runs)} runs")
    loaded, rows, skipped = {}, [], []
    for run in runs:
        R = load_offline(run)
        if R is None:
            skipped.append(f"{run.condition}/seed_{run.seed}")
            continue
        loaded[(run.condition, run.seed)] = R
        rows.extend(derive_rows(R))
    if skipped:
        print(f"skipped {len(skipped)} runs without summary.json / analysis_offline/metrics.csv:")
        for s in skipped:
            print("   ", s)

    tab = pd.DataFrame(rows)
    # joint-only trees (e.g. the pos-emb ablation) have no exception columns; add them as NaN so every
    # figure/summary sees an empty selection instead of a missing attribute
    for c in EXCEPTION_COLUMNS:
        if c not in tab.columns:
            tab[c] = np.nan
    tab.to_csv(out / "runs_table.csv", index=False)
    print(f"saved {out / 'runs_table.csv'}  ({len(tab)} run x task rows)")

    go = tab[tab.joint][["arch", "seed", "task", "grok_step", "t_switch_rule_or_rulex",
                         "t_half_mech", "t_half_mech_corrected", "mech_window_start", "mech_metric"]]
    go.to_csv(supp / "grok_order_clean.csv", index=False)

    print_summaries(tab)
    hidden_progress_stats(tab, out)

    if not args.no_figures:
        fig_regime_shares(loaded, tab, out)             # C1
        fig_hidden_progress(tab, out)                   # C1
        fig_order_vs_overreg(tab, out)                  # C2 (+ stats txt)
        fig_grok_order(tab, out)                        # C4
        fig_ucurve(loaded, tab, supp)                   # C2 illustration
        fig_grok_delay(tab, supp)                       # matrix at a glance
        fig_cognitive_regimes(loaded, tab, supp)        # C1 detail
        print("\nfigures written to", out)


if __name__ == "__main__":
    main()
