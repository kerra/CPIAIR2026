"""Aggregate the lesion sweeps written by grok/lesions.py (final checkpoint; the development figure uses all tags).

Reads  <base_out>/paper/lesions/tables/lesions_table*.csv
Writes <base_out>/paper/lesions/
    lesion_headline.png        ebottom: rule / exceptions / overregularisation vs energy removed
    lesion_cognitive.png       low-pass: best-model family shares along f_cut + GCM lambda vs f_cut
                               (+ lesion_cognitive_table.csv)
    lesion_development.png     the same lesions at 0.25 / 0.5 / 1 x grok and at the end (+ .csv)
    lesion_summary.csv         per run x lesion: D50 doses (rule / exceptions), dissociation index
    lesion_stats.txt           pooled dose-response Spearman, dissociation index, specificity vs
                               random controls, unit lesions, best-model path along the low-pass sweep

Dissociation index: SI = (drop in acc_exc) - (drop in test acc) at matched dose; positive = exceptions hurt
more than the rule. Reported at q = .25 / .5 for the energy lesions and at k = 64 / 256 for unit lesions.

Usage: python3 grok/lesions_report.py --base-out ./exceptions_runs [--task add|div|max]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.lesion_summary import load_lesion_tables, mean_over_controls, stats_text, summary_table, with_deltas
from ..plots.lesions import fig_cognitive, fig_development, fig_headline

SIDE = "in"      # the stored sweeps also hold answer-head ('out') rows; the paper reports the input embedding


# Entry point that aggregates the raw lesion sweeps into the lesion figures, the summary table and
# the statistics text.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes <base_out>/paper/lesions/
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--task", default="add", choices=["add", "div", "max"], help="task whose runs are reported (default add)")
    args = ap.parse_args()
    out = args.base_out / "paper" / "lesions"
    sfx = "" if args.task == "add" else f"_{args.task}"
    d = load_lesion_tables(args.base_out, args.task)
    x_all = mean_over_controls(with_deltas(d))
    fig_development(x_all, SIDE, out, sfx)
    x = x_all[x_all.step_req == "final"]
    print(f"reporting the final checkpoint: {x.groupby(['arch','condition','seed']).ngroups} runs")
    summ = summary_table(x, SIDE)
    summ.to_csv(out / f"lesion_summary{sfx}.csv", index=False)
    txt = f"task = {args.task}\n" + stats_text(x, summ, SIDE)
    (out / f"lesion_stats{sfx}.txt").write_text(txt + "\n")
    print(txt)
    if args.task != "max":
        fig_headline(x, SIDE, out, sfx)
    if args.task == "add":
        fig_cognitive(x, SIDE, out, sfx)
    print("\nfigures written to", out)


if __name__ == "__main__":
    main()
