"""H3b: is the ORDER in which individual test items are learned shared across
architectures? Per-item "grok time" = first dumped step after the item's LAST error on the
full-table argmax dumps (preds/<task>/step_*.npz), i.e. the onset of persistent correctness.

Reads   <run>/preds/<task>/step_*.npz, <run>/data_split.npz   (single-task runs, rho as selected)
Writes  <base_out>/paper/supp/
    item_order_stats_<tag>.txt   within-arch seed reliability, cross-arch Spearman of seed-mean item times
    item_order_<tag>.png         cross-arch scatter of seed-mean item times + a P x P map per arch

Usage: python3 grok/item_order.py --base-out ./exceptions_runs [--task add] [--rho 0] [--wd 0.1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analysis.items import item_order_stats, item_times
from ..plots.items import fig_item_order
from ..runs.discover import find_runs
from ..runs.io import grok_step_of


# Entry point of the H3b item-order analysis: is the order in which individual test items are learned
# shared across architectures?
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes the item-order statistics text and figure into paper/supp/
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--task", default="add")
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--P", type=int, default=149)
    args = ap.parse_args()
    out = args.base_out / "paper" / "supp"          # a boundary result: supplementary
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_rho{args.rho:g}_wd{args.wd:g}"

    runs = [r for r in find_runs(args.base_out, kind="single")
            if r.tasks == (args.task,) and r.rho == args.rho and r.wd == args.wd]
    groks = [grok_step_of(r, args.task) for r in runs]
    print(f"{len(runs)} runs: " + ", ".join(f"{r.arch} s{r.seed} (grok {g})" for r, g in zip(runs, groks)))
    T = pd.concat([item_times(r, args.task, args.P, g) for r, g in zip(runs, groks)], ignore_index=True)
    means, txt = item_order_stats(T, runs, groks, args.task, args.rho, args.wd, args.P)
    (out / f"item_order_stats_{tag}.txt").write_text(txt + "\n")
    print(txt)
    fig_item_order(means, args.P, args.task, args.rho, args.wd, out / f"item_order_{tag}.png")


if __name__ == "__main__":
    main()
