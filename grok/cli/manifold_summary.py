"""Minimal manifold summary for the paper (replaces the grids of per-run manifold figures).

Reads the location_metrics.csv tables written by grok/jonts.py for the matrix joint runs (no positional
embeddings; the reference) and for the matched_2M_pos ablation, plus the two runs_table.csv for the grok order.
Writes <out>/:
    manifold_heatmaps.png      per architecture: tasks x locations heatmap of the task metric (seed mean, final)
    manifold_table.csv / .md   per arch x task: best location, metric mean ± sd, dominant mode per seed, circle corr
    pos_ablation.csv / .md     layer-0 locations: metric without vs with positional embeddings (+ grok order lines)

Usage: python3 -m grok.cli.manifold_summary   (defaults point at exceptions_runs_pos/paper/…; --out exceptions_runs/paper/manifolds)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.manifold_summary import (METRIC_LABEL, best_locations, grok_order_lines, heatmap_grid,
                                         load_location_metrics, pos_ablation, seed_mean, to_markdown)
from ..core.tasks import ARCHS
from ..plots.manifold_summary import fig_manifold_heatmaps


# Entry point of the paper's minimal manifold summary: reads the location_metrics.csv tables of the
# reference runs and the positional-embedding ablation and renders the heatmaps and tables.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes the heatmap figure and the manifold / ablation tables into --out
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nopos", type=Path, default=Path("exceptions_runs_pos/paper/manifolds_nopos_reference/location_metrics.csv"),
                    help="location_metrics.csv of the matrix joint runs (no positional embeddings)")
    ap.add_argument("--pos", type=Path, default=Path("exceptions_runs_pos/paper/manifolds/location_metrics.csv"),
                    help="location_metrics.csv of the matched_2M_pos joint runs")
    ap.add_argument("--runs-table", type=Path, default=Path("exceptions_runs/paper/runs_table.csv"))
    ap.add_argument("--runs-table-pos", type=Path, default=Path("exceptions_runs_pos/paper/runs_table.csv"))
    ap.add_argument("--out", type=Path, default=Path("exceptions_runs/paper/manifolds"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sm_nopos = seed_mean(load_location_metrics(args.nopos))
    grids = {a: heatmap_grid(sm_nopos, a) for a in ARCHS if (sm_nopos.arch == a).any()}
    fig_manifold_heatmaps(grids, args.out / "manifold_heatmaps.png",
                          "Where each task's structure lives along the forward pass (joint runs, final checkpoint, "
                          "mean of 3 seeds, no positional embeddings)")

    table = best_locations(sm_nopos)
    table.to_csv(args.out / "manifold_table.csv", index=False)
    md = ["# Where each task's structure lives (joint runs, final checkpoint, mean ± sd over 3 seeds, no positional embeddings)",
          "", "Metric per task: " + "; ".join(f"{t}: {METRIC_LABEL[t]}" for t in METRIC_LABEL) +
          ". `location` = best of the comparable EQ locations shown in manifold_heatmaps.png; `best_anywhere` = best of every "
          "captured location (block internals, token positions). `modes_per_seed` = dominant Fourier mode per seed "
          "(add: among modes ≥ 3; div: dlog basis, 74 = Legendre).",
          "", to_markdown(table)]
    (args.out / "manifold_table.md").write_text("\n".join(md) + "\n")

    if args.pos.exists():
        sm_pos = seed_mean(load_location_metrics(args.pos))
        abl = pos_ablation(sm_nopos, sm_pos)
        abl.to_csv(args.out / "pos_ablation.csv", index=False)
        short = abl[abl.best != ""][["arch", "task", "location", "value_mean_nopos", "value_mean_pos", "delta",
                                      "modes_nopos", "modes_pos", "best"]]
        lines = ["# Positional-embedding ablation (matched_2M_pos): layer-0 structure without vs with learned positional embeddings",
                 "", "Joint runs, final checkpoint, seed means; layer-0 locations only (full table in pos_ablation.csv). "
                 "`best` marks the best layer-0 location under each condition.", "", to_markdown(short), ""]
        for name, path in (("matrix (no pos)", args.runs_table), ("matched_2M_pos", args.runs_table_pos)):
            gl = grok_order_lines(path)
            if gl:
                lines += [f"Grok order, {name}:", ""] + [f"    {l}" for l in gl] + [""]
        (args.out / "pos_ablation.md").write_text("\n".join(lines))
        print((args.out / "pos_ablation.md").read_text())
    print((args.out / "manifold_table.md").read_text())
    print("written to", args.out)


if __name__ == "__main__":
    main()
