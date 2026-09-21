""""Where do the algorithms live" for the joint (B1) runs: class-mean Fourier geometry along the forward pass.

For every joint run (one model that learned add + div + max) this walks the forward pass of the final
checkpoint, records the residual stream / sublayer outputs at the EQ position (plus the a/b token embeddings,
and with the `positions` preset the residual stream at the a, op and b positions; with `deep` the sublayer
internals), computes class means over the FULL task table (EQ-position locations grouped by the answer class;
the a/b token-embedding panels grouped by the token's own value), re-orders them into each task's privileged
basis (add: natural; div: discrete log, generator 2; max: natural / ordinal) and writes the per-location
metrics: Fourier concentration (top-4, and among modes >= 3), dominant modes, best-mode circle correlation,
linear-distance correlation (max).

Output (in --out-dir, default <base_out>/paper/manifolds/):
    location_metrics.csv     arch, seed, step, task, location, preset, metrics -- MERGED across calls
                             (rows with the same arch/seed/step/task/location are replaced), the input of
                             `python3 -m grok.cli.manifold_summary` (paper: manifold heatmaps, best-location
                             table, positional-embedding ablation)

Usage:
    python3 grok/jonts.py --base-out ./exceptions_runs --preset final --out-dir ./exceptions_runs_pos/paper/manifolds_nopos_reference
    python3 grok/jonts.py --base-out ./exceptions_runs_pos --preset positions --out-dir ./exceptions_runs_pos/paper/manifolds

Needs torch; mamba is slow on CPU (SSD einsums, batch 512). Presets: final (comparable EQ locations),
positions (residual stream at a / op / b / EQ at every level), deep (first-sublayer internals).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ..analysis.manifolds import PRESETS, merge_location_metrics, task_class_means, task_metrics
from ..core.tasks import ordered_rows
from ..models.common import device
from ..runs.checkpoints import load_model, resolve_checkpoint
from ..runs.discover import find_runs
from ..runs.io import load_summary


# Entry point of the "where do the algorithms live" analysis: walks the forward pass of each joint
# run's final checkpoint and records the class-mean geometry at every location.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - merges the per-location metrics into location_metrics.csv
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--runs", default="joint", help="regex on run dir (default: joint)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--preset", choices=list(PRESETS), default="final",
                    help="which locations to capture: final (comparable EQ locations), positions (residual "
                         "stream at a / op / b / eq at every level), deep (first-sublayer internals)")
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write (default <base_out>/paper/manifolds)")
    args = ap.parse_args()
    deep = args.preset == "deep"
    positions = args.preset == "positions"

    out_dir = args.out_dir or (args.base_out / "paper" / "manifolds")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args.base_out, args.runs, args.seeds, kind="joint")
    print(f"joint runs: {len(runs)}  | device: {device} | preset: {args.preset}")

    rows_out = []
    for run in runs:
        step, path = resolve_checkpoint(run, "final")
        if path is None:
            print(f"  [skip] {run.dir.name}: no final checkpoint")
            continue
        summ = load_summary(run) or {}
        grok_steps = summ.get("grok_step") or {}
        step_num = int(summ.get("steps_run") or -1)
        print(f"[{run.arch} seed {run.seed}] final")
        model = load_model(run, path)
        for t in run.tasks:
            M_by_loc = task_class_means(run.arch, model, t, deep=deep, positions=positions, locs=PRESETS[args.preset])
            for k, (loc, M) in enumerate(M_by_loc.items()):
                rows_out.append({"arch": run.arch, "seed": run.seed, "step": "final",
                                 "step_num": step_num, "grok_step": grok_steps.get(t),
                                 "task": t, "location": loc, "loc_index": k,
                                 "preset": args.preset, **task_metrics(ordered_rows(M, t), t)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rows_out:
        csv_path = out_dir / "location_metrics.csv"
        df = merge_location_metrics(csv_path, pd.DataFrame(rows_out))
        print("saved:", csv_path, f"({len(df)} rows)")


if __name__ == "__main__":
    main()
