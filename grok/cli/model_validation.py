"""Held-out likelihood, permutation null and extra cognitive models on a checkpoint subset of the single-task
add runs. Writes <base_out>/paper/supp/model_validation.csv, model_validation_stats.txt and atrium_gate.png.
Usage: python3 -m grok.cli.model_validation --base-out ./exceptions_runs [--runs REGEX] [--n-perm 50]"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.model_validation import summarise, validate_run
from ..plots.validation import fig_atrium_gate
from ..runs.discover import find_runs


# Entry point of the cognitive-model validation: held-out likelihood, a permutation null and the
# three extra models on a checkpoint subset of the single-task add runs.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes model_validation.csv, its statistics text and atrium_gate.png into paper/supp/
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--runs", default="_add_", help="regex on the run path (default: every single-task add run)")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--figure-only", action="store_true", help="redraw atrium_gate.png from the stored model_validation.csv")
    args = ap.parse_args()
    out = args.base_out / "paper" / "supp"
    out.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        fig_atrium_gate(pd.read_csv(out / "model_validation.csv"), out / "atrium_gate.png")
        print("wrote", out / "atrium_gate.png")
        return
    runs = [r for r in find_runs(args.base_out, args.runs, kind="single") if r.task == "add"]
    print(f"{len(runs)} runs")
    rng = np.random.default_rng(0)
    frames = []
    for i, run in enumerate(runs):
        t0 = time.time()
        frames.append(validate_run(run, rng, args.n_perm))
        pd.concat(frames, ignore_index=True).to_csv(out / "model_validation.csv", index=False)
        print(f"[{i+1}/{len(runs)}] {run.label}: {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    txt = summarise(df)
    (out / "model_validation_stats.txt").write_text(txt + "\n")
    fig_atrium_gate(df, out / "atrium_gate.png")
    print(txt)


if __name__ == "__main__":
    main()
