"""Causal lesions on finished checkpoints (inference only, GPU-aware).

Fourier lesions of the numeric token embedding (dose = share of spectral energy; natural basis for add,
discrete-log basis for div; none for max): lowpass keeps DC and f <= f_cut; etop / ebottom remove the most /
least energetic modes until a share q of the energy is gone; erand removes random modes to the same q (control,
--n-control draws). Unit lesions zero the top-k down-projection input units ranked by the gradient x activation
attribution of the exception-vs-rule logit gap (neurons) against k random units (randunits). Every lesioned
prediction table is read out behaviourally and in cognitive-model space on the common probes.

Writes <base_out>/paper/lesions/tables/
    lesions_table<tag>.csv        one row per run x checkpoint x lesion x dose (x control draw)
    embedding_spectrum<tag>.csv   per-mode energy share of the numeric embedding rows

Usage:
    python3 grok/lesions.py --base-out ./exceptions_runs                       # add rho0.02 wd0.1, all archs/seeds, final.pt
    python3 grok/lesions.py --base-out ./exceptions_runs --runs kimi_add_rho0.05 --seeds 1000 --steps final 0.5g
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analysis.lesions import ALL_LESIONS, LesionOptions, dose_grids, sweep_checkpoint
from ..models.common import device
from ..runs.discover import find_runs


# Entry point of the causal lesion sweeps on finished checkpoints (inference only, GPU-aware).
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes the raw sweep tables and embedding spectra under <base_out>/paper/lesions/tables/
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--runs", default=r"_add_rho0\.02_wd0\.1_", help="regex on run dir (default: add rho0.02 wd0.1)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", default=["final"],
                    help="'final', snapshot steps (nearest is used) and/or grok-relative 'Xg' tokens, e.g. 0.5g 1g")
    ap.add_argument("--lesions", nargs="*", default=list(ALL_LESIONS), choices=list(ALL_LESIONS))
    ap.add_argument("--n-control", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for the output table name")
    args = ap.parse_args()

    runs = find_runs(args.base_out, args.runs, args.seeds, kind="single")    # exceptions live in single-task runs
    if not runs:
        raise SystemExit("no runs matched")
    print(f"{len(runs)} runs | device {device} | lesions {args.lesions} | steps {args.steps}")
    opts = LesionOptions(lesions=tuple(args.lesions))
    G = dose_grids(args.n_control)
    out_dir = args.base_out / "paper" / "lesions" / "tables"      # raw sweep tables; figures go one level up
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, spec = [], []
    for run in runs:
        print(f"[run] {run.condition} seed={run.seed}")
        for st in args.steps:
            r, s = sweep_checkpoint(run, st, opts, G)
            rows.extend(r)
            spec.extend(s)
        # write after every run so a long sweep can be inspected while it runs
        pd.DataFrame(rows).to_csv(out_dir / f"lesions_table{args.tag}.csv", index=False)
        pd.DataFrame(spec).to_csv(out_dir / f"embedding_spectrum{args.tag}.csv", index=False)
    tab = pd.DataFrame(rows)
    print(f"\nsaved {out_dir / f'lesions_table{args.tag}.csv'} ({len(tab)} rows)")
    if len(tab):
        pd.set_option("display.width", 220)
        cols = ["arch", "seed", "step", "lesion", "param", "dose", "test_acc", "test_ce", "acc_exc",
                "overreg_rate", "exc_logit_gap", "best_model", "gcm_circular_lambda", "rulex_m"]
        print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
