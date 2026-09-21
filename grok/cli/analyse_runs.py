"""Per-run offline analysis (expensive layer: needs torch, snapshots and preds).

Writes <run>/analysis_offline/: metrics.csv, fourier_trajectory_<task>.csv, cognitive_trajectory_<task>.csv.
Reuses cached CSVs unless --force; --skip-fourier / --skip-cognitive keep the other half. Aggregate afterwards with `python3 grok/aggpaper.py`.

Usage: python3 grok/exceptions_analysis.py --base-out ./exceptions_runs [--runs REGEX] [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.per_run import PerRunOptions, analyze_run
from ..runs.discover import find_runs


# Entry point of the per-run offline analysis - the expensive layer that needs torch, snapshots and
# prediction dumps. Cached CSVs are reused unless --force is given.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes <run>/analysis_offline/ for every matching run
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--runs", type=str, default=None, help="regex filter on run dir path, e.g. 'kimi_add_rho0.02'")
    ap.add_argument("--max-fourier-snaps", type=int, default=80)
    ap.add_argument("--max-cog-checkpoints", type=int, default=80)
    ap.add_argument("--skip-fourier", action="store_true")
    ap.add_argument("--skip-cognitive", action="store_true")
    ap.add_argument("--no-strict-artifacts", action="store_true", help="downgrade artifact/processing mismatches to warnings")
    ap.add_argument("--force", action="store_true", help="recompute per-run CSVs even if they exist")
    args = ap.parse_args()
    opts = PerRunOptions(base_out=args.base_out, max_fourier_snaps=args.max_fourier_snaps,
                         max_cog_checkpoints=args.max_cog_checkpoints, skip_fourier=args.skip_fourier,
                         skip_cognitive=args.skip_cognitive,
                         strict=not args.no_strict_artifacts, force=args.force)

    runs = find_runs(args.base_out, args.runs, verbose=True)
    if not runs:
        raise SystemExit(f"no runs found under {args.base_out}")
    print(f"found {len(runs)} runs")
    for r in runs:
        analyze_run(r, opts)
    print(f"\nPER-RUN ANALYSIS DONE — {len(runs)} runs; aggregate with aggpaper.py")


if __name__ == "__main__":
    main()
