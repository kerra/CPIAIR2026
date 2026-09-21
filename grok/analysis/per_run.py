from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..runs.checkpoints import select_steps, snapshot_steps
from ..runs.io import load_metrics
from ..runs.records import Run
from ..runs.verify import verify_run_artifacts
from .trajectories import cognitive_trajectory, fourier_trajectory


# Options of the per-run offline analysis: where the run tree is, the caps on how many snapshots
# and checkpoints are processed, which stages to skip, and whether mismatches raise or warn.
@dataclass(frozen=True)
class PerRunOptions:
    base_out: Path
    max_fourier_snaps: int = 80
    max_cog_checkpoints: int = 80
    skip_fourier: bool = False
    skip_cognitive: bool = False
    strict: bool = True               # artifact / processing mismatches raise instead of warn
    force: bool = False               # recompute per-run CSVs even if they exist


# per-run driver

# Runs the offline analysis of ONE run into <run>/analysis_offline/: metrics.csv plus a Fourier and
# a cognitive trajectory per task. Existing CSVs are reused unless --force is given, and a reused
# cache is audited against the current disk selection so stale files are reported.
# Input: run - Run; opts - PerRunOptions
# Output: dict - run, metrics and the per-task fourier / cognitive frames
def analyze_run(run: Run, opts: PerRunOptions) -> dict:
    d = run.dir
    out = d / "analysis_offline"
    out.mkdir(exist_ok=True)
    print(f"[run] {run.condition} seed={run.seed} "
          f"(arch={run.arch}, tasks={'+'.join(run.tasks)})")

    strict = opts.strict
    verify_run_artifacts(run, opts.base_out, strict=strict)

    mdf = load_metrics(run)
    mdf.to_csv(out / "metrics.csv", index=False)

    def audit_cache(what, df, avail, max_n, task, **sel_kwargs):
        cur = (set(select_steps(run, task, avail, max_n,
                                what=f"{what} cache-check", **sel_kwargs))
               if avail else set())
        got = set(int(s) for s in df["step"]) if len(df) else set()
        if got == cur:
            print(f"  [{what}] cache OK: {len(got)}/{len(cur)} cached rows match "
                  f"current disk selection")
        else:
            print(f"  WARNING [{what}] cached CSV rows != current disk selection — "
                  f"missing {sorted(cur - got) or '[]'}, stale-extra {sorted(got - cur) or '[]'}; "
                  f"rerun with --force to recompute")

    fdfs, cdfs = {}, {}
    for task in run.tasks:
        fpath = out / f"fourier_trajectory_{task}.csv"
        snaps = list(snapshot_steps(run))
        if opts.skip_fourier:
            fdf = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()
            print(f"  [fourier:{task}] --skip-fourier — "
                  + (f"cached CSV with {len(fdf)} rows" if fpath.exists() else "no cache"))
        elif fpath.exists() and not opts.force:
            fdf = pd.read_csv(fpath)
            print(f"  fourier_trajectory_{task}.csv exists — reusing (--force to recompute)")
            audit_cache(f"fourier:{task}", fdf, snaps, opts.max_fourier_snaps, task)
        else:
            fdf = fourier_trajectory(run, task, opts.max_fourier_snaps, strict=strict)
            fdf.to_csv(fpath, index=False)
            print("  saved:", fpath)
        fdfs[task] = fdf

        cpath = out / f"cognitive_trajectory_{task}.csv"
        preds_avail = [int(p.stem.split("_")[1])
                       for p in (d / "preds" / task).glob("step_*.npz")]
        if opts.skip_cognitive:
            cdf = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame()
            print(f"  [cognitive:{task}] --skip-cognitive — "
                  + (f"cached CSV with {len(cdf)} rows" if cpath.exists() else "no cache"))
        elif cpath.exists() and not opts.force:
            cdf = pd.read_csv(cpath)
            print(f"  cognitive_trajectory_{task}.csv exists — reusing (--force to recompute)")
            audit_cache(f"cognitive:{task}", cdf, preds_avail, opts.max_cog_checkpoints,
                        task, window_hi_mult=3.0, take_full_window=True)
        else:
            cdf = cognitive_trajectory(run, task, opts.max_cog_checkpoints, strict=strict)
            cdf.to_csv(cpath, index=False)
            print("  saved:", cpath)
        cdfs[task] = cdf
    return {"run": run, "metrics": mdf, "fourier": fdfs, "cognitive": cdfs}
