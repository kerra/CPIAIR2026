from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .records import Run


# Asserts the on-disk snapshot and preds counts of a run against what the run itself recorded in
# summary.json, run_config.json and the manifest. A run still in progress is reported and skipped.
# Input: run - Run; base_out - Path of the run tree; strict - raise instead of warning
# Output: None - prints the verdict; raises AssertionError under strict when counts disagree
def verify_run_artifacts(run: Run, base_out: Path, strict: bool = True):
    d = run.dir
    snaps_disk = sorted(int(p.stem.split("_")[1]) for p in (d / "snapshots").glob("step_*.pt"))
    spath = d / "summary.json"
    if not spath.exists():
        n_preds = sum(1 for _ in (d / "preds").rglob("step_*.npz"))
        print(f"  [artifacts] no summary.json (run in progress?) — count assertions "
              f"skipped; on disk: {len(snaps_disk)} snapshots, {n_preds} preds dumps")
        return
    with open(spath) as f:
        summary = json.load(f)
    steps_run = int(summary["steps_run"])
    expected_snaps = sorted(int(s) for s in summary.get("snap_steps", []))
    with open(d / "run_config.json") as f:
        cfg = json.load(f)
    pe = int(cfg.get("preds_every", 250))
    de = int(cfg.get("preds_dense_every", 0))
    dfrom = int(cfg.get("preds_dense_from", 0))
    duntil = int(cfg.get("preds_dense_until", 0))
    expected_preds = set(range(pe, steps_run + 1, pe))
    if de > 0:
        expected_preds |= {s for s in range(de, min(duntil, steps_run) + 1, de)
                           if s >= dfrom}
    expected_preds = sorted(expected_preds | set(expected_snaps))

    problems = []
    if snaps_disk != expected_snaps:
        problems.append(f"snapshots: missing {sorted(set(expected_snaps) - set(snaps_disk))}, "
                        f"unexpected {sorted(set(snaps_disk) - set(expected_snaps))}")
    n_preds_total = 0
    for t in run.tasks:
        on_disk = sorted(int(p.stem.split("_")[1])
                         for p in (d / "preds" / t).glob("step_*.npz"))
        n_preds_total += len(on_disk)
        if on_disk != expected_preds:
            problems.append(f"preds[{t}]: missing {sorted(set(expected_preds) - set(on_disk))}, "
                            f"unexpected {sorted(set(on_disk) - set(expected_preds))}")

    mpath = Path(base_out) / "runs_manifest.csv"
    if mpath.exists():
        mf = pd.read_csv(mpath)
        row = mf[(mf["arch"] == run.arch) & (mf["rho"] == run.rho)
                 & (mf["wd"] == run.wd) & (mf["seed"] == run.seed)
                 & (mf["tasks"] == "+".join(run.tasks))]
        if len(row) and int(row["steps_run"].iloc[0]) != steps_run:
            problems.append(f"manifest steps_run={int(row['steps_run'].iloc[0])} "
                            f"!= summary steps_run={steps_run}")

    if problems:
        msg = f"[artifacts] {run['condition']} seed={run['seed']}: " + "; ".join(problems)
        if strict:
            raise AssertionError(msg + "  (rerun with --no-strict-artifacts to continue anyway)")
        print("  WARNING " + msg)
    else:
        grid_txt = (f"dense({de} in [{dfrom},{duntil}]) ∪ coarse({pe})" if de > 0
                    else f"coarse({pe})")
        print(f"  [artifacts] ok: {len(snaps_disk)} snapshots == summary.snap_steps, "
              f"{n_preds_total} preds dumps == {grid_txt} grid ∪ snapshots per task "
              f"(steps_run={steps_run})")


# Unconditional selected-versus-processed accounting for one analysis stage; always prints, even
# when everything passed.
# Input: what - stage label; rows - the rows produced; excluded - (step, reason) pairs;
# selected - the steps that were selected; strict - raise when anything was excluded
# Output: None - prints the tally; raises AssertionError when the accounting does not add up
def report_processed(what: str, rows: list, excluded: list, selected: list, strict: bool):
    print(f"  [{what}] processed {len(rows)}/{len(selected)} selected -> {len(rows)} CSV rows"
          + ("" if not excluded else "  (EXCLUSIONS BELOW)"))
    for step, reason in excluded:
        print(f"    step {step} EXCLUDED: {reason}")
    if len(rows) + len(excluded) != len(selected):
        raise AssertionError(f"{what}: accounting broken — {len(rows)} rows + "
                             f"{len(excluded)} exclusions != {len(selected)} selected")
    if excluded and strict:
        raise AssertionError(
            f"{what}: {len(excluded)}/{len(selected)} selected checkpoints failed "
            f"(reasons above). Fix the cause or rerun with --no-strict-artifacts.")
