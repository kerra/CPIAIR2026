from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..cognitive.exceptions import ExceptionInfo
from .records import Run

SPLIT_KEYS = ("train_a", "train_b", "train_c", "test_a", "test_b", "test_c")




# Reads a run's summary.json.
# Input: run - Run
# Output: dict, or None when the run has not finished and written one yet
def load_summary(run: Run) -> dict | None:
    p = run.dir / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# Reads a run's training metrics, falling back to the copy inside a training-state checkpoint
# when metrics.json has not been written.
# Input: run - Run
# Output: pd.DataFrame, one row per logged step; raises FileNotFoundError when there is none
def load_metrics(run: Run) -> pd.DataFrame:
    d = run.dir
    if (d / "metrics.json").exists():
        return pd.DataFrame(json.loads((d / "metrics.json").read_text()))
    for name in ("last_training_state.pt", "final.pt"):
        if (d / name).exists():
            import torch
            return pd.DataFrame(torch.load(d / name, map_location="cpu", weights_only=False)["metrics"])
    raise FileNotFoundError(f"no metrics for {d}")


# Reads the train/test split of one task out of data_split.npz.
# Input: run - Run; task - task name
# Output: dict[str, np.ndarray[int64]] - train_a/b/c and test_a/b/c
def load_task_split(run: Run, task: str) -> dict[str, np.ndarray]:
    z = np.load(run.dir / "data_split.npz")
    return {k: np.asarray(z[f"{task}_{k}"], dtype=np.int64) for k in SPLIT_KEYS}




# Reads the exception record of a run. The stored files also carry unused tier and weight
# arrays, left over from a sampler the matrix never used; those are ignored.
# Input: run - Run
# Output: ExceptionInfo, or None when the run has no exceptions
def load_exceptions(run: Run) -> ExceptionInfo | None:
    p = run.dir / "exceptions.npz"
    if not p.exists():
        return None
    z = np.load(p)      # stored files also carry unused tier / weights arrays (a sampler the matrix never used)
    return ExceptionInfo(idx=np.asarray(z["idx"]), a=np.asarray(z["a"]), b=np.asarray(z["b"]),
                         exc_label=np.asarray(z["exc_label"]), rule_label=np.asarray(z["rule_label"]),
                         task=run.tasks[0])


# The task that carries exceptions: the single task of a rho > 0 single-task run.
# Input: run - Run
# Output: str, or None for joint runs and for rho = 0
def exc_task_of(run: Run) -> str | None:
    return run.tasks[0] if (run.rho > 0 and len(run.tasks) == 1) else None


# The train set as the network actually saw it, with exception labels written over the rule
# labels.
# Input: split_t - dict from load_task_split; exc - ExceptionInfo or None
# Output: tuple (train_a, train_b, train_c) of np.ndarray
def corrupted_train(split_t: dict, exc: ExceptionInfo | None):
    c = split_t["train_c"].copy()
    if exc is not None:
        c[exc.idx] = exc.exc_label
    return split_t["train_a"], split_t["train_b"], c


# The step at which test accuracy first crossed 0.95. summary.json is authoritative and the
# metrics are only the fallback.
# Input: run - Run; task - task name, defaulting to the run's single task
# Output: int, or None when the run never grokked
def grok_step_of(run: Run, task: str | None = None) -> int | None:
    task = task or run.task
    summ = load_summary(run)
    if summ is not None:
        g = (summ.get("grok_step") or {}).get(task)
        if g is not None:
            return int(g)
    try:
        m = load_metrics(run)
    except FileNotFoundError:
        return None
    if f"test_{task}" not in m.columns:
        return None
    hit = m[m[f"test_{task}"] > 0.95]
    return int(hit["step"].iloc[0]) if len(hit) else None


# Steps at which the test-accuracy and overregularisation thresholds were first crossed.
# Input: run - Run; task - task name
# Output: sorted list[int] - the crossing steps, empty when thresholds.json is missing
def threshold_steps_of(run: Run, task: str) -> list[int]:
    tpath = run.dir / "thresholds.json"
    if not tpath.exists():
        return []
    crossings = json.loads(tpath.read_text())
    return sorted({int(v) for k, v in crossings.items()
                   if v is not None and (k.startswith(f"test_{task}>=") or k.startswith("overreg_rate>="))})


# Reads a CSV, returning an empty frame instead of raising when it is missing or empty.
# Input: path - Path of the CSV
# Output: pd.DataFrame
def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


# The light per-run outputs of the offline analysis that the aggregation layer works from.
# Input: run - Run
# Output: dict - run, summary, metrics, and the fourier / cognitive trajectory frames per
# task; None when the run has no summary or no metrics yet
def load_offline(run: Run) -> dict | None:
    summ = load_summary(run)
    if summ is None:
        return None
    ao = run.dir / "analysis_offline"
    mdf = read_csv_or_empty(ao / "metrics.csv")
    if mdf.empty:
        return None
    return {"run": run, "summary": summ, "metrics": mdf.sort_values("step").reset_index(drop=True),
            "fourier": {t: read_csv_or_empty(ao / f"fourier_trajectory_{t}.csv") for t in run.tasks},
            "cognitive": {t: read_csv_or_empty(ao / f"cognitive_trajectory_{t}.csv") for t in run.tasks}}
