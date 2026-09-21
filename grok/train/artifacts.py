from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..cognitive.exceptions import ExceptionInfo
from ..core.tasks import P, TASK_OP
from ..models.common import device
from ..models.zoo import full_table_predictions
from .config import (ACC_THRESHOLDS, BATCH, GROK_THRESH, LR, METRICS_EVERY, OVERREG_THRESHOLDS, PREDS_EVERY,
                     SAVE_STATE_EVERY, SNAPSHOT_FP16, SPLIT_SEED)
from .data import clean_split


# The fixed log-spaced snapshot schedule, trimmed to a run's step cap (the cap itself is always
# included).
# Input: cap - the run's maximum step
# Output: sorted list[int] - the steps at which a snapshot is saved
def make_snap_steps(cap: int):
    base = [1, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
            6000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000]
    steps = [s for s in base if s <= cap]
    if cap not in steps:
        steps.append(cap)
    return sorted(set(steps))


# Whether a prediction dump is due at this step, under the coarse schedule or inside the dense window.
# Input: step - current step; preds_every - coarse period; dense_every, dense_from, dense_until -
# the dense window (dense_every = 0 disables it)
# Output: bool
def preds_due(step: int, preds_every: int, dense_every: int,
              dense_from: int, dense_until: int) -> bool:
    if dense_every > 0 and dense_from <= step <= dense_until and step % dense_every == 0:
        return True
    return step % preds_every == 0


# Writes run_config.json on the first start and asserts the run's identity on every later one, so
# one condition can never share an out_dir with another. The dump schedule is recorded at creation
# and read back, so a resumed run keeps the schedule it started with and offline verification
# reads the same file.
# Input: out_dir - Path of the seed directory; arch, tasks, seed, rho, wd, variant - the identity;
# dense_every, dense_from, dense_until - the preds dump window
# Output: dict - the config as stored; raises RuntimeError on an out_dir collision
def check_or_write_run_config(out_dir: Path, arch: str, tasks, seed: int,
                              rho: float, wd: float,
                              dense_every: int, dense_from: int, dense_until: int,
                              variant: str = "matched_2M") -> dict:
    identity = {"arch": arch, "tasks": list(tasks), "rho": float(rho),
                "wd": float(wd), "freq_tiers": None,          # key kept for the stored configs' format
                "seed": int(seed), "variant": variant}
    path = out_dir / "run_config.json"
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        saved_id = {k: saved.get(k) for k in identity}
        if saved_id != identity:
            raise RuntimeError(
                f"out_dir collision: {path} holds {saved_id}, but this run is "
                f"{identity}. Use a fresh out_dir per condition.")
        return saved
    info = dict(identity)
    info.update({"P": P, "lr": LR, "batch": BATCH,
                 "per_task_batch": BATCH // len(tasks),
                 "split_seed": SPLIT_SEED,
                 "metrics_every": METRICS_EVERY, "grok_thresh": GROK_THRESH,
                 "preds_every": PREDS_EVERY,
                 "preds_dense_every": dense_every,
                 "preds_dense_from": dense_from,
                 "preds_dense_until": dense_until,
                 "save_state_every": SAVE_STATE_EVERY,
                 "snapshot_fp16": SNAPSHOT_FP16})
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


# Loads data_split.npz, or creates it from the clean splits on the first start. It holds the clean
# train/test arrays for EVERY task of the run, namespaced {task}_{train|test}_{a|b|c}. The split
# comes from a device-seeded randperm and CUDA and CPU differ, so these arrays - not the seed -
# are authoritative on resume and in the offline analysis.
# Input: out_dir - Path of the seed directory; tasks - sequence of task names
# Output: dict {task: (train tuple, test tuple)} of long tensors on the active device
def load_or_create_splits(out_dir: Path, tasks):
    split_path = out_dir / "data_split.npz"
    if split_path.exists():
        z = np.load(split_path)

        def to_t(name):
            return torch.as_tensor(np.asarray(z[name]), dtype=torch.long, device=device)

        out = {}
        for t in tasks:
            tr = (to_t(f"{t}_train_a"), to_t(f"{t}_train_b"), to_t(f"{t}_train_c"))
            te = (to_t(f"{t}_test_a"), to_t(f"{t}_test_b"), to_t(f"{t}_test_c"))
            out[t] = (tr, te)
        print("loaded split(s) from", split_path)
        return out

    out = {}
    arrays = {}
    for t in tasks:
        tr, te = clean_split(t)
        out[t] = (tr, te)
        for name, tup in (("train", tr), ("test", te)):
            for col, ten in zip("abc", tup):
                arrays[f"{t}_{name}_{col}"] = ten.cpu().numpy()
    np.savez(split_path, **arrays)
    print("saved split(s) to", split_path)
    return out


# Writes exceptions.npz on the first start, and on resume asserts that the regenerated exceptions
# are identical to the stored ones - a mismatch means the directory holds a different
# condition or seed, and the run refuses to continue.
# Input: out_dir - Path of the seed directory; exc - ExceptionInfo, or None for a clean run
# Output: None; raises RuntimeError on a mismatch
def persist_or_assert_exceptions(out_dir: Path, exc: Optional[ExceptionInfo]):
    if exc is None:
        return
    path = out_dir / "exceptions.npz"
    fields = ("idx", "a", "b", "exc_label", "rule_label")     # stored files may also carry the unused tier/weights arrays
    if path.exists():
        z = np.load(path)
        for k in fields:
            saved = np.asarray(z[k])
            regen = np.asarray(getattr(exc, k))
            ok = (np.allclose(saved, regen) if saved.dtype.kind == "f"
                  else np.array_equal(saved, regen))
            if not ok:
                raise RuntimeError(
                    f"exceptions.npz mismatch on '{k}' in {out_dir} — the saved "
                    f"run was a different condition/seed. Refusing to resume.")
        print("resume check: regenerated exceptions match", path)
    else:
        np.savez(path, **{k: np.asarray(getattr(exc, k)) for k in fields})
        print(f"saved {exc.n} exceptions to", path)


# Saves the model parameters at one step, in fp16 when SNAPSHOT_FP16 is set (the analysis is
# inference-only, and fp32 would cost about 20 MB per snapshot).
# Input: model - nn.Module; snap_dir - Path of the snapshots directory; step - int
# Output: None - writes step_<step>.pt
def save_snapshot(model, snap_dir: Path, step: int):
    snap_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if SNAPSHOT_FP16 else torch.float32
    torch.save(
        {"step": step,
         "state_dict": {name: p.detach().cpu().to(dtype).clone()
                        for name, p in model.named_parameters()}},
        snap_dir / f"step_{step:06d}.pt")


# Dumps the model's argmax predictions over the complete task table, as compressed int16.
# Input: model - nn.Module; task - task name; path - Path of the .npz to write
# Output: None
def dump_preds(model, task: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = full_table_predictions(model, task, TASK_OP[task], P, device)
    np.savez_compressed(path, **{k: np.asarray(v).astype(np.int16)
                                 for k, v in preds.items()})


# Saves the resumable training state, plus metrics.json and thresholds.json beside it.
# Input: model, opt - the module and optimiser; step - int; metrics - list of logged rows;
# grok_step - per-task grok steps; crossings - threshold crossings; snap_steps_saved - list;
# out_dir - Path of the seed directory
# Output: None
def save_training_state(model, opt, step, metrics, grok_step, crossings,
                        snap_steps_saved, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": step,
         "model_state_dict": model.state_dict(),
         "optimizer_state_dict": opt.state_dict(),
         "metrics": metrics,
         "grok_step": grok_step,
         "threshold_crossings": crossings,
         "snap_steps_saved": snap_steps_saved},
        out_dir / "last_training_state.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(crossings, f, indent=2)


# The empty threshold-crossing table of a run: one entry per accuracy threshold per task, plus the
# overregularisation thresholds when the run carries exceptions.
# Input: tasks - sequence of task names; has_exceptions - bool
# Output: dict[str, None] keyed by threshold name
def init_crossings(tasks, has_exceptions: bool):
    d = {}
    for t in tasks:
        for thr in ACC_THRESHOLDS:
            d[f"test_{t}>={thr:g}"] = None
    if has_exceptions:
        for thr in OVERREG_THRESHOLDS:
            d[f"overreg_rate>={thr:g}"] = None
    return d


# Inserts or replaces one run's row in runs_manifest.csv at the root of the run tree.
# Input: base_out - Path of the run tree; row - dict with at least condition and seed
# Output: None - rewrites the manifest sorted by condition and seed
def update_manifest(base_out: Path, row: dict):
    path = Path(base_out) / "runs_manifest.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        mask = (df["condition"] == row["condition"]) & (df["seed"] == row["seed"])
        df = pd.concat([df[~mask], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["condition", "seed"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print("manifest updated:", path)
