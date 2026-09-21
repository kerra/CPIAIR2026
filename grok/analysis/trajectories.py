from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ..cognitive.fits import compare_cognitive_models
from ..core.fourier import best_fourier_distance_corr, fourier_concentration, linear_corr
from ..core.gcm import GCMProbeCache, probe_mask
from ..core.tasks import P, TASK_OP, full_table, n_classes, ordered_rows
from ..models.common import device
from ..models.zoo import eq_states
from ..runs.checkpoints import load_model, select_steps, snapshot_steps
from ..runs.io import corrupted_train, exc_task_of, load_exceptions, load_task_split
from ..runs.records import Run
from ..runs.verify import report_processed


# Class means of the EQ-position state (after the final norm) over the full task table, batched.
# Input: run - Run; model - the loaded checkpoint; task - task name; batch - items per forward pass
# Output: dict {location: np.ndarray [P, d]} - currently just final_eq
@torch.no_grad()
def class_mean_manifolds(run: Run, model, task: str, batch: int = 4096) -> dict[str, np.ndarray]:
    aa, bb, cc = full_table(task)
    sums = {}
    counts = np.zeros(P, dtype=np.int64)
    for s in range(0, len(aa), batch):
        a = torch.as_tensor(aa[s:s + batch], dtype=torch.long, device=device)
        b = torch.as_tensor(bb[s:s + batch], dtype=torch.long, device=device)
        locs = eq_states(run.arch, model, a, TASK_OP[task], b)
        for k, v in locs.items():
            if k not in sums:
                sums[k] = np.zeros((P, v.shape[1]), dtype=np.float64)
            np.add.at(sums[k], cc[s:s + batch], v.astype(np.float64))
        np.add.at(counts, cc[s:s + batch], 1)
    counts = np.maximum(counts, 1)
    return {k: (v / counts[:, None]).astype(np.float32) for k, v in sums.items()}


# The mechanistic trajectory of a run: at each selected weight snapshot, the Fourier concentration,
# best-mode circle correlation and linear correlation of the class-mean manifold.
# Input: run - Run; task - task name; max_snaps - cap on the snapshots processed;
# strict - raise instead of warning when a snapshot fails
# Output: pd.DataFrame - one row per snapshot, empty when the run has no snapshots
def fourier_trajectory(run, task: str, max_snaps: int, strict: bool = True) -> pd.DataFrame:
    snaps = snapshot_steps(run)
    if not snaps:
        print(f"  [fourier:{task}] processed 0/0 — no snapshots on disk")
        return pd.DataFrame()
    steps = select_steps(run, task, list(snaps), max_snaps, what=f"fourier:{task} snapshots")
    rows_out, excluded = [], []
    for i, step in enumerate(steps):
        try:
            model = load_model(run, snaps[step])
            Ms = class_mean_manifolds(run, model, task)
            row = {"step": step}
            for loc, M in Ms.items():
                rows = ordered_rows(M, task)
                row[f"fourier_conc_{loc}"] = fourier_concentration(rows, top_k=4)[0]
                mode, corr = best_fourier_distance_corr(rows, top_k_modes=8)
                row[f"best_mode_{loc}"] = mode
                row[f"best_corr_{loc}"] = corr
                row[f"linear_corr_{loc}"] = linear_corr(rows)
            rows_out.append(row)
            del model
        except Exception as e:
            excluded.append((step, repr(e)))
        if (i + 1) % 10 == 0 and (i + 1) < len(steps):
            print(f"    progress: {i + 1}/{len(steps)} snapshots done")
    report_processed(f"fourier:{task}", rows_out, excluded, steps, strict)
    return pd.DataFrame(rows_out)


# cognitive-model + exemplar-consistency trajectory (per task)

# The cognitive trajectory of a run: the six models compared on common probes at each selected
# prediction dump. The probe set and the exact GCM vote tables depend only on the train set and the
# task table, not on the network, so they are built once and every checkpoint is then fitted by a
# gather - which is what keeps all six models scored on identical probes.
# Input: run - Run; task - task name; max_ckpts - cap on the dumps processed;
# strict - raise instead of warning when a dump fails
# Output: pd.DataFrame - one row per checkpoint with BICs, log-likelihoods, fitted parameters
# and the pairwise BIC differences; empty when the run has no prediction dumps
def cognitive_trajectory(run, task: str, max_ckpts: int, strict: bool = True) -> pd.DataFrame:
    preds_files = {int(p.stem.split("_")[1]): p
                   for p in sorted((run.dir / "preds" / task).glob("step_*.npz"))}
    if not preds_files:
        print(f"  [cognitive:{task}] processed 0/0 — no preds dumps on disk")
        return pd.DataFrame()
    split = load_task_split(run, task)
    exc = load_exceptions(run) if exc_task_of(run) == task else None
    tr_a, tr_b, tr_c = corrupted_train(split, exc)

    steps = select_steps(run, task, list(preds_files), max_ckpts,
                         what=f"cognitive:{task} checkpoints",
                         window_hi_mult=3.0, take_full_window=True)
    # The probe set and the exact GCM vote tables depend only on the train set
    # and the task table, not on the network: build once, fit every checkpoint
    # by a gather (all six models are scored on the same probes).
    z0 = np.load(preds_files[steps[0]])
    a0 = np.asarray(z0["a"], dtype=np.int64); b0 = np.asarray(z0["b"], dtype=np.int64)
    mask0 = probe_mask(a0, b0, tr_a, tr_b, exc)
    t0 = time.time()
    cache = GCMProbeCache(tr_a, tr_b, tr_c, a0[mask0], b0[mask0], n_classes(task, P), P)
    print(f"  [cognitive:{task}] probes = test items + exception items: n={int(mask0.sum())} of {len(a0)} table items; "
          f"exact GCM cache ({len(cache.lambdas)} lambdas x 2 metrics) built in {time.time() - t0:.1f}s")
    rows, excluded = [], []
    for i, step in enumerate(steps):
        try:
            z = np.load(preds_files[step])
            preds = {k: np.asarray(z[k], dtype=np.int64) for k in ("a", "b", "pred", "rule")}
            report = compare_cognitive_models(
                preds, exc, P=P, task=task,
                train_a=tr_a, train_b=tr_b, train_c=tr_c, gcm_cache=cache)
            row = {"step": step, "best_model": report["best_model"],
                   "probe": report["probe"], "n_probe": report["n_probe"],
                   "n_probe_test": report["n_probe_test"], "n_probe_exc": report["n_probe_exc"]}
            for name in ("uniform", "gcm_hamming", "gcm_circular", "gcm_linear",
                         "rule", "rulex"):
                row[f"bic_{name}"] = report[name]["bic"]
                row[f"ll_{name}"] = report[name]["ll"]
            gcm_bics = {k: report[k]["bic"] for k in ("gcm_hamming", "gcm_circular", "gcm_linear")}
            gcm_best = min(gcm_bics, key=gcm_bics.get)
            rule_fam = min(report["rule"]["bic"], report["rulex"]["bic"])
            row["gcm_best"] = gcm_best
            row["dbic_gcm_vs_uniform"] = report["uniform"]["bic"] - gcm_bics[gcm_best]   # >0: exemplar beats uniform
            row["dbic_rulefam_vs_gcm"] = gcm_bics[gcm_best] - rule_fam                    # >0: rule family beats exemplar
            row["dbic_rulex_vs_rule"] = report["rule"]["bic"] - report["rulex"]["bic"]    # >0: exception route needed
            row["gcm_hamming_lambda"] = report["gcm_hamming"].get("lambda")
            row["gcm_circular_lambda"] = report["gcm_circular"].get("lambda")
            row["gcm_linear_lambda"] = report["gcm_linear"].get("lambda")
            row["rule_eps"] = report["rule"].get("eps")
            row["rulex_eps"] = report["rulex"].get("eps")
            row["rulex_m"] = report["rulex"].get("m")
            row["delta_bic_rulex_minus_gcmh"] = (report["rulex"]["bic"]
                                                 - report["gcm_hamming"]["bic"])
            rows.append(row)
        except Exception as e:
            excluded.append((step, repr(e)))
        if (i + 1) % 10 == 0 and (i + 1) < len(steps):
            print(f"    progress: {i + 1}/{len(steps)} checkpoints fitted")
    report_processed(f"cognitive:{task}", rows, excluded, steps, strict)
    return pd.DataFrame(rows)
