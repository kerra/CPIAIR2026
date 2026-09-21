from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..cognitive.exceptions import ExceptionInfo
from ..cognitive.fits import compare_cognitive_models
from ..core.fourier import N_MODES, apply_scale, mode_energy, modes_by_energy_fraction, top_modes
from ..core.gcm import GCMProbeCache, probe_mask
from ..core.tasks import P, TASK_OP, full_table, n_classes
from ..models.common import device
from ..runs.checkpoints import load_model, resolve_checkpoint
from ..runs.io import grok_step_of, load_exceptions, load_task_split
from ..runs.records import Run

LESIONS_FOURIER = ("lowpass", "etop", "ebottom", "erand")


LESIONS_UNITS = ("neurons", "randunits")


ALL_LESIONS = LESIONS_FOURIER + LESIONS_UNITS


TASK_BASIS = {"add": "natural", "div": "dlog", "max": None}   # max: no group structure >> no Fourier lesions


# Which lesion families a sweep runs: the Fourier-subspace ones, the unit ones, or both.
@dataclass(frozen=True)
class LesionOptions:
    lesions: tuple[str, ...] = ALL_LESIONS


# The dose grids of each lesion family - exactly the grids the stored sweeps were run on.
# Input: n_control - how many random-control draws per dose
# Output: dict - the lowpass, q and units grids plus n_control
def dose_grids(n_control: int = 3) -> dict:
    return {"lowpass": [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, N_MODES],
            "q": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            "units": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], "n_control": n_control}


# Argmax predictions and per-item log-softmax of a model over the full task table, batched.
# Input: model - nn.Module; task - task name; a, b - full-table coordinates; batch - items per pass
# Output: tuple (np.ndarray[int64] predictions, np.ndarray[float32] [n_items, K] log-probabilities)
@torch.no_grad()
def predict(model, task: str, a: np.ndarray, b: np.ndarray, batch: int = 4096):
    op = TASK_OP[task]
    ta = torch.as_tensor(a, dtype=torch.long, device=device)
    tb = torch.as_tensor(b, dtype=torch.long, device=device)
    preds, logps = [], []
    for s in range(0, len(a), batch):
        lg = model(ta[s:s + batch], op, tb[s:s + batch]).float()
        preds.append(lg.argmax(dim=-1).cpu().numpy())
        logps.append(torch.log_softmax(lg, dim=-1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64), np.concatenate(logps).astype(np.float32)


# behavioural readout

# Behavioural readout of one lesioned prediction table: test, regular-train and exception
# accuracies with their cross-entropies, plus the overregularisation rate and logit gap when the
# run carries exceptions.
# Input: pred, logp - the lesioned predictions; a, b, rule - the full table; split - the
# train/test split; exc - ExceptionInfo or None
# Output: dict of floats; the exception columns are NaN for a clean run
def behavioural_metrics(pred, logp, a, b, rule, split, exc: ExceptionInfo | None) -> dict:
    key = a * P + b
    row = -np.ones(P * P, dtype=np.int64); row[key] = np.arange(len(a))
    te_i = row[split["test_a"] * P + split["test_b"]]
    te_pred = pred[te_i]
    out = {"test_acc": float((te_pred == split["test_c"]).mean()),
           "test_ce": float(-logp[te_i, split["test_c"]].mean())}
    tr_i = row[split["train_a"] * P + split["train_b"]]
    tr_pred = pred[tr_i]
    tr_c = split["train_c"].copy()
    if exc is not None and exc.n > 0:
        tr_c[exc.idx] = exc.exc_label
        reg = np.ones(len(tr_i), bool); reg[exc.idx] = False
        out["train_reg_acc"] = float((tr_pred[reg] == tr_c[reg]).mean())
        out["train_reg_ce"] = float(-logp[tr_i[reg], tr_c[reg]].mean())
        ex_i = row[np.asarray(exc.a) * P + np.asarray(exc.b)]
        pe = pred[ex_i]
        el, rl = np.asarray(exc.exc_label), np.asarray(exc.rule_label)
        hit_e = pe == el; hit_r = pe == rl
        out.update({"acc_exc": float(hit_e.mean()), "overreg_rate": float(hit_r.mean()),
                    "other_err_rate": float(1.0 - hit_e.mean() - hit_r.mean()),
                    "exc_ce": float(-logp[ex_i, el].mean()),
                    "exc_logit_gap": float((logp[ex_i, el] - logp[ex_i, rl]).mean())})
    else:
        out["train_reg_acc"] = float((tr_pred == tr_c).mean())
        out["train_reg_ce"] = float(-logp[tr_i, tr_c].mean())
        out.update({"acc_exc": np.nan, "overreg_rate": np.nan, "other_err_rate": np.nan,
                    "exc_ce": np.nan, "exc_logit_gap": np.nan})
    return out


# Cognitive-model readout of one lesioned prediction table on the common probes: which model wins
# and by how much.
# Input: pred - lesioned predictions; a, b, rule - the full table; exc - ExceptionInfo or None;
# task - task name; train - the (possibly corrupted) train arrays; cache - GCMProbeCache
# Output: dict - best_model, the pairwise BIC differences, the fitted parameters and every BIC
def cognitive_readout(pred, a, b, rule, exc, task, train, cache) -> dict:
    rep = compare_cognitive_models({"a": a, "b": b, "pred": pred, "rule": rule}, exc, P, task,
                                   train_a=train[0], train_b=train[1], train_c=train[2], gcm_cache=cache)
    gcm = {k: rep[k]["bic"] for k in ("gcm_hamming", "gcm_circular", "gcm_linear")}
    gbest = min(gcm, key=gcm.get)
    row = {"best_model": rep["best_model"], "gcm_best": gbest, "n_probe": rep["n_probe"],
           "dbic_gcm_vs_uniform": rep["uniform"]["bic"] - gcm[gbest],
           "dbic_rulefam_vs_gcm": gcm[gbest] - min(rep["rule"]["bic"], rep["rulex"]["bic"]),
           "dbic_rulex_vs_rule": rep["rule"]["bic"] - rep["rulex"]["bic"],
           "gcm_circular_lambda": rep["gcm_circular"]["lambda"],
           "gcm_hamming_lambda": rep["gcm_hamming"]["lambda"],
           "rule_eps": rep["rule"]["eps"], "rulex_m": rep["rulex"].get("m"),
           "rulex_eps": rep["rulex"].get("eps")}
    for k in ("uniform", "gcm_hamming", "gcm_circular", "gcm_linear", "rule", "rulex"):
        row[f"bic_{k}"] = rep[k]["bic"]
    return row


# The scale vector of one lesion at one dose. lowpass keeps DC and every mode f <= param;
# etop / ebottom / erand instead remove the most energetic, the least energetic or random modes
# until the energy share param is gone.
# Input: lesion - lesion name; en - per-mode energy; param - cut-off or energy share;
# rng - Generator, needed only by erand
# Output: tuple (np.ndarray scale over modes 0..N_MODES, int number of modes touched)
def fourier_scale(lesion: str, en: np.ndarray, param: float, rng: np.random.Generator | None):
    scale = np.ones(N_MODES + 1)
    if lesion == "lowpass":
        f_cut = int(param)
        scale[f_cut + 1:] = 0.0
        return scale, N_MODES - f_cut
    if lesion in ("etop", "ebottom", "erand"):
        modes = modes_by_energy_fraction(en, param, {"etop": "top", "ebottom": "bottom", "erand": "rand"}[lesion], rng)
        scale[modes] = 0.0
        return scale, len(modes)
    raise ValueError(lesion)


# Swaps the numeric rows of tok_emb for a lesioned version, keeping the original so it can
# always be restored.
class EmbeddingSurgeon:

    def __init__(self, model):
        self.emb = model.backbone.tok_emb.weight          # [VOCAB, d]
        self.E0 = self.emb.data[:P].detach().float().cpu().numpy().copy()

    def set(self, E_in: np.ndarray | None):
        with torch.no_grad():
            self.emb.data[:P] = torch.as_tensor(E_in if E_in is not None else self.E0,
                                                dtype=self.emb.dtype, device=self.emb.device)

    def restore(self):
        self.set(None)


# unit lesions on the down-projection inputs

# The down-projection layers of an architecture, whose inputs the unit lesions act on.
# Input: model - nn.Module; arch - architecture name
# Output: list of (module name, nn.Linear) pairs
def unit_sites(model, arch: str) -> list[tuple[str, torch.nn.Module]]:
    suffix = {"transformer": "down_proj", "kimi": "W_down", "mamba": "out_proj"}[arch]
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and name.endswith(suffix)]


# Forward pre-hooks on the down-projection inputs: they zero the masked units and, while an
# attribution pass is running, keep the live tensors with retain_grad.
class UnitMasker:
    def __init__(self, sites):
        self.masks = {}
        self.live = None          # dict >> live tensors with retain_grad (for attribution)
        self.handles = [m.register_forward_pre_hook(self._hook(name)) for name, m in sites]

    def _hook(self, name):
        def fn(module, inputs):
            x = inputs[0]
            if self.live is not None:
                x.retain_grad()
                self.live[name] = x
            mask = self.masks.get(name)
            if mask is None:
                return None
            return (x * mask.to(x.dtype).view(1, 1, -1),) + tuple(inputs[1:])
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


# Gradient x activation attribution of the exception-versus-rule logit gap (logit[exc_label] -
# logit[rule_label], summed over the exception items) to every unit at the EQ position. A positive
# value means the unit pushes the memorised answer.
# Input: model - nn.Module; masker - UnitMasker over the sites; task - task name;
# exc - ExceptionInfo; batch - exception items per pass
# Output: dict {site: np.ndarray [n_units]} of attributions
def unit_attribution(model, masker: UnitMasker, task, exc: ExceptionInfo, batch: int = 512) -> dict:
    op = TASK_OP[task]
    a = torch.as_tensor(np.asarray(exc.a), dtype=torch.long, device=device)
    b = torch.as_tensor(np.asarray(exc.b), dtype=torch.long, device=device)
    el = torch.as_tensor(np.asarray(exc.exc_label), dtype=torch.long, device=device)
    rl = torch.as_tensor(np.asarray(exc.rule_label), dtype=torch.long, device=device)
    attr: dict[str, np.ndarray] = {}
    for s in range(0, len(a), batch):
        masker.live = {}
        with torch.enable_grad():
            lg = model(a[s:s + batch], op, b[s:s + batch]).float()
            r = torch.arange(lg.shape[0], device=device)
            gap = (lg[r, el[s:s + batch]] - lg[r, rl[s:s + batch]]).sum()
            gap.backward()
        for name, x in masker.live.items():
            g = (x[:, -1, :] * x.grad[:, -1, :]).detach().float().sum(dim=0).cpu().numpy()
            attr[name] = attr.get(name, 0) + g
        model.zero_grad(set_to_none=True)
    masker.live = None
    return attr


# Flattens the per-site attributions into one ranked table, most exception-pushing first.
# Input: attr - dict {site: attribution array}
# Output: pd.DataFrame - one row per unit with site, unit and attr
def unit_ranking(attr: dict) -> pd.DataFrame:
    rows = [{"site": site, "unit": u, "attr": float(v[u])} for site, v in attr.items() for u in range(len(v))]
    return pd.DataFrame(rows).sort_values("attr", ascending=False).reset_index(drop=True)


# Builds the multiplicative masks that switch off a chosen set of units.
# Input: sites - (name, module) pairs; chosen - rows of a unit ranking to lesion
# Output: dict {site: float tensor [in_features]} with 0 at the chosen units
def masks_from_units(sites, chosen: pd.DataFrame) -> dict:
    masks = {}
    for name, m in sites:
        v = torch.ones(m.in_features, device=device)
        units = np.asarray(chosen[chosen.site == name].unit.values, dtype=np.int64)
        if len(units):
            v[torch.as_tensor(units.copy(), device=device)] = 0.0
        masks[name] = v
    return masks


# Runs every lesion of opts on one checkpoint of one single-task run: the unlesioned baseline, the
# Fourier lesions of the numeric embedding (natural basis for add, discrete-log for div; skipped
# for max, which has no group structure) and the unit lesions with their matched random controls.
# Input: run - Run; step_req - "final", an int or an "Xg" tag; opts - LesionOptions;
# G - the dose grids from dose_grids
# Output: tuple (list[dict] result rows, list[dict] embedding-spectrum rows); empty when the
# requested checkpoint does not exist
def sweep_checkpoint(run: Run, step_req, opts: LesionOptions, G: dict) -> tuple[list[dict], list[dict]]:
    step, path = resolve_checkpoint(run, step_req)
    if path is None:
        print(f"  [skip] no checkpoint for step {step_req} (no grok or no snapshots)")
        return [], []
    out_rows: list[dict] = []
    spec_rows: list[dict] = []
    t0 = time.time()
    model = load_model(run, path)
    task = run.task
    a, b, rule = full_table(task)
    split = load_task_split(run, task)
    exc = load_exceptions(run)
    tr_c = split["train_c"].copy()
    if exc is not None:
        tr_c[exc.idx] = exc.exc_label
    train = (split["train_a"], split["train_b"], tr_c)
    mask = probe_mask(a, b, train[0], train[1], exc)
    cache = GCMProbeCache(train[0], train[1], train[2], a[mask], b[mask], n_classes(task, P), P)
    base = {"arch": run.arch, "condition": run.condition, "seed": run.seed,
            "task": task, "rho": run.rho, "wd": run.wd, "step": step,
            "step_req": str(step_req), "grok_step": grok_step_of(run)}

    def record(lesion, side, param, dose, control_seed, pred, logp, extra=None):
        row = dict(base, lesion=lesion, side=side, param=param, dose=dose, control_seed=control_seed)
        row.update(behavioural_metrics(pred, logp, a, b, rule, split, exc))
        row.update(cognitive_readout(pred, a, b, rule, exc, task, train, cache))
        if extra:
            row.update(extra)
        out_rows.append(row)
        return row

    # ---- baseline ----
    p0, l0 = predict(model, task, a, b)
    r0 = record("none", "-", 0, 0.0, None, p0, l0)
    print(f"  [{run.arch} s{run.seed} step {step}] baseline: test {r0['test_acc']:.3f} "
          f"acc_exc {r0['acc_exc']:.3f} overreg {r0['overreg_rate']:.3f} best={r0['best_model']}")

    # ---- Fourier lesions of the numeric token embedding (basis by task: natural for add, dlog for div) ----
    basis = TASK_BASIS[task]
    fourier_lesions = [l for l in opts.lesions if l in LESIONS_FOURIER]
    if basis is None:
        print(f"    [fourier] skipped for task '{task}' (no group structure; unit lesions only)")
    elif fourier_lesions:
        surgeon = EmbeddingSurgeon(model)
        en, dc = mode_energy(surgeon.E0, basis)
        share = en / en.sum()
        for f in range(1, N_MODES + 1):
            spec_rows.append(dict(base, basis=basis, side="in", mode=f, energy_share=float(share[f - 1]),
                                  rank=int(np.argsort(-en).tolist().index(f - 1)) + 1,
                                  dc_share=float(dc / (dc + en.sum()))))
        print(f"    basis {basis}: top modes in: {top_modes(en, 6).tolist()}")
        for lesion in fourier_lesions:
            doses = [float(f) for f in G["lowpass"]] if lesion == "lowpass" else list(G["q"])
            is_ctrl = lesion == "erand"
            for c in range(G["n_control"] if is_ctrl else 1):
                for dose in doses:
                    crng = np.random.default_rng(1000 + c) if is_ctrl else None   # same draw per dose
                    scale, n_touched = fourier_scale(lesion, en, dose, crng)
                    surgeon.set(apply_scale(surgeon.E0, scale, basis))
                    pred, logp = predict(model, task, a, b)
                    param = int(dose) if lesion == "lowpass" else n_touched
                    record(lesion, "in", param, float(dose), c if is_ctrl else None, pred, logp,
                           {"n_modes_touched": n_touched, "basis": basis})
            surgeon.restore()
        surgeon.restore()

    # ---- unit lesions ----
    unit_lesions = [l for l in opts.lesions if l in LESIONS_UNITS]
    if unit_lesions and exc is not None and exc.n > 0:
        sites = unit_sites(model, run.arch)
        masker = UnitMasker(sites)
        sel = unit_ranking(unit_attribution(model, masker, task, exc))
        n_units = len(sel)
        print(f"    unit sites: {[n for n, _ in sites]} -> {n_units} units, ranked by gradient x activation; "
              f"share of positive attribution in the top 32 = {(sel.attr.head(32).sum() / sel.attr.clip(lower=0).sum()):.2f}")
        sel.to_csv(run.dir / f"unit_selectivity_{step}.csv", index=False)
        ks = [k for k in G["units"] if k <= n_units]
        for lesion in unit_lesions:
            n_ctrl = G["n_control"] if lesion == "randunits" else 1
            for c in range(n_ctrl):
                crng = np.random.default_rng(2000 + c)
                for k in ks:
                    chosen = sel.head(k) if lesion == "neurons" else sel.iloc[crng.choice(n_units, size=k, replace=False)]
                    masker.masks = masks_from_units(sites, chosen)
                    pred, logp = predict(model, task, a, b)
                    record(lesion, "units", k, float(k) / n_units, c if lesion == "randunits" else None, pred, logp,
                           {"n_units_total": n_units,
                            "attr_min_selected": float(chosen.attr.min()) if k else np.nan})
        masker.masks = {}
        masker.remove()
    elif unit_lesions:
        print("    [units] skipped: run has no exceptions (rho = 0)")
    print(f"    done in {time.time() - t0:.0f}s, {len(out_rows)} rows")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_rows, spec_rows
