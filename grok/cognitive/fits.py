from __future__ import annotations

import numpy as np

from ..core.gcm import DEFAULT_GCM_LAMBDAS, GCMProbeCache, bic, gcm_votes, label_table, loglik, probe_mask
from ..core.tasks import n_classes
from .exceptions import ExceptionInfo


# GCM with the hamming distance d((a,b),(a',b')) = [a != a'] + [b != b'], in closed form via
# row and column label counts, so no n x m distance matrix is ever built.
# Input: train_* - the exemplar memory; probe_a, probe_b, probe_pred - the probe set and the
# network's predictions on it; K - classes; lambdas - decay grid to search
# Output: dict - ll, lambda and bic of the best-fitting lambda
def fit_gcm_hamming(train_a, train_b, train_c, probe_a, probe_b, probe_pred,
                    K: int, lambdas=None) -> dict:
    train_a = np.asarray(train_a); train_b = np.asarray(train_b)
    train_c = np.asarray(train_c)
    A = int(max(train_a.max(), np.max(probe_a))) + 1
    B = int(max(train_b.max(), np.max(probe_b))) + 1
    cnt_a = np.zeros((A, K)); cnt_b = np.zeros((B, K))
    np.add.at(cnt_a, (train_a, train_c), 1.0)
    np.add.at(cnt_b, (train_b, train_c), 1.0)
    tot = cnt_a.sum(axis=0)                      # [K]
    # self-match table: label of train item at (a,b), else -1
    L = -np.ones((A, B), dtype=np.int64)
    L[train_a, train_b] = train_c

    pa = np.asarray(probe_a); pb = np.asarray(probe_b)
    pp = np.asarray(probe_pred)
    ca = cnt_a[pa]; cb = cnt_b[pb]               # [n, K]
    self_lab = L[pa, pb]
    S = np.zeros((len(pa), K))
    has_self = self_lab >= 0
    S[np.where(has_self)[0], self_lab[has_self]] = 1.0

    if lambdas is None:
        lambdas = np.geomspace(0.05, 8.0, 25)
    best = {"ll": -np.inf, "lambda": None}
    n1 = ca + cb - 2.0 * S                       # distance-1 counts
    n2 = tot[None, :] - ca - cb + S              # distance-2 counts
    r = np.arange(len(pa))
    for lam in lambdas:
        w1, w2 = np.exp(-lam), np.exp(-2.0 * lam)
        votes = S + w1 * n1 + w2 * n2 + 1e-9
        p = votes[r, pp] / votes.sum(axis=1)
        ll = loglik(p)
        if ll > best["ll"]:
            best = {"ll": ll, "lambda": float(lam)}
    best["bic"] = bic(best["ll"], 1, len(pa))
    return best


# GCM with a per-digit distance summed over both digits - the "structured similarity" exemplar
# model. "circular" uses mod-P wraparound distance (cyclic tasks: add, div), "linear" uses plain
# |x - y| (ordinal tasks: max). Exact on every probe, with no subsampling, so its likelihood and
# BIC sample size match every other model in compare_cognitive_models. Before 2026-09-10 this fit
# subsampled 3000 probes and returned a BIC on n = 3000 next to full-table BICs, which handed GCM
# about 2*(n - 3000)*log K for free. Either metric is a theory-laden similarity assumption, so it
# is reported alongside the hamming variant.
# Input: train_* - the exemplar memory; probe_a, probe_b, probe_pred - probes and predictions;
# K, P - classes and modulus; lambdas - decay grid; metric - "circular" | "linear";
# cache - GCMProbeCache built for exactly these probes, or None
# Output: dict - ll, lambda, bic and n_probe of the best-fitting lambda
def fit_gcm_circular(train_a, train_b, train_c, probe_a, probe_b, probe_pred,
                     K: int, P: int, lambdas=None, metric: str = "circular",
                     cache: GCMProbeCache | None = None) -> dict:
    pa = np.asarray(probe_a, dtype=np.int64); pb = np.asarray(probe_b, dtype=np.int64)
    pp = np.asarray(probe_pred, dtype=np.int64)
    if cache is not None:
        if not cache.matches(pa, pb):
            raise ValueError("gcm cache was built for a different probe set")
        if metric in cache.logp:
            return cache.fit(pp, metric)
    lambdas = np.asarray(DEFAULT_GCM_LAMBDAS if lambdas is None else lambdas, float)
    T = label_table(train_a, train_b, train_c, K, P)
    r = np.arange(len(pp))
    best = {"ll": -np.inf, "lambda": None}
    for lam in lambdas:
        V = gcm_votes(T, P, K, float(lam), metric)[pa, pb, :] + 1e-9
        ll = loglik(V[r, pp] / V.sum(axis=1))
        if ll > best["ll"]:
            best = {"ll": float(ll), "lambda": float(lam)}
    best["bic"] = bic(best["ll"], 1, len(pp))
    best["n_probe"] = int(len(pp))
    return best


# Pure rule with a lapse rate: P(pred = rule) = (1 - eps) + eps/K, and eps/K on every other class.
# Input: probe_pred - predicted labels; probe_rule - rule labels; K - classes; eps_grid - lapse grid
# Output: dict - ll, eps and bic of the best-fitting lapse rate
def fit_rule(probe_pred, probe_rule, K: int, eps_grid=None) -> dict:
    pp, rr = np.asarray(probe_pred), np.asarray(probe_rule)
    n = len(pp); n_hit = int((pp == rr).sum())
    if eps_grid is None:
        eps_grid = np.linspace(1e-4, 0.9999, 400)
    best = {"ll": -np.inf, "eps": None}
    for e in eps_grid:
        ll = n_hit * np.log((1 - e) + e / K) + (n - n_hit) * np.log(e / K)
        if ll > best["ll"]:
            best = {"ll": float(ll), "eps": float(e)}
    best["bic"] = bic(best["ll"], 1, n)
    return best


# Rule plus exception route: the rule everywhere, plus a memorised-exception route that fires with
# probability m on the exception items.
# Input: probe_a, probe_b - probe coordinates; probe_pred, probe_rule - predicted and rule labels;
# exc - ExceptionInfo or None; K - classes; eps_grid, m_grid - grids for the two free parameters
# Output: dict - ll, eps, m and bic of the best-fitting pair
def fit_rulex(probe_a, probe_b, probe_pred, probe_rule, exc: ExceptionInfo | None,
              K: int, eps_grid=None, m_grid=None) -> dict:
    pp = np.asarray(probe_pred); rr = np.asarray(probe_rule)
    pa = np.asarray(probe_a); pb = np.asarray(probe_b)
    n = len(pp)
    is_exc = np.zeros(n, dtype=bool)
    exc_lab = np.zeros(n, dtype=np.int64)
    if exc is not None and exc.n > 0:
        key = {(int(a), int(b)): int(e) for a, b, e in zip(exc.a, exc.b, exc.exc_label)}
        for i in range(n):
            e = key.get((int(pa[i]), int(pb[i])))
            if e is not None:
                is_exc[i] = True
                exc_lab[i] = e
    reg = ~is_exc
    n_reg = int(reg.sum()); n_reg_hit = int((pp[reg] == rr[reg]).sum())
    e_hit = int((pp[is_exc] == exc_lab[is_exc]).sum())
    e_rule = int((pp[is_exc] == rr[is_exc]).sum())
    e_other = int(is_exc.sum()) - e_hit - e_rule

    if eps_grid is None:
        eps_grid = np.linspace(1e-4, 0.9999, 60)
    if m_grid is None:
        m_grid = np.linspace(0.0, 1.0, 51)
    best = {"ll": -np.inf}
    for e in eps_grid:
        ll_reg = (n_reg_hit * np.log((1 - e) + e / K)
                  + (n_reg - n_reg_hit) * np.log(e / K))
        for m in m_grid:
            ll = (ll_reg
                  + e_hit * np.log((1 - e) * m + e / K)
                  + e_rule * np.log((1 - e) * (1 - m) + e / K)
                  + e_other * np.log(e / K))
            if ll > best["ll"]:
                best = {"ll": float(ll), "eps": float(e), "m": float(m)}
    best["bic"] = bic(best["ll"], 2, n)
    return best


# Fits all six models to the network's predictions on ONE common probe set (test items plus
# exception items, see probe_mask) and ranks them by BIC. Because uniform, the three GCM
# variants, rule and rulex are all scored on the same n items, their BICs are comparable.
# Input: preds - dict from full_table_predictions ({a, b, pred, rule}); exc - ExceptionInfo
# or None; P, task - modulus and task name; train_a, train_b, train_c - the (possibly
# corrupted) train arrays that form the exemplar memory; gcm_cache - GCMProbeCache built
# for exactly these probes, or None
# Output: dict - per-model {ll, bic, params...}, best_model by BIC, and the probe counts
# n_probe, n_probe_test, n_probe_exc
def compare_cognitive_models(preds: dict, exc: ExceptionInfo | None,
                             P: int, task: str,
                             train_a, train_b, train_c,
                             gcm_cache: GCMProbeCache | None = None) -> dict:
    K = n_classes(task, P)
    a, b = np.asarray(preds["a"], dtype=np.int64), np.asarray(preds["b"], dtype=np.int64)
    pred, rule = np.asarray(preds["pred"], dtype=np.int64), np.asarray(preds["rule"], dtype=np.int64)
    mask = probe_mask(a, b, train_a, train_b, exc)
    a, b, pred, rule = a[mask], b[mask], pred[mask], rule[mask]
    n = len(pred)

    out = {"uniform": {"ll": n * np.log(1.0 / K),
                       "bic": bic(n * np.log(1.0 / K), 0, n)}}
    out["gcm_hamming"] = fit_gcm_hamming(train_a, train_b, train_c, a, b, pred, K)
    out["gcm_circular"] = fit_gcm_circular(train_a, train_b, train_c, a, b, pred, K, P,
                                           metric="circular", cache=gcm_cache)
    out["gcm_linear"] = fit_gcm_circular(train_a, train_b, train_c, a, b, pred, K, P,
                                         metric="linear", cache=gcm_cache)
    out["rule"] = fit_rule(pred, rule, K)
    out["rulex"] = fit_rulex(a, b, pred, rule, exc, K)
    models = [k for k in out if isinstance(out[k], dict)]
    out["best_model"] = min(models, key=lambda k: out[k]["bic"])

    ta = np.asarray(train_a, dtype=np.int64); tb = np.asarray(train_b, dtype=np.int64)
    A = int(max(a.max(), ta.max())) + 1 if n else P
    B = int(max(b.max(), tb.max())) + 1 if n else P
    is_train = np.zeros((A, B), dtype=bool); is_train[ta, tb] = True
    n_exc = 0
    if exc is not None and exc.n > 0:
        is_exc = np.zeros((A, B), dtype=bool)
        is_exc[np.asarray(exc.a), np.asarray(exc.b)] = True
        n_exc = int(is_exc[a, b].sum())
    out["probe"] = "test+exc"
    out["n_probe"] = int(n)
    out["n_probe_test"] = int((~is_train[a, b]).sum())
    out["n_probe_exc"] = n_exc
    return out
