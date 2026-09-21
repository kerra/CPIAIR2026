from __future__ import annotations

import numpy as np


# Log-likelihood of a set of predictions, clipped away from log(0).
# Input: prob_of_pred - array of the probability the model gave to each predicted label
# Output: float - the summed log-likelihood
def loglik(prob_of_pred: np.ndarray) -> float:
    return float(np.log(np.clip(prob_of_pred, 1e-12, None)).sum())


# Bayesian information criterion of a fit; lower is better.
# Input: ll - float log-likelihood; k_params - int free parameters; n - int number of items
# Output: float - the BIC score
def bic(ll: float, k_params: int, n: int) -> float:
    return k_params * np.log(n) - 2.0 * ll


# Similarity between two token values, decaying exponentially with the distance between them.
# Input: P - int number of token values; lam - float decay rate; metric - "circular" (mod-P
# wraparound) or "linear" (plain |x - x'|)
# Output: np.ndarray [P, P] - the kernel K[x, x']
def digit_kernel(P: int, lam: float, metric: str) -> np.ndarray:
    x = np.arange(P, dtype=float)
    d = np.abs(x[:, None] - x[None, :])
    if metric == "circular":
        d = np.minimum(d, P - d)
    elif metric != "linear":
        raise ValueError(metric)
    return np.exp(-lam * d / (P / 2.0))


# Counts how many train items sit in each (a, b) cell carrying each label.
# Input: train_a, train_b, train_c - int arrays of the train items; K - classes; P - modulus
# Output: np.ndarray [P, P, K] - the counts T[a, b, c]
def label_table(train_a, train_b, train_c, K: int, P: int) -> np.ndarray:
    T = np.zeros((P, P, K))
    np.add.at(T, (np.asarray(train_a, dtype=np.int64), np.asarray(train_b, dtype=np.int64),
                  np.asarray(train_c, dtype=np.int64)), 1.0)
    return T


# Exact GCM vote table over the whole (a, b) grid, with no probe subsampling. The kernel is
# separable in the two digits, so the double sum over train items is just two matrix products.
# Input: T - [P, P, K] label table; P, K - grid size and classes; lam - decay; metric - "circular" | "linear"
# Output: np.ndarray [P, P, K] - votes[a, b, c], summed over the train items
def gcm_votes(T: np.ndarray, P: int, K: int, lam: float, metric: str) -> np.ndarray:
    Kx = digit_kernel(P, lam, metric)
    # exp(-lam*d) lies in [exp(-80), 1] on the default grid, so over/underflow
    # is impossible here; macOS Accelerate BLAS nevertheless raises spurious
    # FP-exception flags inside matmul, which numpy surfaces as
    # RuntimeWarnings. Silence them and assert finiteness instead.
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        M = Kx @ T.reshape(P, P * K)                                  # sum over a' >> [a, (b', c)]
        M = M.reshape(P, P, K).transpose(1, 0, 2).reshape(P, P * K)   # >> [b', (a, c)]
        V = (Kx @ M).reshape(P, P, K).transpose(1, 0, 2)              # sum over b' >> [a, b, c]
    assert np.all(np.isfinite(V)), "non-finite GCM votes (real numerical problem)"
    return V


DEFAULT_GCM_LAMBDAS = np.geomspace(0.2, 40.0, 25)


# Exact GCM class log-probabilities at a FIXED probe set, for a lambda grid and both digit
# metrics. Depends only on the (corrupted) train set and the probes, not on the network, so it is
# built once per run/task and every checkpoint is then fitted by a gather. Memory: n_lambda x
# n_probe x K float32 per metric, about 170 MB for 11k probes, K = 149 and 25 lambdas.
class GCMProbeCache:

    def __init__(self, train_a, train_b, train_c, probe_a, probe_b, K: int, P: int,
                 lambdas=None, metrics=("circular", "linear")):
        self.pa = np.asarray(probe_a, dtype=np.int64)
        self.pb = np.asarray(probe_b, dtype=np.int64)
        self.K, self.P = int(K), int(P)
        self.lambdas = np.asarray(DEFAULT_GCM_LAMBDAS if lambdas is None else lambdas, float)
        T = label_table(train_a, train_b, train_c, self.K, self.P)
        self.logp = {}
        for metric in metrics:
            L = np.empty((len(self.lambdas), len(self.pa), self.K), dtype=np.float32)
            for i, lam in enumerate(self.lambdas):
                V = gcm_votes(T, self.P, self.K, float(lam), metric)[self.pa, self.pb, :] + 1e-9
                L[i] = np.log(V / V.sum(axis=1, keepdims=True))
            self.logp[metric] = L

    # Checks that a probe set is the one this cache was built for.
    # Input: probe_a, probe_b - int arrays of probe coordinates
    # Output: bool - True when they are identical to the cached probes
    def matches(self, probe_a, probe_b) -> bool:
        pa = np.asarray(probe_a); pb = np.asarray(probe_b)
        return (len(pa) == len(self.pa) and np.array_equal(pa, self.pa)
                and np.array_equal(pb, self.pb))

    # Picks the lambda that best explains one checkpoint's predictions, by maximum log-likelihood.
    # Input: probe_pred - int array of predicted labels at the cached probes; metric - "circular" | "linear"
    # Output: dict - ll, lambda, bic and n_probe of the best-fitting lambda
    def fit(self, probe_pred, metric: str) -> dict:
        pp = np.asarray(probe_pred, dtype=np.int64)
        assert len(pp) == len(self.pa), "prediction vector does not match the cached probes"
        L = self.logp[metric]
        lls = L[:, np.arange(len(pp)), pp].sum(axis=1, dtype=np.float64)   # [n_lambda]
        i = int(np.argmax(lls))
        return {"ll": float(lls[i]), "lambda": float(self.lambdas[i]),
                "bic": bic(float(lls[i]), 1, len(pp)), "n_probe": int(len(pp))}


# Selects the probe items on which rule, exemplar and RULEX can actually disagree: everything
# outside the train set, plus the exception items. Memorised regular train items carry no
# information - there the GCM matches itself at distance 0 and the rule label IS the train label.
# Input: a, b - full-table coordinates; train_a, train_b - train coordinates; exc - exception record or None
# Output: np.ndarray[bool] over the full table - True where the item is a probe
def probe_mask(a, b, train_a, train_b, exc) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    ta = np.asarray(train_a, dtype=np.int64); tb = np.asarray(train_b, dtype=np.int64)
    A = int(max(a.max(), ta.max())) + 1
    B = int(max(b.max(), tb.max())) + 1
    is_train = np.zeros((A, B), dtype=bool)
    is_train[ta, tb] = True
    mask = ~is_train[a, b]
    if exc is not None and exc.n > 0:
        is_exc = np.zeros((A, B), dtype=bool)
        is_exc[np.asarray(exc.a), np.asarray(exc.b)] = True
        mask |= is_exc[a, b]
    return mask
