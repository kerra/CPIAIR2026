from __future__ import annotations

import numpy as np

from .tasks import P, dlog_order

N_MODES = (P - 1) // 2          # 74 real Fourier modes for odd P (f and P-f are one mode)


# Share of spectral energy carried by the top_k Fourier modes of the class means.
# min_mode = 3 ignores the smooth f = 1, 2 structure any ordinal/max solution imprints.
# Input: rows - [n_classes, d] centred class means; top_k - modes counted; min_mode - lowest mode kept
# Output: tuple (float concentration in [0, 1], list[int] dominant mode indices)
def fourier_concentration(rows: np.ndarray, top_k: int = 4, min_mode: int = 1):
    fft = np.fft.rfft(rows, axis=0)
    energy = (np.abs(fft) ** 2).sum(1)
    energy[:min(min_mode, len(energy))] = 0.0
    if energy.sum() <= 0:
        return np.nan, -1
    idx = np.argsort(energy)[::-1][:top_k]
    return float(energy[idx].sum() / (energy.sum() + 1e-12)), [int(i) for i in idx]


# Pairwise Euclidean distances between class means, as a flat upper-triangle vector.
# Input: rows - [n_classes, d] centred class means
# Output: tuple (distances, class indices, triu index pair, n_classes)
def _pairwise(rows: np.ndarray):
    n = rows.shape[0]
    Dm = np.sqrt(((rows[:, None, :] - rows[None, :, :]) ** 2).sum(-1))
    idx = np.arange(n)
    tri = np.triu_indices(n, k=1)
    return Dm[tri], idx, tri, n


# How linear the manifold is: correlation of manifold distances with |i - j| in class order.
# Input: rows - [n_classes, d] centred class means
# Output: float - Pearson r, NaN when the distances are degenerate
def linear_corr(rows: np.ndarray) -> float:
    a, idx, tri, n = _pairwise(rows)
    b = np.abs(idx[:, None] - idx[None, :])[tri].astype(float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 else np.nan


# True when the class means have collapsed to numerically zero.
# Input: rows - [n_classes, d] class means; tol - float threshold on the largest absolute value
# Output: bool
def is_degenerate(rows: np.ndarray, tol: float = 1e-6) -> bool:
    return float(np.abs(rows).max()) < tol


# Among the top-energy modes, the one whose ideal circle best matches the manifold's pairwise
# distances. Stays high for a clean ring at any frequency.
# Input: rows - [n_classes, d] centred class means; top_k_modes - how many modes to try
# Output: tuple (int mode, float corr), or (-1, NaN) when no mode fits
def best_mode_corr(rows: np.ndarray, top_k_modes: int = 8):
    a, idx, tri, n = _pairwise(rows)
    if a.std() < 1e-12 or n < 3:
        return -1, np.nan
    energy = (np.abs(np.fft.rfft(rows, axis=0)) ** 2).sum(1); energy[0] = 0.0
    cands = [int(m) for m in np.argsort(energy)[::-1] if m > 0][:top_k_modes]
    delta = np.abs(idx[:, None] - idx[None, :])[tri]
    best = (-1, -np.inf)
    for m in cands:
        b = np.sqrt(2.0 - 2.0 * np.cos(2.0 * np.pi * m * delta / n))
        if b.std() < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best[1]:
            best = (m, r)
    return best if np.isfinite(best[1]) else (-1, np.nan)


# best_mode_corr with the None sentinel that the per-run trajectories expect.
# Input: rows - [n_classes, d] centred class means; top_k_modes - how many modes to try
# Output: tuple (int mode or None, float corr)
def best_fourier_distance_corr(rows: np.ndarray, top_k_modes: int = 8):
    mode, corr = best_mode_corr(rows, top_k_modes)
    return (None, np.nan) if mode < 0 else (int(mode), float(corr))


# ---- spectra of a [P, d] matrix in the natural or the dlog basis (lesions) ----

# Which rows of a [P, d] matrix enter the transform, in transform order.
# Input: basis - "natural" (token values) or "dlog" (multiplicative exponent)
# Output: np.ndarray[int64] - row indices
def _rows_for_basis(basis: str) -> np.ndarray:
    return np.arange(P) if basis == "natural" else dlog_order()


# Energy per real Fourier mode of an embedding matrix, plus its DC energy. natural: L = P = 149
# over token values; dlog: L = P-1 = 148 over the exponent, where bin 74 is the Legendre character.
# Input: E - [P, d] embedding matrix; basis - "natural" or "dlog"
# Output: tuple (np.ndarray energy of modes 1..N_MODES, float DC energy)
def mode_energy(E: np.ndarray, basis: str = "natural") -> tuple[np.ndarray, float]:
    rows = _rows_for_basis(basis)
    L = len(rows)
    F = np.fft.fft(E[rows].astype(np.float64), axis=0)         # [L, d]
    pw = (np.abs(F) ** 2).sum(axis=1)                          # [L]
    en = np.array([pw[f] + (pw[L - f] if L - f != f else 0.0) for f in range(1, N_MODES + 1)])
    return en, float(pw[0])


# Rescales each Fourier bin of an embedding matrix by scale[f] - the lesion operator. DC is left
# alone, and so are rows outside the basis (the value 0 in the dlog basis).
# Input: E - [P, d] embedding matrix; scale - array indexed 1..N_MODES; basis - "natural" or "dlog"
# Output: np.ndarray[float32] [P, d] - the rescaled matrix
def apply_scale(E: np.ndarray, scale: np.ndarray, basis: str = "natural") -> np.ndarray:
    rows = _rows_for_basis(basis)
    L = len(rows)
    F = np.fft.fft(E[rows].astype(np.float64), axis=0)
    full = np.ones(L)
    for f in range(1, N_MODES + 1):
        full[f] = scale[f]
        full[L - f] = scale[f]
    out = E.astype(np.float32).copy()
    out[rows] = np.real(np.fft.ifft(F * full[:, None], axis=0)).astype(np.float32)
    return out


# The k most energetic modes.
# Input: en - per-mode energy array; k - how many modes to take
# Output: np.ndarray[int] - mode indices 1..N_MODES
def top_modes(en: np.ndarray, k: int) -> np.ndarray:
    return (np.argsort(-en)[:k] + 1)          # mode indices 1..N_MODES


# How many of the strongest modes are needed to reach a given share of the total energy.
# Input: en - per-mode energy array; share - target fraction in [0, 1]
# Output: int - number of modes
def k_for_energy_share(en: np.ndarray, share: float) -> int:
    cs = np.cumsum(np.sort(en)[::-1]) / en.sum()
    return int(np.searchsorted(cs, share) + 1)


# Modes whose cumulative energy share first reaches q, taken from the most energetic
# ("top"), the least energetic ("bottom") or in a random order ("rand").
# Input: en - per-mode energy; q - target share; order - "top" | "bottom" | "rand"; rng - Generator, for "rand"
# Output: np.ndarray[int] - mode indices 1..N_MODES
def modes_by_energy_fraction(en: np.ndarray, q: float, order: str, rng=None) -> np.ndarray:
    if q <= 0:
        return np.array([], dtype=int)
    if order == "top":
        idx = np.argsort(-en)
    elif order == "bottom":
        idx = np.argsort(en)
    else:
        idx = rng.permutation(len(en))
    cs = np.cumsum(en[idx]) / en.sum()
    n = int(np.searchsorted(cs, q - 1e-9) + 1)
    return idx[:n] + 1
