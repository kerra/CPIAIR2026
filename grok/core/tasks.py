from __future__ import annotations

import numpy as np

P = 149
OP_DIV, OP_ADD, OP_MAX, OP_PARITY, EQ = P, P + 1, P + 2, P + 3, P + 4   # OP_PARITY: reserved token, no task
VOCAB = P + 5
TASK_OP = {"div": OP_DIV, "add": OP_ADD, "max": OP_MAX}
TASKS = ("add", "div", "max")
ARCHS = ("transformer", "mamba", "kimi")
DLOG_GENERATOR = 2


# ----------------------------------------------------------------------
# Task rules (must mirror make_task_data in the training scripts)
# ----------------------------------------------------------------------

# Ground-truth label of the task rule for every (a, b) pair.
# Input: task - "div" | "add" | "max"; a, b - int arrays of token values; P - prime modulus
# Output: np.ndarray[int64] - the correct class of each pair
def rule_label(task: str, a: np.ndarray, b: np.ndarray, P: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if task == "div":
        inv = np.array([pow(int(x), P - 2, P) if x != 0 else 0 for x in range(P)],
                       dtype=np.int64)
        return (a * inv[b]) % P
    if task == "add":
        return (a + b) % P
    if task == "max":
        return np.maximum(a, b)
    raise ValueError(task)


def n_classes(task: str, P: int) -> int:
    if task not in ("add", "div", "max"):
        raise ValueError(task)
    return P



# Every (a, b) pair of a task together with its rule label, in meshgrid "ij" order.
# Input: task - "div" | "add" | "max"; for div the b = 0 column is dropped
# Output: tuple of three np.ndarray[int64] - a, b and the label of each pair
def full_table(task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aa, bb = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    if task == "div":
        keep = bb != 0
        aa, bb = aa[keep], bb[keep]
    return aa.astype(np.int64), bb.astype(np.int64), rule_label(task, aa, bb, P)


# The multiplicative group of Z_P in discrete-log order, order[k] = g^k mod P. A Fourier
# transform over k then reads out multiplicative characters; 0 has no dlog and is excluded.
# Input: generator - int, a primitive root modulo P (asserted)
# Output: np.ndarray[int64] of length P-1 - the class order used for div
def dlog_order(generator: int = DLOG_GENERATOR) -> np.ndarray:
    out, x = [], 1
    for _ in range(P - 1):
        out.append(x)
        x = (x * generator) % P
    assert len(set(out)) == P - 1, f"{generator} is not a primitive root modulo {P}"
    return np.asarray(out, dtype=np.int64)


# Privileged class order of a task: natural for add/max, discrete-log for div.
# Input: task - "div" | "add" | "max"
# Output: np.ndarray[int64] - class indices in transform order
def row_order(task: str) -> np.ndarray:
    return dlog_order() if task == "div" else np.arange(P)


# Re-orders class means into the task's own basis and centres them.
# Input: M - [n_classes, d] matrix of class means; task - "div" | "add" | "max"
# Output: np.ndarray[float64] - the centred rows in basis order
def ordered_rows(M: np.ndarray, task: str) -> np.ndarray:
    rows = M[row_order(task)].astype(np.float64)
    return rows - rows.mean(axis=0, keepdims=True)
