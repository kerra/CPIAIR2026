from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.tasks import n_classes, rule_label


# Bookkeeping for the corrupted train items; all arrays are aligned with each other.
@dataclass(frozen=True)
class ExceptionInfo:
    idx: np.ndarray          # positions inside the train tensors
    a: np.ndarray
    b: np.ndarray
    exc_label: np.ndarray    # trained (irregular) label
    rule_label: np.ndarray   # what the rule would say
    task: str = "add"

    @property
    def n(self) -> int:
        return len(self.idx)


# Corrupts a fraction of an already-split train set: round(frac * n_train) items are drawn
# uniformly without replacement and each is relabelled with a deranged label, uniform over the
# K - 1 non-rule classes. The RNG call order (choice then integers) is the one the matrix was
# generated with, so regenerated exceptions are asserted equal to the stored ones.
# Input: a, b, c - train arrays; task - task name; P - modulus; exception_frac - fraction to
# corrupt; exception_seed - int RNG seed
# Output: tuple (np.ndarray[int64] new labels, ExceptionInfo or None when frac <= 0)
def corrupt_train(a, b, c, task: str, P: int, exception_frac: float,
                  exception_seed: int = 0) -> tuple[np.ndarray, ExceptionInfo | None]:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64).copy()
    if exception_frac <= 0:
        return c, None
    n_train = len(a)
    K = n_classes(task, P)
    rng = np.random.default_rng(exception_seed)

    n_exc = max(1, min(int(round(exception_frac * n_train)), n_train))
    idx = rng.choice(n_train, size=n_exc, replace=False)
    rl = rule_label(task, a[idx], b[idx], P)
    exc = (rl + rng.integers(1, K, size=n_exc)) % K       # uniform over the K-1 non-rule classes
    c[idx] = exc
    info = ExceptionInfo(idx=idx, a=a[idx], b=b[idx], exc_label=exc, rule_label=rl, task=task)
    return c, info


# Fit-free behavioural readouts on the exception items: accuracy, overregularisation ("goed"),
# other errors and the exception-minus-rule logit gap.
# Input: logits_exc - [n_exc, K] logits on the exception items; exc - ExceptionInfo
# Output: dict - acc_exc, overreg_rate, other_err_rate, logit_gap_exc_minus_rule
def exception_metrics(logits_exc: np.ndarray, exc: ExceptionInfo) -> dict:
    pred = logits_exc.argmax(axis=1)
    hit_exc = pred == exc.exc_label
    hit_rule = pred == exc.rule_label
    r = np.arange(exc.n)
    gap = logits_exc[r, exc.exc_label] - logits_exc[r, exc.rule_label]
    return {
        "acc_exc": float(hit_exc.mean()),
        "overreg_rate": float(hit_rule.mean()),
        "other_err_rate": float(1.0 - hit_exc.mean() - hit_rule.mean()),
        "logit_gap_exc_minus_rule": float(gap.mean()),
    }
