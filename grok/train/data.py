from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from ..cognitive.exceptions import corrupt_train
from ..core.tasks import P, TASK_OP
from ..models.common import device
from .config import SPLIT_SEED


# Builds the complete table of one task as torch tensors and, optionally, splits it in half
# into train and test. The permutation comes from a device-seeded randperm, and CUDA and CPU
# disagree there, so a run's stored data_split.npz is what is authoritative afterwards.
# Input: task - "div" | "add" | "max"; split_seed - int seed of the permutation;
# split - False returns the whole table instead of a split
# Output: (a, b, c) long tensors when split is False, otherwise a (train, test) pair of them
def make_task_data(task: str, split_seed: int = SPLIT_SEED, split: bool = True):
    aa, bb, cc = [], [], []
    for a in range(P):
        for b in range(P):
            if task == "div" and b == 0:
                continue
            if task == "div":
                c = (a * pow(b, P - 2, P)) % P
            elif task == "add":
                c = (a + b) % P
            elif task == "max":
                c = max(a, b)
            else:
                raise ValueError(task)
            aa.append(a)
            bb.append(b)
            cc.append(c)

    aa = torch.tensor(aa, device=device, dtype=torch.long)
    bb = torch.tensor(bb, device=device, dtype=torch.long)
    cc = torch.tensor(cc, device=device, dtype=torch.long)

    if not split:
        return aa, bb, cc

    g = torch.Generator(device=device)
    g.manual_seed(split_seed)
    perm = torch.randperm(len(aa), device=device, generator=g)
    nt = len(aa) // 2
    train = (aa[perm[:nt]], bb[perm[:nt]], cc[perm[:nt]])
    test = (aa[perm[nt:]], bb[perm[nt:]], cc[perm[nt:]])
    return train, test


# Accuracy of a model on one split of one task, evaluated in batches.
# Input: model - nn.Module; data - {task: (train tuple, test tuple)}; task - task name;
# split - "train" or "test"; batch_size - items per forward pass
# Output: float - the fraction of correct predictions
@torch.no_grad()
def eval_task(model, data: dict, task: str, split: str = "test", batch_size: int = 4096) -> float:
    model.eval()
    a, b, c = data[task][1 if split == "test" else 0]
    correct = 0
    total = 0
    for start in range(0, len(a), batch_size):
        logits = model(a[start:start + batch_size], TASK_OP[task], b[start:start + batch_size])
        pred = logits.argmax(dim=-1)
        correct += int((pred == c[start:start + batch_size]).sum().item())
        total += len(pred)
    return correct / total


# The pristine (uncorrupted) train/test split of a task, computed once per process and cached.
# Input: task - task name; split_seed - int seed of the permutation
# Output: tuple (train, test), each an (a, b, c) triple of long tensors
@lru_cache(maxsize=None)
def clean_split(task: str, split_seed: int = SPLIT_SEED):
    return make_task_data(task, split_seed=split_seed, split=True)


# Torch wrapper over the irregulars machinery: corrupts a fraction of the already-split train
# tensors. It reuses the original split, so a run stays comparable to its clean condition.
# Input: train_tuple - (a, b, c) long tensors; task - task name; P - modulus;
# exception_frac - fraction to corrupt; exception_seed - int RNG seed
# Output: tuple (new train tuple, ExceptionInfo or None when the fraction is 0)
def add_exceptions_to_train(train_tuple, task: str, P: int, exception_frac: float, exception_seed: int = 0):
    if exception_frac <= 0:
        return train_tuple, None
    a_t, b_t, c_t = train_tuple
    as_np = lambda t: t.detach().cpu().numpy().astype(np.int64)  # noqa: E731
    new_c, info = corrupt_train(as_np(a_t), as_np(b_t), as_np(c_t), task, P, exception_frac, exception_seed)
    return (a_t, b_t, torch.as_tensor(new_c, dtype=c_t.dtype, device=c_t.device)), info


# The mini-batch sampler the matrix was trained with: uniform without replacement (randperm) on a
# clean train set, and uniform WITH replacement (equal-weight multinomial) once the train set
# carries exceptions. Every item is sampled uniformly either way, but the two draw from the
# generator differently, so the distinction is kept for exact reproducibility of the stored runs.
# Input: has_exceptions - bool; n_train - train set size; batch_size - items per step;
# device - torch device; generator - torch.Generator
# Output: long tensor [batch_size] of train indices
def batch_indices(has_exceptions: bool, n_train: int, batch_size: int, device, generator):
    if not has_exceptions:
        return torch.randperm(n_train, device=device, generator=generator)[:batch_size]
    w = torch.ones(n_train, dtype=torch.float32, device=device)
    return torch.multinomial(w, batch_size, replacement=True, generator=generator)
