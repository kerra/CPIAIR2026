from __future__ import annotations

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_TARGET = 2.0e6
PARAM_TOL = 0.10


# Root-mean-square layer norm, identical in all three architectures.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x_float * rms * self.weight.float()
        return out.to(dtype)


# ======================================================================
# matched_2M builders + shared interface
# ======================================================================

# Counts the trainable parameters of a model.
# Input: model - nn.Module
# Output: int - number of parameters with requires_grad
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Asserts that an architecture really sits inside the matched_2M budget of 2.0M +- 10 %.
# Input: model - nn.Module; arch - architecture name, used only in the error message
# Output: int - the parameter count; raises AssertionError when it is out of band
def assert_param_budget(model: nn.Module, arch: str) -> int:
    n = count_params(model)
    lo = PARAM_TARGET * (1 - PARAM_TOL)
    hi = PARAM_TARGET * (1 + PARAM_TOL)
    assert lo <= n <= hi, (
        f"{arch} matched_2M is {n / 1e6:.3f}M params — outside "
        f"{PARAM_TARGET / 1e6:.1f}M ± {PARAM_TOL:.0%} [{lo / 1e6:.2f}M, {hi / 1e6:.2f}M]")
    return n
