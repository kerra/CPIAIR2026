from __future__ import annotations

import functools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend selection)

ARCH_COLOR = {"transformer": "#1f77b4", "mamba": "#d62728", "kimi": "#2ca02c"}
TASK_COLOR = {"add": "#4c72b0", "div": "#dd8452", "max": "#55a868"}
WD_MARKER = {0.1: "o", 0.3: "s"}
FAMILY = {"uniform": "uniform", "gcm_hamming": "exemplar", "gcm_circular": "exemplar", "gcm_linear": "exemplar",
          "rule": "rule family", "rulex": "rule family"}
FAMILY_COLOR = {"rule family": "#e6550d", "exemplar": "#3182bd", "uniform": "#9e9e9e"}
PAPER_RC = {"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25}


# Decorator that runs a figure function inside plt.rc_context, so an rc set applies for the duration
# of one call and nothing mutates the global rcParams. Figures are saved inside the wrapped call.
# Input: rc - the rcParams to apply
# Output: a decorator wrapping a figure function
def styled(rc: dict):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with plt.rc_context(rc):
                return fn(*args, **kwargs)
        return wrapper
    return deco
