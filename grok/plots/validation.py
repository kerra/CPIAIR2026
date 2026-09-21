from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.tasks import ARCHS
from .style import ARCH_COLOR, PAPER_RC, plt, styled

TAG_ORDER = ["0.5g", "1g", "2g", "final"]
TAG_LABEL = {"0.5g": "0.5 x grok", "1g": "1 x grok", "2g": "2 x grok", "final": "final"}


# The one figure of the model validation: the ATRIUM-lite rule weight pi per architecture at the
# four validation checkpoints (0.5, 1 and 2 x grok, and the end), which shows the exemplar-to-rule
# sequence as a continuous gate. Median as a line, IQR as a band, every run as a point.
# Input: df - the model_validation table; path - Path of the figure file
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_atrium_gate(df: pd.DataFrame, path: Path) -> None:
    at = df[df.model == "atrium_lite"].copy()
    tags = [t for t in TAG_ORDER if t in set(at.tag)]
    xs = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, arch in enumerate([a for a in ARCHS if (at.arch == a).any()]):
        g = at[at.arch == arch]
        q = g.groupby("tag").pi.quantile([0.25, 0.5, 0.75]).unstack().reindex(tags)
        jit = (k - 1) * 0.12
        ax.plot(xs + jit, q[0.5].values, marker="o", ms=5, lw=1.8, color=ARCH_COLOR[arch], label=arch)
        ax.fill_between(xs + jit, q[0.25].values, q[0.75].values, color=ARCH_COLOR[arch], alpha=0.15, lw=0)
        for i, t in enumerate(tags):
            v = g[g.tag == t].pi.values
            ax.scatter(np.full(len(v), xs[i] + jit) + np.random.default_rng(k).normal(0, 0.02, len(v)), v,
                       s=9, color=ARCH_COLOR[arch], alpha=0.45, lw=0)
    ax.set_xticks(xs); ax.set_xticklabels([TAG_LABEL[t] for t in tags])
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel("ATRIUM-lite rule weight $\\pi$")
    ax.set_xlabel("checkpoint (grok units)")
    ax.set_title("Rule weight of the rule + exemplar mixture along training (single-task add)", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
