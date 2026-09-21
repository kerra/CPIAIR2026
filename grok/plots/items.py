from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .style import PAPER_RC, plt, styled


# The H3b figure: pairwise scatters of the seed-mean item grok times across architectures, plus a
# P x P map of the item time per architecture.
# Input: means - seed-mean item times indexed by item key; P - modulus; task, rho, wd - the
# condition; path - Path of the figure file
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_item_order(means: pd.DataFrame, P: int, task: str, rho: float, wd: float, path: Path) -> None:
    key = means.index.values
    a, b = key // P, key % P
    # figure: pairwise scatter of seed-mean item times + P x P map of item time per arch
    archs = list(means.columns)
    fig, axes = plt.subplots(2, max(3, len(archs)), figsize=(4.2 * max(3, len(archs)), 8))
    pairs = list(combinations(archs, 2))
    for j in range(max(3, len(archs))):
        ax = axes[0, j]
        if j < len(pairs):
            a1, a2 = pairs[j]
            ax.scatter(means[a1], means[a2], s=4, alpha=0.3, color="k")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(f"{a1}: item grok step (seed mean)"); ax.set_ylabel(a2)
            r = sps.spearmanr(means[a1], means[a2], nan_policy="omit")[0]; ax.set_title(f"Spearman {r:+.2f}", fontsize=10)
        else:
            ax.axis("off")
    for j, arch in enumerate(archs):
        ax = axes[1, j]
        M = np.full((P, P), np.nan); M[a, b] = np.log10(means[arch].values)
        im = ax.imshow(M, cmap="viridis", origin="lower"); ax.set_title(f"{arch}: log10 item grok step (a x b)", fontsize=10)
        ax.set_xlabel("b"); ax.set_ylabel("a"); ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(len(archs), max(3, len(archs))):
        axes[1, j].axis("off")
    fig.suptitle(f"Per-item learning order across architectures ({task}, rho {rho:g}, wd {wd:g})")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path); plt.close(fig)
