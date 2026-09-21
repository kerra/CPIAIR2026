from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from ..analysis.manifold_summary import METRIC_LABEL
from .style import PAPER_RC, plt, styled


# The one manifold figure of the paper: per architecture, a tasks x locations heatmap of the task
# metric at the final checkpoint, averaged over seeds - where each algorithm lives along the
# forward pass. NaN cells are locations that are constant across items.
# Input: grids - {arch: (tasks, locations, [tasks x locations] values)}; path - figure file;
# title - figure title; cell - size of one heatmap cell in inches
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_manifold_heatmaps(grids: dict, path: Path, title: str, cell: float = 0.85) -> None:
    archs = list(grids)
    n_cols = max(len(grids[a][1]) for a in archs)
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#e8e8e8")
    fig, axes = plt.subplots(len(archs), 1, figsize=(cell * n_cols + 2.8, len(archs) * (cell * 3 + 1.35) + 0.6),
                             squeeze=False, constrained_layout=True)
    for ax, arch in zip(axes[:, 0], archs):
        tasks, locs, grid = grids[arch]
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v < 0.6 else "black")
                else:
                    ax.text(j, i, "const", ha="center", va="center", fontsize=6.5, color="#777777")
        ax.set_xticks(range(len(locs)))
        ax.set_xticklabels(locs, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([f"{t} ({METRIC_LABEL[t]})" for t in tasks], fontsize=8)
        ax.set_title(arch, fontsize=10, loc="left")
        ax.grid(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.5, pad=0.02, label="structure (0–1); grey = constant across items")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
