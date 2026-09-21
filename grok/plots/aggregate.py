from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.stats import order_vs_overreg_stats
from ..analysis.table import single_add_runs
from ..core.tasks import ARCHS
from .style import ARCH_COLOR, FAMILY, FAMILY_COLOR, PAPER_RC, TASK_COLOR, WD_MARKER, plt, styled


# figures

# Draws the U-curve of one condition: test accuracy, exception accuracy and overregularisation over
# training, one line per seed and architecture.
# Input: loaded - {(condition, seed): offline dict}; tab - the runs table; task, rho, wd - the
# condition; out - Path of the output directory; relative - put steps on a grok-relative axis
# Output: None - saves the figure
def plot_ucurve(loaded, tab, task, rho, wd, out: Path, relative: bool):
    rows = tab[(~tab.joint) & (tab.task == task) & (tab.rho == rho) & (tab.wd == wd)]
    if rows.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
    n_ok, n_all = {}, {}
    for _, r in rows.iterrows():
        R = loaded[(r.condition, r.seed)]
        m = R["metrics"]
        n_all[r.arch] = n_all.get(r.arch, 0) + 1
        g = r.grok_step
        if relative:
            if g is None or not np.isfinite(g) or g <= 0:
                continue
            x = m["step"].values / float(g)
        else:
            x = m["step"].values
        n_ok[r.arch] = n_ok.get(r.arch, 0) + 1
        col = ARCH_COLOR[r.arch]
        ls = "-" if not r.non_grokking else "--"
        axes[0].plot(x, m[f"test_{task}"], color=col, alpha=0.75, lw=1.3, ls=ls)
        if "acc_exc" in m:
            axes[1].plot(x, m["acc_exc"], color=col, alpha=0.75, lw=1.3, ls=ls)
            axes[2].plot(x, m["overreg_rate"], color=col, alpha=0.75, lw=1.3, ls=ls)
    for arch in ARCHS:
        if arch in n_all:
            lab = f"{arch} ({n_ok.get(arch, 0)}/{n_all[arch]} seeds"
            lab += " grokked)" if relative else ", dashed = hit cap)"
            axes[0].plot([], [], color=ARCH_COLOR[arch], label=lab)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel(f"test acc ({task})"); axes[1].set_ylabel("acc_exc")
    axes[2].set_ylabel("overreg_rate")
    for ax in axes:
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    if relative:
        for ax in axes:
            ax.axvline(1.0, color="k", lw=0.8, ls=":")
        axes[2].set_xlabel("step / grok_step  (1 = behavioral grok)")
        tag = "rel"
    else:
        axes[2].set_xlabel("step"); tag = "abs"
    fig.suptitle(f"{task}: rho={rho:g}, WD={wd:g} — exception memory through the grok", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"ucurve_{tag}_{task}_rho{rho:g}_wd{wd:g}.png"); plt.close(fig)


# The one representative U-curve of the paper: add, rho 0.02, WD 0.1, on grok-relative steps.
# Input: loaded - {(condition, seed): offline dict}; tab - the runs table; out - output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_ucurve(loaded, tab, out: Path):
    plot_ucurve(loaded, tab, "add", 0.02, 0.1, out, relative=True)


# How much the exceptions delay the grok: grok step against rho on the single-task add runs.
# Input: tab - the runs table; out - Path of the output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_grok_delay(tab: pd.DataFrame, out: Path):
    d = tab[(~tab.joint) & (tab.task == "add")].copy()
    if d.empty:
        return
    rhos = sorted(d.rho.unique())
    xpos = {r: i for i, r in enumerate(rhos)}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for arch in ARCHS:
        for wd in sorted(d.wd.unique()):
            c = d[(d.arch == arch) & (d.wd == wd)]
            if c.empty:
                continue
            jit = (0.06 if wd == 0.3 else -0.06) + (ARCHS.index(arch) - 1) * 0.02
            means = []
            for rho in rhos:
                cc = c[c.rho == rho]
                ok = cc[~cc.non_grokking & cc.grok_step.notna()]
                ng = cc[cc.non_grokking | cc.grok_step.isna()]
                x = xpos[rho] + jit
                if len(ok):
                    ax.scatter([x] * len(ok), ok.grok_step, color=ARCH_COLOR[arch],
                               marker=WD_MARKER.get(wd, "o"), s=28, alpha=0.8, zorder=3)
                    means.append((x, float(np.median(ok.grok_step))))
                if len(ng):
                    capv = ng.cap.fillna(ng.steps_run).values
                    ax.scatter([x] * len(ng), capv, facecolors="none",
                               edgecolors=ARCH_COLOR[arch], marker="^", s=60, zorder=4)
            if means:
                xs, ys = zip(*means)
                ax.plot(xs, ys, color=ARCH_COLOR[arch], lw=1.2,
                        ls="-" if wd == 0.1 else "--", alpha=0.9)
    for arch in ARCHS:
        ax.plot([], [], color=ARCH_COLOR[arch], label=arch)
    ax.plot([], [], color="grey", ls="-", label="WD 0.1 (circles)")
    ax.plot([], [], color="grey", ls="--", label="WD 0.3 (squares)")
    ax.scatter([], [], facecolors="none", edgecolors="grey", marker="^", s=60,
               label="hit cap, no grok (plotted at cap)")
    ax.set_yscale("log"); ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlabel("rho (exception fraction)"); ax.set_ylabel("grok step (add, single-task)")
    ax.set_title("Cost of exceptions: grok step vs rho (points = seeds, line = median)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "grok_delay.png"); plt.close(fig)


# The order in which the tasks of the joint runs grok, per architecture and seed.
# Input: tab - the runs table; out - Path of the output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_grok_order(tab: pd.DataFrame, out: Path):
    d = tab[tab.joint].copy()
    if d.empty:
        return
    keys = sorted(d[["arch", "seed"]].drop_duplicates().itertuples(index=False),
                  key=lambda k: (ARCHS.index(k.arch), k.seed))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(keys) + 1.8))
    for y, (arch, seed) in enumerate(keys):
        c = d[(d.arch == arch) & (d.seed == seed)]
        for _, r in c.iterrows():
            if r.grok_step is not None and np.isfinite(r.grok_step):
                ax.scatter(r.grok_step, y, color=TASK_COLOR[r.task], s=70, zorder=3)
            if np.isfinite(r.t_switch_rule_or_rulex):
                ax.scatter(r.t_switch_rule_or_rulex, y, facecolors="none",
                           edgecolors=TASK_COLOR[r.task], s=140, lw=1.4, zorder=2)
            if np.isfinite(r.t_half_mech_corrected):
                ax.scatter(r.t_half_mech_corrected, y, color=TASK_COLOR[r.task],
                           marker="|", s=160, lw=1.6, zorder=2)
        ax.axhline(y, color="lightgrey", lw=0.6, zorder=0)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{a} s{s}" for a, s in keys])
    ax.set_xscale("log"); ax.set_xlabel("step")
    for t, col in TASK_COLOR.items():
        ax.scatter([], [], color=col, s=70, label=f"{t}: behavioral grok")
    ax.scatter([], [], facecolors="none", edgecolors="grey", s=140, label="cognitive switch (rule/rulex)")
    ax.scatter([], [], color="grey", marker="|", s=160, label="mech half-rise (add: post-max window)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title("Track B': order of crystallization in joint training")
    fig.tight_layout(); fig.savefig(out / "grok_order.png"); plt.close(fig)


# Hidden progress (C1): the mechanistic half-rise against the behavioural one over the single-task
# runs, which is where mechanism leading behaviour becomes visible.
# Input: tab - the runs table; out - Path of the output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_hidden_progress(tab: pd.DataFrame, out: Path):
    d = tab[(~tab.joint)].copy()
    if "t_half_mech_use" in d.columns:
        d["t_half_mech"] = d["t_half_mech_use"]
    d = d.dropna(subset=["t_half_mech", "t_half_test_acc"])
    d = d[(d.t_half_mech > 0) & (d.t_half_test_acc > 0)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    mk = {"add": "o", "div": "s", "max": "^"}
    for arch in ARCHS:
        for task in mk:
            c = d[(d.arch == arch) & (d.task == task)]
            if c.empty:
                continue
            ax.scatter(c.t_half_test_acc, c.t_half_mech, color=ARCH_COLOR[arch],
                       marker=mk[task], s=34, alpha=0.8)
    lo = min(d.t_half_mech.min(), d.t_half_test_acc.min()) * 0.7
    hi = max(d.t_half_mech.max(), d.t_half_test_acc.max()) * 1.4
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="k", lw=0.8, label="mech = behavior")
    ax.plot(xs, xs / 2, color="grey", lw=0.8, ls="--", label="mech leads 2x")
    ax.plot(xs, xs / 4, color="grey", lw=0.8, ls=":", label="mech leads 4x")
    for arch in ARCHS:
        ax.scatter([], [], color=ARCH_COLOR[arch], label=arch)
    for t, m in mk.items():
        ax.scatter([], [], color="grey", marker=m, label=t)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("behavioral half-rise (test acc)"); ax.set_ylabel("mechanistic half-rise")
    ax.set_title("Hidden progress: representation forms before behavior flips")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(out / "hidden_progress.png"); plt.close(fig)


# Does the ORDER of the two routes predict overregularisation? x is how much of the exception route
# already exists at the grok (exception accuracy at grok_step), y is the peak overregularisation,
# over all single-task rho > 0 runs.
# Input: tab - the runs table; out - Path of the output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_order_vs_overreg(tab: pd.DataFrame, out: Path):
    d = tab[(~tab.joint) & (tab.rho > 0)].dropna(subset=["peak_overreg", "acc_exc_at_grok"])
    if d.empty:
        return
    mk = {"add": "o", "div": "s", "max": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for arch in ARCHS:
        for task in mk:
            c = d[(d.arch == arch) & (d.task == task)]
            if c.empty:
                continue
            axes[0].scatter(c.acc_exc_at_grok, c.peak_overreg, color=ARCH_COLOR[arch],
                            marker=mk[task], s=40, alpha=0.8, edgecolors="white", lw=0.5)
            cc = c.dropna(subset=["mem_to_grok_ratio"])
            axes[1].scatter(cc.mem_to_grok_ratio, cc.peak_overreg, color=ARCH_COLOR[arch],
                            marker=mk[task], s=40, alpha=0.8, edgecolors="white", lw=0.5)
    # honest reference: at the grok instant, overreg + acc_exc + other_err = 1, so the left
    # panel is close to an accounting identity; the timing panel on the right is not.
    axes[0].plot([0, 1], [1, 0], color="grey", lw=0.8, ls="--", label="1 - acc_exc (identity)")
    axes[0].set_xlabel("acc_exc at the behavioral grok  (exception route already built?)")
    axes[0].set_ylabel("peak overreg_rate")
    axes[0].set_xlim(-0.03, 1.03); axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_xscale("log"); axes[1].axvline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("t(exceptions memorized) / t(grok)   (<1: exceptions first, >1: rule first)")
    axes[1].set_ylabel("peak overreg_rate"); axes[1].set_ylim(-0.03, 1.03)
    for arch in ARCHS:
        axes[0].scatter([], [], color=ARCH_COLOR[arch], label=arch)
    for t, m in mk.items():
        axes[0].scatter([], [], color="grey", marker=m, label=t)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("Overregularization is set by which route arrives first (all archs, tasks, seeds)")
    fig.tight_layout(); fig.savefig(out / "order_vs_overreg.png"); plt.close(fig)
    order_vs_overreg_stats(d, out)


# Per-run Delta-BIC trajectories against step / grok on the single-task add runs: does the exemplar
# description beat uniform before the grok, and when does the rule family take over?
# Input: loaded - {(condition, seed): offline dict}; tab - the runs table; out - output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_cognitive_regimes(loaded, tab: pd.DataFrame, out: Path):
    items = list(single_add_runs(loaded, tab))
    if not items:
        return
    cols = [("dbic_gcm_vs_uniform", "exemplar (best GCM) vs uniform"),
            ("dbic_rulefam_vs_gcm", "rule family vs exemplar"),
            ("dbic_rulex_vs_rule", "rulex vs rule (exception route)")]
    fig, axes = plt.subplots(len(ARCHS), 3, figsize=(13, 3.0 * len(ARCHS)), sharex=True)
    for i, arch in enumerate(ARCHS):
        for j, (col, title) in enumerate(cols):
            ax = axes[i, j]
            for r, cdf in items:
                if r.arch != arch or (col == "dbic_rulex_vs_rule" and r.rho == 0):
                    continue
                x = cdf["step"].values / float(r.grok_step)
                y = cdf[col].values / cdf["n_probe"].values
                ax.plot(x, y, color=ARCH_COLOR[arch], lw=0.9, alpha=0.7,
                        ls="-" if r.rho == 0 else "--")
            ax.axhline(0, color="k", lw=0.6); ax.axvline(1, color="k", lw=0.6, ls=":")
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{arch}\ndBIC per probe (nats)")
            if i == len(ARCHS) - 1:
                ax.set_xlabel("step / grok_step")
    axes[0, 0].plot([], [], color="grey", ls="-", label="rho = 0")
    axes[0, 0].plot([], [], color="grey", ls="--", label="rho > 0")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Cognitive description over training (common-probe fits: test + exception items)")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / "cognitive_regimes.png"); plt.close(fig)


# The readable version of the regime strips: per architecture, the share of single-task add runs
# whose best model belongs to each family (uniform / exemplar / rule family) as a function of
# step / grok_step.
# Input: loaded - {(condition, seed): offline dict}; tab - the runs table; out - output directory
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_regime_shares(loaded, tab: pd.DataFrame, out: Path):
    items = list(single_add_runs(loaded, tab))
    if not items:
        return
    grid = np.geomspace(1e-3, 10, 60)
    archs = [a for a in ARCHS if any(r.arch == a for r, _ in items)]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.3 * len(archs), 3.6), squeeze=False, sharey=True)
    rows = []
    for j, arch in enumerate(archs):
        runs = [(r, c) for r, c in items if r.arch == arch]
        fam_at = []
        for r, cdf in runs:
            x = cdf["step"].values / float(r.grok_step)
            fam = np.array([FAMILY.get(str(m), "uniform") for m in cdf["best_model"].values])
            idx = np.searchsorted(x, grid, side="right") - 1          # last checkpoint at or before the grid point
            f = np.where(idx >= 0, fam[np.clip(idx, 0, len(fam) - 1)], None)
            fam_at.append(f)
        F = np.array(fam_at, dtype=object)
        ax = axes[0, j]
        bottom = np.zeros(len(grid))
        for fname in FAMILY_COLOR:
            valid = (F != None)                                          # noqa: E711
            share = np.where(valid.sum(0) > 0, (F == fname).sum(0) / np.maximum(valid.sum(0), 1), np.nan)
            ax.fill_between(grid, bottom, bottom + np.nan_to_num(share), color=FAMILY_COLOR[fname], lw=0, label=fname)
            bottom = bottom + np.nan_to_num(share)
            for gx, sh in zip(grid, share):
                rows.append({"arch": arch, "step_over_grok": gx, "family": fname, "share": sh})
        ax.axvline(1, color="k", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(0, 1)
        ax.set_title(f"{arch}  (n = {len(runs)} runs)"); ax.set_xlabel("step / grok step")
        if j == 0:
            ax.set_ylabel("share of runs, best model")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which cognitive model describes the network best, over training (single-task add; dotted line = behavioural grok)")
    fig.tight_layout(rect=[0, 0.07, 1, 0.92]); fig.savefig(out / "regime_shares.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "regime_shares.csv", index=False)
