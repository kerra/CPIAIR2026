from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.lesion_summary import TAG_ORDER, summary_table
from ..core.tasks import ARCHS
from .style import FAMILY, FAMILY_COLOR, PAPER_RC, plt, styled


# figures

# The low-pass sweep read in cognitive-model space, one panel per architecture: the share of runs
# whose best model is the rule family, the exemplar family or uniform at each bandwidth.
# Input: x - sweep rows with deltas; side - Fourier side; out - output directory; sfx - filename suffix
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_cognitive(x: pd.DataFrame, side: str, out: Path, sfx: str = ""):
    lp = x[(x.lesion == "lowpass") & (x.side == side)].copy()
    if lp.empty:
        return
    lp["family"] = lp.best_model.map(FAMILY).fillna("uniform")
    fcuts = sorted(lp.dose.unique(), reverse=True)            # 74 (intact) ... 0 (DC only)
    xs = np.arange(len(fcuts))
    labels = [("intact" if f == fcuts[0] else f"{int(f)}") for f in fcuts]
    archs = [a for a in ARCHS if (lp.arch == a).any()]
    fig, axes = plt.subplots(2, len(archs), figsize=(4.3 * len(archs), 6.2), squeeze=False, sharex=True)
    rows = []
    for j, arch in enumerate(archs):
        g = lp[lp.arch == arch]
        n_runs = g.groupby(["condition", "seed"]).ngroups
        share = (g.groupby(["dose", "family"]).size().unstack(fill_value=0)
                   .reindex(index=fcuts, columns=list(FAMILY_COLOR), fill_value=0))
        share = share.div(share.sum(axis=1), axis=0)
        ax = axes[0, j]
        bottom = np.zeros(len(fcuts))
        for fam in FAMILY_COLOR:
            ax.bar(xs, share[fam].values, bottom=bottom, color=FAMILY_COLOR[fam], width=0.85, label=fam)
            bottom += share[fam].values
        ax.set_ylim(0, 1); ax.set_title(f"{arch}  (n = {n_runs} runs)")
        if j == 0:
            ax.set_ylabel("share of runs, best model"); ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        lam = g.groupby("dose").gcm_circular_lambda.quantile([0.25, 0.5, 0.75]).unstack().reindex(fcuts)
        ax2 = axes[1, j]
        ax2.plot(xs, lam[0.5].values, marker="o", ms=4, color="k")
        ax2.fill_between(xs, lam[0.25].values, lam[0.75].values, color="k", alpha=0.12, lw=0)
        ax2.set_yscale("log"); ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_xlabel("Fourier modes kept in the numeric embedding")
        if j == 0:
            ax2.set_ylabel("GCM kernel sharpness λ\n(median, IQR)")
        for f, lab in zip(fcuts, labels):
            rows.append({"arch": arch, "modes_kept": lab, "share_rule_family": round(float(share.loc[f, "rule family"]), 2),
                         "share_exemplar": round(float(share.loc[f, "exemplar"]), 2), "share_uniform": round(float(share.loc[f, "uniform"]), 2),
                         "gcm_lambda_median": round(float(lam.loc[f, 0.5]), 2),
                         "test_acc_median": round(float(g[g.dose == f].test_acc.median()), 3),
                         "acc_exc_median": round(float(g[(g.dose == f) & (g.rho > 0)].acc_exc.median()), 3) if (g.rho > 0).any() else np.nan})
    fig.suptitle("Band-limiting the embedding retraces development in reverse: rule → exemplar → uniform, with a widening kernel")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out / f"lesion_cognitive{sfx}.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / f"lesion_cognitive_table{sfx}.csv", index=False)


# The one-row causal figure: removing the LEAST energetic modes of the embedding, over the pooled
# rho > 0 runs - test accuracy (the rule), exception accuracy and overregularisation against the
# energy removed, as median and IQR.
# Input: x - sweep rows with deltas; side - Fourier side; out - output directory; sfx - filename suffix
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_headline(x: pd.DataFrame, side: str, out: Path, sfx: str = ""):
    g0 = x[(x.lesion == "ebottom") & (x.side == side) & (x.rho > 0)]
    if g0.empty:
        return
    archs = [a for a in ARCHS if (g0.arch == a).any()]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.3 * len(archs), 3.6), squeeze=False, sharey=True)
    for j, arch in enumerate(archs):
        ax = axes[0, j]
        g = g0[g0.arch == arch]
        for col, color, ls, lab in (("test_acc", "#e6550d", "-", "rule (test acc)"),
                                    ("acc_exc", "#3182bd", "-", "exceptions (acc_exc)"),
                                    ("overreg_rate", "#9e9e9e", "--", "rule answer on exceptions (overreg)")):
            q = g.groupby("dose")[col].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index, q[0.5], color=color, ls=ls, lw=2, label=lab)
            ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.15, lw=0)
        ax.set_title(f"{arch}  (n = {g.groupby(['condition','seed']).ngroups} runs)")
        ax.set_xlabel("energy removed, weakest modes first")
        ax.set_ylim(-0.03, 1.03)
        if j == 0:
            ax.set_ylabel("accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Removing the low-energy part of the embedding deletes the exceptions and induces overregularization; the rule survives")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93]); fig.savefig(out / f"lesion_headline{sfx}.png", bbox_inches="tight"); plt.close(fig)


# How the dissociation builds up over training: the baseline state, the D50 of rule and exceptions
# under ebottom and etop, and SI at q = 0.5 under ebottom, per checkpoint tag (median over the
# rho > 0 runs).
# Input: x_all - sweep rows with deltas at every checkpoint tag; side - Fourier side;
# out - output directory; sfx - filename suffix
# Output: None - saves the figure
@styled(PAPER_RC)
def fig_development(x_all: pd.DataFrame, side: str, out: Path, sfx: str = ""):
    tags = [t for t in TAG_ORDER if t in set(x_all.step_req)]
    if len(tags) < 2:
        return
    rows = []
    for tag in tags:
        xt = x_all[x_all.step_req == tag]
        st = summary_table(xt, side)
        st["tag"] = tag
        b = xt[xt.lesion == "none"][["arch", "condition", "seed", "rho", "test_acc", "acc_exc", "overreg_rate"]].assign(tag=tag)
        rows.append((st, b))
    S = pd.concat([r[0] for r in rows]); B = pd.concat([r[1] for r in rows])
    S.to_csv(out / f"lesion_development{sfx}.csv", index=False)
    # actual checkpoint position: median step / grok_step per (arch, tag) -- the snapshot grid is coarse,
    # so "0.5g" can sit at 0.6-0.7 of the grok in the fast-grokking transformer runs
    base = x_all[(x_all.lesion == "none")].copy()
    if "grok_step" in base.columns:
        base["ratio"] = pd.to_numeric(base["step"], errors="coerce") / base["grok_step"]
    else:
        base["ratio"] = np.nan
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), sharex=False)
    xs = np.arange(len(tags))
    for j, arch in enumerate(ARCHS):
        ax = axes[0, j]
        bb = B[(B.arch == arch)]
        if bb.empty:
            continue
        rat = base[base.arch == arch].groupby("step_req")["ratio"].median().reindex(tags)
        labels = [t if (t == "final" or not np.isfinite(rat.get(t, np.nan))) else f"{t}\n({rat[t]:.2f}g)" for t in tags]
        for ax_ in (axes[0, j], axes[1, j]):
            ax_.set_xticks(xs); ax_.set_xticklabels(labels, fontsize=8)
        for col, ls, lab in (("test_acc", "-", "test acc (rule)"), ("acc_exc", "--", "acc_exc"), ("overreg_rate", ":", "overreg")):
            m = bb[bb.rho > 0].groupby("tag")[col].median().reindex(tags)
            ax.plot(xs, m.values, ls=ls, marker="o", ms=4, color="k", label=lab)
        ax.set_ylim(-0.03, 1.03); ax.set_title(f"{arch}: unlesioned state")
        if j == 0:
            ax.legend(fontsize=7)
        ax = axes[1, j]
        for les, color in (("ebottom", "#1f77b4"), ("etop", "#d62728")):
            ss = S[(S.arch == arch) & (S.lesion == les) & (S.rho > 0)]
            if ss.empty:
                continue
            for col, ls, lab in (("D50_rule", "-", f"{les}: D50 rule"), ("D50_exc", "--", f"{les}: D50 exc")):
                m = ss.groupby("tag")[col].median().reindex(tags)
                ax.plot(xs, m.values, ls=ls, marker="o", ms=4, color=color, label=lab)
        ss = S[(S.arch == arch) & (S.lesion == "ebottom") & (S.rho > 0)]
        if "SI_q0.5" in ss:
            m = ss.groupby("tag")["SI_q0.5"].median().reindex(tags)
            ax.plot(xs, m.values, ls="-.", marker="s", ms=4, color="#2ca02c", label="ebottom: SI (q=.5)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-0.6, 1.05); ax.set_title(f"{arch}: lesion doses (energy fraction) & SI")
        if j == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Dissociation along development: same lesions at 0.25 / 0.5 / 1.0 x grok and at the end (rho > 0, medians)")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out / f"lesion_development{sfx}.png"); plt.close(fig)
