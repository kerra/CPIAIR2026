from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core.tasks import ARCHS
from .lesion_summary import paired_bottom_vs_random

ARCH_NAME = {"transformer": "transformer", "mamba": "Mamba-2", "kimi": "KDA"}
FAMILY = {"uniform": "uniform", "gcm_hamming": "exemplar", "gcm_circular": "exemplar", "gcm_linear": "exemplar",
          "alcove_lite": "exemplar", "prototype": "prototype", "rule": "rule family", "rulex": "rule family",
          "atrium_lite": "mixture"}
STANDARD_MODELS = ("uniform", "gcm_hamming", "gcm_circular", "gcm_linear", "rule", "rulex")
MODEL_SHORT = {"atrium_lite": "ATRIUM-l.", "gcm_hamming": "GCM-h.", "gcm_circular": "GCM-c.", "gcm_linear": "GCM-l.",
               "alcove_lite": "ALCOVE-l.", "rulex": "RULEX", "rule": "rule", "uniform": "uniform", "prototype": "prototype"}


# ------------------------------------------------------------------------------------------- formatting

# Monospace LaTeX with the underscores escaped.
# Input: s - any value
# Output: str - a \texttt{...} fragment
def tt(s: str) -> str:
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"


# A float at a fixed number of decimals, blank when it is missing or non-finite.
# Input: x - the value; nd - decimals
# Output: str
def num(x, nd: int = 2) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# Two decimals, with negative zero normalised to 0.00.
# Input: x - the value
# Output: str
def two(x) -> str:
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s


# A float with an explicit LaTeX plus or minus sign.
# Input: x - the value; nd - decimals
# Output: str, blank when the value is non-finite
def signed(x, nd: int = 2) -> str:
    return "" if not np.isfinite(x) else (f"$+${x:.{nd}f}" if x >= 0 else f"$-${abs(x):.{nd}f}")


# A step count as an integer with a thousands separator.
# Input: x - the value
# Output: str, blank when the value is missing or non-finite
def steps(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{int(round(x)):,}"


# A p-value: two decimals down to 0.01, scientific notation in LaTeX below that.
# Input: p - the p-value
# Output: str, blank when it is non-finite
def pval(p) -> str:
    if not np.isfinite(p):
        return ""
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# The median of a series with its min-max range, collapsing to the median alone when the range is
# a single value.
# Input: v - the values; fmt - the formatter applied to each number
# Output: str like "1,200 [900--1,800]"
def med_range(v, fmt=steps) -> str:
    v = pd.Series(v).dropna()
    if v.empty:
        return ""
    lo, hi = fmt(v.min()), fmt(v.max())
    return fmt(v.median()) if lo == hi else f"{fmt(v.median())} [{lo}--{hi}]"


# The min-max range of a series, collapsing to one value when both ends format the same.
# Input: v - the values; fmt - the formatter applied to each number
# Output: str
def rng(v, fmt) -> str:
    v = pd.Series(v).dropna()
    return "" if v.empty else (fmt(v.min()) if fmt(v.min()) == fmt(v.max()) else f"{fmt(v.min())}--{fmt(v.max())}")


# A full-width LaTeX tabularx environment with booktabs rules.
# Input: colspec - the column specification; header - header cells; rows - the body rows;
# tabcolsep - inter-column padding
# Output: str - the LaTeX source
def tabularx(colspec: str, header: list[str], rows: list[list[str]], tabcolsep: str = "3pt") -> str:
    L = [r"\small", f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}", f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(L) + "\n"


# A plain LaTeX tabular environment with booktabs rules and a multi-line header.
# Input: colspec - the column specification; header_lines - raw header lines; rows - the body rows
# Output: str - the LaTeX source
def tabular(colspec: str, header_lines: list[str], rows: list[list[str]]) -> str:
    L = [r"\small", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header_lines + [r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# The header row that runs one column per architecture.
# Input: first - the label of the leading column
# Output: list[str] of header cells
def arch_header(first: str = "") -> list[str]:
    return [first] + [r"\textbf{" + ARCH_NAME[a] + "}" for a in ARCHS]


# ------------------------------------------------------------------------------------------- the tables

# Grok step per single-task cell: the median with its min-max range over the seeds that grokked,
# with cap hits counted separately.
# Input: tab - the runs table
# Output: str - the LaTeX table
def grok_steps_table(tab: pd.DataFrame) -> str:
    s = tab[~tab.joint]
    rows = []
    for (task, wd, rho), g in s.groupby(["task", "wd", "rho"], sort=True):
        row = [f"{task}, {wd:g}", f"{rho:g}"]
        for a in ARCHS:
            c = g[g.arch == a]
            ok = c[~c.non_grokking & c.grok_step.notna()]
            cell = med_range(ok.grok_step)
            n_cap = int(c.non_grokking.sum())
            if n_cap:
                cell += f", {n_cap} cap"
            row.append(cell)
        rows.append((["add", "div", "max"].index(task), wd, rho, row))
    rows = [r[-1] for r in sorted(rows, key=lambda r: r[:3])]
    header = [r"\textbf{task,} $\lambda_{WD}$", r"$\rho$"] + [r"\textbf{" + ARCH_NAME[a] + "}" for a in ARCHS]
    return tabularx("@{}YYYYY@{}", header, rows)


# The collapsed best-model sequences of the single-task add runs, counted per architecture.
# Input: tab - the runs table
# Output: str - the LaTeX table
def regimes_table(tab: pd.DataFrame) -> str:
    s = tab[(~tab.joint) & (tab.task == "add") & (tab.regime_seq != "")]
    rows = []
    for a in ARCHS:
        vc = s[s.arch == a].regime_seq.value_counts()
        arrow = r" $\rightarrow$ "
        seqs = "; ".join(k.replace(">", arrow) + r" $\times$" + str(v) for k, v in vc.items())
        rows.append([ARCH_NAME[a], seqs])
    n = int(s.groupby("arch").size().max())
    return tabularx("@{}p{2.6cm}Y@{}", [r"\textbf{architecture}", f"\\textbf{{sequences ({n} runs)}}"], rows)


# The pooled H1 statistics over the single-task run x task rows, on the same definitions as
# hidden_progress_stats.
# Input: tab - the runs table
# Output: str - the LaTeX table
def hidden_progress_table(tab: pd.DataFrame) -> str:
    d = tab[~tab.joint].copy()
    rows = []
    for beh_col, name in (("t_half_test_acc", "behavioural half-rise"), ("t_switch_rule_or_rulex", "cognitive switch")):
        dd = d.dropna(subset=["t_half_mech_use", beh_col]).copy()
        dd["tm"] = dd["t_half_mech_use"].astype(float)
        dd = dd[(dd.tm > 0) & (dd[beh_col] > 0)]
        rho, p = sps.spearmanr(dd.tm, dd[beh_col])
        lead = dd.tm < dd[beh_col]
        bt = sps.binomtest(int(lead.sum()), len(dd), 0.5)
        ratio = dd[beh_col] / dd.tm
        rows.append([f"$t_{{mech}}$ (first Fourier structure) vs {name}",
                     f"{rho:.3f} (p = {pval(p)})",
                     f"{int(lead.sum())}/{len(dd)} (sign test p = {pval(bt.pvalue)})",
                     f"{ratio.median():.2f} ({ratio.quantile(.25):.2f}--{ratio.quantile(.75):.2f})"])
    header = [r"\textbf{relation}", r"\textbf{Spearman}", r"\textbf{mechanism leads}", r"\textbf{median} $t_{beh}/t_{mech}$ \textbf{(IQR)}"]
    return tabularx("@{}YYYY@{}", header, rows)


# Route order and overregularisation on add for the rho > 0 runs, with both weight decays pooled.
# Input: tab - the runs table
# Output: str - the LaTeX table
def route_order_table(tab: pd.DataFrame) -> str:
    r = tab[(~tab.joint) & (tab.task == "add") & (tab.rho > 0)]

    def label(v):
        m = v.median()
        return "rule first" if m > 1.5 else ("exceptions first" if m < 1 / 1.5 else "together")

    def col(a):
        g = r[r.arch == a]
        cell_means = g.groupby(["rho", "wd"]).peak_overreg.mean()
        return [med_range(g.mem_to_grok_ratio, lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}") + " --- " + label(g.mem_to_grok_ratio.dropna()),
                rng(g.t_exc_memorized, steps),
                num(g.acc_exc_at_grok.median(), 3 if g.acc_exc_at_grok.median() > 0.99 else 2),
                rng(g.peak_overreg, lambda v: f"{v:.3f}" if v < 0.1 else f"{v:.2f}"),
                rng(cell_means, lambda v: f"{v:.2f}"),
                f"{g.final_acc_exc.median():.2f} / {g.final_overreg.median():.2f}"]

    names = [r"$t_{exc}/t_{grok}$, median [min--max]", "exceptions memorized at (steps)",
             r"$\mathrm{acc}_{\mathrm{exc}}$ at the grok instant (median)", "peak overregularization, per run",
             "peak overregularization, cell means", r"final $\mathrm{acc}_{\mathrm{exc}}$ / final overreg (median)"]
    cols = {a: col(a) for a in ARCHS}
    rows = [[names[i]] + [cols[a][i] for a in ARCHS] for i in range(len(names))]
    return tabularx("@{}YYYY@{}", arch_header(), rows)


# Spectral lesions of the numeric embedding at one checkpoint tag - medians over runs, the same
# quantities lesion_stats.txt reports.
# Input: x - the sweep rows with deltas; summ - the lesion summary table; side - Fourier side
# Output: str - the LaTeX table
def lesion_table(x: pd.DataFrame, summ: pd.DataFrame, side: str = "in") -> str:
    cells = {a: [] for a in ARCHS}
    for a in ARCHS:
        g = x[(x.lesion == "etop") & (x.side == side) & (x.arch == a)]
        s_top = summ[(summ.lesion == "etop") & (summ.arch == a)]
        s_bot = summ[(summ.lesion == "ebottom") & (summ.arch == a) & (summ.rho > 0)]
        si = s_bot["SI_q0.5"].dropna()
        ov = s_bot.overreg_at_rule_intact
        m = paired_bottom_vs_random(x, side, a)
        w1 = sps.wilcoxon(m.d_test_bot, m.d_test_rnd, alternative="less")
        w2 = sps.wilcoxon(m.d_exc_bot, m.d_exc_rnd, alternative="greater")
        cells[a] = [signed(sps.spearmanr(g.dose, g.test_acc)[0]),
                    f"{two(s_top.D50_rule.median())} / {two(s_top.D50_exc.median())}",
                    f"{two(s_bot.D50_exc.median())} / {two(s_bot.D50_rule.median())}",
                    f"{two(s_bot['d_test_q0.5'].median())} / {two(s_bot['d_exc_q0.5'].median())}",
                    f"{signed(si.median())}; {int((si >= 0.3).sum())}/{len(si)}",
                    f"{ov.median():.2f} ($\\geq$ 0.2 in {int((ov >= 0.2).sum())}/{len(ov)})",
                    f"{int((m.d_test_bot < m.d_test_rnd).sum())}/{len(m)}, p = {pval(w1.pvalue)}",
                    f"{int((m.d_exc_bot > m.d_exc_rnd).sum())}/{len(m)}, p = {pval(w2.pvalue)}"]
    names = [r"\textbf{etop} (most energetic modes removed): Spearman(dose, test acc)",
             "etop: D50 rule / D50 exceptions",
             r"\textbf{ebottom} (least energetic modes removed): D50 exceptions / D50 rule",
             r"ebottom, $q = 0.5$: drop test acc / drop $\mathrm{acc}_{\mathrm{exc}}$",
             r"ebottom, $q = 0.5$: SI median; runs with SI $\geq$ 0.3",
             r"ebottom: max overregularization with the rule intact (test $\geq$ 0.9)",
             r"specificity vs random modes: rule spared (bottom $<$ random)",
             r"specificity: exceptions hit harder (bottom $>$ random)"]
    rows = [[names[i]] + [cells[a][i] for a in ARCHS] for i in range(len(names))]
    return tabularx("@{}YYYY@{}", arch_header(), rows)


# Band-limiting: the share of runs best described by the rule family and the median GCM kernel
# sharpness, as a function of how many Fourier modes are kept.
# Input: lc - the lesion cognitive table; kept - which mode counts become columns
# Output: str - the LaTeX table
def lowpass_table(lc: pd.DataFrame, kept=("intact", "32", "16", "8", "4", "0")) -> str:
    lc = lc.assign(modes_kept=lc.modes_kept.astype(str))
    rows = []
    for qname, col, fmt in (("share of runs best described by the rule family", "share_rule_family", lambda v: f"{v:.2f}" if v > 0 else "0"),
                            (r"median GCM kernel sharpness $\lambda$", "gcm_lambda_median", lambda v: f"{v:g}")):
        for i, a in enumerate(ARCHS):
            g = lc[lc.arch == a].set_index("modes_kept")
            rows.append([qname if i == 0 else "", ARCH_NAME[a]] + [fmt(float(g.loc[k, col])) for k in kept])
        if col == "share_rule_family":
            rows.append([r"\midrule"])
    head = [r" & & \multicolumn{" + str(len(kept)) + r"}{c}{Fourier modes kept in the numeric embedding} \\",
            r"\cmidrule(lr){3-" + str(2 + len(kept)) + "}",
            r"\textbf{quantity} & \textbf{architecture} & " + " & ".join(("74 (intact)" if k == "intact" else k) for k in kept) + r" \\"]
    body = []
    for r in rows:
        body.append(r[0] if r == [r"\midrule"] else " & ".join(r) + r" \\")
    L = [r"\small", r"\begin{tabular}{@{}ll" + "c" * len(kept) + "@{}}", r"\toprule"] + head + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# The best comparable EQ location per architecture and task, with the dominant mode per seed and
# the circle correlation.
# Input: mt - the manifold table
# Output: str - the LaTeX table
def manifold_table(mt: pd.DataFrame) -> str:
    rows = []
    for a in ARCHS:
        row = [ARCH_NAME[a]]
        for t in ("add", "div", "max"):
            r = mt[(mt.arch == a) & (mt.task == t)].iloc[0]
            value = str(r.value).replace("±", r"$\pm$")
            cell = f"{tt(r.location)} {value}"
            if t != "max":
                cell += f", modes {r.modes_per_seed}, circle corr {r.circle_corr:.2f}"
            best_loc = str(r.best_anywhere).split(" (")[0]
            if best_loc != r.location:
                best_val = str(r.best_anywhere).split("(")[1].split(";")[0]
                cell += f"; best anywhere {tt(best_loc)} {best_val}"
            row.append(cell)
        rows.append(row)
    header = [r"\textbf{architecture}", r"\textbf{add (}$F$\textbf{, modes $\geq$ 3)}", r"\textbf{div (}$F$\textbf{, dlog basis)}", r"\textbf{max (linear corr)}"]
    return tabularx("@{}YYYY@{}", header, rows)


# Layer-0 locations without versus with learned positional embeddings, flagging the rows that are
# best under either condition.
# Input: pa - the positional-ablation table
# Output: str - the LaTeX table
def pos_ablation_table(pa: pd.DataFrame) -> str:
    d = pa[pa.best.fillna("") != ""]
    rows = []
    for r in d.itertuples():
        modes = "--" if str(r.modes_nopos).startswith("–") or str(r.modes_nopos).startswith("-") else f"{r.modes_nopos} $\\rightarrow$ {r.modes_pos}"
        rows.append([ARCH_NAME[r.arch], r.task, tt(r.location), f"{r.value_mean_nopos:.2f}", f"{r.value_mean_pos:.2f}",
                     signed(r.delta) if abs(r.delta) >= 0.005 else "$0.00$", modes])
    header = [r"\textbf{arch}", r"\textbf{task}", r"\textbf{location}", r"\textbf{no pos}", r"\textbf{pos}", r"$\Delta$",
              r"\textbf{dominant modes per seed, no pos $\rightarrow$ pos}"]
    return tabularx("@{}llYcccY@{}", header, rows, tabcolsep="4pt")


# Threshold sensitivity of the C2 route-order regularity.
# Input: sens - the sensitivity grid from robustness.sensitivity
# Output: str - the LaTeX table
def robustness_table(sens: pd.DataFrame) -> str:
    rows = [[f"{r.grok_thr:.2f}", f"{r.exc_thr:.2f}", str(int(r.n)), f"{r.spearman:.3f}", f"{r.r2_logistic:.3f}",
             f"{r.r2_logistic_arch:.3f}", f"{r.delta_r2_arch:.3f}", f"{r.b:.2f}"] for r in sens.itertuples()]
    head = [r"\textbf{grok thr} & \textbf{exc.\ thr} & $n$ & \textbf{Spearman} & $R^2$ \textbf{logistic} & $R^2$ \textbf{+ arch} & $\Delta R^2$ & \textbf{slope} $b$ \\"]
    return tabular("@{}cccccccc@{}", head, rows)


# Held-out agreement, the nine-model winners, ATRIUM-lite / ALCOVE-lite / prototype and the
# permutation null per checkpoint tag - the same counting rules as the validation report.
# Input: df - the stacked validation frames; tags - which checkpoint tags become rows
# Output: str - the LaTeX table
def validation_table(df: pd.DataFrame, tags=("0.5g", "1g", "2g", "final")) -> str:
    std = list(STANDARD_MODELS)
    cols = {}
    for tag in tags:
        d = df[df.tag == tag]
        n = agree = fam = 0
        winners, alc_w, alc_beats, mix_pi, ranks = {}, [], 0, [], []
        for _, g in d.groupby(["condition", "seed"]):
            g6 = g[g.model.isin(std)]
            bb = g6.loc[g6.bic.idxmin(), "model"]; bc = g6.loc[g6.cvll.idxmax(), "model"]
            agree += bb == bc; fam += FAMILY[bb] == FAMILY[bc]; n += 1
            b9 = g.loc[g.bic.idxmin(), "model"]; winners[b9] = winners.get(b9, 0) + 1
            alc = g[g.model == "alcove_lite"].iloc[0]; alc_w.append(alc.w)
            alc_beats += alc.bic < g[g.model == "gcm_circular"].bic.iloc[0]
            mix_pi.append(float(g[g.model == "atrium_lite"].pi.iloc[0]))
            ranks.append(g[g.model.isin(std + ["prototype"])].sort_values("bic").model.tolist().index("prototype") + 1)
        one = d.groupby(["condition", "seed"]).first()
        win_txt = ", ".join(f"{MODEL_SHORT[m]} {c}" for m, c in sorted(winners.items(), key=lambda kv: -kv[1]))
        cols[tag] = [str(n), f"{agree} / {fam}", win_txt,
                     f"{winners.get('atrium_lite', 0)} ({np.median(mix_pi):.2f})",
                     f"{np.median(alc_w):.1f}; {alc_beats}",
                     f"{np.mean(ranks):.1f}",
                     f"{int((one.obs_dbic_exemplar > one.perm_dbic_exemplar_max).sum())} / {int((one.obs_dbic_rulefam > one.perm_dbic_rulefam_max).sum())}"]
    names = ["checkpoints ($n$)", "BIC winner = held-out winner (same model / same family)", "best of nine models by BIC (counts)",
             r"ATRIUM-lite wins (median rule weight $\pi$)", r"ALCOVE-lite: median $w$; checkpoints where it beats the equal-attention GCM",
             "prototype: mean BIC rank among seven models (7 = worst)", "checkpoints (of $n$) whose observed $\\Delta$BIC exceeds the permutation-null maximum: exemplar vs uniform / rule family vs exemplar"]
    header = [""] + [r"\textbf{" + (t.replace("g", r" $\times$ grok") if t != "final" else "final") + "}" for t in tags]
    rows = [[names[i]] + [cols[t][i] for t in tags] for i in range(len(names))]
    return tabularx("@{}p{5.6cm}YYYY@{}", header, rows, tabcolsep="5pt")


STAT_LABEL = {
    "Spearman(t_mech, behavioural half-rise)": r"Spearman($t_{mech}$, behavioral half-rise), 90 run $\times$ task rows",
    "Spearman(t_mech, cognitive switch)": r"Spearman($t_{mech}$, cognitive switch), 90 rows",
    "Spearman(peak overreg, log10 t_exc/t_grok)": r"Spearman(peak overregularization, $\log_{10} t_{exc}/t_{grok}$)",
    "median t_switch/t_grok (add)": r"median $t_{switch}/t_{grok}$ (add)",
    "median lead t_beh/t_mech (add)": r"median lead $t_{beh}/t_{mech}$ (add)",
    "median t_exc/t_grok (add)": r"median $t_{exc}/t_{grok}$ (add)",
    "median D50 exceptions (ebottom, add)": "median D50 of the exceptions, low-energy lesion (add)",
    "median D50 rule (ebottom, add)": "median D50 of the rule, low-energy lesion (add)",
    "median SI at q = 0.5 (ebottom, add)": "median SI at $q = 0.5$, low-energy lesion (add)",
}


# The 95 % cluster-bootstrap intervals of the statistics quoted in the text.
# Input: hi - the frame from robustness.headline_intervals
# Output: str - the LaTeX table
def headline_intervals_table(hi: pd.DataFrame) -> str:
    rows = []
    for r in hi.itertuples():
        nd = 3 if r.estimate < 1 and "median t_exc" not in r.statistic else 2
        rows.append([STAT_LABEL.get(r.statistic, r.statistic), "pooled" if r.arch == "all" else ARCH_NAME.get(r.arch, r.arch),
                     f"{r.estimate:.{nd}f}", f"{r.ci_lo:.{nd}f}--{r.ci_hi:.{nd}f}", str(int(r.n))])
    header = [r"\textbf{statistic}", r"\textbf{architecture}", r"\textbf{estimate}", r"\textbf{95\,\% interval}", r"$n$"]
    return tabularx("@{}Ylccc@{}", header, rows)
