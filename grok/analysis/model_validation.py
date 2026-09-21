from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.gcm import DEFAULT_GCM_LAMBDAS, GCMProbeCache, bic, digit_kernel, label_table, probe_mask
from ..core.tasks import P, n_classes
from ..runs.io import corrupted_train, grok_step_of, load_exceptions, load_task_split
from ..runs.records import Run

TAGS = ("0.5g", "1g", "2g", "final")
ALCOVE_LAMBDAS = np.geomspace(0.2, 40.0, 7)
ALCOVE_W = (0.1, 0.3, 0.5, 0.7, 0.9)
MIX_PI = np.linspace(0.0, 1.0, 21)
MIX_EPS = np.geomspace(1e-4, 0.5, 20)
PROTO_SENS = np.geomspace(0.1, 100.0, 25)
FAMILY = {"uniform": "uniform", "gcm_hamming": "exemplar", "gcm_circular": "exemplar", "gcm_linear": "exemplar",
          "alcove_lite": "exemplar", "prototype": "prototype", "rule": "rule family", "rulex": "rule family",
          "atrium_lite": "mixture"}


# ---------------------------------------------------------------- per-probe log-probability tables

# Per-probe log-probabilities of the shared-digit (hamming) GCM, in closed form from the label
# counts, for a whole grid of decay rates at once.
# Input: train_a, train_b, train_c - the exemplar memory; pa, pb - probe coordinates;
# K - classes; lambdas - the decay grid
# Output: np.ndarray[float32] [n_lambda, n_probe, K]
def hamming_logp(train_a, train_b, train_c, pa, pb, K: int, lambdas) -> np.ndarray:
    A = int(max(train_a.max(), pa.max())) + 1; B = int(max(train_b.max(), pb.max())) + 1
    cnt_a = np.zeros((A, K)); cnt_b = np.zeros((B, K))
    np.add.at(cnt_a, (train_a, train_c), 1.0); np.add.at(cnt_b, (train_b, train_c), 1.0)
    tot = cnt_a.sum(axis=0)
    L = -np.ones((A, B), dtype=np.int64); L[train_a, train_b] = train_c
    ca, cb = cnt_a[pa], cnt_b[pb]
    S = np.zeros((len(pa), K)); sl = L[pa, pb]; hs = sl >= 0
    S[np.where(hs)[0], sl[hs]] = 1.0
    n1 = ca + cb - 2.0 * S; n2 = tot[None, :] - ca - cb + S
    out = np.empty((len(lambdas), len(pa), K), dtype=np.float32)
    for i, lam in enumerate(lambdas):
        v = S + np.exp(-lam) * n1 + np.exp(-2.0 * lam) * n2 + 1e-9
        out[i] = np.log(v / v.sum(axis=1, keepdims=True))
    return out


# ALCOVE-lite: a circular GCM whose two digits get different attention weights,
# d = 2[w d(a,a') + (1-w) d(b,b')]. Still separable, so it is two matrix products.
# Input: T - [P, P, K] label table; pa, pb - probe coordinates; K - classes; lam - decay;
# w - attention on the first digit, in (0, 1)
# Output: np.ndarray [n_probe, K] of log-probabilities
def alcove_logp(T, pa, pb, K: int, lam: float, w: float) -> np.ndarray:
    Ka = digit_kernel(P, 2.0 * lam * w, "circular"); Kb = digit_kernel(P, 2.0 * lam * (1.0 - w), "circular")
    with np.errstate(all="ignore"):
        M = Ka @ T.reshape(P, P * K)
        M = M.reshape(P, P, K).transpose(1, 0, 2).reshape(P, P * K)
        V = (Kb @ M).reshape(P, P, K).transpose(1, 0, 2)
    v = V[pa, pb, :] + 1e-9
    return np.log(v / v.sum(axis=1, keepdims=True))


# Prototype model: each class prototype is the mean torus embedding (cos and sin of both digits)
# of its training items, and votes decay as exp(-c ||x - mu_k||).
# Input: train_a, train_b, train_c - the training items; pa, pb - probe coordinates;
# K - classes; sens - the grid of sensitivities c
# Output: np.ndarray[float32] [n_sens, n_probe, K]
def prototype_logp(train_a, train_b, train_c, pa, pb, K: int, sens) -> np.ndarray:
    def emb(a, b):
        return np.stack([np.cos(2 * np.pi * a / P), np.sin(2 * np.pi * a / P), np.cos(2 * np.pi * b / P), np.sin(2 * np.pi * b / P)], 1)
    X = emb(train_a, train_b)
    mu = np.zeros((K, 4)); cnt = np.zeros(K)
    np.add.at(mu, train_c, X); np.add.at(cnt, train_c, 1.0)
    mu = mu / np.maximum(cnt, 1)[:, None]
    Q = emb(pa, pb)
    D = np.sqrt(((Q[:, None, :] - mu[None, :, :]) ** 2).sum(-1))          # [n_probe, K]
    out = np.empty((len(sens), len(pa), K), dtype=np.float32)
    for i, c in enumerate(sens):
        v = np.exp(-c * D) + 1e-12
        out[i] = np.log(v / v.sum(axis=1, keepdims=True))
    return out


# ---------------------------------------------------------------- scoring helpers

# Log-likelihood of a prediction vector on a subset of the probes, for every parameter row at once.
# Input: logp - [n_params, n_probe, K] log-probability table; pred - predicted labels;
# idx - the probe indices to score
# Output: np.ndarray [n_params] of log-likelihoods
def ll_grid(logp: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return logp[:, idx, pred[idx]].sum(axis=1, dtype=np.float64)


# Log-likelihood of the pure rule with a lapse rate, over a grid of lapse rates.
# Input: pred, rule - predicted and rule labels; idx - probe indices; eps_grid - lapse grid
# Output: np.ndarray [n_eps] of log-likelihoods
def rule_ll(pred, rule, idx, eps_grid):
    n = len(idx); n_hit = int((pred[idx] == rule[idx]).sum())
    return n_hit * np.log((1 - eps_grid) + eps_grid / P) + (n - n_hit) * np.log(eps_grid / P)


# Log-likelihood of the rule-plus-exception model over the joint grid of lapse rate and
# exception-route probability.
# Input: pred, rule - predicted and rule labels; is_exc, exc_lab - which probes are exceptions and
# their memorised labels; idx - probe indices; eps_grid, m_grid - the two parameter grids
# Output: np.ndarray [n_eps, n_m] of log-likelihoods
def rulex_ll(pred, rule, is_exc, exc_lab, idx, eps_grid, m_grid):
    reg = idx[~is_exc[idx]]; ex = idx[is_exc[idx]]
    n_reg, n_reg_hit = len(reg), int((pred[reg] == rule[reg]).sum())
    e_hit = int((pred[ex] == exc_lab[ex]).sum()); e_rule = int((pred[ex] == rule[ex]).sum()); e_other = len(ex) - e_hit - e_rule
    E, M = np.meshgrid(eps_grid, m_grid, indexing="ij")
    ll_reg = n_reg_hit * np.log((1 - E) + E / P) + (n_reg - n_reg_hit) * np.log(E / P)
    with np.errstate(divide="ignore"):
        ll = (ll_reg + e_hit * np.log((1 - E) * M + E / P) + e_rule * np.log((1 - E) * (1 - M) + E / P) + e_other * np.log(E / P))
    return ll   # [n_eps, n_m]


# ATRIUM-lite: a mixture p = pi * p_rule(eps) + (1 - pi) * p_gcm(lambda), scored over the joint grid.
# Input: logp_gcm - [n_lambda, n_probe, K] exemplar table; pred, rule - predicted and rule labels;
# idx - probe indices; eps_grid, pi_grid - the two mixing grids
# Output: np.ndarray [n_lambda, n_eps, n_pi] of log-likelihoods
def mixture_ll(logp_gcm: np.ndarray, pred, rule, idx, eps_grid, pi_grid) -> np.ndarray:
    hit = (pred[idx] == rule[idx]).astype(float)
    p_gcm = np.exp(logp_gcm[:, idx, pred[idx]])                     # [n_lambda, n]
    out = np.empty((logp_gcm.shape[0], len(eps_grid), len(pi_grid)))
    for j, eps in enumerate(eps_grid):
        p_rule = hit * ((1 - eps) + eps / P) + (1 - hit) * (eps / P)   # [n]
        for k, pi in enumerate(pi_grid):
            out[:, j, k] = np.log(pi * p_rule[None, :] + (1 - pi) * p_gcm + 1e-300).sum(axis=1)
    return out


# ---------------------------------------------------------------- one checkpoint

# Everything a validation needs that depends on the run but not on the checkpoint: the corrupted
# train set, the probe set and its exception flags, and the pre-computed log-probability tables of
# every exemplar-style model.
class RunContext:

    def __init__(self, run: Run, a: np.ndarray, b: np.ndarray, rule: np.ndarray):
        split = load_task_split(run, run.task); exc = load_exceptions(run)
        self.tr_a, self.tr_b, self.tr_c = corrupted_train(split, exc)
        self.mask = probe_mask(a, b, self.tr_a, self.tr_b, exc)
        self.pa, self.pb, self.rule = a[self.mask], b[self.mask], rule[self.mask]
        self.K = n_classes(run.task, P)
        self.is_exc = np.zeros(len(self.pa), bool); self.exc_lab = np.zeros(len(self.pa), np.int64)
        if exc is not None and exc.n > 0:
            key = {(int(x), int(y)): int(e) for x, y, e in zip(exc.a, exc.b, exc.exc_label)}
            for i, (x, y) in enumerate(zip(self.pa, self.pb)):
                e = key.get((int(x), int(y)))
                if e is not None:
                    self.is_exc[i] = True; self.exc_lab[i] = e
        self.cache = GCMProbeCache(self.tr_a, self.tr_b, self.tr_c, self.pa, self.pb, self.K, P)
        self.logp = {"gcm_circular": self.cache.logp["circular"], "gcm_linear": self.cache.logp["linear"],
                     "gcm_hamming": hamming_logp(self.tr_a, self.tr_b, self.tr_c, self.pa, self.pb, self.K, np.geomspace(0.05, 8.0, 25)),
                     "prototype": prototype_logp(self.tr_a, self.tr_b, self.tr_c, self.pa, self.pb, self.K, PROTO_SENS)}
        self.T = label_table(self.tr_a, self.tr_b, self.tr_c, self.K, P)
        self.alcove = [(lam, w, alcove_logp(self.T, self.pa, self.pb, self.K, lam, w)) for lam in ALCOVE_LAMBDAS for w in ALCOVE_W]
        self.exc = exc


# Scores all nine models on one checkpoint: BIC on the full probe set, two-fold held-out
# log-likelihood (fit on one half, score on the other), and a permutation null for the family
# Delta-BICs obtained by shuffling the answers over probes.
# Input: ctx - RunContext; pred_full - predictions over the full table; rng - Generator;
# n_perm - how many shuffles the null uses
# Output: list[dict] - one row per model, each also carrying the permutation-null columns
def evaluate_checkpoint(ctx: RunContext, pred_full: np.ndarray, rng: np.random.Generator, n_perm: int = 50) -> list[dict]:
    pred = pred_full[ctx.mask]; n = len(pred); K = ctx.K
    all_idx = np.arange(n)
    perm = rng.permutation(n); halves = (perm[: n // 2], perm[n // 2:])
    eps400 = np.linspace(1e-4, 0.9999, 400); eps60 = np.linspace(1e-4, 0.9999, 60); m51 = np.linspace(0, 1, 51)
    rows = []

    def add(model, ll_full, k, cvll, params):
        rows.append({"model": model, "family": FAMILY[model], "ll": float(ll_full), "bic": float(bic(ll_full, k, n)),
                     "cvll": float(cvll), "k": k, "n_probe": n, **params})

    # uniform
    add("uniform", n * np.log(1 / K), 0, n * np.log(1 / K), {})
    # kernel models with one parameter (grid rows of a logp table)
    for name in ("gcm_hamming", "gcm_circular", "gcm_linear", "prototype"):
        lp = ctx.logp[name]
        grid = (np.geomspace(0.05, 8.0, 25) if name == "gcm_hamming" else PROTO_SENS if name == "prototype" else DEFAULT_GCM_LAMBDAS)
        lls = ll_grid(lp, pred, all_idx); i = int(np.argmax(lls))
        cv = 0.0
        for A, B in (halves, halves[::-1]):
            j = int(np.argmax(ll_grid(lp, pred, A))); cv += float(ll_grid(lp, pred, B)[j])
        add(name, lls[i], 1, cv, {"param": float(grid[i])})
    # ALCOVE-lite (lambda, w)
    lls = np.array([ll_grid(lp[None], pred, all_idx)[0] for _, _, lp in ctx.alcove]); i = int(np.argmax(lls))
    cv = 0.0
    for A, B in (halves, halves[::-1]):
        j = int(np.argmax([ll_grid(lp[None], pred, A)[0] for _, _, lp in ctx.alcove]))
        cv += float(ll_grid(ctx.alcove[j][2][None], pred, B)[0])
    add("alcove_lite", lls[i], 2, cv, {"param": float(ctx.alcove[i][0]), "w": float(ctx.alcove[i][1])})
    # rule
    ll = rule_ll(pred, ctx.rule, all_idx, eps400); i = int(np.argmax(ll))
    cv = sum(float(rule_ll(pred, ctx.rule, B, eps400)[int(np.argmax(rule_ll(pred, ctx.rule, A, eps400)))]) for A, B in (halves, halves[::-1]))
    add("rule", ll[i], 1, cv, {"param": float(eps400[i])})
    # rulex
    ll = rulex_ll(pred, ctx.rule, ctx.is_exc, ctx.exc_lab, all_idx, eps60, m51); ij = np.unravel_index(np.argmax(ll), ll.shape)
    cv = 0.0
    for A, B in (halves, halves[::-1]):
        ab = np.unravel_index(np.argmax(rulex_ll(pred, ctx.rule, ctx.is_exc, ctx.exc_lab, A, eps60, m51)), ll.shape)
        cv += float(rulex_ll(pred, ctx.rule, ctx.is_exc, ctx.exc_lab, B, eps60, m51)[ab])
    add("rulex", ll[ij], 2, cv, {"param": float(eps60[ij[0]]), "m": float(m51[ij[1]])})
    # ATRIUM-lite mixture on the hamming GCM
    lp = ctx.logp["gcm_hamming"]; lamg = np.geomspace(0.05, 8.0, 25)
    ll = mixture_ll(lp, pred, ctx.rule, all_idx, MIX_EPS, MIX_PI); ijk = np.unravel_index(np.argmax(ll), ll.shape)
    cv = 0.0
    for A, B in (halves, halves[::-1]):
        abc = np.unravel_index(np.argmax(mixture_ll(lp, pred, ctx.rule, A, MIX_EPS, MIX_PI)), ll.shape)
        cv += float(mixture_ll(lp, pred, ctx.rule, B, MIX_EPS, MIX_PI)[abc])
    add("atrium_lite", ll[ijk], 3, cv, {"param": float(lamg[ijk[0]]), "eps": float(MIX_EPS[ijk[1]]), "pi": float(MIX_PI[ijk[2]])})
    # permutation null of the two family contrasts (shuffle the network's answers across probes)
    d_exemplar, d_rule = [], []
    for _ in range(n_perm):
        pp = pred[rng.permutation(n)]
        b_u = bic(n * np.log(1 / K), 0, n)
        b_h = bic(float(ll_grid(ctx.logp["gcm_hamming"], pp, all_idx).max()), 1, n)
        b_r = min(bic(float(rule_ll(pp, ctx.rule, all_idx, eps400).max()), 1, n),
                  bic(float(rulex_ll(pp, ctx.rule, ctx.is_exc, ctx.exc_lab, all_idx, eps60, m51).max()), 2, n))
        d_exemplar.append(b_u - b_h); d_rule.append(b_h - b_r)
    obs = {r["model"]: r["bic"] for r in rows}
    null = {"perm_dbic_exemplar_max": float(np.max(d_exemplar)), "perm_dbic_exemplar_p99": float(np.percentile(d_exemplar, 99)),
            "perm_dbic_rulefam_max": float(np.max(d_rule)), "perm_dbic_rulefam_p99": float(np.percentile(d_rule, 99)),
            "obs_dbic_exemplar": obs["uniform"] - obs["gcm_hamming"], "obs_dbic_rulefam": obs["gcm_hamming"] - min(obs["rule"], obs["rulex"])}
    for r in rows:
        r.update(null)
    return rows


# The prediction dumps nearest the requested grok-relative tags, plus the final dump.
# Input: run - Run
# Output: dict {tag: (step, npz path)}; only "final" when the run never grokked
def checkpoint_files(run: Run) -> dict[str, tuple[int, object]]:
    files = sorted((run.dir / "preds" / run.task).glob("step_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    steps = np.array([int(p.stem.split("_")[1]) for p in files])
    out = {"final": (int(steps[-1]), files[-1])}
    g = grok_step_of(run)
    if g is not None:
        for tag, mult in (("0.5g", 0.5), ("1g", 1.0), ("2g", 2.0)):
            k = int(np.argmin(np.abs(steps - mult * g))); out[tag] = (int(steps[k]), files[k])
    return out


# Runs the whole validation for one run: builds the run context once, then evaluates every
# grok-relative checkpoint.
# Input: run - Run; rng - Generator; n_perm - shuffles per permutation null
# Output: pd.DataFrame - one row per (checkpoint, model)
def validate_run(run: Run, rng: np.random.Generator, n_perm: int = 50) -> pd.DataFrame:
    ck = checkpoint_files(run)
    z0 = np.load(ck["final"][1]); a = np.asarray(z0["a"], np.int64); b = np.asarray(z0["b"], np.int64); rule = np.asarray(z0["rule"], np.int64)
    ctx = RunContext(run, a, b, rule)
    rows = []
    for tag, (step, path) in ck.items():
        pred = np.asarray(np.load(path)["pred"], np.int64)
        for r in evaluate_checkpoint(ctx, pred, rng, n_perm):
            rows.append({"arch": run.arch, "condition": run.condition, "seed": run.seed, "rho": run.rho, "wd": run.wd,
                         "tag": tag, "step": step, "grok_step": grok_step_of(run), **r})
    return pd.DataFrame(rows)


# The validation report: how often the BIC winner agrees with the held-out-likelihood winner, which
# models win with and without the three extra ones, the fitted ALCOVE attention weight, where the
# prototype model ranks, and whether the observed family Delta-BICs clear the permutation null.
# Input: df - the stacked per-run validation frames
# Output: str - the report text
def summarise(df: pd.DataFrame) -> str:
    L = ["Validation of the cognitive-model comparison (single-task add runs; checkpoints nearest 0.5, 1, 2 x grok and the final dump)"]
    L.append(f"runs: {df.groupby(['condition','seed']).ngroups}; checkpoints: {df.groupby(['condition','seed','tag']).ngroups}; models: {sorted(df.model.unique())}")
    std = ["uniform", "gcm_hamming", "gcm_circular", "gcm_linear", "rule", "rulex"]
    for tag in TAGS:
        d = df[df.tag == tag]
        if d.empty:
            continue
        L.append(f"\n[{tag}]  n checkpoints = {d.groupby(['condition','seed']).ngroups}")
        agree = fam_same = rich_changes = 0; n = 0; winners_bic, winners_cv, winners_rich = {}, {}, {}
        alc_w, alc_beats, mix_wins = [], 0, 0
        for (_, _), g in d.groupby(["condition", "seed"]):
            g6 = g[g.model.isin(std)]
            bb = g6.loc[g6.bic.idxmin(), "model"]; bc = g6.loc[g6.cvll.idxmax(), "model"]
            agree += bb == bc; fam_same += FAMILY[bb] == FAMILY[bc]; n += 1
            winners_bic[bb] = winners_bic.get(bb, 0) + 1; winners_cv[bc] = winners_cv.get(bc, 0) + 1
            b9 = g.loc[g.bic.idxmin(), "model"]; winners_rich[b9] = winners_rich.get(b9, 0) + 1
            rich_changes += FAMILY[b9] != FAMILY[bb]
            alc = g[g.model == "alcove_lite"].iloc[0]; alc_w.append(alc.w)
            alc_beats += alc.bic < g[g.model == "gcm_circular"].bic.iloc[0]
            mix_wins += b9 == "atrium_lite"
        L.append(f"  best-by-BIC == best-by-held-out-LL (six standard models): {agree}/{n} same model, {fam_same}/{n} same family")
        L.append(f"  winners by BIC: {winners_bic}; by held-out LL: {winners_cv}")
        L.append(f"  with the three extra models: winners {winners_rich}; family of the winner changes in {rich_changes}/{n}; ATRIUM-lite wins {mix_wins}/{n}")
        L.append(f"  ALCOVE-lite: fitted attention w median {np.median(alc_w):.2f} (IQR {np.percentile(alc_w,25):.2f}-{np.percentile(alc_w,75):.2f}); beats the equal-attention circular GCM by BIC in {alc_beats}/{n}")
        pr = d[d.model == "prototype"]; g6 = d[d.model.isin(std)]
        rank = np.mean([ (g[g.model.isin(std + ['prototype'])].sort_values('bic').model.tolist().index('prototype') + 1) for _, g in d.groupby(['condition','seed']) ])
        L.append(f"  prototype model: mean BIC rank among 7 models = {rank:.1f} (7 = worst); median dBIC(prototype - uniform)/n = {np.median((pr.bic.values - d[d.model=='uniform'].bic.values) / pr.n_probe.values):.3f} nats/probe")
        one = d.groupby(["condition", "seed"]).first()
        L.append(f"  permutation null ({int(50)} shuffles of the answers over probes): observed dBIC(uniform - gcm_hamming) exceeds the null maximum in "
                 f"{int((one.obs_dbic_exemplar > one.perm_dbic_exemplar_max).sum())}/{len(one)} checkpoints (null max median {one.perm_dbic_exemplar_max.median():.1f} nats, observed median {one.obs_dbic_exemplar.median():.0f}); "
                 f"dBIC(gcm - rule family) exceeds the null maximum in {int((one.obs_dbic_rulefam > one.perm_dbic_rulefam_max).sum())}/{len(one)} (null max median {one.perm_dbic_rulefam_max.median():.1f}, observed median {one.obs_dbic_rulefam.median():.0f})")
    return "\n".join(L)
