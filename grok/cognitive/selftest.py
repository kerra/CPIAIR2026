from __future__ import annotations

import numpy as np

from ..core.gcm import GCMProbeCache, gcm_votes, label_table, probe_mask
from ..core.tasks import n_classes, rule_label
from .exceptions import corrupt_train, exception_metrics
from .fits import compare_cognitive_models


# Self-test on three synthetic developmental stages of a fake network at P = 29: exemplar-like,
# pure rule, and rule plus memorised exceptions. Asserts that each stage is recovered by the right
# cognitive model, that the separable GCM kernel equals brute force, and that no model beats
# uniform at chance. Pure numpy, no torch; run it with `python3 -m grok.cognitive.selftest`.
# Input: none
# Output: None - prints a line per check and raises AssertionError on the first failure
def run_selftest() -> None:
    P_, task_ = 29, "add"
    K_ = n_classes(task_, P_)
    rng = np.random.default_rng(0)

    aa, bb = np.meshgrid(np.arange(P_), np.arange(P_), indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    cc = rule_label(task_, aa, bb, P_)
    perm = rng.permutation(len(aa)); half = len(aa) // 2
    tr_i, te_i = perm[:half], perm[half:]

    tr_c, EXC = corrupt_train(aa[tr_i], bb[tr_i], cc[tr_i], task_, P_, exception_frac=0.05, exception_seed=1)
    tr_a, tr_b = aa[tr_i], bb[tr_i]
    assert EXC.n == round(0.05 * half)
    assert np.all(tr_c[EXC.idx] == EXC.exc_label)
    assert np.all(EXC.exc_label != EXC.rule_label)
    print(f"[exceptions] n={EXC.n}, deranged labels ok")

    full_pred_rule = rule_label(task_, aa, bb, P_)
    preds_base = {"a": aa, "b": bb, "rule": full_pred_rule}
    train_args = dict(train_a=tr_a, train_b=tr_b, train_c=tr_c)

    # separable-kernel GCM (K.T.K) must equal brute force over train items
    T_ = label_table(tr_a, tr_b, tr_c, K_, P_)
    x_ = np.arange(P_, dtype=float)
    for metric_ in ("circular", "linear"):
        d_ = np.abs(x_[:, None] - x_[None, :])
        if metric_ == "circular":
            d_ = np.minimum(d_, P_ - d_)
        d_ = d_ / (P_ / 2.0)
        w_ = np.exp(-3.0 * (d_[aa][:, tr_a] + d_[bb][:, tr_b]))          # [n_table, n_train]
        oh_ = np.zeros((len(tr_c), K_)); oh_[np.arange(len(tr_c)), tr_c] = 1.0
        with np.errstate(all="ignore"):                                   # denormal noise in the BLAS matmul
            V_slow = (w_ @ oh_).reshape(P_, P_, K_)                      # meshgrid 'ij' ravel >> [a, b, c]
        V_fast = gcm_votes(T_, P_, K_, 3.0, metric_)
        assert np.allclose(V_fast, V_slow, rtol=1e-9, atol=1e-12), metric_
    print("[gcm kernel] K.T.K == brute force on both metrics")

    # common probe set: cached == direct, same n for all six models,
    # and at chance no model beats uniform (the old subsampling bug did)
    mask_ = probe_mask(aa, bb, tr_a, tr_b, EXC)
    cache_ = GCMProbeCache(tr_a, tr_b, tr_c, aa[mask_], bb[mask_], K_, P_)
    rnd_pred = rng.integers(0, K_, len(aa))
    r_d = compare_cognitive_models({**preds_base, "pred": rnd_pred}, EXC, P_, task_, **train_args)
    r_c = compare_cognitive_models({**preds_base, "pred": rnd_pred}, EXC, P_, task_,
                                   **train_args, gcm_cache=cache_)
    for m_ in ("gcm_circular", "gcm_linear"):
        assert abs(r_d[m_]["ll"] - r_c[m_]["ll"]) < 1e-3 * max(1.0, abs(r_d[m_]["ll"])), m_
        assert r_d[m_]["lambda"] == r_c[m_]["lambda"], m_
    assert r_d["n_probe"] == int(mask_.sum()) == r_d["gcm_circular"]["n_probe"]
    assert r_d["n_probe_exc"] == EXC.n and r_d["n_probe_test"] == len(te_i)
    assert r_d["best_model"] == "uniform", {m: round(r_d[m]["bic"] - r_d["uniform"]["bic"], 1)
                                            for m in ("gcm_hamming", "gcm_circular",
                                                      "gcm_linear", "rule", "rulex")}
    print(f"[probe set] test+exc: n={r_d['n_probe']} (test {r_d['n_probe_test']}, "
          f"exc {r_d['n_probe_exc']}); at chance best = {r_d['best_model']}")

    # Stage 1: exemplar-like network (GCM-hamming generator, lam=3.0)
    lam = 3.0
    cnt_a = np.zeros((P_, K_)); cnt_b = np.zeros((P_, K_))
    np.add.at(cnt_a, (tr_a, tr_c), 1.0); np.add.at(cnt_b, (tr_b, tr_c), 1.0)
    tot = cnt_a.sum(0)
    L = -np.ones((P_, P_), int); L[tr_a, tr_b] = tr_c
    S = np.zeros((len(aa), K_)); sl = L[aa, bb]; hs = sl >= 0
    S[np.where(hs)[0], sl[hs]] = 1.0
    n1 = cnt_a[aa] + cnt_b[bb] - 2 * S
    n2 = tot[None] - cnt_a[aa] - cnt_b[bb] + S
    votes = S + np.exp(-lam) * n1 + np.exp(-2 * lam) * n2
    p = votes / votes.sum(1, keepdims=True)
    stage1 = np.array([rng.choice(K_, p=pi) for pi in p])
    r1 = compare_cognitive_models({**preds_base, "pred": stage1}, EXC, P_, task_,
                                  **train_args)
    print("[stage 1 memorization] best =", r1["best_model"],
          f"(gcm lam_hat={r1['gcm_hamming']['lambda']:.2f})")
    assert r1["best_model"] == "gcm_hamming"
    assert np.isfinite(r1["gcm_linear"]["ll"]) and r1["gcm_linear"]["lambda"] > 0

    # Stage 2: pure rule (full overregularization of exceptions)
    stage2 = full_pred_rule.copy()
    r2 = compare_cognitive_models({**preds_base, "pred": stage2}, EXC, P_, task_,
                                  **train_args)
    logits2 = np.full((EXC.n, K_), -10.0)
    logits2[np.arange(EXC.n), EXC.rule_label] = 10.0
    m2 = exception_metrics(logits2, EXC)
    print("[stage 2 rule] best =", r2["best_model"],
          f"overreg={m2['overreg_rate']:.2f}")
    assert r2["best_model"] in ("rule", "rulex") and m2["overreg_rate"] == 1.0

    # Stage 3: rule + memorized exceptions (RULEX endpoint)
    stage3 = full_pred_rule.copy()
    key = {(int(x), int(y)): int(e) for x, y, e in zip(EXC.a, EXC.b, EXC.exc_label)}
    for i in range(len(aa)):
        e = key.get((int(aa[i]), int(bb[i])))
        if e is not None:
            stage3[i] = e
    r3 = compare_cognitive_models({**preds_base, "pred": stage3}, EXC, P_, task_,
                                  **train_args)
    logits3 = np.full((EXC.n, K_), -10.0)
    logits3[np.arange(EXC.n), EXC.exc_label] = 10.0
    m3 = exception_metrics(logits3, EXC)
    print("[stage 3 rulex] best =", r3["best_model"],
          f"m_hat={r3['rulex'].get('m'):.2f} acc_exc={m3['acc_exc']:.2f}")
    assert r3["best_model"] == "rulex" and m3["acc_exc"] == 1.0

    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    run_selftest()
