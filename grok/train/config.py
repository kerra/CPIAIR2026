from __future__ import annotations


METRICS_EVERY = 25


GROK_THRESH = 0.95


BATCH = 512


LR = 1e-3


BETAS = (0.9, 0.98)


GRAD_CLIP = 1.0


WARMUP_STEPS = 10


SPLIT_SEED = 0          # unified (kimi's legacy script used 42; all others 0)


SAVE_STATE_EVERY = 250


# stopping rules
CAP_SINGLE = 20000


CAP_JOINT = 30000


MARGIN_RHO0 = 1000


MARGIN_JOINT = 1000


RHOPOS_GROK_MULT = 10


RHOPOS_GROK_ADD = 2000


# preds dump schedules (recorded per run in run_config.json)
PREDS_EVERY = 250


SINGLE_DENSE_EVERY = 25


SINGLE_DENSE_FROM = 0


SINGLE_DENSE_UNTIL = 1500


JOINT_DENSE_EVERY = 25


JOINT_DENSE_FROM = 100        # the dense window the 9 joint runs were trained with


JOINT_DENSE_UNTIL = 4000


ACC_THRESHOLDS = (0.25, 0.50, 0.75, 0.90, 0.95)


OVERREG_THRESHOLDS = (0.25, 0.50, 0.75)


SNAPSHOT_FP16 = True    # storage budget: fp32 ~20MB/snapshot; analysis is inference-only


JOINT_TASKS = ("add", "div", "max")


MATRIX_SEEDS = (1000, 1001, 1002)


A1_PASSES = {
    1: [(0.0, 0.1), (0.02, 0.1)],
    2: [(0.05, 0.1)],
    3: [(0.0, 0.3), (0.02, 0.3), (0.05, 0.3)],
}


A2_TASKS = ("div", "max")


A2_RHOS = (0.0, 0.02)


A2_WD = 0.1


B1_WD = 0.3
