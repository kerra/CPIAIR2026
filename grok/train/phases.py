from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..cognitive.selftest import run_selftest
from ..core.tasks import ARCHS
from ..models.common import assert_param_budget
from ..models.zoo import make_model
from .config import A1_PASSES, A2_RHOS, A2_TASKS, A2_WD, B1_WD, JOINT_TASKS
from .loop import train_seed
from .naming import run_dir_for


# Cognitive-layer self-test (pure numpy) plus the parameter budget of all three models. CPU is fine.
# Input: variant - model variant to build
# Output: None - prints each check; raises AssertionError on the first failure
def phase_selftest(variant: str = "matched_2M") -> None:
    run_selftest()
    print("cognitive selftest OK")
    for arch in ARCHS:
        m = make_model(arch, seed=0, variant=variant)
        n = assert_param_budget(m, arch)
        print(f"  {arch}: {n:,} params ({variant} ok)")
        del m
    print("models selftest OK")


# Track A1: {arch} x add x rho {0, .02, .05} x WD {.1, .3}, in the three execution passes of A1_PASSES.
# Input: base_out - Path of the run tree; archs - architectures to train; passes - A1_PASSES keys;
# seeds - seeds to train; cap - step cap or None; variant - model variant
# Output: None - trains each run and updates the manifest
def phase_a1(base_out: Path, archs, passes, seeds, cap: Optional[int], variant: str = "matched_2M"):
    for arch in archs:
        for pass_id in passes:
            for rho, wd in A1_PASSES[pass_id]:
                for seed in seeds:
                    d = run_dir_for(base_out, arch, ["add"], rho, wd, seed)
                    train_seed(arch, ["add"], seed, d, rho=rho, wd=wd, cap=cap,
                               base_out=base_out, variant=variant)
    print("A1 requested passes done — see", Path(base_out) / "runs_manifest.csv")


# Track A2, the task-axis marker validation: {arch} x {div, max} x rho {0, .02} x WD .1.
# Input: base_out - Path of the run tree; archs - architectures; seeds - seeds; cap - step cap or
# None; variant - model variant
# Output: None - trains each run and updates the manifest
def phase_a2(base_out: Path, archs, seeds, cap: Optional[int], variant: str = "matched_2M"):
    for arch in archs:
        for task in A2_TASKS:
            for rho in A2_RHOS:
                for seed in seeds:
                    d = run_dir_for(base_out, arch, [task], rho, A2_WD, seed)
                    train_seed(arch, [task], seed, d, rho=rho, wd=A2_WD, cap=cap,
                               base_out=base_out, variant=variant)
    print("A2 done (task-axis marker validation).")


# Track B', the grok-order track: {arch} x joint {add, div, max} x rho 0 x WD .3.
# Input: base_out - Path of the run tree; archs - architectures; seeds - seeds; cap - step cap or
# None; variant - model variant
# Output: None - trains each run and updates the manifest
def phase_b1(base_out: Path, archs, seeds, cap: Optional[int], variant: str = "matched_2M"):
    for arch in archs:
        for seed in seeds:
            d = run_dir_for(base_out, arch, list(JOINT_TASKS), 0.0, B1_WD, seed)
            train_seed(arch, list(JOINT_TASKS), seed, d, rho=0.0, wd=B1_WD, cap=cap,
                       base_out=base_out, variant=variant)
    print("B1 done (grok-order track).")
