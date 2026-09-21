"""Rule-vs-exemplar grokking experiment — TRAINING driver for the three
architectures (transformer / mamba / kimi via grok.models, ONE size variant
everywhere: matched_2M, asserted 2.0M ± 10% at startup). The vanilla RNN was
removed from the pipeline 2026-08-29: at matched_2M under the unified knobs it
never fit the training set (max train acc = chance over 12k+ steps).

Tracks
------
  Track A  (exceptions, single-task):
    A1  {arch} x add x rho {0, 0.02, 0.05} x WD {0.1, 0.3}, 3 seeds  (54 runs)
        pass 1: rho {0, 0.02} x WD 0.1     (core story per arch)
        pass 2: rho 0.05      x WD 0.1     (phase boundary)
        pass 3: rho {0,.02,.05} x WD 0.3   (WD knob / erosion)
    A2  {arch} x {div, max} x rho {0, 0.02} x WD 0.1, 3 seeds        (36 runs)
  Track B' (grok order, joint-task {add, div, max}, no exceptions):
    B1  {arch} x joint x rho=0 x WD 0.3, 3 seeds                     (9 runs)
        PER_TASK_BATCH = 512 // 3; grok detection, threshold snapshots and
        preds dumps run PER TASK.

Stopping rules (no fixed MAX_STEPS)
-----------------------------------
  single-task, rho > 0 : stop at max(10 * grok_step, grok_step + 2000)
  single-task, rho = 0 : early stop at grok_step + 1000
  joint-task           : stop when ALL tasks grokked, + 1000 margin
  hard cap             : 20000 (single) / 30000 (joint); hitting the cap
                         without grok records non_grokking=True in the
                         manifest — a phase-boundary datapoint, never a
                         retune trigger.

Usage (run each architecture separately / on separate machines)
---------------------------------------------------------------
  python3 grok/exceptions_experiment.py --phase selftest
  python3 grok/exceptions_experiment.py --phase a1 --arch kimi --pass 1
  python3 grok/exceptions_experiment.py --phase a2 --arch all
  python3 grok/exceptions_experiment.py --phase b1 --arch mamba
  python3 grok/exceptions_experiment.py --phase single --arch transformer --task add --rho 0.02 --wd 0.1 --seed 1000 [--cap 20000]
  --arch takes one or more names, or "all".
  --variant matched_2M_pos (learned positional embeddings) must go to a --base-out other than ./exceptions_runs.
  --phase selftest runs on CPU; every other phase needs CUDA.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models.common import device
from ..models.zoo import VARIANTS
from ..train.config import JOINT_TASKS, MATRIX_SEEDS
from ..train.loop import train_seed
from ..train.naming import parse_archs, run_dir_for
from ..train.phases import phase_a1, phase_a2, phase_b1, phase_selftest


# Entry point of the training driver: parses the phase, architectures, seeds and variant, then runs
# the requested matrix phase or a single explicitly specified run.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - trains the runs into --base-out and updates the manifest
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["selftest", "a1", "a2", "b1", "single"])
    ap.add_argument("--arch", nargs="+", default=["all"], help="one or more of transformer/mamba/kimi, or 'all'")
    ap.add_argument("--base-out", type=Path, default=Path("./exceptions_runs"))
    ap.add_argument("--variant", default="matched_2M", choices=list(VARIANTS),
                    help="model variant (grok.models.zoo.VARIANTS). matched_2M_pos = same size + learned positional "
                         "embeddings; must go to a --base-out other than ./exceptions_runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(MATRIX_SEEDS))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--task", type=str, default="add", choices=["add", "div", "max"])
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--pass", dest="passes", type=int, nargs="+", default=[1, 2, 3], help="A1 execution pass(es): 1, 2, 3")
    ap.add_argument("--cap", type=int, default=None, help="override the hard cap (default 20000 single / 30000 joint)")
    ap.add_argument("--joint", action="store_true", help="single phase: run the joint task set instead of --task")
    ap.add_argument("--no-stop-rule", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()
    variant = args.variant
    if variant != "matched_2M" and args.base_out.resolve() == Path("./exceptions_runs").resolve():
        raise SystemExit(f"variant {variant} is an ablation: give it its own --base-out (e.g. ./exceptions_runs_pos), "
                         "never ./exceptions_runs")

    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    if args.phase == "selftest":
        phase_selftest(variant)
        return
    assert device.type == "cuda", "CUDA is not found"

    archs = parse_archs(args.arch)
    args.base_out.mkdir(parents=True, exist_ok=True)

    if args.phase == "a1":
        phase_a1(args.base_out, archs, passes=args.passes, seeds=args.seeds, cap=args.cap, variant=variant)
    elif args.phase == "a2":
        phase_a2(args.base_out, archs, seeds=args.seeds, cap=args.cap, variant=variant)
    elif args.phase == "b1":
        phase_b1(args.base_out, archs, seeds=args.seeds, cap=args.cap, variant=variant)
    elif args.phase == "single":
        assert args.rho is not None and args.wd is not None, "--rho and --wd required"
        assert len(archs) == 1, "single phase takes exactly one --arch"
        tasks = list(JOINT_TASKS) if args.joint else [args.task]
        d = run_dir_for(args.base_out, archs[0], tasks, args.rho, args.wd, args.seed)
        train_seed(archs[0], tasks, args.seed, d, rho=args.rho, wd=args.wd, cap=args.cap,
                   disable_stop_rule=args.no_stop_rule, base_out=args.base_out,
                   force_retrain=args.force_retrain, variant=variant)


if __name__ == "__main__":
    main()
