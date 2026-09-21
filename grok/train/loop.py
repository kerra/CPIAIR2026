from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..cognitive.exceptions import exception_metrics
from ..core.tasks import P, TASK_OP
from ..models.common import assert_param_budget, device
from ..models.zoo import batched_logits, make_model
from .artifacts import (check_or_write_run_config, dump_preds, init_crossings, load_or_create_splits, make_snap_steps,
                        persist_or_assert_exceptions, preds_due, save_snapshot, save_training_state, update_manifest)
from .config import (ACC_THRESHOLDS, BATCH, BETAS, CAP_JOINT, CAP_SINGLE, GRAD_CLIP, GROK_THRESH, JOINT_DENSE_EVERY,
                     JOINT_DENSE_FROM, JOINT_DENSE_UNTIL, LR, MARGIN_JOINT, MARGIN_RHO0, METRICS_EVERY,
                     OVERREG_THRESHOLDS, PREDS_EVERY, RHOPOS_GROK_ADD, RHOPOS_GROK_MULT, SAVE_STATE_EVERY,
                     SINGLE_DENSE_EVERY, SINGLE_DENSE_FROM, SINGLE_DENSE_UNTIL, WARMUP_STEPS)
from .data import add_exceptions_to_train, batch_indices, eval_task
from .naming import condition_name


# The grok-relative stopping rule. A joint run stops once every task has grokked plus a margin; a
# single-task run stops at max(10*grok, grok + 2000) when rho > 0, and at grok + 1000 when rho = 0.
# Input: tasks - sequence of task names; rho - exception fraction; grok_step - {task: step or None};
# step - the current step
# Output: tuple (bool should_stop, str reason - empty when not stopping)
def stop_decision(tasks, rho: float, grok_step: dict, step: int):
    if len(tasks) > 1:
        if all(v is not None for v in grok_step.values()):
            if step >= max(grok_step.values()) + MARGIN_JOINT:
                return True, f"all tasks grokked + {MARGIN_JOINT}"
        return False, ""
    g = grok_step[tasks[0]]
    if g is None:
        return False, ""
    if rho > 0:
        stop_at = max(RHOPOS_GROK_MULT * g, g + RHOPOS_GROK_ADD)
        if step >= stop_at:
            return True, f"max(10*grok, grok+{RHOPOS_GROK_ADD}) = {stop_at}"
    else:
        if step >= g + MARGIN_RHO0:
            return True, f"grok + {MARGIN_RHO0}"
    return False, ""


# Trains one seed of one condition into out_dir. Resumable: an existing final.pt is returned as is,
# and a last_training_state.pt is picked up and continued from. Along the way it writes threshold
# snapshots, prediction dumps, the per-run summary and the manifest row.
# Input: arch - architecture; tasks - one or more task names; seed - int; out_dir - Path;
# rho - exception fraction; wd - weight decay; cap - step cap, defaulting to the per-kind cap;
# dense_window - (every, from, until) for the preds dumps; disable_stop_rule - train to the cap;
# base_out - run tree root for the manifest; force_retrain - ignore existing checkpoints;
# resume_from_last_state - pick up a training state; variant - model variant (models.zoo.VARIANTS)
# Output: dict - the final checkpoint
def train_seed(arch: str, tasks, seed: int, out_dir: Path, rho: float, wd: float,
               cap: Optional[int] = None,
               dense_window: Optional[tuple] = None,   # (every, from, until)
               disable_stop_rule: bool = False,
               base_out: Optional[Path] = None,
               force_retrain: bool = False,
               resume_from_last_state: bool = True,
               variant: str = "matched_2M"):
    tasks = list(tasks)
    joint = len(tasks) > 1
    if joint:
        assert rho == 0.0, "Track B' (joint) runs carry no exceptions"
    if cap is None:
        cap = CAP_JOINT if joint else CAP_SINGLE
    if dense_window is None:
        dense_window = ((JOINT_DENSE_EVERY, JOINT_DENSE_FROM, JOINT_DENSE_UNTIL) if joint
                        else (SINGLE_DENSE_EVERY, SINGLE_DENSE_FROM, SINGLE_DENSE_UNTIL))

    cond = condition_name(arch, tasks, rho, wd)
    print("=" * 70)
    print(f"{cond} | seed={seed} | cap={cap} | stop_rule="
          f"{'OFF (extend)' if disable_stop_rule else 'grok-relative'}")
    print("=" * 70)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final.pt"
    last_state_path = out_dir / "last_training_state.pt"

    run_cfg = check_or_write_run_config(out_dir, arch, tasks, seed, rho, wd, *dense_window, variant=variant)
    preds_every = int(run_cfg.get("preds_every", PREDS_EVERY))
    d_every = int(run_cfg.get("preds_dense_every", 0))
    d_from = int(run_cfg.get("preds_dense_from", 0))
    d_until = int(run_cfg.get("preds_dense_until", 0))

    if final_path.exists() and not force_retrain:
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        print("loaded existing final checkpoint:", final_path)
        print("grok:", ckpt.get("grok_step"), "| final accs:", ckpt.get("final_accs"))
        return ckpt

    # ---- per-run data: corrupt FRESH from the clean split, never re-corrupt ----
    splits = load_or_create_splits(out_dir, tasks)
    data: dict[str, tuple] = {}          # {task: (train tuple as the network sees it, test tuple)}
    EXC = None
    for t in tasks:
        tr_clean, te = splits[t]
        if not joint and rho > 0 and t == tasks[0]:
            tr, EXC = add_exceptions_to_train(tr_clean, task=t, P=P, exception_frac=rho, exception_seed=seed)
            persist_or_assert_exceptions(out_dir, EXC)
            data[t] = (tr, te)   # eval split="train" sees corrupted labels (intended)
        else:
            data[t] = (tr_clean, te)
    exc_task = tasks[0] if EXC is not None else None
    if EXC is not None:
        print(f"exceptions on '{exc_task}': n={EXC.n} ({rho:g} of {len(data[exc_task][0][0])})")

    model = make_model(arch, seed, variant)
    n_params = assert_param_budget(model, arch)
    print(f"{arch} params: {n_params:,} ({n_params / 1e6:.3f}M, matched_2M ok)")

    per_task_batch = BATCH // len(tasks)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=BETAS)
    pg = torch.Generator(device=device)
    pg.manual_seed(seed + 7)

    metrics = []
    grok_step = {t: None for t in tasks}
    crossings = init_crossings(tasks, EXC is not None)
    snap_steps_saved = []
    start_step = 1

    if resume_from_last_state and last_state_path.exists() and not force_retrain:
        state = torch.load(last_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        opt.load_state_dict(state["optimizer_state_dict"])
        metrics = state.get("metrics", [])
        grok_step = state.get("grok_step", {t: None for t in tasks})
        crossings = state.get("threshold_crossings", init_crossings(tasks, EXC is not None))
        snap_steps_saved = state.get("snap_steps_saved", [])
        start_step = int(state["step"]) + 1
        print(f"RESUMED from {last_state_path}; continuing from step {start_step}; "
              f"grok={grok_step}")

    snap_set = set(make_snap_steps(cap))
    preds_root = out_dir / "preds"

    t0 = time.time()
    last_log = t0
    stop_reason = "cap"

    step = start_step - 1   # keep defined if the loop body never runs
    for step in range(start_step, cap + 1):
        model.train()

        if step <= WARMUP_STEPS:
            for group in opt.param_groups:
                group["lr"] = LR * step / WARMUP_STEPS

        total_loss = None
        for t in tasks:
            tr_t = data[t][0]
            pm = batch_indices(t == exc_task and EXC is not None, len(tr_t[0]), per_task_batch, device, pg)
            logits = model(tr_t[0][pm], TASK_OP[t], tr_t[1][pm])
            loss = F.cross_entropy(logits, tr_t[2][pm])
            total_loss = loss if total_loss is None else total_loss + loss

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        want_snapshot = step in snap_set

        if step % METRICS_EVERY == 0 or step == 1:
            row = {"step": step, "loss": float(total_loss.detach().item())}
            for t in tasks:
                test_acc = eval_task(model, data, t, split="test")
                row[f"test_{t}"] = test_acc
                row[f"train_{t}"] = eval_task(model, data, t, split="train")
                if grok_step[t] is None and test_acc > GROK_THRESH:
                    grok_step[t] = step
                    print(f"  step {step}: {t.upper()} grokked (test > {GROK_THRESH})")
                for thr in ACC_THRESHOLDS:
                    key = f"test_{t}>={thr:g}"
                    if crossings.get(key) is None and test_acc >= thr:
                        crossings[key] = step
                        want_snapshot = True
                        print(f"  step {step}: crossed {key} -> snapshot")
            if EXC is not None:
                logits_exc = batched_logits(model, EXC.a, TASK_OP[exc_task], EXC.b, device)
                row.update(exception_metrics(logits_exc, EXC))
                for thr in OVERREG_THRESHOLDS:
                    key = f"overreg_rate>={thr:g}"
                    if crossings.get(key) is None and row["overreg_rate"] >= thr:
                        crossings[key] = step
                        want_snapshot = True
                        print(f"  step {step}: crossed {key} -> snapshot")
            metrics.append(row)

        if want_snapshot and step not in snap_steps_saved:
            save_snapshot(model, out_dir / "snapshots", step)
            snap_steps_saved.append(step)
            print(f"  saved snapshot at step {step}")

        if preds_due(step, preds_every, d_every, d_from, d_until) or want_snapshot:
            for t in tasks:
                preds_path = preds_root / t / f"step_{step:06d}.npz"
                if not preds_path.exists():
                    dump_preds(model, t, preds_path)

        if step % SAVE_STATE_EVERY == 0 or step == cap:
            save_training_state(model, opt, step, metrics, grok_step, crossings,
                                snap_steps_saved, out_dir)

        if time.time() - last_log > 30:
            if metrics:
                latest = metrics[-1]
                acc_str = " ".join(f"{t}={latest[f'test_{t}']:.3f}" for t in tasks)
                if "acc_exc" in latest:
                    acc_str += (f" acc_exc={latest['acc_exc']:.3f} "
                                f"overreg={latest['overreg_rate']:.3f}")
            else:
                acc_str = ""
            print(f"  step {step:6d}: loss={total_loss.detach().item():.3f} | {acc_str} | "
                  f"grok={grok_step} | {time.time() - t0:.0f}s "
                  f"({(step - start_step + 1) / max(time.time() - t0, 1e-9):.2f} steps/s)")
            last_log = time.time()

        if not disable_stop_rule:
            should_stop, reason = stop_decision(tasks, rho, grok_step, step)
            if should_stop:
                stop_reason = reason
                print(f"  step {step}: stop rule hit — {reason}")
                break

    steps_run = step
    wall = time.time() - t0
    s_per_step = wall / max(steps_run - start_step + 1, 1)
    non_grokking = any(v is None for v in grok_step.values())
    if non_grokking and steps_run >= cap:
        stop_reason = f"cap {cap} WITHOUT grok (phase-boundary datapoint, do not retune)"

    # capture the endpoint even when the stop rule cut before the cap
    if steps_run >= 1 and steps_run not in snap_steps_saved:
        save_snapshot(model, out_dir / "snapshots", steps_run)
        snap_steps_saved.append(steps_run)
    if steps_run >= 1:
        for t in tasks:
            preds_path = preds_root / t / f"step_{steps_run:06d}.npz"
            if not preds_path.exists():
                dump_preds(model, t, preds_path)

    # self-verify preds dumps per task against the recorded schedule
    expected = set(s for s in range(preds_every, steps_run + 1, preds_every))
    if d_every > 0:
        expected |= {s for s in range(d_every, min(d_until, steps_run) + 1, d_every)
                     if s >= d_from}
    expected = sorted(expected | set(snap_steps_saved))
    n_preds_total = 0
    for t in tasks:
        on_disk = sorted(int(p.stem.split("_")[1])
                         for p in (preds_root / t).glob("step_*.npz"))
        if on_disk != expected:
            raise AssertionError(
                f"preds[{t}] on disk != recorded schedule in {out_dir}: "
                f"missing {sorted(set(expected) - set(on_disk))}, "
                f"unexpected {sorted(set(on_disk) - set(expected))}")
        n_preds_total += len(on_disk)
    grid_txt = (f"dense({d_every} in [{d_from},{d_until}]) + coarse({preds_every})"
                if d_every > 0 else f"coarse({preds_every})")
    print(f"  [preds] {n_preds_total} dumps ({len(expected)}/task x {len(tasks)} task"
          f"{'s' if len(tasks) > 1 else ''}) == {grid_txt} grid ∪ "
          f"{len(snap_steps_saved)} snapshots (self-verified)")

    final_accs = {t: eval_task(model, data, t, split="test") for t in tasks}
    final_train = {t: eval_task(model, data, t, split="train") for t in tasks}
    final_exc = {}
    if EXC is not None:
        logits_exc = batched_logits(model, EXC.a, TASK_OP[exc_task], EXC.b, device)
        final_exc = exception_metrics(logits_exc, EXC)

    ckpt = {
        "state_dict": {name: p.detach().cpu().clone() for name, p in model.named_parameters()},
        "metrics": metrics,
        "grok_step": grok_step,
        "final_accs": final_accs,
        "config": {**run_cfg, "cap": cap, "n_params": n_params,
                   "stop_reason": stop_reason, "non_grokking": non_grokking},
        "threshold_crossings": crossings,
        "snap_steps": sorted(snap_steps_saved),
    }
    torch.save(ckpt, final_path)
    save_training_state(model, opt, steps_run, metrics, grok_step, crossings,
                        snap_steps_saved, out_dir)

    summary = {
        "condition": cond, "arch": arch, "tasks": tasks, "variant": variant,
        "seed": seed, "rho": rho, "wd": wd, "freq_tiers": None,     # key kept for the stored summaries' format
        "n_params": n_params,
        "grok_step": grok_step, "non_grokking": non_grokking,
        "stop_reason": stop_reason, "cap": cap,
        "final_accs": final_accs, "final_train_accs": final_train,
        "final_exception_metrics": final_exc,
        "threshold_crossings": crossings,
        "steps_run": steps_run, "wall_seconds": wall, "s_per_step": s_per_step,
        "snap_steps": sorted(snap_steps_saved),
        "n_snapshots": len(snap_steps_saved),
        "n_preds_dumps": n_preds_total,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if base_out is not None:
        row = {
            "condition": cond, "arch": arch, "tasks": "+".join(tasks),
            "variant": variant, "rho": rho, "wd": wd,
            "tiers": "None", "seed": seed,
            "out_dir": str(out_dir), "status": "done",
            "n_params": n_params,
            "steps_run": steps_run, "cap": cap,
            "non_grokking": non_grokking, "stop_reason": stop_reason,
            "s_per_step": round(s_per_step, 4),
            "wall_seconds": round(wall, 1),
            "n_snapshots": len(snap_steps_saved),
            "n_preds_dumps": n_preds_total,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for t in tasks:
            row[f"grok_step_{t}"] = grok_step[t]
            row[f"final_test_{t}"] = final_accs[t]
            row[f"final_train_{t}"] = final_train[t]
        row["final_acc_exc"] = final_exc.get("acc_exc")
        row["final_overreg_rate"] = final_exc.get("overreg_rate")
        update_manifest(base_out, row)

    print(f"done {cond} seed={seed}; grok={grok_step}; "
          f"non_grokking={non_grokking}; final={ {k: round(v, 4) for k, v in final_accs.items()} }; "
          f"exc={ {k: round(v, 4) for k, v in final_exc.items()} }; "
          f"{n_preds_total} preds dumps; {len(snap_steps_saved)} snapshots; "
          f"wall={wall:.0f}s ({s_per_step:.3f} s/step)")

    del model, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt
