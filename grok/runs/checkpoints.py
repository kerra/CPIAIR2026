from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..models.common import device
from ..models.zoo import make_model
from .io import grok_step_of, threshold_steps_of
from .records import Run


# Inventory of the snapshots a run has on disk.
# Input: run - Run
# Output: dict[int, Path] - step number to snapshot file
def snapshot_steps(run: Run) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in sorted((run.dir / "snapshots").glob("step_*.pt"))}


# Resolves a requested checkpoint: "final" gives final.pt, an int gives the nearest snapshot,
# and "Xg" (e.g. "0.5g") gives the snapshot nearest to X times the grok step of the task.
# Input: run - Run; step_req - "final", an int, or an "Xg" string; task - task name,
# defaulting to the run's single task
# Output: tuple (step, Path), or (None, None) when the run never grokked or has no snapshots
def resolve_checkpoint(run: Run, step_req, task: str | None = None):
    if step_req == "final":
        p = run.dir / "final.pt"
        return ("final", p) if p.exists() else (None, None)
    if str(step_req).endswith("g"):
        g = grok_step_of(run, task)
        if g is None:
            return None, None
        want = float(str(step_req)[:-1]) * g
    else:
        want = int(step_req)
    snaps = snapshot_steps(run)
    if not snaps:
        return None, None
    step = min(snaps, key=lambda s: abs(s - want))
    return step, snaps[step]


# Rebuilds the run's architecture, seed and variant, loads the checkpoint weights and puts the
# model in eval mode - dropout is 0.2 in all three archs, so activations must never be captured
# in train mode. fp16 snapshots are cast into fp32 parameters.
# Input: run - Run; path - Path of the checkpoint
# Output: torch.nn.Module in eval mode on the active device
def load_model(run: Run, path: Path) -> torch.nn.Module:
    model = make_model(run.arch, run.seed, run.variant)
    sd = torch.load(path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# Thins a list of steps down to at most max_n, spaced evenly on a log axis.
# Input: steps - iterable of step numbers; max_n - cap
# Output: sorted list[int]
def subsample_log(steps, max_n: int):
    steps = sorted(set(int(s) for s in steps))
    if max_n <= 0:
        return []
    if len(steps) <= max_n:
        return steps
    logs = np.log10(np.maximum(np.array(steps, float), 1.0))
    targets = np.linspace(logs[0], logs[-1], max_n)
    idx = sorted(set(int(np.argmin(np.abs(logs - t))) for t in targets))
    return [steps[i] for i in idx]


# Chooses which checkpoints the offline analysis processes: every threshold-crossing step, plus
# coverage of the grok window, plus a log-spaced fill up to the cap. take_full_window keeps EVERY
# available step inside [0.5*grok, window_hi_mult*grok] at full resolution, log-spacing within the
# window only once the budget runs out.
# Input: run - Run; task - task name; available - the steps on disk; max_n - cap on the selection;
# min_in_grok_window - floor on in-window steps; what - label used in the printout;
# window_hi_mult - upper edge of the window as a multiple of grok; take_full_window - keep the
# whole window; outside_reserve - steps held back for outside the window
# Output: sorted list[int] - the selected steps; also prints the selection and every skip reason
def select_steps(run, task: str, available, max_n: int, min_in_grok_window: int = 6,
                 what: str = "checkpoints", window_hi_mult: float = 2.0,
                 take_full_window: bool = False, outside_reserve: int = 20):
    available = sorted(set(int(s) for s in available))
    if not available:
        return []
    must = [s for s in threshold_steps_of(run, task) if s in set(available)]
    sel = set(must)

    grok = grok_step_of(run, task)
    win_txt = "no grok step — window rule skipped"
    if grok is not None:
        window = (0.5 * grok, window_hi_mult * grok)
        in_win = [s for s in available if window[0] <= s <= window[1]]
        if take_full_window:
            must_outside = len([s for s in must if not (window[0] <= s <= window[1])])
            budget = max(min_in_grok_window, max_n - must_outside - outside_reserve)
            if len(in_win) <= budget:
                sel |= set(in_win)
                win_txt = (f"grok={grok}, window=[{window[0]:.0f}, {window[1]:.0f}]: "
                           f"ALL {len(in_win)} in-window steps kept (full resolution)")
            else:
                sel |= set(subsample_log(in_win, budget))
                win_txt = (f"grok={grok}, window=[{window[0]:.0f}, {window[1]:.0f}]: "
                           f"{budget}/{len(in_win)} in-window steps (log-spaced within "
                           f"window — cap {max_n} minus {outside_reserve} outside-reserve)")
        else:
            have = len([s for s in sel if window[0] <= s <= window[1]])
            if in_win and have < min_in_grok_window:
                sel |= set(subsample_log(in_win, min_in_grok_window))
            win_txt = (f"grok={grok}, window=[{window[0]:.0f}, {window[1]:.0f}] "
                       f"({len([s for s in sel if window[0] <= s <= window[1]])} selected inside)")

    fill_budget = max(0, max_n - len(sel))
    rest = [s for s in available if s not in sel]
    sel |= set(subsample_log(rest, fill_budget))
    sel = sorted(sel)

    n_skip = len(available) - len(sel)
    print(f"  [{what}] {len(sel)}/{len(available)} steps selected: {sel}")
    print(f"    threshold-crossing steps included: {must or 'none'}; {win_txt}")
    if n_skip:
        print(f"    skipped {n_skip} steps outside the transition by log-spaced "
              f"subsampling (cap {max_n}; raise --max-* to keep more)")
    return sel
