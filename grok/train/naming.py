from __future__ import annotations

from pathlib import Path

from ..core.tasks import ARCHS


# The task part of a condition name: "joint" for a multi-task run, otherwise the task itself.
# Input: tasks - sequence of task names
# Output: str
def tasks_tag(tasks) -> str:
    return "joint" if len(tasks) > 1 else tasks[0]


# The canonical condition name. The _tiersNone suffix is historical - it names a frequency-tier
# sampler the matrix never used - and is kept because it is part of the on-disk directory names.
# Input: arch - architecture; tasks - sequence of task names; rho - exception fraction; wd - weight decay
# Output: str - the condition directory name
def condition_name(arch: str, tasks, rho: float, wd: float) -> str:
    return f"{arch}_{tasks_tag(tasks)}_rho{rho:g}_wd{wd:g}_tiersNone"


# The directory one seed of one condition is trained into.
# Input: base_out - Path of the run tree; arch, tasks, rho, wd - the condition; seed - int
# Output: Path - <base_out>/<condition>/seed_<seed>
def run_dir_for(base_out: Path, arch: str, tasks, rho: float, wd: float, seed: int) -> Path:
    return Path(base_out) / condition_name(arch, tasks, rho, wd) / f"seed_{seed}"


# Expands the --arch CLI values, where "all" means every architecture in the pipeline.
# Input: vals - list of architecture names, or the single value "all"
# Output: list[str]; raises AssertionError on an unknown name
def parse_archs(vals) -> list:
    if len(vals) == 1 and vals[0].lower() == "all":
        return list(ARCHS)
    for v in vals:
        assert v in ARCHS, f"unknown arch {v!r}; expected {ARCHS} or 'all'"
    return list(vals)
