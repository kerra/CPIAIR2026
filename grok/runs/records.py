from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# The immutable record of one training run (one seed of one condition) as it sits on disk, at
# <base_out>/<condition>/seed_<s>/ with run_config.json, summary.json, final.pt, snapshots/,
# preds/ and analysis_offline/.
@dataclass(frozen=True)
class Run:
    dir: Path
    condition: str
    arch: str
    tasks: tuple[str, ...]
    seed: int
    rho: float
    wd: float
    variant: str = "matched_2M"

    @property
    def is_joint(self) -> bool:
        return len(self.tasks) > 1

    # The single task of a single-task run - the exception-carrying task whenever rho > 0.
    # Input: none
    # Output: str - the task name; raises ValueError on a joint run, where it is ambiguous
    @property
    def task(self) -> str:
        if self.is_joint:
            raise ValueError(f"{self.label} is a joint run ({'+'.join(self.tasks)}); name the task explicitly")
        return self.tasks[0]

    @property
    def label(self) -> str:
        return f"{self.condition} seed {self.seed}"

    # Dict-style access, kept for the call sites that were moved verbatim from the old scripts.
    # Input: key - attribute name, or "task"
    # Output: the attribute value
    def __getitem__(self, key: str):
        return self.task if key == "task" else getattr(self, key)


# Builds the run record from <run_dir>/run_config.json; the condition is the parent directory.
# Input: run_dir - Path of the seed directory; base_out - Path of the run tree root
# Output: Run
def run_from_dir(run_dir: Path, base_out: Path) -> Run:
    cfg = json.loads((run_dir / "run_config.json").read_text())
    rel = run_dir.relative_to(base_out)
    cond = str(rel.parent) if str(rel.parent) != "." else run_dir.parent.name
    return Run(dir=run_dir, condition=cond, arch=cfg["arch"], tasks=tuple(cfg["tasks"]), seed=int(cfg["seed"]),
               rho=float(cfg["rho"]), wd=float(cfg["wd"]), variant=cfg.get("variant", "matched_2M"))
