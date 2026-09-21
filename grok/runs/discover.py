from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.tasks import ARCHS
from .records import Run, run_from_dir

NON_RUN_PARTS = frozenset({"analysis_offline", "paper", "lesions"})


# Finds every run under a run tree - the one function that replaces the four near-copies the old
# scripts each carried. Directories of removed architectures are skipped and left untouched.
# Input: base_out - Path of the run tree; pattern - regex over the full path, or None;
# seeds - iterable of seeds to keep, or None; kind - "any" | "single" | "joint";
# verbose - report the skipped directories
# Output: tuple[Run, ...] in sorted path order
def find_runs(base_out: Path, pattern: str | None = None, seeds=None, kind: str = "any",
              verbose: bool = False) -> tuple[Run, ...]:
    base_out = Path(base_out)
    wanted_seeds = {int(s) for s in seeds} if seeds else None
    out = []
    for cfg_path in sorted(base_out.rglob("run_config.json")):
        run_dir = cfg_path.parent
        if NON_RUN_PARTS & set(run_dir.relative_to(base_out).parts):
            continue
        if pattern and not re.search(pattern, str(run_dir)):
            continue
        arch = json.loads(cfg_path.read_text()).get("arch")
        if arch not in ARCHS:
            if verbose:
                print(f"[find_runs] SKIP {run_dir} — arch {arch!r} is not in the pipeline")
            continue
        run = run_from_dir(run_dir, base_out)
        if wanted_seeds and run.seed not in wanted_seeds:
            continue
        if (kind == "single" and run.is_joint) or (kind == "joint" and not run.is_joint):
            continue
        out.append(run)
    return tuple(out)
