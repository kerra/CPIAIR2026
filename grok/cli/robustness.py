"""Robustness statistics for C1 / C2 (threshold sensitivity, cluster bootstrap, leave-one-architecture-out,
between/within cells, random-intercept model). Writes <base_out>/paper/supp/robustness_stats.txt and
robustness_sensitivity.csv and headline_intervals.csv. Usage: python3 -m grok.cli.robustness --base-out ./exceptions_runs"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..analysis.robustness import headline_intervals, report
from ..runs.discover import find_runs


# Entry point of the C1 / C2 robustness statistics: threshold sensitivity, cluster bootstrap,
# leave-one-architecture-out, the between/within split and the random-intercept model.
# Input: command-line arguments (see the module docstring, which is also the --help text)
# Output: None - writes robustness_stats.txt, robustness_sensitivity.csv and headline_intervals.csv
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-out", type=Path, required=True)
    args = ap.parse_args()
    out = args.base_out / "paper" / "supp"
    out.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(args.base_out / "paper" / "runs_table.csv")
    txt, sens = report(find_runs(args.base_out), tab)
    lesions = args.base_out / "paper" / "lesions" / "lesion_summary.csv"
    if lesions.exists():
        hi = headline_intervals(tab, pd.read_csv(lesions))
        hi.to_csv(out / "headline_intervals.csv", index=False)
        txt += "\n\n[headline intervals] 95% cluster-bootstrap (cells) intervals of the statistics quoted in the text\n" + hi.round(3).to_string(index=False)
    (out / "robustness_stats.txt").write_text(txt + "\n")
    sens.to_csv(out / "robustness_sensitivity.csv", index=False)
    print(txt)


if __name__ == "__main__":
    main()
