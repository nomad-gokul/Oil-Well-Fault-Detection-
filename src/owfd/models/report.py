"""Keep the results table in ``docs/results.md`` up to date from saved metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"
HEADER = (
    "| Model | Split | Macro F1 | Balanced acc. | Fault ROC-AUC | Fault PR-AUC | Fault F1 "
    "| Windows |\n|---|---|---:|---:|---:|---:|---:|---:|"
)


def _row(name: str, split: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {split} | {m['macro_f1']:.3f} | {m['balanced_accuracy']:.3f} | "
        f"{m['fault_roc_auc']:.3f} | {m['fault_pr_auc']:.3f} | {m['fault_f1']:.3f} | {m['n']:,} |"
    )


def update_results(path: Path, name: str, metrics: dict[str, dict[str, Any]]) -> None:
    """Insert or replace the rows of model ``name`` (one per split) in the results table."""
    text = path.read_text() if path.exists() else "# Results\n"
    if BEGIN not in text:
        text = text.rstrip() + f"\n\n{BEGIN}\n{HEADER}\n{END}\n"
    head, rest = text.split(BEGIN, 1)
    table, tail = rest.split(END, 1)
    rows = [r for r in table.strip().splitlines()[2:] if not r.startswith(f"| {name} |")]
    rows += [_row(name, split, m) for split, m in metrics.items()]
    path.write_text(f"{head}{BEGIN}\n{HEADER}\n" + "\n".join(rows) + f"\n{END}{tail}")


def results_rows(path: Path) -> list[str]:
    """Rows currently in the results table (for tests and reports)."""
    match = re.search(f"{BEGIN}\n(.*?){END}", path.read_text(), re.S)
    return match.group(1).strip().splitlines()[2:] if match else []
