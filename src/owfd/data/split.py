"""Leak-free train/val/test splits: real instances are split *by well*; val/test are real only.

Usage::

    python -m owfd.data.split  # writes configs/splits.json (commit it; later tasks reuse it)

No well appears in more than one split, so the model is always evaluated on wells it has
never seen. Simulated and hand-drawn instances only ever go to training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from owfd.config import SplitConfig, get_settings
from owfd.data.schema import Source

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def classes_without_synthetic(index: pd.DataFrame) -> set[int]:
    """Event classes that only exist in real data (their real wells must stay in training)."""
    synthetic = set(index.loc[index["source"] != Source.REAL.value, "event_class"])
    return set(index["event_class"]) - synthetic


def _score(
    assign: np.ndarray, counts: np.ndarray, config: SplitConfig, must_train: np.ndarray
) -> float:
    """Higher is better. ``counts`` is wells x classes; ``assign`` maps well -> 0/1/2."""
    per_split = np.stack([counts[assign == s].sum(axis=0) for s in range(3)])
    present = per_split > 0
    if not present[0, must_train].all():
        return -np.inf
    shares = per_split.sum(axis=1) / counts.sum()
    coverage = 3.0 * present[2].sum() + 2.0 * present[1].sum()
    size_penalty = abs(shares[2] - config.test_frac) + abs(shares[1] - config.val_frac)
    return float(coverage - 50.0 * size_penalty)


def assign_wells(
    real: pd.DataFrame, config: SplitConfig, seed: int, must_train: set[int]
) -> dict[str, str]:
    """Search random well -> split assignments; keep the one with the best class coverage.

    Coverage of every class in test (then val) is rewarded; val/test sizes (as shares of real
    instances) are kept near the configured fractions; classes in ``must_train`` must keep at
    least one real well in training.
    """
    table = pd.crosstab(real["well_id"], real["event_class"])
    counts = table.to_numpy()
    must = np.isin(table.columns.to_numpy(), sorted(must_train))
    rng = np.random.default_rng(seed)
    best_score, best = -np.inf, None
    for _ in range(config.n_trials):
        r = rng.random(len(table))
        assign = np.where(
            r < config.test_frac, 2, np.where(r < config.test_frac + config.val_frac, 1, 0)
        )
        score = _score(assign, counts, config, must)
        if score > best_score:
            best_score, best = score, assign
    if best is None:
        raise RuntimeError("No valid well assignment found; increase split.n_trials")
    return {well: SPLITS[s] for well, s in zip(table.index, best, strict=True)}


def make_splits(index: pd.DataFrame, config: SplitConfig, seed: int) -> dict[str, Any]:
    """Build the split manifest: wells and instance ids per split."""
    real = index[index["source"] == Source.REAL.value]
    wells = assign_wells(real, config, seed, classes_without_synthetic(index))
    split_of = index["well_id"].map(wells).fillna("train")  # synthetic -> train
    return {
        "seed": seed,
        "config": config.model_dump(),
        "wells": {s: sorted(w for w, a in wells.items() if a == s) for s in SPLITS},
        "instances": {s: sorted(index.loc[split_of == s, "instance_id"]) for s in SPLITS},
    }


def validate_splits(splits: dict[str, Any], index: pd.DataFrame) -> None:
    """Raise ``ValueError`` on any leakage or inconsistency between the splits."""
    inst = {s: set(splits["instances"][s]) for s in SPLITS}
    wells = {s: set(splits["wells"][s]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if inst[a] & inst[b]:
                raise ValueError(f"Instances shared by {a} and {b}")
            if wells[a] & wells[b]:
                raise ValueError(f"Wells shared by {a} and {b}")
    if set().union(*inst.values()) != set(index["instance_id"]):
        raise ValueError("Splits do not cover exactly the indexed instances")
    by_id = index.set_index("instance_id")
    for s in ("val", "test"):
        if (by_id.loc[sorted(inst[s]), "source"] != Source.REAL.value).any():
            raise ValueError(f"{s} must contain real instances only")
    for s in SPLITS:
        used = set(by_id.loc[sorted(inst[s]), "well_id"].dropna())
        if not used <= wells[s]:
            raise ValueError(f"{s} contains instances from wells assigned elsewhere")


def split_of_instances(splits: dict[str, Any]) -> dict[str, str]:
    """instance_id -> split name."""
    return {i: s for s in SPLITS for i in splits["instances"][s]}


def summarize_splits(splits: dict[str, Any], index: pd.DataFrame) -> pd.DataFrame:
    """Instances per event class (rows) and split/source (columns)."""
    df = index.assign(split=index["instance_id"].map(split_of_instances(splits)))
    table = pd.crosstab(df["event_class"], [df["split"], df["source"]], margins=True)
    return table.reindex(columns=[c for c in table.columns if c[0] in SPLITS or c[0] == "All"])


def save_splits(splits: dict[str, Any], path: Path) -> None:
    """Write the manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1) + "\n")


def load_splits(path: Path) -> dict[str, Any]:
    """Read a manifest written by :func:`save_splits`."""
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description="Create configs/splits.json.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    index = pd.read_parquet(settings.resolve(settings.processed_data_dir) / "instances.parquet")
    splits = make_splits(index, settings.split, settings.random_seed)
    validate_splits(splits, index)
    save_splits(splits, settings.resolve(settings.splits_file))
    for s in SPLITS:
        logger.info("%s wells: %s", s, ", ".join(splits["wells"][s]))
    print(summarize_splits(splits, index).to_string())


if __name__ == "__main__":
    main()
