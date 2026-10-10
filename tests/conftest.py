"""Shared fixtures: a tiny synthetic 3W-like dataset (the real one is never used in tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from owfd.data.schema import SENSOR_VARIABLES, VALVE_STATE_VARIABLES


def make_instance(
    labels: list[int | None], start: str = "2017-01-01 00:00:00", seed: int = 0
) -> pd.DataFrame:
    """A 1 Hz instance in 3W v2 layout with the given per-second labels."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    index = pd.date_range(start, periods=n, freq="1s", name="timestamp")
    data = {col: rng.normal(1e6, 1e3, n) for col in SENSOR_VARIABLES}
    data |= {col: np.ones(n) for col in VALVE_STATE_VARIABLES}
    df = pd.DataFrame(data, index=index)
    df["P-JUS-BS"] = np.nan  # an always-missing variable, as in real 3W files
    df["class"] = pd.array(labels, dtype="Int16")
    df["state"] = pd.array([0 if lab is not None else None for lab in labels], dtype="Int16")
    return df


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """dataset/{0,1}/ with one real, one simulated and one drawn instance."""
    root = tmp_path / "dataset"
    files = {
        "0/WELL-00001_20170101000000.parquet": make_instance([0] * 10 + [None] * 5),
        "1/WELL-00002_20170102000000.parquet": make_instance(
            [0] * 5 + [101] * 5 + [1] * 5, start="2017-01-02"
        ),
        "1/SIMULATED_00001.parquet": make_instance([0] * 4 + [101] * 4, start="2018-01-01"),
        "1/DRAWN_00001.parquet": make_instance([101] * 6, start="2018-02-01"),
    }
    for rel, df in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(root / rel)
    return root
