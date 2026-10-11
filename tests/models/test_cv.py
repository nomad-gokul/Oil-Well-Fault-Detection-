import numpy as np
import pandas as pd
import pytest

from owfd.models.cv import assign_folds, bootstrap_by_well, cap_per_instance


def test_folds_cover_every_well_once_and_spread_fault_wells() -> None:
    counts = pd.Series({f"W{i}": c for i, c in enumerate([30, 20, 9, 8, 5, 3, 2, 1, 0, 0])})
    folds = assign_folds(counts, 3, seed=1)
    assert set(folds) == set(counts.index) and set(folds.values()) == {0, 1, 2}
    per_fold = pd.Series(counts).groupby(pd.Series(folds)).sum()
    assert per_fold.max() - per_fold.min() <= 20  # snake order: the two big wells split
    assert folds["W0"] != folds["W1"]
    assert assign_folds(counts, 3, seed=1) == folds  # deterministic


def test_cap_per_instance() -> None:
    df = pd.DataFrame({"instance_id": ["a"] * 10 + ["b"] * 3, "x": range(13)})
    capped = cap_per_instance(df, 4, seed=0)
    assert capped.groupby("instance_id").size().to_dict() == {"a": 4, "b": 3}
    assert capped["x"].is_monotonic_increasing and set(capped["x"]) <= set(df["x"])


def test_bootstrap_by_well_interval_contains_the_estimate() -> None:
    rng = np.random.default_rng(0)
    report = pd.DataFrame(
        {
            "well_id": [f"W{i % 8}" for i in range(80)],
            "has_fault": True,
            "detected": rng.random(80) < 0.6,
            "false_alarms": rng.integers(0, 3, 80),
            "normal_hours": 10.0,
        }
    )
    ci = bootstrap_by_well(report, n=300)
    lo, hi = ci["detection_rate"]
    assert lo <= report["detected"].mean() <= hi and hi - lo < 0.5
    fa = report["false_alarms"].sum() / report["normal_hours"].sum() * 24
    assert ci["false_alarms_per_day"][0] <= fa <= ci["false_alarms_per_day"][1]
    assert lo == pytest.approx(bootstrap_by_well(report, n=300)["detection_rate"][0])
