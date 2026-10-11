import numpy as np
import pandas as pd
import pytest

from owfd.monitoring.drift import (
    build_reference,
    check_well,
    deviations_from_baseline,
    fleet_drift,
    window_stats,
)


def well(seed: int, level: float = 0.0, offset: float = 0.0, noise: float = 1.0) -> pd.DataFrame:
    """A well at its own operating level; ``offset`` shifts it after the first hour."""
    rng = np.random.default_rng(seed)
    n = 7200
    a = level + rng.normal(0, noise, n)
    a[3600:] += offset
    return pd.DataFrame({"a": a, "b": level + rng.exponential(1, n)})


def reference() -> dict:
    windows = [
        deviations_from_baseline(well(s, level=s * 10.0), ["a", "b"], 3600) for s in range(60)
    ]
    return build_reference(windows, ["a", "b"])


def test_window_stats_and_reference_ranges() -> None:
    assert window_stats(pd.Series([1.0] * 10)) is None  # too few observations
    ref = reference()
    lo, hi = ref["variables"]["a"]["median"]
    assert lo < 0 < hi and ref["variables"]["a"]["n_windows"] == 60


def test_operating_level_does_not_matter_offset_does() -> None:
    ref = reference()
    typical = deviations_from_baseline(well(100, level=500.0), ["a", "b"], 3600)
    assert check_well(ref, typical)["a"]["status"] == "typical"
    shifted = deviations_from_baseline(well(101, offset=8.0), ["a", "b"], 3600)
    assert check_well(ref, shifted)["a"]["status"] == "unusual"
    noisy = deviations_from_baseline(well(102, noise=50.0), ["a", "b"], 3600)
    assert "log_iqr" in check_well(ref, noisy)["a"]["outside"]


def test_fleet_drift_needs_many_wells() -> None:
    ref = reference()
    dev = lambda seed, off=0.0: deviations_from_baseline(well(seed, offset=off), ["a", "b"], 3600)  # noqa: E731
    one_bad = {f"w{i}": dev(200 + i, 8.0 if i == 0 else 0.0) for i in range(5)}
    report = fleet_drift(ref, one_bad)
    # The faulty well (and, by chance, at most one normal well) is unusual: no fleet drift.
    assert not report["drift"]
    assert "w0" in report["sensors"]["a"]["unusual_wells"]
    assert report["sensors"]["a"]["wells_unusual"] <= 2
    fleet_shift = {f"w{i}": dev(300 + i, 8.0) for i in range(5)}
    report = fleet_drift(ref, fleet_shift)
    assert report["drift"] and report["sensors"]["a"]["status"] == "drift"
    assert report["sensors"]["b"]["status"] == "stable"
    assert fleet_drift(ref, {"w": dev(1, 8.0)})["sensors"]["a"]["status"] == "insufficient_data"


def test_constant_sensor_is_typical() -> None:
    ref = reference()
    flat = deviations_from_baseline(well(5), ["a", "b"], 3600)
    flat["a"] = 0.0  # e.g. gas lift switched off: constant, perfectly normal
    flat["a"] += np.random.default_rng(0).normal(0, 1e-4, len(flat))
    stats = check_well(ref, flat)["a"]
    assert stats["log_iqr"] == pytest.approx(-2.0, abs=0.01)  # floored, not -6
