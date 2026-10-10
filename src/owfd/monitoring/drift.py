"""Fleet-level data drift: has the data from *many* wells shifted away from what we trained on?

Per well and sensor, two window statistics of the deviation from the well's own baseline
(the mean of its first hour, as in the model's relative features):

- ``median``: typical offset from the baseline,
- ``log_iqr``: log of the inter-quartile range (spread / noise level), floored.

The reference holds the full range (plus a 10% margin) of each statistic over 2-hour windows
of real training and validation wells. A well is *unusual* on a sensor when a statistic falls
outside that range, i.e. beyond anything seen on those wells; shifts smaller than the natural
well-to-well variation are, honestly, not detectable this way. Drift is flagged for a sensor
when at least half of the streaming wells (and at least ``min_wells``) are unusual on it: a
fleet-wide shift such as a recalibrated sensor or a unit change in the data pipeline. A single
faulty well does not raise it (that is what alerts are for).

Why not PSI against the pooled training distribution: wells differ so much that a single
normal well's window already reaches the PSI maximum, so PSI cannot tell normal from shifted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

STATS = ("median", "log_iqr")
MIN_OBSERVED = 1800  # at least 30 min of observed readings in a window
IQR_FLOOR = 0.01  # scaled units
MARGIN = 0.10  # widen the reference range by 10% of its span on each side


def deviations_from_baseline(
    df: pd.DataFrame, variables: list[str], baseline_s: int
) -> pd.DataFrame:
    """Each value minus the mean of its first ``baseline_s`` rows (one well or instance)."""
    return df[variables] - df[variables].iloc[:baseline_s].mean()


def window_stats(dev: pd.Series) -> dict[str, float] | None:
    """Statistics of one sensor's deviations in one window (None if too few observations)."""
    x = dev.dropna().to_numpy(dtype=np.float64)
    if len(x) < MIN_OBSERVED:
        return None
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    # Floor the spread: constant or heavily quantised sensors (e.g. gas lift off) would
    # otherwise give an extreme log-spread for perfectly normal behaviour.
    return {"median": float(med), "log_iqr": float(np.log10(q3 - q1 + IQR_FLOOR))}


def build_reference(windows: Iterable[pd.DataFrame], variables: list[str]) -> dict[str, Any]:
    """Per sensor and statistic, the observed range over reference windows (+ margin)."""
    collected: dict[str, dict[str, list[float]]] = {v: {s: [] for s in STATS} for v in variables}
    for w in windows:
        for v in variables:
            stats = window_stats(w[v]) if v in w else None
            if stats:
                for s in STATS:
                    collected[v][s].append(stats[s])
    ref: dict[str, Any] = {"variables": {}}
    for v, by_stat in collected.items():
        n = len(by_stat["median"])
        if n < 20:
            continue
        ranges = {}
        for s in STATS:
            lo, hi = min(by_stat[s]), max(by_stat[s])
            pad = MARGIN * (hi - lo)
            ranges[s] = [lo - pad, hi + pad]
        ref["variables"][v] = {"n_windows": n, **ranges}
    return ref


def check_well(reference: dict[str, Any], dev: pd.DataFrame) -> dict[str, Any]:
    """Per sensor: the window statistics and whether any is outside the reference range."""
    out: dict[str, Any] = {}
    for v, r in reference["variables"].items():
        stats = window_stats(dev[v]) if v in dev else None
        if stats is None:
            out[v] = {"status": "insufficient_data"}
            continue
        outside = [s for s in STATS if not r[s][0] <= stats[s] <= r[s][1]]
        out[v] = {
            **{s: round(stats[s], 4) for s in STATS},
            "status": "unusual" if outside else "typical",
            "outside": outside,
        }
    return out


def fleet_drift(
    reference: dict[str, Any],
    wells: dict[str, pd.DataFrame],
    min_wells: int = 3,
    share: float = 0.5,
) -> dict[str, Any]:
    """Drift report over all streaming wells (``wells``: well id -> deviations frame)."""
    per_well = {w: check_well(reference, dev) for w, dev in wells.items()}
    sensors: dict[str, Any] = {}
    for v in reference["variables"]:
        checked = [w for w, r in per_well.items() if r[v]["status"] != "insufficient_data"]
        unusual = [w for w in checked if per_well[w][v]["status"] == "unusual"]
        frac = len(unusual) / len(checked) if checked else 0.0
        drift = len(checked) >= min_wells and frac >= share
        sensors[v] = {
            "wells_checked": len(checked),
            "wells_unusual": len(unusual),
            "share_unusual": round(frac, 3),
            "unusual_wells": unusual,
            "status": "drift"
            if drift
            else "insufficient_data"
            if len(checked) < min_wells
            else "stable",
        }
    return {
        "drift": any(s["status"] == "drift" for s in sensors.values()),
        "sensors": sensors,
        "wells": per_well,
    }
