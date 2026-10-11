import numpy as np
import pandas as pd

from owfd.models.evaluate import AlertRule
from owfd.models.well_profiles import (
    MIN_WINDOWS,
    UNSTABLE,
    chronic_report,
    fit_profile,
    prequential,
    summarize,
)

FEATURES = ["a", "b"]


def recording(iid: str, well: str, start: str, cls: int, n: int = 90, overlap_h: int = 0) -> tuple:
    """Windows of one recording: feature ``a`` is high while the well is unstable."""
    rng = np.random.default_rng(abs(hash(iid)) % 2**32)
    t0 = pd.Timestamp(start)
    times = t0 + pd.to_timedelta(np.arange(n), "min")
    level = 3.0 if cls == UNSTABLE else 0.0
    w = pd.DataFrame({"instance_id": iid, "well_id": well, "end_time": times, "event_class": cls,
                      "a": level + rng.normal(0, 0.3, n), "b": rng.normal(0, 1, n)})  # fmt: skip
    meta = {"start_time": t0 - pd.Timedelta(hours=overlap_h), "end_time": times[-1],
            "well_id": well, "event_class": cls, "source": "real"}  # fmt: skip
    return w, pd.Series(meta, name=iid)


def dataset(spec: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [recording(*s) for s in spec]
    return pd.concat([w for w, _ in parts], ignore_index=True), pd.DataFrame([m for _, m in parts])


def test_fit_profile_needs_both_kinds() -> None:
    x = pd.DataFrame({"a": np.r_[np.zeros(MIN_WINDOWS), np.ones(MIN_WINDOWS - 1)], "b": 0.0})
    assert fit_profile(x, x["a"].to_numpy() > 0.5) is None
    x = pd.DataFrame({"a": np.r_[np.zeros(MIN_WINDOWS), np.ones(MIN_WINDOWS)], "b": 0.0})
    model = fit_profile(x, x["a"].to_numpy() > 0.5, n_jobs=1)
    assert (
        model is not None
        and model.predict_proba(pd.DataFrame({"a": [1.0], "b": [0.0]}))[0, 1] > 0.5
    )


def test_prequential_learns_only_from_earlier_recordings() -> None:
    windows, instances = dataset([
        ("r1", "W", "2020-01-01 00:00", 0),
        ("r2", "W", "2020-01-01 03:00", UNSTABLE),  # first unstable episode: no profile yet
        ("r3", "W", "2020-01-01 06:00", UNSTABLE),  # profile from r1 + r2
        ("r4", "W", "2020-01-01 09:00", 0),
        ("q1", "Q", "2020-01-01 00:00", 0),  # never unstable: no profile, not scored
    ])  # fmt: skip
    preds = prequential(windows, instances, FEATURES)
    assert set(preds["well_id"]) == {"W"}
    first = preds.groupby("instance_id")["has_profile"].first()
    assert not first["r1"] and not first["r2"] and first["r3"] and first["r4"]
    report = chronic_report(preds, AlertRule(0.8, 5)).set_index("instance_id")
    assert not report.at["r2", "caught"] and report.at["r3", "caught"]
    assert report.at["r4", "false_alarms"] == 0
    s = summarize(report.reset_index(), instances)
    assert s["unstable_episodes"] == [1, 2] and s["after_first_profile"] == [1, 1]
    assert s["continuations"] == [1, 1] and s["new_onsets"] == [0, 1]


def test_overlapping_recordings_are_not_used_for_training() -> None:
    # r2 starts (with its unlabelled hour) before r1 has ended: r1 must not train r2's profile.
    windows, instances = dataset([
        ("r0", "W", "2020-01-01 00:00", UNSTABLE),
        ("r1", "W", "2020-01-02 00:00", 0),
        ("r2", "W", "2020-01-02 02:00", UNSTABLE, 90, 1),
    ])  # fmt: skip
    preds = prequential(windows, instances, FEATURES)
    assert not preds.groupby("instance_id")["has_profile"].first()["r2"]  # only r0 is usable


def test_partial_labels_are_deterministic() -> None:
    windows, instances = dataset([(f"r{i}", "W", f"2020-01-{i + 1:02d}", UNSTABLE if i % 2 else 0)
                                  for i in range(8)])  # fmt: skip
    a = prequential(windows, instances, FEATURES, label_frac=0.5, seed=3)
    b = prequential(windows, instances, FEATURES, label_frac=0.5, seed=3)
    pd.testing.assert_frame_equal(a, b)
