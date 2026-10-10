import numpy as np
import pandas as pd
import pytest

from owfd.models.evaluate import (
    AlertRule,
    alert_onsets,
    alert_state,
    apply_temperature,
    early_warning_report,
    expected_calibration_error,
    fit_temperature,
    instance_report,
    per_class_summary,
    summarize,
    tune_rule,
)


def instance(
    iid: str, labels: list[int], p_fault: list[float], start: str = "2020-01-01"
) -> pd.DataFrame:
    """One window per minute; labels use raw 3W codes (0, 1..9, 101..109)."""
    raw = np.array(labels)
    return pd.DataFrame(
        {
            "instance_id": iid,
            "end_time": pd.date_range(start, periods=len(raw), freq="min"),
            "event_class": np.where(raw >= 100, raw - 100, raw),
            "is_transient": raw >= 100,
            "p_fault": p_fault,
        }
    )


def test_alert_state_and_onsets() -> None:
    p = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1])
    rule = AlertRule(0.5, 2)
    assert alert_state(p, rule).tolist() == [0, 1, 0, 0, 1, 1, 1, 0]
    assert alert_onsets(p, rule).tolist() == [0, 1, 0, 0, 1, 0, 0, 0]
    assert alert_onsets(p, AlertRule(0.5, 1)).sum() == 2


def test_known_lead_time_and_delay() -> None:
    # 10 min normal, 20 min transient, 10 min steady; P(fault) rises 5 min into the transient
    labels = [0] * 10 + [105] * 20 + [5] * 10
    p = [0.1] * 15 + [0.9] * 25
    r = instance_report(instance("a", labels, p), AlertRule(0.5, 3))
    # alert state turns on at minute 17 (3 consecutive from minute 15)
    assert r["detected"] and r["delay_min"] == 7 and r["lead_min"] == 13
    assert r["false_alarms"] == 0 and r["normal_hours"] == pytest.approx(10 / 60)


def test_false_alarm_and_missed_fault() -> None:
    labels = [0] * 30 + [101] * 10
    p = [0.1] * 5 + [0.9] * 3 + [0.1] * 32  # alert on normal data, nothing during the fault
    r = instance_report(instance("b", labels, p), AlertRule(0.5, 2))
    assert r["false_alarms"] == 1 and not r["detected"] and np.isnan(r["lead_min"])


def test_alert_already_on_at_fault_start_counts_and_no_transient_gives_no_lead() -> None:
    labels = [0] * 5 + [3] * 10  # severe slugging: no transient label
    r = instance_report(instance("c", labels, [0.9] * 15), AlertRule(0.5, 1))
    assert r["detected"] and r["delay_min"] == 0 and np.isnan(r["lead_min"])
    assert r["false_alarms"] == 1  # it was on during normal operation too


def test_summaries_and_rule_tuning() -> None:
    preds = pd.concat([
        instance("n1", [0] * 1440, [0.2] * 1440),  # one day of normal operation
        instance("f1", [0] * 10 + [101] * 30, [0.2] * 15 + [0.8] * 25),
        instance("f2", [0] * 10 + [102] * 30, [0.2] * 20 + [0.6] * 20),
    ])  # fmt: skip
    report = early_warning_report(preds, AlertRule(0.5, 2))
    s = summarize(report)
    assert s["fault_instances"] == 2 and s["detection_rate"] == 1.0
    assert s["false_alarms_per_day"] == 0
    assert per_class_summary(report).loc[1, "instances"] == 1
    rule, grid = tune_rule(preds, np.array([0.1, 0.5, 0.7]), ks=(1, 2))
    assert rule.threshold == 0.5  # 0.1 fires on normal data; 0.7 misses f2
    assert len(grid) == 6


def test_calibration_helpers() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    sharp = np.full((2000, 3), 0.005)
    sharp[np.arange(2000), np.where(rng.random(2000) < 0.7, y, (y + 1) % 3)] = 0.99
    t = fit_temperature(sharp, y)
    assert t > 1  # over-confident (99% vs 70% accuracy) -> soften
    soft = apply_temperature(sharp, t)
    np.testing.assert_allclose(soft.sum(axis=1), 1)
    assert abs(soft.max(axis=1).mean() - 0.7) < 0.1
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0
