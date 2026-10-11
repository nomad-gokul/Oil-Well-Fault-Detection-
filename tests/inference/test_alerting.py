from pathlib import Path

import pandas as pd
import pytest

from owfd.inference.alerting import AlertConfig, AlertEngine, AlertStatus
from owfd.inference.predictor import Prediction

CFG = AlertConfig(threshold=0.6, k=3, resolve_margin=0.1, resolve_after=4, cooldown_s=600,
                  severity={8: "high", 3: "low"})  # fmt: skip
T0 = pd.Timestamp("2020-01-01")


def pred(minute: int, p_fault: float, fault_class: int = 8, well: str = "w") -> Prediction:
    probs = [1 - p_fault] + [0.0] * 9
    probs[fault_class] = p_fault
    return Prediction(well, T0 + pd.Timedelta(minutes=minute), 0, "x", p_fault, probs)


def run(
    engine: AlertEngine, ps: list[float], start: int = 0, cls: int = 8
) -> list[tuple[int, str]]:
    events = []
    for i, p in enumerate(ps):
        events += [(start + i, e.kind) for e in engine.process(pred(start + i, p, cls))]
    return events


def test_opens_once_after_k_consecutive_windows() -> None:
    eng = AlertEngine(CFG)
    events = run(eng, [0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.95, 0.9, 0.99, 0.9])
    opened = [e for e in events if e[1] == "opened"]
    assert opened == [(5, "opened")]  # the first run of 2 is too short; then exactly one alert
    a = eng.active("w")
    assert a is not None and a.peak_probability == 0.99 and a.severity == "high"
    assert "updated" in {k for _, k in events}


def test_hysteresis_and_resolution() -> None:
    eng = AlertEngine(CFG)
    run(eng, [0.9] * 3)
    # 0.55 is below the open threshold but above threshold - margin (0.5): not "low" yet
    events = run(eng, [0.55] * 6 + [0.4] * 3, start=3)
    assert all(k != "resolved" for _, k in events)
    events = run(eng, [0.4], start=12)  # 4th consecutive low window
    assert events == [(12, "resolved")] and eng.active("w") is None


def test_cooldown_blocks_immediate_realert() -> None:
    eng = AlertEngine(CFG)
    assert run(eng, [0.9] * 3 + [0.1] * 4) == [(2, "opened"), (6, "resolved")]
    # P(fault) high again every minute: blocked for 10 min after the resolution at minute 6,
    # then the fault is still there, so it reopens at the first window after the cool-down.
    assert run(eng, [0.9] * 14, start=7) == [(16, "opened")]


def test_new_alert_after_cooldown_needs_k_windows() -> None:
    eng = AlertEngine(CFG)
    run(eng, [0.9] * 3 + [0.1] * 4)  # resolves at 6
    assert run(eng, [0.1] * 10 + [0.9] * 3, start=7) == [(19, "opened")]


def test_wells_are_independent() -> None:
    eng = AlertEngine(CFG)
    for i in range(3):
        eng.process(pred(i, 0.9, well="a"))
        eng.process(pred(i, 0.1, well="b"))
    assert eng.active("a") is not None and eng.active("b") is None


def test_acknowledge_and_restore() -> None:
    eng = AlertEngine(CFG)
    run(eng, [0.9] * 3)
    alert = eng.active("w")
    assert alert is not None
    AlertEngine.acknowledge(alert, T0)
    assert alert.status == AlertStatus.ACKNOWLEDGED and alert.acknowledged_at == T0
    AlertEngine.acknowledge(alert, T0 + pd.Timedelta(hours=1))  # idempotent
    assert alert.acknowledged_at == T0
    fresh = AlertEngine(CFG)
    fresh.restore([alert])
    assert run(fresh, [0.9] * 5, start=3) == []  # no duplicate after a restart
    assert fresh.active("w") is alert


def test_reported_class_follows_the_evidence() -> None:
    eng = AlertEngine(CFG)
    run(eng, [0.9] * 3, cls=3)
    assert eng.active("w").event_class == 3  # type: ignore[union-attr]
    run(eng, [0.9] * 5, start=3, cls=8)
    a = eng.active("w")
    assert a is not None and a.event_class == 8 and a.severity == "high"


def test_config_from_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("resolve_after: 7\nseverity:\n  2: high\n")
    cfg = AlertConfig.from_files({"threshold": 0.7, "k": 2}, tmp_path / "a.yaml")
    assert (cfg.threshold, cfg.k, cfg.resolve_after) == (0.7, 2, 7)
    assert cfg.severity_of(2) == "high" and cfg.severity_of(5) == "medium"
    real = AlertConfig.from_files({"threshold": 0.5, "k": 3}, Path("configs/alerts.yaml"))
    assert real.severity_of(8) == "high"


@pytest.mark.parametrize("p", [0.0, 0.6])
def test_values_at_or_below_threshold_never_open(p: float) -> None:
    assert run(AlertEngine(CFG), [p] * 20) == []
