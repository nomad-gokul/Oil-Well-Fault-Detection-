"""Alert lifecycle per well: OPEN -> ACKNOWLEDGED -> RESOLVED, one alert per fault episode.

Pure logic (no I/O): feed it predictions in time order, get :class:`AlertEvent`s back. The
API persists the events and restores open alerts on restart with :meth:`AlertEngine.restore`.

Per-well thresholds (optional, from the bundle): the first ``burnin_windows`` predictions of a
well's stream are its burn-in. Meanwhile the global threshold applies; afterwards the well
uses ``max(threshold, min(quantile(burn-in P(fault), burnin_quantile) + burnin_margin,
max_threshold))``, as tuned in ``owfd.models.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from owfd.data.schema import CLASS_NAMES
from owfd.inference.predictor import Prediction


class AlertStatus(StrEnum):
    """Lifecycle states."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class AlertConfig:
    """Opening rule (from the serving bundle) and lifecycle settings (configs/alerts.yaml)."""

    threshold: float
    k: int
    resolve_margin: float = 0.10
    resolve_after: int = 10
    cooldown_s: float = 1800
    severity: dict[int, str] = field(default_factory=dict)
    burnin_quantile: float | None = None
    burnin_margin: float = 0.0
    max_threshold: float = 0.99
    burnin_windows: int = 0

    @classmethod
    def from_files(cls, alert_rule: dict[str, Any], path: Path) -> AlertConfig:
        """Combine the bundle's ``alert`` section with ``configs/alerts.yaml``."""
        extra = yaml.safe_load(path.read_text()) if path.exists() else {}
        severity = {int(c): str(s) for c, s in (extra.pop("severity", None) or {}).items()}
        q = alert_rule.get("burnin_quantile")
        return cls(
            threshold=float(alert_rule["threshold"]),
            k=int(alert_rule["k"]),
            severity=severity,
            burnin_quantile=None if q is None else float(q),
            burnin_margin=float(alert_rule.get("burnin_margin", 0.0)),
            max_threshold=float(alert_rule.get("max_threshold", 0.99)),
            burnin_windows=int(alert_rule.get("burnin_windows", 0)),
            **extra,
        )

    def well_threshold(self, burnin_p: list[float]) -> float:
        """Threshold after a burn-in with these P(fault) values (global one if disabled)."""
        if self.burnin_quantile is None or not burnin_p:
            return self.threshold
        adaptive = float(np.quantile(burnin_p, self.burnin_quantile)) + self.burnin_margin
        return max(self.threshold, min(adaptive, self.max_threshold))

    def severity_of(self, event_class: int) -> str:
        """Severity label of an event class."""
        return self.severity.get(event_class, "medium")


@dataclass
class Alert:
    """An alert as tracked by the engine."""

    well_id: str
    event_class: int
    severity: str
    opened_at: pd.Timestamp
    peak_probability: float
    status: AlertStatus = AlertStatus.OPEN
    id: int | None = None
    acknowledged_at: pd.Timestamp | None = None
    resolved_at: pd.Timestamp | None = None
    class_votes: dict[int, float] = field(default_factory=dict)
    #: "onset" (the main model) or "chronic" (a well's flow-instability profile).
    source: str = "onset"

    @property
    def class_name(self) -> str:
        """Name of the event class."""
        return CLASS_NAMES[self.event_class]


@dataclass(frozen=True)
class AlertEvent:
    """Something that happened to an alert: ``kind`` is opened, updated or resolved."""

    kind: str
    alert: Alert


@dataclass
class _WellState:
    run_above: int = 0
    run_below: int = 0
    active: Alert | None = None
    last_resolved: pd.Timestamp | None = None
    seen: int = 0
    burnin_p: list[float] = field(default_factory=list)
    threshold: float | None = None  # set when the burn-in ends


class AlertEngine:
    """Applies :class:`AlertConfig` to each well's prediction stream."""

    def __init__(self, config: AlertConfig, source: str = "onset") -> None:
        self.config = config
        self.source = source
        self.wells: dict[str, _WellState] = {}

    def restore(self, alerts: list[Alert]) -> None:
        """Re-attach unresolved alerts (e.g. loaded from the database after a restart)."""
        for alert in alerts:
            if alert.status != AlertStatus.RESOLVED and alert.source == self.source:
                self.wells.setdefault(alert.well_id, _WellState()).active = alert

    def active(self, well_id: str) -> Alert | None:
        """The well's unresolved alert, if any."""
        state = self.wells.get(well_id)
        return state.active if state else None

    def threshold(self, well_id: str) -> float:
        """The well's current alert threshold (global until its burn-in is complete)."""
        state = self.wells.get(well_id)
        if state is None or state.threshold is None:
            return self.config.threshold
        return state.threshold

    def _burnin(self, s: _WellState, p_fault: float) -> None:
        cfg = self.config
        s.seen += 1
        if cfg.burnin_quantile is None or s.threshold is not None:
            return
        if s.seen <= cfg.burnin_windows:
            s.burnin_p.append(p_fault)
        if s.seen >= cfg.burnin_windows:
            s.threshold = cfg.well_threshold(s.burnin_p)
            s.burnin_p = []

    def process(self, p: Prediction) -> list[AlertEvent]:
        """Update the well's state with one prediction (in time order)."""
        cfg = self.config
        s = self.wells.setdefault(p.well_id, _WellState())
        threshold = self.threshold(p.well_id)  # the burn-in window itself uses the global one
        self._burnin(s, p.p_fault)
        s.run_above = s.run_above + 1 if p.p_fault > threshold else 0
        below = p.p_fault < threshold - cfg.resolve_margin
        s.run_below = s.run_below + 1 if below else 0
        fault_class = int(np.argmax(p.probabilities[1:])) + 1  # most likely fault class

        if s.active is not None:
            a = s.active
            a.class_votes[fault_class] = a.class_votes.get(fault_class, 0.0) + p.p_fault
            if s.run_below >= cfg.resolve_after:
                a.status, a.resolved_at = AlertStatus.RESOLVED, p.timestamp
                s.active, s.last_resolved, s.run_above = None, p.timestamp, 0
                return [AlertEvent("resolved", a)]
            changed = p.p_fault > a.peak_probability
            a.peak_probability = max(a.peak_probability, p.p_fault)
            # Report the class with the most accumulated evidence over the episode.
            best = max(a.class_votes, key=lambda c: a.class_votes[c])
            if best != a.event_class:
                a.event_class, a.severity, changed = best, cfg.severity_of(best), True
            return [AlertEvent("updated", a)] if changed else []

        in_cooldown = s.last_resolved is not None and (
            (p.timestamp - s.last_resolved).total_seconds() < cfg.cooldown_s
        )
        if s.run_above >= cfg.k and not in_cooldown:
            s.active = Alert(p.well_id, fault_class, cfg.severity_of(fault_class), p.timestamp,
                             p.p_fault, class_votes={fault_class: p.p_fault},
                             source=self.source)  # fmt: skip
            return [AlertEvent("opened", s.active)]
        return []

    @staticmethod
    def acknowledge(alert: Alert, when: pd.Timestamp) -> Alert:
        """Mark an open alert as acknowledged (idempotent; resolved alerts are unchanged)."""
        if alert.status == AlertStatus.OPEN:
            alert.status, alert.acknowledged_at = AlertStatus.ACKNOWLEDGED, when
        return alert
