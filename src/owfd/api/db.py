"""Persistence for predictions, alerts and operator feedback (SQLite by default, Postgres via
OWFD_DATABASE_URL)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from owfd.inference.alerting import Alert, AlertStatus
from owfd.inference.predictor import Prediction


class Base(DeclarativeBase):
    """Declarative base."""


class PredictionRow(Base):
    """One model output (one per well per minute)."""

    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    well_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    predicted_class: Mapped[int] = mapped_column(Integer)
    p_fault: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[list[float]] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(64))


class AlertRow(Base):
    """One alert episode."""

    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    well_id: Mapped[str] = mapped_column(String(64), index=True)
    event_class: Mapped[int] = mapped_column(Integer)
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    peak_probability: Mapped[float] = mapped_column(Float)
    top_drivers: Mapped[list[dict[str, float | str]] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="onset", server_default="onset")


class FeedbackRow(Base):
    """An operator's verdict on an alert (labels for retraining well profiles)."""

    __tablename__ = "alert_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(Integer, index=True)
    well_id: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


def make_engine(url: str) -> Engine:
    """Create the engine and tables (SQLite folders are created as needed)."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            # Every new connection to ":memory:" is a separate, empty database: share one.
            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    _migrate(engine)
    return engine


def _migrate(engine: Engine) -> None:
    """Add columns introduced after a database was created (create_all only adds tables)."""
    columns = {c["name"] for c in inspect(engine).get_columns("alerts")}
    if "source" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE alerts ADD COLUMN source VARCHAR(16) DEFAULT 'onset'"))


def _naive(ts: pd.Timestamp | None) -> datetime | None:
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t).to_pydatetime()


class Store:
    """Small repository API used by the service."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save_predictions(self, preds: list[Prediction], model_version: str) -> None:
        """Insert predictions."""
        if not preds:
            return
        with Session(self.engine) as s:
            s.add_all(PredictionRow(well_id=p.well_id, ts=_naive(p.timestamp),
                                    predicted_class=p.predicted_class, p_fault=p.p_fault,
                                    probabilities=p.probabilities, model_version=model_version)
                      for p in preds)  # fmt: skip
            s.commit()

    def save_alert(
        self, alert: Alert, drivers: list[dict[str, float | str]] | None = None
    ) -> Alert:
        """Insert or update an alert; sets ``alert.id`` on first save."""
        with Session(self.engine) as s:
            row = s.get(AlertRow, alert.id) if alert.id is not None else None
            if row is None:
                row = AlertRow(well_id=alert.well_id, top_drivers=drivers, source=alert.source)
                s.add(row)
            row.event_class, row.severity, row.status = (
                alert.event_class,
                alert.severity,
                alert.status,
            )
            row.opened_at = _naive(alert.opened_at)  # type: ignore[assignment]
            row.acknowledged_at = _naive(alert.acknowledged_at)
            row.resolved_at = _naive(alert.resolved_at)
            row.peak_probability = alert.peak_probability
            if drivers is not None:
                row.top_drivers = drivers
            s.commit()
            alert.id = row.id
        return alert

    @staticmethod
    def _to_alert(row: AlertRow) -> Alert:
        ts = lambda v: pd.Timestamp(v) if v is not None else None  # noqa: E731
        return Alert(row.well_id, row.event_class, row.severity, pd.Timestamp(row.opened_at),
                     row.peak_probability, AlertStatus(row.status), row.id,
                     ts(row.acknowledged_at), ts(row.resolved_at),
                     source=row.source or "onset")  # fmt: skip

    def get_alert(self, alert_id: int) -> tuple[Alert, list[dict[str, float | str]] | None] | None:
        """An alert and its stored drivers, or None."""
        with Session(self.engine) as s:
            row = s.get(AlertRow, alert_id)
            return (self._to_alert(row), row.top_drivers) if row else None

    def list_alerts(
        self,
        well_id: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[Alert, list[dict[str, float | str]] | None]]:
        """Alerts, newest first, with optional filters."""
        q = select(AlertRow).order_by(AlertRow.opened_at.desc()).limit(limit)
        if well_id:
            q = q.where(AlertRow.well_id == well_id)
        if status:
            q = q.where(AlertRow.status == status)
        if since:
            q = q.where(AlertRow.opened_at >= _naive(pd.Timestamp(since)))
        if until:
            q = q.where(AlertRow.opened_at <= _naive(pd.Timestamp(until)))
        with Session(self.engine) as s:
            return [(self._to_alert(r), r.top_drivers) for r in s.scalars(q)]

    def unresolved_alerts(self) -> list[Alert]:
        """Open or acknowledged alerts (to restore the engine after a restart)."""
        return [a for a, _ in self.list_alerts(limit=10_000) if a.status != AlertStatus.RESOLVED]

    def predictions(self, well_id: str, since: datetime | None = None,
                    limit: int = 1440) -> pd.DataFrame:  # fmt: skip
        """A well's recent predictions (oldest first)."""
        q = select(PredictionRow).where(PredictionRow.well_id == well_id)
        if since:
            q = q.where(PredictionRow.ts >= _naive(pd.Timestamp(since)))
        q = q.order_by(PredictionRow.ts.desc()).limit(limit)
        with Session(self.engine) as s:
            rows = [{"timestamp": r.ts, "predicted_class": r.predicted_class, "p_fault": r.p_fault,
                     "probabilities": json.loads(json.dumps(r.probabilities)),
                     "model_version": r.model_version} for r in s.scalars(q)]  # fmt: skip
        return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

    def save_feedback(
        self, alert: Alert, label: str, note: str | None, when: datetime
    ) -> dict[str, Any]:
        """Store an operator's verdict on an alert."""
        assert alert.id is not None
        with Session(self.engine) as s:
            row = FeedbackRow(alert_id=alert.id, well_id=alert.well_id, label=label, note=note,
                              created_at=_naive(pd.Timestamp(when)))  # fmt: skip
            s.add(row)
            s.commit()
            return self._feedback_dict(row)

    @staticmethod
    def _feedback_dict(row: FeedbackRow) -> dict[str, Any]:
        return {"id": row.id, "alert_id": row.alert_id, "well_id": row.well_id,
                "label": row.label, "note": row.note, "created_at": row.created_at}  # fmt: skip

    def feedback(self, well_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        """Operator feedback, newest first."""
        q = select(FeedbackRow).order_by(FeedbackRow.created_at.desc()).limit(limit)
        if well_id:
            q = q.where(FeedbackRow.well_id == well_id)
        with Session(self.engine) as s:
            return [self._feedback_dict(r) for r in s.scalars(q)]

    def wells(self) -> list[str]:
        """All wells that have predictions."""
        with Session(self.engine) as s:
            return sorted(s.scalars(select(PredictionRow.well_id).distinct()))
