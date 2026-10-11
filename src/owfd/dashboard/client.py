"""Thin client for the API, used by the dashboard (kept separate so it can be tested)."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

DEFAULT_URL = "http://localhost:8000"


class ApiUnavailableError(RuntimeError):
    """The API could not be reached (down, or a free-tier instance waking up)."""


class Client:
    """Synchronous API client; every call raises :class:`ApiUnavailableError` on network errors."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 10,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OWFD_API_URL", DEFAULT_URL)).rstrip("/")
        key = api_key if api_key is not None else os.environ.get("OWFD_API_KEY")
        headers = {"X-API-Key": key} if key else {}
        self.http = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers=headers, transport=transport
        )

    def _get(self, path: str, **params: Any) -> Any:
        try:
            r = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(str(exc)) from exc
        if r.status_code >= 500:
            raise ApiUnavailableError(f"{r.status_code} from {path}")
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **params: Any) -> Any:
        try:
            r = self.http.post(path, params=params)
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(str(exc)) from exc
        r.raise_for_status()
        return r.json()

    def health(self) -> dict[str, Any]:
        """``GET /health``."""
        return dict(self._get("/health"))

    def model(self) -> dict[str, Any]:
        """``GET /model``."""
        return dict(self._get("/model"))

    def wells(self) -> list[str]:
        """``GET /wells``."""
        return list(self._get("/wells"))

    def predictions(self, well_id: str, limit: int = 240) -> pd.DataFrame:
        """A well's latest predictions as a DataFrame (oldest first)."""
        df = pd.DataFrame(self._get(f"/wells/{well_id}/predictions", limit=limit))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def readings(self, well_id: str, minutes: int = 60, every_s: int = 10) -> pd.DataFrame:
        """A well's recent sensor readings (DatetimeIndex, one column per sensor)."""
        data = self._get(f"/wells/{well_id}/readings", minutes=minutes, every_s=every_s)
        if not data["timestamps"]:
            return pd.DataFrame()
        return pd.DataFrame(data["values"], index=pd.to_datetime(data["timestamps"]))

    def alerts(
        self, well_id: str | None = None, status: str | None = None, limit: int = 500
    ) -> pd.DataFrame:
        """Alerts as a DataFrame (newest first)."""
        df = pd.DataFrame(self._get("/alerts", well_id=well_id, status=status, limit=limit))
        for col in ("opened_at", "acknowledged_at", "resolved_at"):
            if col in df:
                df[col] = pd.to_datetime(df[col])
        return df

    def acknowledge(self, alert_id: int) -> dict[str, Any]:
        """``POST /alerts/{id}/ack``."""
        return dict(self._post(f"/alerts/{alert_id}/ack"))

    def feedback(self, alert_id: int, label: str) -> dict[str, Any]:
        """Record an operator verdict on an alert (needs the API key)."""
        try:
            r = self.http.post(f"/alerts/{alert_id}/feedback", json={"label": label})
        except httpx.HTTPError as exc:
            raise ApiUnavailableError(str(exc)) from exc
        r.raise_for_status()
        return dict(r.json())

    def drift(self) -> dict[str, Any] | None:
        """``GET /monitoring/drift`` (None if the bundle has no training reference)."""
        try:
            return dict(self._get("/monitoring/drift"))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    def demo(self) -> dict[str, Any]:
        """``GET /demo``."""
        return dict(self._get("/demo"))

    def demo_start(self, speed: int, classes: list[int] | None) -> dict[str, Any]:
        """``POST /demo/start``."""
        params: dict[str, Any] = {"speed": speed, "loop": True}
        if classes:
            params["classes"] = classes
        return dict(self._post("/demo/start", **params))

    def demo_stop(self) -> dict[str, Any]:
        """``POST /demo/stop``."""
        return dict(self._post("/demo/stop"))


def well_status(p_fault: float | None, threshold: float, alert_status: str | None) -> str:
    """Fleet card status: Fault (alert open/acknowledged), Warning (close to the threshold),
    Normal, or No data."""
    if alert_status in ("open", "acknowledged"):
        return "Fault"
    if p_fault is None:
        return "No data"
    return "Warning" if p_fault > threshold - 0.15 else "Normal"


def scenario_source(info: dict[str, Any]) -> str:
    """Where a demo well's data comes from, e.g. ``"WELL-00004, Aug 2014"`` (``""`` if unknown)."""
    well = info.get("source_well")
    recorded = info.get("recorded_at")
    when = pd.Timestamp(recorded).strftime("%b %Y") if recorded else ""
    return ", ".join(part for part in (well, when) if part)


def scenario_caption(info: dict[str, Any]) -> str:
    """One line for a demo well's fleet card: the real well it replays and its true condition."""
    source = scenario_source(info)
    truth = (
        "no fault (normal operation)"
        if int(info.get("event_class", -1)) == 0
        else str(info.get("class_name", "unknown"))
    )
    return f"Real well: {source} · actual: {truth}" if source else f"Actual: {truth}"


def display_units(readings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Readings in engineering units for charts: pressures Pa -> bar; temperatures in degC,
    gas-lift flow in m3/s. Physically impossible placeholder values (> 1e4 bar) are hidden."""
    out = readings.copy()
    units = {}
    for col in out.columns:
        if col.startswith("P"):
            out[col] = out[col] / 1e5
            out[col] = out[col].where(out[col].abs() < 1e4)
            units[col] = "bar"
        elif col.startswith("T"):
            units[col] = "°C"
        elif col.startswith("Q"):
            units[col] = "m³/s"
        else:
            units[col] = ""
    return out, units
