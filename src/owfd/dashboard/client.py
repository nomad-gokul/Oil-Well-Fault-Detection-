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


#: Each fault class in plain words: (what is happening, what it means, what to do).
PLAIN_FAULTS: dict[int, tuple[str, str, str]] = {
    1: (
        "Sudden rise in water",
        "The well is suddenly producing much more water than usual.",
        "Tell the production engineer and check the water handling.",
    ),
    2: (
        "Safety valve closed by itself",
        "The safety valve deep in the well shut unexpectedly. The well stops producing.",
        "Act now: check the safety valve system with the control room.",
    ),
    3: (
        "Severe slugging",
        "Liquid and gas arrive in large alternating surges that shake the pipeline.",
        "Keep watching. Adjust the choke or gas lift if it continues.",
    ),
    4: (
        "Unstable flow",
        "Pressures and flow keep swinging up and down instead of staying steady.",
        "Keep watching. Check the choke and gas-lift settings.",
    ),
    5: (
        "Production dropping fast",
        "The well is quickly losing its ability to produce.",
        "Tell the production engineer today.",
    ),
    6: (
        "Choke valve suddenly blocked",
        "The valve that controls the flow at the surface closed or clogged quickly.",
        "Act now: check the choke valve.",
    ),
    7: (
        "Scale building up in the choke valve",
        "Mineral deposits are slowly clogging the valve that controls the flow.",
        "Plan a cleaning or chemical treatment.",
    ),
    8: (
        "Ice-like plug forming in the production line",
        "Hydrates (ice-like crystals of gas and water) are forming and can block the pipeline.",
        "Act now: start hydrate prevention (e.g. inhibitor injection) as per procedure.",
    ),
    9: (
        "Ice-like plug forming in the service line",
        "Hydrates (ice-like crystals of gas and water) are forming and can block the line.",
        "Act now: start hydrate prevention as per procedure.",
    ),
}

#: Display levels, worst first: (label, icon, colour, what the level means).
LEVELS: dict[str, tuple[str, str, str, str]] = {
    "critical": ("Critical", "🔴", "#c0392b", "Can stop production or block a line. Act now."),
    "warning": ("Warning", "🟠", "#d35400", "Hurts production. Check today."),
    "watch": ("Watch", "🟡", "#b7950b", "Minor or not yet confirmed. Keep an eye on it."),
    "normal": ("Normal", "🟢", "#1e8449", "No fault detected."),
    "no_data": ("No data", "⚪", "#7f8c8d", "The well is not sending readings."),
}
_SEVERITY_LEVEL = {"high": "critical", "medium": "warning", "low": "watch"}


def duration_text(start: Any, end: Any) -> str:
    """How long between two timestamps, in words: ``"just now"``, ``"25 min"``, ``"3 h 5 min"``,
    ``"2 days"``."""
    a, b = (t.tz_convert(None) if t.tzinfo else t for t in (pd.Timestamp(start), pd.Timestamp(end)))
    minutes = max(int((b - a).total_seconds() // 60), 0)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min"
    if minutes < 48 * 60:
        h, m = divmod(minutes, 60)
        return f"{h} h {m} min" if m else f"{h} h"
    return f"{minutes // (24 * 60)} days"


def fault_text(
    event_class: int, class_name: str = "", source: str | None = None
) -> tuple[str, str, str]:
    """(headline, what it means, what to do) for a fault class; ``source="chronic"`` marks a
    long-running condition recognised by the well's own profile."""
    headline, meaning, action = PLAIN_FAULTS.get(
        event_class, (class_name or "Fault", "The system detected a fault.", "Check the well.")
    )
    if source == "chronic":
        headline = f"Long-running {headline[0].lower()}{headline[1:]}"
        meaning += " It matches this well's earlier unstable periods."
    return headline, meaning, action


def plain_status(
    alerts: list[dict[str, Any]],
    p_fault: float | None,
    threshold: float,
    now: Any = None,
) -> dict[str, Any]:
    """A well's current state for non-specialists: its worst unresolved alert (or, without one,
    whether its readings look unusual) as a level, a headline, what it means and what to do.

    ``alerts`` are the well's alerts as returned by the API; ``now`` (the well's latest data
    time) gives how long the problem has lasted."""
    open_ = [a for a in alerts if a.get("status") in ("open", "acknowledged")]
    order = list(LEVELS)
    if open_:
        worst = min(
            open_, key=lambda a: order.index(_SEVERITY_LEVEL.get(a.get("severity", ""), "warning"))
        )
        level = _SEVERITY_LEVEL.get(worst.get("severity", ""), "warning")
        headline, meaning, action = fault_text(
            int(worst["event_class"]), str(worst.get("class_name", "")), worst.get("source")
        )
        since = worst.get("opened_at")
        return {
            "level": level,
            "headline": headline,
            "meaning": meaning,
            "action": action,
            "duration": duration_text(since, now) if since is not None and now is not None else "",
            "acknowledged": worst.get("status") == "acknowledged",
            "more": len(open_) - 1,
        }
    base = well_status(p_fault, threshold, None)
    if base == "No data":
        level, headline, meaning = "no_data", "No readings", "No sensor data received yet."
    elif base == "Warning":
        level, headline = "watch", "Readings look unusual"
        meaning = "Something may be starting, but no fault is confirmed yet."
    else:
        level, headline, meaning = "normal", "Working normally", "No fault detected."
    action = "Keep an eye on it." if level == "watch" else ""
    return {
        "level": level,
        "headline": headline,
        "meaning": meaning,
        "action": action,
        "duration": "",
        "acknowledged": False,
        "more": 0,
    }


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
