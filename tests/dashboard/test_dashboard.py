import json

import httpx
import pytest

from owfd.config import PROJECT_ROOT
from owfd.dashboard.client import ApiUnavailableError, Client, well_status


def mock_client(routes: dict[str, object], status: int = 200) -> Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"error": {}})
        return httpx.Response(status, content=json.dumps(body))

    return Client("http://api", api_key="k", transport=httpx.MockTransport(handler))


def test_client_parses_frames() -> None:
    c = mock_client({
        "/wells/w/predictions": [{"timestamp": "2020-01-01T00:05:00", "p_fault": 0.2,
                                  "predicted_class": 0, "probabilities": [0.8] + [0.02] * 9}],
        "/wells/w/readings": {"timestamps": ["2020-01-01T00:00:00", "2020-01-01T00:00:10"],
                              "values": {"P-PDG": [1.0, None]}},
        "/alerts": [{"id": 1, "well_id": "w", "status": "open", "opened_at": "2020-01-01T00:05:00",
                     "acknowledged_at": None, "resolved_at": None}],
    })  # fmt: skip
    preds = c.predictions("w")
    assert str(preds["timestamp"].dtype).startswith("datetime64")
    readings = c.readings("w")
    assert len(readings) == 2 and readings["P-PDG"].isna().sum() == 1
    alerts = c.alerts()
    assert alerts.loc[0, "opened_at"].year == 2020
    assert c.http.headers["X-API-Key"] == "k"


def test_client_reports_unavailable_api() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    c = Client("http://api", transport=httpx.MockTransport(down))
    with pytest.raises(ApiUnavailableError):
        c.health()
    with pytest.raises(ApiUnavailableError):
        mock_client({"/health": {}}, status=503).health()


@pytest.mark.parametrize(
    ("p", "alert", "expected"),
    [(0.9, "open", "Fault"), (0.1, "acknowledged", "Fault"), (0.5, None, "Warning"),
     (0.1, None, "Normal"), (None, None, "No data"), (0.9, "resolved", "Warning")],
)  # fmt: skip
def test_well_status(p: float | None, alert: str | None, expected: str) -> None:
    assert well_status(p, threshold=0.6, alert_status=alert) == expected


def test_app_renders_when_api_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OWFD_API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(str(PROJECT_ROOT / "src/owfd/dashboard/app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("API unavailable" in w.value for w in at.warning)


def test_display_units() -> None:
    import pandas as pd

    from owfd.dashboard.client import display_units

    df = pd.DataFrame({"P-PDG": [2.2e7, 1.18e42], "T-TPT": [101.0, 102.0], "QGL": [0.5, None]})
    out, units = display_units(df)
    assert out["P-PDG"].iloc[0] == 220.0 and pd.isna(out["P-PDG"].iloc[1])  # placeholder hidden
    assert units == {"P-PDG": "bar", "T-TPT": "°C", "QGL": "m³/s"}
    assert out["T-TPT"].tolist() == [101.0, 102.0]


def test_app_reads_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Streamlit Community Cloud the API address comes from secrets, not the environment."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("OWFD_API_URL", raising=False)
    at = AppTest.from_file(str(PROJECT_ROOT / "src/owfd/dashboard/app.py"), default_timeout=30)
    at.secrets["OWFD_API_URL"] = "http://127.0.0.1:9"
    at.run()
    assert not at.exception
    assert any("http://127.0.0.1:9" in w.value for w in at.warning)


def test_scenario_caption() -> None:
    from owfd.dashboard.client import scenario_caption, scenario_source

    info = {"event_class": 4, "class_name": "Flow Instability", "source_well": "WELL-00004",
            "recorded_at": "2014-08-06T01:01:07"}  # fmt: skip
    assert scenario_source(info) == "WELL-00004, Aug 2014"
    assert scenario_caption(info) == "Real well: WELL-00004, Aug 2014 · actual: Flow Instability"
    normal = {"event_class": 0, "class_name": "Normal Operation", "source_well": None,
              "recorded_at": None}  # fmt: skip
    assert scenario_caption(normal) == "Actual: no fault (normal operation)"


def test_duration_text() -> None:
    from owfd.dashboard.client import duration_text

    t0 = "2020-01-01T00:00:00"
    assert duration_text(t0, "2020-01-01T00:00:30") == "just now"
    assert duration_text(t0, "2020-01-01T00:25:00") == "25 min"
    assert duration_text(t0, "2020-01-01T03:05:00") == "3 h 5 min"
    assert duration_text(t0, "2020-01-01T02:00:00+00:00") == "2 h"  # naive vs UTC-aware
    assert duration_text(t0, "2020-01-05T00:00:00") == "4 days"


def test_plain_status_uses_the_worst_open_alert() -> None:
    from owfd.dashboard.client import plain_status

    alerts = [
        {"status": "open", "severity": "low", "event_class": 4, "class_name": "Flow Instability",
         "opened_at": "2020-01-01T00:00:00", "source": "chronic"},
        {"status": "acknowledged", "severity": "high", "event_class": 8,
         "class_name": "Hydrate in Production Line", "opened_at": "2020-01-01T00:20:00"},
        {"status": "resolved", "severity": "high", "event_class": 2,
         "class_name": "Spurious Closure of DHSV", "opened_at": "2019-12-31T00:00:00"},
    ]  # fmt: skip
    s = plain_status(alerts, 0.99, 0.85, now="2020-01-01T00:45:00")
    assert s["level"] == "critical"
    assert s["headline"] == "Ice-like plug forming in the production line"
    assert s["action"].startswith("Act now")
    assert s["duration"] == "25 min" and s["acknowledged"] and s["more"] == 1

    chronic = plain_status(alerts[:1], 0.1, 0.85, now="2020-01-01T01:00:00")
    assert chronic["level"] == "watch" and chronic["headline"] == "Long-running unstable flow"


@pytest.mark.parametrize(
    ("p", "level"), [(0.1, "normal"), (0.8, "watch"), (None, "no_data")]
)  # fmt: skip
def test_plain_status_without_alerts(p: float | None, level: str) -> None:
    from owfd.dashboard.client import plain_status

    resolved = [{"status": "resolved", "severity": "high", "event_class": 2, "class_name": "x",
                 "opened_at": "2020-01-01T00:00:00"}]  # fmt: skip
    s = plain_status(resolved, p, 0.85)
    assert s["level"] == level and s["duration"] == "" and s["more"] == 0


def test_every_fault_class_has_plain_text() -> None:
    from owfd.dashboard.client import PLAIN_FAULTS, fault_text
    from owfd.data.schema import CLASS_NAMES

    assert set(PLAIN_FAULTS) == set(CLASS_NAMES) - {0}
    assert fault_text(42, "Something new")[0] == "Something new"  # unknown class: API name
