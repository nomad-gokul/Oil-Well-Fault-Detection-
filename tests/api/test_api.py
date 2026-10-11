from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from owfd.api.main import Service, create_app
from owfd.config import Settings
from owfd.inference.predictor import Prediction
from tests.bundle import build_bundle, raw_instance

KEY = {"X-API-Key": "secret"}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_bundle(tmp_path_factory.mktemp("bundle"))


def settings(bundle: Path, db: Path, **kw: Any) -> Settings:
    return Settings(
        serving_bundle=bundle,
        database_url=f"sqlite:///{db}",
        api_key="secret",
        alerts_file=Path("configs/alerts.yaml"),
        **kw,
    )


@pytest.fixture
def client(bundle: Path, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings(bundle, tmp_path / "db.sqlite"))) as c:
        yield c


def payload(df: pd.DataFrame, well: str = "WELL-T") -> dict[str, Any]:
    readings = [
        {
            "timestamp": ts.isoformat(),
            "values": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
        }
        for ts, row in df.iterrows()
    ]
    return {"well_id": well, "readings": readings}


def test_health_model_and_request_id(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["model_version"] == "test" and len(r.headers["X-Request-ID"]) == 12
    assert client.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"
    m = client.get("/model").json()
    assert m["window_s"] == 300 and m["classes"]["8"] == "Hydrate in Production Line"
    assert "P-PDG" in m["variables"]


def test_predict(client: TestClient) -> None:
    r = client.post("/predict", json=payload(raw_instance(400)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert 0 <= body["p_fault"] <= 1 and len(body["drivers"]) == 5
    assert body["timestamp"].startswith("2017-01-01T00:06:39")


def test_predict_validation_errors(client: TestClient) -> None:
    short = client.post("/predict", json=payload(raw_instance(100)))
    assert short.status_code == 422 and "300" in short.json()["error"]["message"]
    bad = client.post("/predict", json={"readings": [{"timestamp": "nope", "values": {}}]})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "validation_error" and bad.json()["error"]["request_id"]
    reversed_ = payload(raw_instance(400))
    reversed_["readings"].reverse()
    assert client.post("/predict", json=reversed_).status_code == 422


def test_ingest_requires_key_and_streams(client: TestClient) -> None:
    body = payload(raw_instance(400))
    assert client.post("/ingest", json=body).status_code == 401
    assert client.post("/ingest", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    r = client.post("/ingest", json=body, headers=KEY)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2  # windows ending at second 299 and 359
    stored = client.get("/wells/WELL-T/predictions").json()
    assert len(stored) == 2 and stored[0]["timestamp"] < stored[1]["timestamp"]
    assert client.get("/wells").json() == ["WELL-T"]


def scripted(svc: Service, p_faults: list[float], fault_class: int = 8) -> None:
    """Replace the model's stream with scripted predictions (one per call to update)."""
    queue = list(p_faults)

    def update(well_id: str, readings: pd.DataFrame, keep_inputs: bool = False) -> list[Prediction]:
        out = []
        for ts in readings.index[59::60]:
            p = queue.pop(0)
            probs = [1 - p] + [0.0] * 9
            probs[fault_class] = p
            x = raw_instance(300).iloc[:0]  # inputs unused by the stub explainer path
            out.append(Prediction(well_id, ts, fault_class if p > 0.5 else 0, "x", p, probs, None))
            del x
        return out

    svc.stream.update = update  # type: ignore[method-assign]


def test_alert_lifecycle_websocket_and_restart(bundle: Path, tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    sent: list[dict[str, Any]] = []
    with TestClient(create_app(settings(bundle, db, alert_webhook_url="http://hook"))) as c:
        svc: Service = c.app.state.service  # type: ignore[attr-defined]
        threshold = svc.alerts.config.threshold
        k = svc.alerts.config.k

        async def fake_notify(alert: Any) -> None:
            sent.append(alert.model_dump())

        svc.notify = fake_notify  # type: ignore[method-assign]
        scripted(svc, [0.05] * 3 + [min(0.99, threshold + 0.3)] * (k + 2))
        readings = raw_instance(60 * (k + 5))
        with c.websocket_connect("/ws/all") as ws:
            r = c.post("/ingest", json=payload(readings), headers=KEY)
            assert r.status_code == 200, r.text
            alerts = r.json()["alerts"]
            assert [a["status"] for a in alerts][0] == "open"
            assert alerts[0]["class_name"] == "Hydrate in Production Line"
            assert alerts[0]["severity"] == "high"
            kinds = [
                ws.receive_json()["type"] for _ in range(len(r.json()["predictions"]) + len(alerts))
            ]
            assert kinds.count("prediction") == k + 5 and "alert" in kinds
        listed = c.get("/alerts", params={"status": "open"}).json()
        assert len(listed) == 1 and sent and sent[0]["well_id"] == "WELL-T"
        alert_id = listed[0]["id"]
        assert c.post(f"/alerts/{alert_id}/ack").status_code == 401
        acked = c.post(f"/alerts/{alert_id}/ack", headers=KEY).json()
        assert acked["status"] == "acknowledged" and acked["acknowledged_at"]
        assert c.post("/alerts/999/ack", headers=KEY).status_code == 404
        assert c.get("/alerts", params={"status": "bogus"}).status_code == 422

    # After a restart the open alert is restored: continued high P(fault) adds no new alert.
    with TestClient(create_app(settings(bundle, db))) as c:
        svc = c.app.state.service  # type: ignore[attr-defined]
        assert svc.alerts.active("WELL-T") is not None
        scripted(svc, [0.99] * 5)
        later = raw_instance(300)
        later.index = later.index + pd.Timedelta(days=1)
        r = c.post("/ingest", json=payload(later), headers=KEY)
        assert [
            a for a in r.json()["alerts"] if a["status"] == "open" and a["id"] != alert_id
        ] == []
        assert len(c.get("/alerts").json()) == 1


def test_bundle_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import owfd.config as config
    from owfd.api.main import latest_bundle

    s = Settings(serving_bundle=None, artifacts_dir=tmp_path, database_url="sqlite:///:memory:")
    assert latest_bundle(s).parts[-2:] == ("deploy", "bundle")  # committed production bundle
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # a checkout without deploy/bundle
    with pytest.raises(RuntimeError, match="No serving bundle"):
        latest_bundle(Settings(serving_bundle=None, artifacts_dir=tmp_path))


def test_readings_endpoint(client: TestClient) -> None:
    client.post("/ingest", json=payload(raw_instance(400)), headers=KEY)
    r = client.get("/wells/WELL-T/readings", params={"minutes": 5, "every_s": 10}).json()
    assert len(r["timestamps"]) == 30 and set(r["values"]) >= {"P-PDG", "QGL"}
    assert r["values"]["QGL"][0] is None  # absent sensor -> null
    assert client.get("/wells/nope/readings").json()["timestamps"] == []


def test_demo_mode_endpoints(bundle: Path, tmp_path: Path) -> None:
    import time

    from tests.simulator.test_replay import make_demo_dir

    demo = make_demo_dir(tmp_path / "demo", {"demo-a": 400, "demo-b": 360})
    s = settings(bundle, tmp_path / "db.sqlite", demo_dir=demo)
    with TestClient(create_app(s)) as c:
        assert c.get("/demo").json() == {"running": False}
        assert c.post("/demo/start").status_code == 401
        started = c.post("/demo/start", params={"speed": 400, "loop": False}, headers=KEY).json()
        assert started["running"] and started["wells"] == ["demo-a", "demo-b"]
        for _ in range(50):
            if not c.get("/demo").json()["running"]:
                break
            time.sleep(0.1)
        assert c.get("/wells").json() == ["demo-a", "demo-b"]
        assert len(c.get("/wells/demo-a/predictions").json()) == 2
        assert c.post("/demo/stop", headers=KEY).json()["running"] is False


def test_demo_mode_starts_on_startup(bundle: Path, tmp_path: Path) -> None:
    from tests.simulator.test_replay import make_demo_dir

    demo = make_demo_dir(tmp_path / "demo", {"demo-a": 400})
    s = settings(bundle, tmp_path / "db.sqlite", demo_dir=demo, demo_mode=True)
    with TestClient(create_app(s)) as c:
        status = c.get("/demo").json()
        assert status["running"] and status["loop"] is True


def test_metrics_and_drift(bundle: Path, tmp_path: Path) -> None:
    import json

    from owfd.monitoring.drift import build_reference

    rng = np.random.default_rng(0)
    windows = [pd.DataFrame({"P-PDG": rng.normal(0, 1, 3600)}) for _ in range(30)]
    (bundle / "reference.json").write_text(json.dumps(build_reference(windows, ["P-PDG"])))
    try:
        with TestClient(create_app(settings(bundle, tmp_path / "db.sqlite"))) as c:
            empty = c.get("/monitoring/drift").json()
            assert empty["drift"] is False and empty["sensors"]["P-PDG"]["wells_checked"] == 0
            c.post("/ingest", json=payload(raw_instance(2400)), headers=KEY)
            c.get("/health")
            body = c.get("/metrics").text
            assert 'owfd_http_requests_total{method="POST",path="/ingest",status="200"} 1.0' in body
            assert "owfd_predictions_total" in body and 'owfd_model_info{model="ensemble"' in body
            drift = c.get("/monitoring/drift").json()
            assert "WELL-T" in drift["wells"] and drift["sensors"]["P-PDG"]["wells_checked"] == 1
            assert "owfd_drift_max_share_unusual" in c.get("/metrics").text
    finally:
        (bundle / "reference.json").unlink()
    with TestClient(create_app(settings(bundle, tmp_path / "db2.sqlite"))) as c:
        assert c.get("/monitoring/drift").status_code == 404  # bundle without reference


def test_rate_limiter_window() -> None:
    from owfd.api.main import RateLimiter

    rl = RateLimiter(2)
    assert rl.allow("a", now=0) and rl.allow("a", now=1) and not rl.allow("a", now=2)
    assert rl.allow("b", now=2)  # per client
    assert rl.allow("a", now=61)  # new minute
    assert all(RateLimiter(0).allow("a", now=0) for _ in range(100))  # disabled


def test_rate_limit_returns_429(bundle: Path, tmp_path: Path) -> None:
    s = settings(bundle, tmp_path / "db.sqlite", rate_limit_per_min=2)
    with TestClient(create_app(s)) as c:
        body = payload(raw_instance(300))
        codes = [c.post("/predict", json=body).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.post("/predict", json=body, headers={"X-Forwarded-For": "203.0.113.9"})
        assert r.status_code == 200  # another client
        assert c.get("/health").status_code == 200  # reads are not limited


def test_in_memory_database_creates_no_file(bundle: Path, tmp_path: Path) -> None:
    from owfd.config import PROJECT_ROOT

    s = Settings(serving_bundle=bundle, database_url="sqlite:///:memory:", api_key="secret")
    with TestClient(create_app(s)) as c:
        assert c.get("/health").status_code == 200
        # Request threads must see the tables created at startup (one shared connection).
        c.post("/ingest", json=payload(raw_instance(400)), headers=KEY)
        assert c.get("/alerts").status_code == 200
        assert len(c.get("/wells/WELL-T/predictions").json()) == 2
    assert not (PROJECT_ROOT / ":memory:").exists()


def test_json_log_lines_keep_tracebacks() -> None:
    import json
    import logging
    import sys

    from owfd.api.main import JsonFormatter

    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    line = json.loads(JsonFormatter().format(record))
    assert line["msg"] == "failed" and "ValueError: boom" in line["exc_info"]


def test_chronic_alerts_feedback_and_profile(tmp_path: Path) -> None:
    from tests.bundle import add_profile

    bundle = add_profile(build_bundle(tmp_path / "b", context=True), "WELL-T", k=3)
    with TestClient(create_app(settings(bundle, tmp_path / "db.sqlite"))) as c:
        svc: Service = c.app.state.service  # type: ignore[attr-defined]
        assert c.get("/wells/WELL-T/profile").json()["has_profile"] is True
        assert c.get("/wells/other/profile").json()["has_profile"] is False
        assert c.get("/model").json()["chronic"]["wells"] == ["WELL-T"]
        svc.profiles.score = lambda well, f: np.full(len(f), 0.97)  # type: ignore[method-assign]
        r = c.post("/ingest", json=payload(raw_instance(60 * 9)), headers=KEY)
        assert r.status_code == 200, r.text
        chronic = [a for a in r.json()["alerts"] if a["source"] == "chronic"]
        assert chronic and chronic[0]["class_name"] == "Flow Instability"
        assert len(chronic[0]["top_drivers"]) == 5
        alert_id = chronic[0]["id"]
        # Acknowledge goes to the chronic engine's live alert.
        assert c.post(f"/alerts/{alert_id}/ack", headers=KEY).json()["status"] == "acknowledged"
        assert svc.chronic.active("WELL-T").status == "acknowledged"
        # Operator feedback is stored and listed.
        body = {"label": "confirmed", "note": "unstable since 10:00"}
        assert c.post(f"/alerts/{alert_id}/feedback", json=body).status_code == 401
        fb = c.post(f"/alerts/{alert_id}/feedback", json=body, headers=KEY).json()
        assert fb["label"] == "confirmed" and fb["well_id"] == "WELL-T"
        assert c.post(f"/alerts/{alert_id}/feedback", json={"label": "maybe"},
                      headers=KEY).status_code == 422  # fmt: skip
        assert c.post("/alerts/999/feedback", json=body, headers=KEY).status_code == 404
        assert [f["alert_id"] for f in c.get("/feedback").json()] == [alert_id]
    # After a restart the open chronic alert is restored into the chronic engine only.
    with TestClient(create_app(settings(bundle, tmp_path / "db.sqlite"))) as c:
        svc = c.app.state.service  # type: ignore[attr-defined]
        restored = svc.chronic.active("WELL-T")
        assert restored is not None and restored.source == "chronic" and restored.id == alert_id
        onset = svc.alerts.active("WELL-T")
        assert onset is None or onset.source == "onset"


def test_old_database_gets_the_source_column(tmp_path: Path) -> None:
    import sqlite3

    from owfd.api.db import make_engine

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE alerts (id INTEGER PRIMARY KEY, well_id VARCHAR(64), event_class INTEGER, "
        "severity VARCHAR(16), status VARCHAR(16), opened_at DATETIME, acknowledged_at DATETIME, "
        "resolved_at DATETIME, peak_probability FLOAT, top_drivers JSON)"
    )
    con.execute("INSERT INTO alerts VALUES (1, 'w', 8, 'high', 'open', '2020-01-01 00:00:00', "
                "NULL, NULL, 0.9, NULL)")  # fmt: skip
    con.commit()
    con.close()
    from owfd.api.db import Store

    store = Store(make_engine(f"sqlite:///{db}"))
    assert store.get_alert(1)[0].source == "onset"
