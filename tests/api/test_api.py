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


def test_missing_bundle_fails_clearly(tmp_path: Path) -> None:
    s = Settings(serving_bundle=None, artifacts_dir=tmp_path, database_url="sqlite:///:memory:")
    with pytest.raises(RuntimeError, match="No serving bundle"), TestClient(create_app(s)):
        pass


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
