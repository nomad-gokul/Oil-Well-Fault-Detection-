"""Load test: N wells streaming 1 Hz readings, plus stateless /predict calls and dashboard reads.

    make serve   # or any running API
    locust -f tests/load/locustfile.py --headless -u 60 -r 10 -t 60s --host http://localhost:8000

Set OWFD_API_KEY if the API requires one. Readings come from the bundled demo wells.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import numpy as np
from locust import HttpUser, between, constant_pacing, task

from owfd.config import PROJECT_ROOT
from owfd.simulator.replay import load_demo

SCENARIOS = load_demo(PROJECT_ROOT / "demo_data")
HEADERS = {"X-API-Key": os.environ["OWFD_API_KEY"]} if os.environ.get("OWFD_API_KEY") else {}
_ids = itertools.count()


def readings(df: Any, start: int, n: int, t0: int) -> list[dict[str, Any]]:
    rows = df.iloc[start : start + n]
    return [
        {
            "timestamp": str(np.datetime64("2030-01-01T00:00:00") + np.timedelta64(t0 + i, "s")),
            "values": {
                k: None if np.isnan(v) else float(v) for k, v in zip(rows.columns, row, strict=True)
            },
        }
        for i, row in enumerate(rows.to_numpy())
    ]


class Well(HttpUser):
    """One live well: posts one reading per second to /ingest."""

    weight = 8
    wait_time = constant_pacing(1.0)

    def on_start(self) -> None:
        n = next(_ids)
        self.scenario = SCENARIOS[n % len(SCENARIOS)]
        self.well_id = f"load-{n:04d}"
        self.pos = 0

    @task
    def ingest(self) -> None:
        df = self.scenario.data
        if self.pos >= len(df):
            self.pos = 0
        body = {"well_id": self.well_id, "readings": readings(df, self.pos, 1, self.pos)}
        self.client.post("/ingest", json=body, headers=HEADERS, name="/ingest (1 reading)")
        self.pos += 1


class Analyst(HttpUser):
    """Stateless scoring of a 5-minute window, and dashboard-style reads."""

    weight = 2
    wait_time = between(0.5, 1.5)

    @task(3)
    def predict(self) -> None:
        df = SCENARIOS[3].data
        start = np.random.randint(0, len(df) - 300)
        body = {"well_id": "adhoc", "readings": readings(df, start, 300, 0)}
        self.client.post("/predict", json=body, name="/predict (300 readings)")

    @task(1)
    def alerts(self) -> None:
        self.client.get("/alerts", params={"status": "open"}, name="/alerts")
