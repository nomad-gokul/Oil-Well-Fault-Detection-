"""Replay 3W instances as live wells (sped up), into the API or straight into the model.

Usage::

    python -m owfd.simulator.replay --api http://localhost:8000 --speed 60
    python -m owfd.simulator.replay --in-process --classes 7 8   # no API needed

Scenarios come from ``demo_data/`` (bundled, see ``owfd.simulator.make_demo``) or from the
full dataset's test split with ``--from-dataset``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from owfd.config import PROJECT_ROOT
from owfd.data.schema import CLASS_NAMES

logger = logging.getLogger(__name__)

DEMO_DIR = PROJECT_ROOT / "demo_data"
SHORT = {
    0: "normal",
    1: "bsw",
    2: "dhsv",
    3: "slugging",
    4: "instability",
    5: "prod-loss",
    6: "pck-restriction",
    7: "pck-scaling",
    8: "hydrate-prod",
    9: "hydrate-service",
}


@dataclass
class Scenario:
    """One instance to replay as a well."""

    well_id: str
    event_class: int
    data: pd.DataFrame  # 1 Hz sensor readings (DatetimeIndex); labels are not sent
    labels: pd.Series = field(repr=False)  # ground truth, for evaluation/display only
    source: str = ""


def describe_source(source: str) -> dict[str, str | None]:
    """The real well and recording start of a 3W source such as ``4/WELL-00004_20140806010107``
    (class folder / file name). Either is None when the source does not follow that pattern."""
    well, _, stamp = source.rsplit("/", 1)[-1].partition("_")
    try:
        recorded = datetime.strptime(stamp, "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        recorded = None
    return {"source_well": well if well.startswith("WELL-") else None, "recorded_at": recorded}


def demo_catalog(directory: Path = DEMO_DIR) -> dict[str, dict[str, Any]]:
    """What each bundled demo well replays, read from ``manifest.json`` alone (no readings):
    ``{well_id: {event_class, class_name, source, source_well, recorded_at}}``."""
    path = directory / "manifest.json"
    if not path.exists():
        return {}
    catalog = {}
    for item in json.loads(path.read_text())["scenarios"]:
        cls, source = int(item["event_class"]), str(item.get("source", ""))
        catalog[item["well_id"]] = {
            "event_class": cls,
            "class_name": CLASS_NAMES[cls],
            "source": source,
            **describe_source(source),
        }
    return catalog


def load_demo(directory: Path = DEMO_DIR, classes: list[int] | None = None) -> list[Scenario]:
    """Scenarios from the bundled demo set (``manifest.json`` + one parquet per scenario)."""
    manifest = json.loads((directory / "manifest.json").read_text())
    out = []
    for item in manifest["scenarios"]:
        if classes is not None and item["event_class"] not in classes:
            continue
        df, labels = read_scenario(directory / item["file"])
        out.append(Scenario(item["well_id"], item["event_class"], df, labels, item["source"]))
    return out


def write_scenario(path: Path, df: pd.DataFrame, labels: pd.Series) -> None:
    """Save readings + labels as compressed ``.npz`` (exact float64; no PyArrow needed)."""
    np.savez_compressed(
        path,
        timestamps=pd.DatetimeIndex(df.index).as_unit("ns").asi8,  # unit is explicit
        values=df.to_numpy(np.float64),
        columns=np.array(df.columns, dtype=str),
        labels=labels.to_numpy(dtype=np.int64, na_value=-1),
    )


def read_scenario(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Inverse of :func:`write_scenario`: (readings, labels with <NA> where unlabelled)."""
    with np.load(path) as z:
        index = pd.DatetimeIndex(z["timestamps"].astype("datetime64[ns]"), name="timestamp")
        df = pd.DataFrame(z["values"], index=index, columns=[str(c) for c in z["columns"]])
        raw = z["labels"]
    labels = pd.Series(pd.array(np.where(raw < 0, None, raw), dtype="Int16"), index=index)
    return df, labels


@dataclass
class Replay:
    """Iterates over all scenarios together, ``speed`` readings per scenario per tick.

    Timestamps are shifted so every stream starts at ``start`` (default: now, UTC).
    """

    scenarios: list[Scenario]
    speed: int = 60
    loop: bool = False
    start: pd.Timestamp | None = None
    positions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start = self.start or pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")
        self._offsets = {s.well_id: self.start - s.data.index[0] for s in self.scenarios}

    def done(self) -> bool:
        """True when every scenario has been fully sent (never, with ``loop``)."""
        return not self.loop and all(
            self.positions.get(s.well_id, 0) >= len(s.data) for s in self.scenarios
        )

    def tick(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """The next chunk of readings for every scenario that has data left."""
        for s in self.scenarios:
            pos = self.positions.get(s.well_id, 0)
            if pos >= len(s.data):
                if not self.loop:
                    continue
                # Restart: shift timestamps so the stream continues forward in time.
                self._offsets[s.well_id] += (
                    s.data.index[-1] - s.data.index[0] + pd.Timedelta(seconds=1)
                )
                pos = 0
            chunk = s.data.iloc[pos : pos + self.speed].copy()
            chunk.index = chunk.index + self._offsets[s.well_id]
            self.positions[s.well_id] = pos + len(chunk)
            yield s.well_id, chunk


Sink = Callable[[str, pd.DataFrame], Awaitable[None]]


async def run(replay: Replay, sink: Sink, tick_s: float = 1.0, max_ticks: int | None = None) -> int:
    """Send one tick per ``tick_s`` seconds to ``sink`` until done; returns ticks sent."""
    ticks = 0
    while not replay.done() and (max_ticks is None or ticks < max_ticks):
        await asyncio.gather(*(sink(well, chunk) for well, chunk in replay.tick()))
        ticks += 1
        if tick_s > 0:
            await asyncio.sleep(tick_s)
    return ticks


def http_sink(
    api: str, api_key: str | None, timeout: float = 30
) -> tuple[Sink, Callable[[], Awaitable[None]]]:
    """A sink posting to ``{api}/ingest``, and a function closing its HTTP client."""
    import httpx

    client = httpx.AsyncClient(
        base_url=api, timeout=timeout, headers={"X-API-Key": api_key} if api_key else {}
    )

    async def send(well_id: str, chunk: pd.DataFrame) -> None:
        readings = [
            {
                "timestamp": ts.isoformat(),
                "values": {k: (None if np.isnan(v) else float(v)) for k, v in row.items()},
            }
            for ts, row in zip(chunk.index, chunk.to_dict("records"), strict=True)
        ]
        r = await client.post("/ingest", json={"well_id": well_id, "readings": readings})
        r.raise_for_status()
        for alert in r.json()["alerts"]:
            logger.info(
                "%s alert %s on %s: %s", alert["status"], alert["id"], well_id, alert["class_name"]
            )

    return send, client.aclose


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Replay 3W instances as live wells.")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--speed", type=int, default=60, help="readings per well per second")
    parser.add_argument("--classes", type=int, nargs="*", help="only these event classes")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="score locally, no API")
    parser.add_argument("--demo-dir", type=Path, default=DEMO_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scenarios = load_demo(args.demo_dir, args.classes)
    logger.info(
        "Replaying %d wells at %dx: %s",
        len(scenarios),
        args.speed,
        ", ".join(s.well_id for s in scenarios),
    )
    replay = Replay(scenarios, speed=args.speed, loop=args.loop)

    if args.in_process:
        from owfd.config import get_settings
        from owfd.inference.alerting import AlertConfig, AlertEngine
        from owfd.inference.predictor import Predictor, StreamingPredictor

        settings = get_settings()
        bundle = sorted((settings.resolve(settings.artifacts_dir) / "serving").iterdir())[-1]
        predictor = Predictor.load(bundle)
        stream = StreamingPredictor(predictor)
        engine = AlertEngine(
            AlertConfig.from_files(
                predictor.manifest["alert"], settings.resolve(settings.alerts_file)
            )
        )

        async def local(well_id: str, chunk: pd.DataFrame) -> None:
            for p in stream.update(well_id, chunk):
                for ev in engine.process(p):
                    logger.info(
                        "%s: %s %s (P=%.2f) at %s",
                        well_id,
                        ev.kind,
                        ev.alert.class_name,
                        ev.alert.peak_probability,
                        p.timestamp,
                    )

        asyncio.run(run(replay, local, tick_s=0))
        return

    send, close = http_sink(args.api, args.api_key)

    async def go() -> None:
        try:
            await run(replay, send)
        finally:
            await close()

    asyncio.run(go())


if __name__ == "__main__":
    main()


__all__ = ["CLASS_NAMES", "SHORT", "Replay", "Scenario", "load_demo", "run"]
