"""FastAPI service: predictions, streaming ingestion, alerts and a live WebSocket feed.

Run locally with ``make serve`` (http://localhost:8000/docs).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from owfd.api.db import Store, make_engine
from owfd.api.schemas import (
    AlertOut,
    Driver,
    ErrorOut,
    HealthOut,
    IngestRequest,
    IngestResponse,
    ModelOut,
    PredictionOut,
    PredictRequest,
    Reading,
)
from owfd.config import Settings, get_settings
from owfd.data.schema import CLASS_NAMES
from owfd.inference.alerting import Alert, AlertConfig, AlertEngine, AlertEvent
from owfd.inference.predictor import Prediction, Predictor, StreamingPredictor

logger = logging.getLogger("owfd.api")

RECENT_S = 2 * 3600  #: seconds of raw readings kept per well for the dashboard


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def latest_bundle(settings: Settings) -> Path:
    """The configured bundle, or the newest one under ``artifacts/serving``."""
    if settings.serving_bundle is not None:
        return settings.resolve(settings.serving_bundle)
    root = settings.resolve(settings.artifacts_dir) / "serving"
    bundles = (
        sorted(p for p in root.glob("*") if (p / "manifest.json").exists()) if root.exists() else []
    )
    if not bundles:
        raise RuntimeError(
            f"No serving bundle found under {root}; run python -m owfd.models.export"
        )
    return bundles[-1]


class Service:
    """Everything the endpoints share (created once at startup)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bundle = latest_bundle(settings)
        self.predictor = Predictor.load(self.bundle)
        self.stream = StreamingPredictor(self.predictor)
        self.manifest = self.predictor.manifest
        self.version: str = self.manifest["version"]
        alerts_file = settings.resolve(settings.alerts_file)
        self.alerts = AlertEngine(AlertConfig.from_files(self.manifest["alert"], alerts_file))
        db_url = settings.database_url
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
            db_url = f"sqlite:///{settings.resolve(Path(db_url.removeprefix('sqlite:///')))}"
        self.store = Store(make_engine(db_url))
        self.alerts.restore(self.store.unresolved_alerts())
        self.explainer = self._explainer()
        self.subscribers: dict[str, set[WebSocket]] = defaultdict(set)
        self.started = time.time()
        self.lock = asyncio.Lock()
        # Recent raw readings per well, for the dashboard's sensor charts (memory only).
        self.recent: dict[str, deque[pd.DataFrame]] = defaultdict(deque)
        self.demo_task: asyncio.Task[int] | None = None
        self.demo_info: dict[str, Any] = {}

    def remember(self, well_id: str, df: pd.DataFrame) -> None:
        """Keep the last ``RECENT_S`` seconds of readings for ``well_id``."""
        chunks = self.recent[well_id]
        chunks.append(df)
        cutoff = df.index[-1] - pd.Timedelta(seconds=RECENT_S)
        while chunks and chunks[0].index[-1] < cutoff:
            chunks.popleft()

    def recent_readings(self, well_id: str, minutes: int, every_s: int) -> pd.DataFrame:
        """The well's readings over the last ``minutes``, one row per ``every_s`` seconds."""
        chunks = self.recent.get(well_id)
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks)
        df = df[df.index > df.index[-1] - pd.Timedelta(minutes=minutes)]
        return df.iloc[::every_s]

    async def ingest(self, well_id: str, df: pd.DataFrame) -> IngestResponse:
        """Score new readings, update alerts, persist, notify and broadcast."""
        async with self.lock:
            self.remember(well_id, df)
            preds = self.stream.update(well_id, df, keep_inputs=self.explainer is not None)
            self.store.save_predictions(preds, self.version)
            events: list[AlertEvent] = []
            for p in preds:
                events += self.alerts.process(p)
            alerts_out = []
            for ev in events:
                drivers = None
                if ev.kind == "opened":
                    trigger = next(p for p in preds if p.timestamp == ev.alert.opened_at)
                    trigger.predicted_class = ev.alert.event_class  # explain the alert's class
                    d = self.drivers(trigger)
                    drivers = [x.model_dump() for x in d] if d else None
                self.store.save_alert(ev.alert, drivers)
                out = self.alert_out(ev.alert, drivers)
                alerts_out.append(out)
                if ev.kind == "opened":
                    asyncio.create_task(self.notify(out))
        pred_out = [self.prediction_out(p) for p in preds]
        for po in pred_out:
            await self.broadcast(
                well_id, {"type": "prediction", "data": po.model_dump(mode="json")}
            )
        for a in alerts_out:
            await self.broadcast(well_id, {"type": "alert", "data": a.model_dump(mode="json")})
        return IngestResponse(predictions=pred_out, alerts=alerts_out)

    def start_demo(self, classes: list[int] | None, speed: int, loop: bool) -> dict[str, Any]:
        """Replay the bundled demo wells through :meth:`ingest` in the background."""
        from owfd.simulator.replay import Replay, load_demo, run

        self.stop_demo()
        scenarios = load_demo(self.settings.resolve(self.settings.demo_dir), classes)
        replay = Replay(scenarios, speed=speed, loop=loop)

        async def sink(well_id: str, chunk: pd.DataFrame) -> None:
            await self.ingest(well_id, chunk)

        self.demo_task = asyncio.create_task(run(replay, sink))
        self.demo_info = {"wells": [s.well_id for s in scenarios], "speed": speed, "loop": loop,
                          "started": pd.Timestamp.now(tz="UTC").isoformat()}  # fmt: skip
        return self.demo_status()

    def stop_demo(self) -> dict[str, Any]:
        """Stop a running demo replay."""
        if self.demo_task is not None and not self.demo_task.done():
            self.demo_task.cancel()
        return self.demo_status()

    def demo_status(self) -> dict[str, Any]:
        """Whether the demo replay is running, and its settings."""
        running = self.demo_task is not None and not self.demo_task.done()
        return {"running": running, **self.demo_info}

    def _explainer(self) -> Any:
        xgb_members = [m for m in self.manifest["members"] if m["kind"] == "xgb"]
        if not xgb_members:
            return None
        from owfd.models.explain import Explainer

        return Explainer.from_files(
            self.bundle / xgb_members[0]["file"], xgb_members[0]["features"]
        )

    def drivers(self, p: Prediction) -> list[Driver] | None:
        """Top SHAP drivers for a prediction (XGBoost bundles only)."""
        if self.explainer is None or p.inputs is None:
            return None
        x, base = p.inputs
        feats = self.predictor.xgb_features(x[None], base[None])
        out = self.explainer.explain_window(feats[0], top=5, target=p.predicted_class or None)
        return [Driver(feature=d.feature, value=d.value, contribution=d.contribution) for d in out]

    def prediction_out(self, p: Prediction, with_drivers: bool = False) -> PredictionOut:
        """API representation of a prediction."""
        return PredictionOut(
            well_id=p.well_id,
            timestamp=p.timestamp.to_pydatetime(),
            predicted_class=p.predicted_class,
            class_name=p.class_name,
            p_fault=p.p_fault,
            is_fault=p.p_fault > self.alerts.config.threshold,
            probabilities={CLASS_NAMES[i]: v for i, v in enumerate(p.probabilities)},
            drivers=self.drivers(p) if with_drivers else None,
        )

    @staticmethod
    def alert_out(a: Alert, drivers: list[dict[str, Any]] | None = None) -> AlertOut:
        """API representation of an alert."""
        ts = lambda v: v.to_pydatetime() if v is not None else None  # noqa: E731
        return AlertOut(
            id=a.id or 0,
            well_id=a.well_id,
            event_class=a.event_class,
            class_name=a.class_name,
            severity=a.severity,
            status=a.status.value,
            opened_at=a.opened_at.to_pydatetime(),
            acknowledged_at=ts(a.acknowledged_at),
            resolved_at=ts(a.resolved_at),
            peak_probability=a.peak_probability,
            top_drivers=[Driver(**d) for d in drivers] if drivers else None,
        )

    async def broadcast(self, well_id: str, message: dict[str, Any]) -> None:
        """Send a message to the well's subscribers and to ``/ws/all``."""
        for ws in list(self.subscribers.get(well_id, ())) + list(self.subscribers.get("all", ())):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - a dead socket must not break ingestion
                self.subscribers[well_id].discard(ws)
                self.subscribers["all"].discard(ws)

    async def notify(self, alert: AlertOut) -> None:
        """Post a new alert to the webhook, if configured (failures are logged, not raised)."""
        url = self.settings.alert_webhook_url
        if not url:
            return
        text = (
            f":rotating_light: {alert.severity.upper()} alert on {alert.well_id}: "
            f"{alert.class_name} (P={alert.peak_probability:.2f}) "
            f"at {alert.opened_at:%Y-%m-%d %H:%M}"
        )
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(url, json={"text": text, "content": text})
        except httpx.HTTPError as exc:
            logger.warning("Alert webhook failed: %s", exc)


def get_service(request: Request) -> Service:
    """The application's :class:`Service` (FastAPI dependency)."""
    svc: Service = request.app.state.service
    return svc


# Module level, so FastAPI can resolve it from the (postponed) annotations.
Svc = Annotated[Service, Depends(get_service)]


def readings_frame(readings: list[Reading]) -> pd.DataFrame:
    """Readings -> DataFrame with a DatetimeIndex (timezone-naive UTC) and sensor columns."""
    index = pd.DatetimeIndex([pd.Timestamp(r.timestamp) for r in readings], name="timestamp")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame([r.values for r in readings], index=index, dtype="float64")
    if not index.is_monotonic_increasing:
        raise HTTPException(422, "readings must be in increasing time order")
    return df


def error(status: int, code: str, message: str, request: Request) -> JSONResponse:
    """The common error envelope."""
    rid = getattr(request.state, "request_id", "")
    body = {"error": {"code": code, "message": message, "request_id": rid}}
    return JSONResponse(status_code=status, content=body)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (tests pass their own settings)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = Service(settings)
        app.state.service = svc
        logger.info("Loaded bundle %s", svc.version)
        if settings.demo_mode:
            svc.start_demo(None, settings.demo_speed, loop=True)
        yield
        svc.stop_demo()

    app = FastAPI(
        title="Oil Well Fault Detection API",
        version="1.0.0",
        description="Early warning of operational faults in offshore oil wells (Petrobras 3W).",
        lifespan=lifespan,
        responses={422: {"model": ErrorOut}, 401: {"model": ErrorOut}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.info(
            "request",
            extra={
                "extra_fields": {
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            },
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error(exc.status_code, "http_error", str(exc.detail), request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []))
        return error(422, "validation_error", f"{where}: {first.get('msg', 'invalid')}", request)

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(401, "missing or invalid X-API-Key")

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health(svc: Svc) -> HealthOut:
        return HealthOut(
            status="ok",
            model_version=svc.version,
            model=svc.manifest["model"],
            uptime_s=round(time.time() - svc.started, 1),
        )

    @app.get("/model", response_model=ModelOut, tags=["meta"])
    def model(svc: Svc) -> ModelOut:
        m = svc.manifest
        return ModelOut(
            version=m["version"],
            model=m["model"],
            created=m.get("created"),
            git_commit=m.get("git_commit"),
            classes=CLASS_NAMES,
            variables=m["preprocessing"]["variables"],
            window_s=m["window_s"],
            stride_s=m["stride_s"],
            alert=m["alert"],
            metrics=m.get("metrics", {}),
        )

    @app.post("/predict", response_model=PredictionOut, tags=["predict"])
    def predict(req: PredictRequest, svc: Svc) -> PredictionOut:
        """Score the last 5 minutes of the given readings (stateless)."""
        df = readings_frame(req.readings)
        if len(df) < svc.predictor.window_s:
            raise HTTPException(422, f"need at least {svc.predictor.window_s} readings (1 Hz)")
        p = svc.predictor.predict_window(df, well_id=req.well_id, keep_inputs=True)
        return svc.prediction_out(p, with_drivers=True)

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        tags=["stream"],
        dependencies=[Depends(require_key)],
    )
    async def ingest(req: IngestRequest, svc: Svc) -> IngestResponse:
        """Feed live readings; returns new predictions (one per minute) and alert changes."""
        return await svc.ingest(req.well_id, readings_frame(req.readings))

    @app.get("/alerts", response_model=list[AlertOut], tags=["alerts"])
    def alerts(
        svc: Svc,
        well_id: str | None = None,
        status: Annotated[str | None, Query(pattern="^(open|acknowledged|resolved)$")] = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> list[AlertOut]:
        return [
            svc.alert_out(a, d)
            for a, d in svc.store.list_alerts(well_id, status, since, until, limit)
        ]

    @app.post(
        "/alerts/{alert_id}/ack",
        response_model=AlertOut,
        tags=["alerts"],
        dependencies=[Depends(require_key)],
    )
    async def acknowledge(alert_id: int, svc: Svc) -> AlertOut:
        async with svc.lock:
            found = svc.store.get_alert(alert_id)
            if found is None:
                raise HTTPException(404, f"alert {alert_id} not found")
            stored, drivers = found
            live = svc.alerts.active(stored.well_id)
            alert = live if live is not None and live.id == alert_id else stored
            AlertEngine.acknowledge(alert, pd.Timestamp.now(tz="UTC").tz_localize(None))
            svc.store.save_alert(alert)
        out = svc.alert_out(alert, drivers)
        await svc.broadcast(alert.well_id, {"type": "alert", "data": out.model_dump(mode="json")})
        return out

    @app.get("/wells", response_model=list[str], tags=["stream"])
    def wells(svc: Svc) -> list[str]:
        return svc.store.wells()

    @app.get("/wells/{well_id}/predictions", tags=["stream"])
    def well_predictions(
        well_id: str,
        svc: Svc,
        since: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=10_000)] = 1440,
    ) -> list[dict[str, Any]]:
        df = svc.store.predictions(well_id, since, limit)
        if df.empty:
            return []
        df["timestamp"] = df["timestamp"].map(lambda t: pd.Timestamp(t).isoformat())
        records: list[dict[str, Any]] = df.replace({np.nan: None}).to_dict("records")
        return records

    @app.get("/wells/{well_id}/readings", tags=["stream"])
    def well_readings(
        well_id: str,
        svc: Svc,
        minutes: Annotated[int, Query(ge=1, le=RECENT_S // 60)] = 60,
        every_s: Annotated[int, Query(ge=1, le=600)] = 10,
    ) -> dict[str, Any]:
        """Recent raw sensor readings (from memory; for charts), downsampled."""
        df = svc.recent_readings(well_id, minutes, every_s)
        return {
            "timestamps": [t.isoformat() for t in df.index],
            "values": {c: [None if pd.isna(v) else float(v) for v in df[c]] for c in df.columns},
        }

    @app.get("/demo", tags=["demo"])
    def demo_status(svc: Svc) -> dict[str, Any]:
        """Whether the built-in replay of bundled demo wells is running."""
        return svc.demo_status()

    @app.post("/demo/start", tags=["demo"], dependencies=[Depends(require_key)])
    async def demo_start(
        svc: Svc,
        speed: Annotated[int, Query(ge=1, le=600)] = 60,
        loop: bool = True,
        classes: Annotated[list[int] | None, Query()] = None,
    ) -> dict[str, Any]:
        """Replay bundled test-set wells through the live pipeline (``speed`` x real time)."""
        return svc.start_demo(classes, speed, loop)

    @app.post("/demo/stop", tags=["demo"], dependencies=[Depends(require_key)])
    async def demo_stop(svc: Svc) -> dict[str, Any]:
        """Stop the demo replay."""
        return svc.stop_demo()

    @app.websocket("/ws/{well_id}")
    async def feed(ws: WebSocket, well_id: str) -> None:
        """Live predictions and alerts for one well (``all`` for every well)."""
        await ws.accept()
        svc: Service = ws.app.state.service
        svc.subscribers[well_id].add(ws)
        try:
            while True:
                await ws.receive_text()  # keep-alive; clients may send pings
        except WebSocketDisconnect:
            svc.subscribers[well_id].discard(ws)

    return app


def configure_logging(level: str = "INFO") -> None:
    """JSON logs on stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


app = create_app() if __name__ != "__main__" else None  # uvicorn owfd.api.main:app
