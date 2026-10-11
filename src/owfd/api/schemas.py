"""Request and response models (Pydantic v2) with OpenAPI examples."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_READING: dict[str, Any] = {
    "timestamp": "2017-06-20T12:29:25",
    "values": {
        "P-PDG": 22362004.0,
        "P-TPT": 14278161.0,
        "T-TPT": 101.4,
        "P-MON-CKP": 1437155.9,
        "T-JUS-CKP": 76.7,
        "P-JUS-CKGL": 17373680.0,
        "QGL": None,
    },
}


class Reading(BaseModel):
    """One 1 Hz sensor reading. Unknown keys are ignored; missing sensors may be null/absent."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_READING})
    timestamp: datetime
    values: dict[str, float | None]


class IngestRequest(BaseModel):
    """Readings for one well, oldest first, at 1 Hz."""

    well_id: str = Field(min_length=1, max_length=64, examples=["WELL-00014"])
    readings: list[Reading] = Field(min_length=1, max_length=86_400)


class PredictRequest(BaseModel):
    """At least one window (300 readings) for one well; all readings are used as history."""

    well_id: str = Field(default="adhoc", min_length=1, max_length=64)
    readings: list[Reading] = Field(min_length=1, max_length=86_400)


class Driver(BaseModel):
    """A feature's SHAP contribution towards the predicted class."""

    feature: str
    value: float
    contribution: float


class PredictionOut(BaseModel):
    """Model output for the window ending at ``timestamp``."""

    well_id: str
    timestamp: datetime
    predicted_class: int
    class_name: str
    p_fault: float = Field(description="1 - P(normal), after calibration")
    is_fault: bool = Field(description="p_fault above the alert threshold")
    probabilities: dict[str, float]
    drivers: list[Driver] | None = None


class AlertOut(BaseModel):
    """An alert episode."""

    id: int
    well_id: str
    event_class: int
    class_name: str
    severity: str
    status: str
    opened_at: datetime
    acknowledged_at: datetime | None
    resolved_at: datetime | None
    peak_probability: float
    top_drivers: list[Driver] | None = None


class IngestResponse(BaseModel):
    """New predictions and alert changes caused by the ingested readings."""

    predictions: list[PredictionOut]
    alerts: list[AlertOut]


class HealthOut(BaseModel):
    """Liveness and model info."""

    status: str
    model_version: str
    model: str
    uptime_s: float


class ModelOut(BaseModel):
    """Serving bundle manifest (subset)."""

    version: str
    model: str
    created: str | None = None
    git_commit: str | None = None
    classes: dict[int, str]
    variables: list[str]
    window_s: int
    stride_s: int
    alert: dict[str, float | None]
    metrics: dict[str, float | int | str | None]


class ErrorOut(BaseModel):
    """Error envelope used by every endpoint."""

    error: dict[str, str]
