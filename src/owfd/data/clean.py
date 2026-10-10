"""Cleaning (per instance, no fitted state) and scaling (fit on training data only).

The same functions run in training and at inference time, so online predictions see exactly
the preprocessing the models were trained on. Every step is causal: a value at time ``t``
depends only on readings at or before ``t``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from owfd.config import PreprocessingConfig
from owfd.data.schema import LABEL_COLUMN

MASK_PREFIX = "missing__"


def mask_column(variable: str) -> str:
    """Name of the binary missing-value mask for ``variable``."""
    return f"{MASK_PREFIX}{variable}"


def select_variables(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Keep ``variables`` (adding all-NaN columns for absent ones) plus the label column."""
    out = df.reindex(columns=variables).astype("float64")
    if LABEL_COLUMN in df.columns:
        out[LABEL_COLUMN] = df[LABEL_COLUMN]
    return out


def stuck_mask(series: pd.Series, stuck_after_s: int) -> pd.Series:
    """True where the value has been unchanged for at least ``stuck_after_s`` samples (1 Hz).

    Causal: the first ``stuck_after_s - 1`` samples of a flat run are still trusted, because
    online we cannot know yet that the sensor is stuck. NaNs are never flagged.
    """
    v = series.to_numpy(dtype=np.float64)
    return pd.Series(_stuck(v, stuck_after_s), index=series.index)


def _stuck(v: np.ndarray, stuck_after_s: int) -> np.ndarray:
    """NumPy core of :func:`stuck_mask` (NaN never equals NaN, so NaNs never form a run)."""
    pos = np.arange(len(v))
    starts = np.ones(len(v), dtype=bool)
    starts[1:] = ~(v[1:] == v[:-1])
    run_length = pos - np.maximum.accumulate(np.where(starts, pos, 0)) + 1
    return (run_length >= stuck_after_s) & ~np.isnan(v)


def _ffill(v: np.ndarray, limit: int) -> np.ndarray:
    """Forward-fill NaNs from the last valid value, at most ``limit`` in a row (as pandas)."""
    pos = np.arange(len(v))
    valid = ~np.isnan(v)
    last = np.maximum.accumulate(np.where(valid, pos, -1))
    fill = ~valid & (last >= 0) & (pos - last <= limit)
    out = v.copy()
    out[fill] = v[last[fill]]
    return out


def clean_instance(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Select variables, mask stuck sensors, forward-fill short gaps and add missing masks.

    Returns float32 sensor columns (NaN where still missing), one bool mask column per variable
    and the label column if present. No fitted state, so it is safe before the train/test split.
    """
    out = select_variables(df, config.variables)
    values = np.array(out[config.variables], dtype=np.float64)  # writable copy
    for j in range(values.shape[1]):
        col = values[:, j]
        col = np.where(_stuck(col, config.stuck_after_s), np.nan, col)
        values[:, j] = _ffill(col, config.ffill_limit_s)
    missing = np.isnan(values)
    with np.errstate(over="ignore"):  # 3W placeholders (~1e42) overflow to inf; clipped later
        sensors = pd.DataFrame(values.astype(np.float32), index=out.index, columns=config.variables)
    masks = pd.DataFrame(
        missing, index=out.index, columns=[mask_column(v) for v in config.variables]
    )
    parts = [sensors, masks]
    if LABEL_COLUMN in out.columns:
        parts.append(out[[LABEL_COLUMN]])
    return pd.concat(parts, axis=1)


@dataclass
class Scaler:
    """Robust per-variable scaling: clip to training quantiles, then (x - median) / IQR.

    Missing values become 0 after scaling (the training median); the mask columns keep the
    information that they were missing.
    """

    variables: list[str]
    median: list[float]
    scale: list[float]
    lower: list[float]
    upper: list[float]

    @classmethod
    def fit(cls, frames: Iterable[pd.DataFrame], config: PreprocessingConfig) -> Scaler:
        """Fit on cleaned *training* instances, subsampling rows to bound memory."""
        step = max(1, config.scaler_sample_every_s)
        sample = pd.concat([f[config.variables].iloc[::step] for f in frames], ignore_index=True)
        if sample.empty:
            raise ValueError("Cannot fit a Scaler on no data")
        q_lo, q_hi = config.clip_quantiles
        stats = sample.astype("float64").quantile([q_lo, 0.25, 0.5, 0.75, q_hi])
        iqr = (stats.loc[0.75] - stats.loc[0.25]).to_numpy()
        spread = (stats.loc[q_hi] - stats.loc[q_lo]).to_numpy()
        # Fall back to the clipped range, then to 1, for (near-)constant variables.
        scale = np.where(iqr > 0, iqr, np.where(spread > 0, spread, 1.0))
        # Variables never observed in training scale to 0 and rely on their mask.
        median = stats.loc[0.5].fillna(0.0).to_numpy()
        return cls(
            variables=list(config.variables),
            median=median.tolist(),
            scale=np.nan_to_num(scale, nan=1.0).tolist(),
            lower=stats.loc[q_lo].fillna(-np.inf).tolist(),
            upper=stats.loc[q_hi].fillna(np.inf).tolist(),
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale a cleaned instance. Mask and label columns pass through unchanged."""
        out = df.copy()
        x = df[self.variables].to_numpy(dtype="float64")
        x = np.clip(x, self.lower, self.upper)
        x = (x - np.asarray(self.median)) / np.asarray(self.scale)
        out[self.variables] = np.nan_to_num(x, nan=0.0).astype("float32")
        return out

    def save(self, path: Path) -> None:
        """Write the scaler as JSON (human-readable, no pickle)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=True))

    @classmethod
    def load(cls, path: Path) -> Scaler:
        """Read a scaler written by :meth:`save`."""
        return cls(**json.loads(path.read_text()))
