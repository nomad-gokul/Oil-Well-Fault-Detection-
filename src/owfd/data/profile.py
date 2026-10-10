"""Per-instance data-quality and label profile of the 3W dataset (input to the EDA).

Usage::

    python -m owfd.data.profile  # writes data/processed/profile.parquet
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from owfd.config import get_settings
from owfd.data.load import load_instance
from owfd.data.schema import (
    LABEL_COLUMN,
    SENSOR_VARIABLES,
    TRANSIENT_OFFSET,
    VALVE_STATE_VARIABLES,
)

logger = logging.getLogger(__name__)

ALL_VARIABLES = SENSOR_VARIABLES + VALVE_STATE_VARIABLES


def _phase_bounds(mask: np.ndarray) -> tuple[int, int] | None:
    """(first, last) row positions where ``mask`` is True, or None."""
    idx = np.flatnonzero(mask)
    return (int(idx[0]), int(idx[-1])) if idx.size else None


def profile_instance(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize one instance: timing, gaps, label phases and per-variable quality.

    Per variable ``v``: ``missing__v`` (fraction of null rows), ``frozen__v`` (present but
    constant) and ``median__v``.
    """
    seconds = df.index.to_series().diff().dt.total_seconds().to_numpy()[1:]
    labels = df[LABEL_COLUMN]
    lab = labels.fillna(-1).to_numpy(dtype=np.int64)
    out: dict[str, Any] = {
        "duration_s": float((df.index[-1] - df.index[0]).total_seconds()) if len(df) else 0.0,
        "n_gaps": int((seconds != 1).sum()),
        "max_gap_s": float(seconds.max()) if seconds.size else 0.0,
        "n_unlabelled": int(labels.isna().sum()),
        "n_normal": int((lab == 0).sum()),
        "n_transient": int((lab >= TRANSIENT_OFFSET).sum()),
        "n_steady": int(((lab >= 1) & (lab < TRANSIENT_OFFSET)).sum()),
    }
    transient = _phase_bounds(lab >= TRANSIENT_OFFSET)
    steady = _phase_bounds((lab >= 1) & (lab < TRANSIENT_OFFSET))
    out["transient_start_s"] = float(transient[0]) if transient else np.nan
    out["steady_start_s"] = float(steady[0]) if steady else np.nan
    # 3W events should progress normal -> transient -> steady state, never backwards.
    labelled = lab[lab >= 0]
    phase = np.where(labelled == 0, 0, np.where(labelled >= TRANSIENT_OFFSET, 1, 2))
    out["phases_ordered"] = bool(np.all(np.diff(phase) >= 0)) if phase.size else True

    for var in ALL_VARIABLES:
        col = df[var] if var in df.columns else pd.Series(np.nan, index=df.index)
        present = col.dropna()
        out[f"missing__{var}"] = 1.0 - len(present) / len(df) if len(df) else 1.0
        out[f"frozen__{var}"] = bool(len(present) > 0 and present.nunique() == 1)
        out[f"median__{var}"] = float(present.median()) if len(present) else np.nan
    return out


def _profile_path(path: Path) -> dict[str, Any]:
    return profile_instance(load_instance(path))


def profile_dataset(index: pd.DataFrame, dataset_dir: Path, workers: int = 4) -> pd.DataFrame:
    """Profile every instance in ``index`` (in parallel) and join the results to the index."""
    paths = [dataset_dir / p for p in index["path"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_profile_path, paths, chunksize=8))
    return pd.concat([index.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Profile every 3W instance.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    index = pd.read_parquet(processed / "instances.parquet")
    profile = profile_dataset(index, settings.resolve(settings.dataset_dir), args.workers)
    profile.to_parquet(processed / "profile.parquet")
    logger.info("Profiled %d instances -> %s", len(profile), processed / "profile.parquet")


if __name__ == "__main__":
    main()
