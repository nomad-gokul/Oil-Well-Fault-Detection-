"""Index and load 3W instances (one Parquet file per instance, 1 Hz, timestamp index)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from owfd.data.schema import LABEL_COLUMN, STATE_COLUMN, Source

#: ``WELL-00001_20170201010207.parquet``, ``SIMULATED_00001.parquet``, ``DRAWN_00001.parquet``
INSTANCE_RE = re.compile(
    r"^(?:(?P<well>WELL-\d{5})_(?P<ts>\d{14})|(?P<kind>SIMULATED|DRAWN)_(?P<num>\d{5}))\.parquet$"
)

INDEX_COLUMNS = [
    "instance_id", "path", "event_class", "source", "well_id",
    "start_time", "end_time", "n_rows",
]  # fmt: skip


def parse_instance_name(name: str) -> tuple[Source, str | None]:
    """Return ``(source, well_id)`` for a 3W file name; ``well_id`` is None if not real.

    Raises:
        ValueError: if the name does not follow a 3W naming pattern.
    """
    match = INSTANCE_RE.match(name)
    if match is None:
        raise ValueError(f"Not a 3W instance file name: {name}")
    if match["well"]:
        return Source.REAL, match["well"]
    return Source(match["kind"].lower()), None


def _time_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pq.read_table(path, columns=["timestamp"]).column("timestamp")
    return pd.Timestamp(timestamps[0].as_py()), pd.Timestamp(timestamps[-1].as_py())


def list_instances(dataset_dir: Path) -> pd.DataFrame:
    """Index every instance under ``dataset_dir/<class>/*.parquet``.

    Returns:
        One row per instance with :data:`INDEX_COLUMNS`. ``path`` is relative to
        ``dataset_dir`` so the index stays valid if the dataset is moved.
    """
    rows = []
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        for file in sorted(class_dir.glob("*.parquet")):
            source, well_id = parse_instance_name(file.name)
            start, end = _time_bounds(file)
            rows.append(
                {
                    "instance_id": f"{class_dir.name}/{file.stem}",
                    "path": file.relative_to(dataset_dir).as_posix(),
                    "event_class": int(class_dir.name),
                    "source": source.value,
                    "well_id": well_id,
                    "start_time": start,
                    "end_time": end,
                    "n_rows": pq.ParquetFile(file).metadata.num_rows,
                }
            )
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return index.astype({"event_class": "int16", "n_rows": "int64"})


def build_index(dataset_dir: Path) -> pd.DataFrame:
    """Build the instance index, failing loudly if the folder holds no instances."""
    index = list_instances(dataset_dir)
    if index.empty:
        raise FileNotFoundError(f"No 3W instances found under {dataset_dir}")
    return index


def summarize_index(index: pd.DataFrame) -> pd.DataFrame:
    """Instance counts per event class x source, with totals."""
    return pd.crosstab(index["event_class"], index["source"], margins=True, margins_name="total")


def load_instance(path: Path) -> pd.DataFrame:
    """Load one instance as a DataFrame with a sorted ``DatetimeIndex`` named ``timestamp``.

    Sensor columns are float64; ``class`` and ``state`` are nullable ``Int16`` (3W leaves
    some periods unlabelled).
    """
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    df = df.sort_index()
    for column in (LABEL_COLUMN, STATE_COLUMN):
        if column in df.columns:
            df[column] = df[column].astype("Int16")
    return df
