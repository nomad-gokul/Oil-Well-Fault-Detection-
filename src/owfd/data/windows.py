"""Sliding windows over cleaned, scaled instances, stored without duplication.

Usage::

    python -m owfd.data.windows  # fits the scaler on train, builds data/processed/windows/

Each split is stored as one float16 array of instance rows (``values.npy``) plus a window table
(``windows.parquet``) whose ``start`` column points into it. A window is the ``window_s`` rows
ending at a labelled row; its label is the label of that last row, so the model only ever
uses the past to classify the present.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from owfd.config import WindowConfig, get_settings
from owfd.data.clean import Scaler, mask_column
from owfd.data.schema import LABEL_COLUMN, TRANSIENT_OFFSET
from owfd.data.split import SPLITS, load_splits, validate_splits

logger = logging.getLogger(__name__)


def channels(variables: list[str]) -> list[str]:
    """Model input channels: the scaled sensors followed by their missing masks."""
    return list(variables) + [mask_column(v) for v in variables]


def map_labels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw 3W labels (0, 1..9, 101..109) -> (event class 0..9, is_transient)."""
    raw = np.asarray(raw, dtype=np.int64)
    transient = raw >= TRANSIENT_OFFSET
    return np.where(transient, raw - TRANSIENT_OFFSET, raw), transient


def window_ends(
    labels: pd.Series,
    config: WindowConfig,
    cap: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Row positions of window ends: every ``stride_s`` rows, labelled, with a full history.

    Ends are aligned to the end of the instance so the latest data is always used. With
    ``cap``, at most ``cap`` ends are kept (random, sorted; needs ``rng``).
    """
    n = len(labels)
    first = config.window_s - 1
    if n <= first:
        return np.empty(0, dtype=np.int64)
    ends = np.arange(n - 1, first - 1, -config.stride_s)[::-1]
    ends = ends[labels.notna().to_numpy()[ends]]
    if cap is not None and len(ends) > cap:
        if rng is None:
            raise ValueError("rng is required when cap is set")
        ends = np.sort(rng.choice(ends, size=cap, replace=False))
    return ends


def _scaled_instances(
    ids: list[str], clean_dir: Path, scaler: Scaler, chans: list[str]
) -> Iterator[tuple[str, pd.DataFrame, np.ndarray]]:
    for instance_id in ids:
        df = scaler.transform(pd.read_parquet(clean_dir / f"{instance_id}.parquet"))
        yield instance_id, df, df[chans].to_numpy(dtype=np.float16)


def build_split(
    ids: list[str],
    index: pd.DataFrame,
    clean_dir: Path,
    scaler: Scaler,
    config: WindowConfig,
    out_dir: Path,
    cap: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Write ``values.npy`` and ``windows.parquet`` for one split and return the window table.

    Only the rows covered by at least one window are stored.
    """
    chans = channels(scaler.variables)
    meta = index.set_index("instance_id")
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    tables: list[pd.DataFrame] = []
    offset = 0
    for instance_id, df, values in _scaled_instances(ids, clean_dir, scaler, chans):
        ends = window_ends(df[LABEL_COLUMN], config, cap, rng)
        if not len(ends):
            continue
        lo, hi = ends[0] - config.window_s + 1, ends[-1] + 1
        blocks.append(values[lo:hi])
        raw = df[LABEL_COLUMN].to_numpy(dtype=np.int64, na_value=-1)[ends]
        event, transient = map_labels(raw)
        tables.append(
            pd.DataFrame(
                {
                    "instance_id": instance_id,
                    "well_id": meta.at[instance_id, "well_id"],
                    "source": meta.at[instance_id, "source"],
                    "end_time": df.index[ends],
                    "end_pos": ends,
                    "start": offset + ends - config.window_s + 1 - lo,
                    "label_raw": raw.astype(np.int16),
                    "event_class": event.astype(np.int8),
                    "is_transient": transient,
                    "is_fault": event > 0,
                }
            )
        )
        offset += hi - lo
    out_dir.mkdir(parents=True, exist_ok=True)
    values_all = np.concatenate(blocks) if blocks else np.empty((0, len(chans)), np.float16)
    np.save(out_dir / "values.npy", values_all)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    table.to_parquet(out_dir / "windows.parquet")
    (out_dir / "meta.json").write_text(
        json.dumps({"channels": chans, "window_s": config.window_s, "stride_s": config.stride_s})
    )
    return table


class WindowStore:
    """Read-only access to one split's windows: ``store.x(i)`` -> [window_s, n_channels]."""

    def __init__(self, split_dir: Path) -> None:
        self.meta: dict[str, Any] = json.loads((split_dir / "meta.json").read_text())
        self.values = np.load(split_dir / "values.npy", mmap_mode="r")
        self.windows = pd.read_parquet(split_dir / "windows.parquet")
        self.window_s: int = self.meta["window_s"]
        self.channels: list[str] = self.meta["channels"]
        self._starts = self.windows["start"].to_numpy() if len(self.windows) else np.empty(0)

    def __len__(self) -> int:
        return len(self.windows)

    def x(self, i: int) -> np.ndarray:
        """One window as float32."""
        s = int(self._starts[i])
        return np.asarray(self.values[s : s + self.window_s], dtype=np.float32)

    def batch(self, idx: np.ndarray) -> np.ndarray:
        """Several windows as float32 [len(idx), window_s, n_channels]."""
        rows = self._starts[np.asarray(idx)][:, None] + np.arange(self.window_s)
        return np.asarray(self.values[rows], dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: fit the scaler on train only, then window every split."""
    argparse.ArgumentParser(description="Build window stores for all splits.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    clean_dir = processed / "clean"
    index = pd.read_parquet(processed / "instances.parquet")
    splits = load_splits(settings.resolve(settings.splits_file))
    validate_splits(splits, index)

    train_ids = splits["instances"]["train"]
    frames = (pd.read_parquet(clean_dir / f"{i}.parquet") for i in train_ids)
    scaler = Scaler.fit(frames, settings.preprocessing)
    scaler_path = settings.resolve(settings.artifacts_dir) / "preprocessing" / "scaler.json"
    scaler.save(scaler_path)
    logger.info("Scaler fit on %d train instances -> %s", len(train_ids), scaler_path)

    for split in SPLITS:
        cap = settings.windows.max_train_windows_per_instance if split == "train" else None
        table = build_split(
            splits["instances"][split], index, clean_dir, scaler, settings.windows,
            processed / "windows" / split, cap=cap, seed=settings.random_seed,
        )  # fmt: skip
        counts = table.groupby(["event_class", "is_transient"]).size().unstack(fill_value=0)
        logger.info("%s: %d windows\n%s", split, len(table), counts.to_string())


if __name__ == "__main__":
    main()
