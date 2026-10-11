"""Build tabular feature files from the cleaned instances.

Usage::

    python -m owfd.features.build  # data/processed/features/

For the windows of every split (``data/processed/windows/<split>``) writes
``{train,val,test}.parquet``. For real instances it also writes

- ``train_full.parquet``: every labelled window of real *training* instances (the training
  table caps windows per instance), used for cross-validation by well;
- ``burnin.parquet``: windows in the unlabelled first hour of every real instance, the
  per-well burn-in period used to set each well's alert threshold.

Each row holds metadata, the 5-minute window statistics (:mod:`owfd.features.stats`) and the
long-context features (:mod:`owfd.features.context`), computed from the same float16-rounded
scaled values the window store and the online predictor use.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from owfd.config import WindowConfig, get_settings
from owfd.data.clean import Scaler
from owfd.data.schema import LABEL_COLUMN
from owfd.data.split import SPLITS, load_splits
from owfd.data.windows import causal_baseline, channels, map_labels, window_ends
from owfd.features.context import all_feature_names, block_means, context_features
from owfd.features.stats import window_features

logger = logging.getLogger(__name__)

META_COLUMNS = [
    "instance_id", "well_id", "source", "end_time", "end_pos",
    "label_raw", "event_class", "is_transient", "is_fault",
]  # fmt: skip


def instance_features(
    df: pd.DataFrame,
    ends: np.ndarray,
    variables: list[str],
    window_s: int,
    baseline_s: int,
    chunk: int = 2048,
) -> np.ndarray:
    """Features [len(ends), n_features] of the windows ending at ``ends`` (row positions) of
    one *scaled* instance ``df``. Every end must have ``window_s`` rows of history."""
    chans = channels(variables)
    k = len(variables)
    values = df[chans].to_numpy(dtype=np.float16).astype(np.float32)  # as stored for training
    ends = np.asarray(ends, dtype=np.int64)
    base = causal_baseline(
        df[variables].to_numpy(np.float64), df[chans[k:]].to_numpy(), ends, baseline_s
    ).astype(np.float32)
    sensors, missing = values[:, :k].astype(np.float64), values[:, k:] > 0.5
    parts = []
    for lo in range(0, len(ends), chunk):
        e, b = ends[lo : lo + chunk], base[lo : lo + chunk]
        x = values[e[:, None] - window_s + 1 + np.arange(window_s)]
        short = window_features(x, variables, b)
        parts.append(
            np.concatenate([short, context_features(block_means(sensors, missing, e), b)], 1)
        )
    n_features = len(all_feature_names(variables))
    return np.concatenate(parts) if parts else np.empty((0, n_features), np.float32)


def burnin_ends(labels: pd.Series, config: WindowConfig) -> np.ndarray:
    """Window ends in the unlabelled stretch before the first labelled row (same stride grid
    as :func:`owfd.data.windows.window_ends`)."""
    n = len(labels)
    first = config.window_s - 1
    if n <= first:
        return np.empty(0, dtype=np.int64)
    ends = np.arange(n - 1, first - 1, -config.stride_s)[::-1]
    labelled = labels.notna().to_numpy()
    if not labelled.any():
        return np.empty(0, dtype=np.int64)
    return ends[ends < int(np.argmax(labelled))]


def _meta(df: pd.DataFrame, ends: np.ndarray, info: dict[str, Any]) -> pd.DataFrame:
    raw = df[LABEL_COLUMN].to_numpy(dtype=np.int64, na_value=-1)[ends]
    event, transient = map_labels(raw)
    unlabelled = raw < 0
    return pd.DataFrame(
        {
            "instance_id": info["instance_id"],
            "well_id": info["well_id"],
            "source": info["source"],
            "end_time": df.index[ends],
            "end_pos": ends,
            "label_raw": raw.astype(np.int16),
            "event_class": np.where(unlabelled, -1, event).astype(np.int8),
            "is_transient": transient & ~unlabelled,
            "is_fault": event > 0,
        }
    )


def _process_instance(
    info: dict[str, Any],
    clean_dir: Path,
    scaler: Scaler,
    config: WindowConfig,
    baseline_s: int,
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Feature rows of one instance for each requested output (``info['outputs']``)."""
    df = scaler.transform(pd.read_parquet(clean_dir / f"{info['instance_id']}.parquet"))
    variables = scaler.variables
    out: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for name, ends in info["outputs"].items():
        if ends is None:  # computed here: "full" = all labelled ends, "burnin" = before labels
            ends = (
                window_ends(df[LABEL_COLUMN], config)
                if name == "train_full"
                else burnin_ends(df[LABEL_COLUMN], config)
            )
        ends = np.asarray(ends, dtype=np.int64)
        feats = instance_features(df, ends, variables, config.window_s, baseline_s)
        meta = _meta(df, ends, info)
        if "rows" in info and name in info["rows"]:
            meta.index = info["rows"][name]
        out[name] = (meta, feats)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build feature files for all splits.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    variables = settings.preprocessing.variables
    out_dir: Path = processed / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = all_feature_names(variables)
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=1))

    index = pd.read_parquet(processed / "instances.parquet").set_index("instance_id")
    splits = load_splits(settings.resolve(settings.splits_file))
    scaler = Scaler.load(settings.resolve(settings.artifacts_dir) / "preprocessing" / "scaler.json")

    # One work item per instance: the window ends each output table needs from it.
    work: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        table = pd.read_parquet(processed / "windows" / split / "windows.parquet")
        for iid, rows in table.groupby("instance_id").groups.items():
            item = work.setdefault(str(iid), {"outputs": {}, "rows": {}})
            item["outputs"][split] = table.loc[rows, "end_pos"].to_numpy()
            item["rows"][split] = np.asarray(rows)
    for split in SPLITS:
        for iid in splits["instances"][split]:
            if index.at[iid, "source"] != "real":
                continue
            item = work.setdefault(iid, {"outputs": {}, "rows": {}})
            item["outputs"]["burnin"] = None
            if split == "train":
                item["outputs"]["train_full"] = None
    for iid, item in work.items():
        item.update(instance_id=iid, well_id=index.at[iid, "well_id"],
                    source=index.at[iid, "source"])  # fmt: skip

    job = partial(_process_instance, clean_dir=processed / "clean", scaler=scaler,
                  config=settings.windows, baseline_s=settings.baseline_s)  # fmt: skip
    collected: dict[str, list[tuple[pd.DataFrame, np.ndarray]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(job, work.values(), chunksize=4)):
            for name, part in result.items():
                collected.setdefault(name, []).append(part)
            if (i + 1) % 200 == 0:
                logger.info("%d / %d instances", i + 1, len(work))

    for name, parts in collected.items():
        meta = pd.concat([m for m, _ in parts])
        feats = pd.DataFrame(np.concatenate([f for _, f in parts]), columns=names, index=meta.index)
        table = pd.concat([meta, feats], axis=1)
        table = table.sort_index() if name in SPLITS else table.reset_index(drop=True)
        table.to_parquet(out_dir / f"{name}.parquet")
        logger.info("%s: %d windows x %d features", name, len(table), len(names))


if __name__ == "__main__":
    main()
