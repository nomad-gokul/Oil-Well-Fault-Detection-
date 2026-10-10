"""Build tabular feature files for every split.

Usage::

    python -m owfd.features.build  # data/processed/features/{train,val,test}.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from owfd.config import get_settings
from owfd.data.split import SPLITS
from owfd.data.windows import WindowStore
from owfd.features.stats import feature_names, window_features

logger = logging.getLogger(__name__)

META_COLUMNS = [
    "instance_id", "well_id", "source", "end_time",
    "label_raw", "event_class", "is_transient", "is_fault",
]  # fmt: skip


def build_features(store: WindowStore, variables: list[str], chunk: int = 4096) -> pd.DataFrame:
    """Feature table (metadata columns + features) for every window in ``store``."""
    names = feature_names(variables)
    parts = []
    for lo in range(0, len(store), chunk):
        idx = np.arange(lo, min(lo + chunk, len(store)))
        parts.append(window_features(store.batch(idx), variables))
    values = np.concatenate(parts) if parts else np.empty((0, len(names)), np.float32)
    features = pd.DataFrame(values, columns=names, index=store.windows.index)
    return pd.concat([store.windows[META_COLUMNS], features], axis=1)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description="Build feature files for all splits.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    variables = settings.preprocessing.variables
    out_dir: Path = processed / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names(variables), indent=1))
    for split in SPLITS:
        table = build_features(WindowStore(processed / "windows" / split), variables)
        table.to_parquet(out_dir / f"{split}.parquet")
        logger.info("%s: %d windows x %d features", split, len(table), len(table.columns) - 8)


if __name__ == "__main__":
    main()
