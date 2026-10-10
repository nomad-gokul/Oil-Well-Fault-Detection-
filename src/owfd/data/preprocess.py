"""Clean every 3W instance and write it to ``data/processed/clean/<class>/<name>.parquet``.

Usage::

    python -m owfd.data.preprocess [--workers 4] [--force]

Scaling is *not* applied here: the scaler is fit on the training split in Task 5.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from owfd.config import PreprocessingConfig, get_settings
from owfd.data.clean import clean_instance
from owfd.data.load import load_instance

logger = logging.getLogger(__name__)


def _clean_one(
    rel_path: str, src: Path, dst: Path, config: PreprocessingConfig, force: bool
) -> int:
    out = dst / rel_path
    if out.exists() and not force:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_instance(load_instance(src / rel_path), config).to_parquet(out)
    return 1


def preprocess_all(
    index: pd.DataFrame, src: Path, dst: Path, config: PreprocessingConfig,
    workers: int = 4, force: bool = False,
) -> int:  # fmt: skip
    """Clean every instance in ``index``; returns the number of files written."""
    work = partial(_clean_one, src=src, dst=dst, config=config, force=force)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(work, index["path"], chunksize=8))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Clean all 3W instances.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    processed = settings.resolve(settings.processed_data_dir)
    index = pd.read_parquet(processed / "instances.parquet")
    written = preprocess_all(
        index, settings.resolve(settings.dataset_dir), processed / "clean",
        settings.preprocessing, args.workers, args.force,
    )  # fmt: skip
    logger.info("Cleaned %d instances (%d written) -> %s", len(index), written, processed / "clean")


if __name__ == "__main__":
    main()
