"""Fetch the Petrobras 3W dataset, or link an existing local copy, then build the instance index.

Usage::

    python -m owfd.data.download                 # sparse git clone into data/raw/3W
    python -m owfd.data.download --path ~/3W     # use an existing copy (repo root or dataset/)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from owfd.config import get_settings
from owfd.data.load import build_index, summarize_index

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/petrobras/3W"
#: Only the data is checked out: class folders and the dataset's metadata files.
SPARSE_PATTERNS = ("/dataset/*.ini", "/dataset/*.md", "/dataset/[0-9]/*")
COMPLETE_MARKER = ".owfd_complete"


def find_dataset_root(path: Path) -> Path:
    """Return the folder that holds the class folders ``0``..``9`` (``path`` or ``path/dataset``).

    Raises:
        FileNotFoundError: if neither location contains 3W class folders with Parquet files.
    """
    for candidate in (path, path / "dataset"):
        if (candidate / "0").is_dir() and any((candidate / "0").glob("*.parquet")):
            return candidate.resolve()
    raise FileNotFoundError(f"No 3W class folders with .parquet files under {path}")


def is_complete(dataset_dir: Path) -> bool:
    """True if a previous download finished (marker present) or the folder is a linked copy."""
    return (dataset_dir / COMPLETE_MARKER).exists() or (
        dataset_dir.is_symlink() and dataset_dir.exists()
    )


def link_existing(path: Path, dataset_dir: Path) -> Path:
    """Symlink ``dataset_dir`` to an existing local 3W copy and return the resolved root."""
    root = find_dataset_root(path.expanduser())
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    if dataset_dir.is_symlink():
        dataset_dir.unlink()
    elif dataset_dir.exists():
        raise FileExistsError(f"{dataset_dir} already exists and is not a symlink")
    dataset_dir.symlink_to(root, target_is_directory=True)
    return root


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def clone_dataset(repo_dir: Path) -> None:
    """Shallow, sparse clone of the 3W repository: downloads only the dataset folder (~GBs)."""
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone", "--progress", "--depth", "1", "--filter=blob:none", "--no-checkout",
            REPO_URL, str(repo_dir),
        )  # fmt: skip
    _git("sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS, cwd=repo_dir)
    logger.info("Checking out the dataset (several GB, this can take a while)...")
    _git("checkout", "--progress", "HEAD", cwd=repo_dir)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, help="existing local 3W copy to link instead")
    parser.add_argument("--force", action="store_true", help="re-run even if already complete")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    dataset_dir = settings.resolve(settings.dataset_dir)

    if args.path is not None:
        root = link_existing(args.path, dataset_dir)
        logger.info("Linked %s -> %s", dataset_dir, root)
    elif is_complete(dataset_dir) and not args.force:
        logger.info("Dataset already present at %s, skipping download", dataset_dir)
    else:
        clone_dataset(dataset_dir.parent)
        find_dataset_root(dataset_dir)  # sanity check before marking complete
        (dataset_dir / COMPLETE_MARKER).touch()

    index = build_index(dataset_dir)
    out = settings.resolve(settings.processed_data_dir) / "instances.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out)
    logger.info("Indexed %d instances -> %s", len(index), out)
    print(summarize_index(index).to_string())


if __name__ == "__main__":
    main()
