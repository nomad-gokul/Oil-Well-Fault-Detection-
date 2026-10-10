import subprocess
from pathlib import Path

import pytest

from owfd.data import download
from owfd.data.download import find_dataset_root, is_complete, link_existing


def test_find_dataset_root_accepts_repo_or_dataset_dir(tiny_dataset: Path) -> None:
    assert find_dataset_root(tiny_dataset) == tiny_dataset.resolve()
    assert find_dataset_root(tiny_dataset.parent) == tiny_dataset.resolve()


def test_find_dataset_root_rejects_other_dirs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path)


def test_link_existing_is_idempotent(tiny_dataset: Path, tmp_path: Path) -> None:
    target = tmp_path / "raw" / "3W" / "dataset"
    assert not is_complete(target)
    link_existing(tiny_dataset.parent, target)
    link_existing(tiny_dataset.parent, target)  # second run replaces the link
    assert target.is_symlink() and is_complete(target)
    assert (target / "0").is_dir()


def test_link_existing_refuses_to_replace_real_dir(tiny_dataset: Path, tmp_path: Path) -> None:
    target = tmp_path / "raw" / "dataset"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        link_existing(tiny_dataset, target)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_clone_dataset_checks_out_only_the_dataset(
    tiny_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clone a local stand-in for the 3W repo and check only dataset files are checked out."""
    upstream = tiny_dataset.parent  # tmp_path, holding dataset/
    (upstream / "dataset" / "dataset.ini").write_text("[VERSION]\nDATASET = 2.0.0\n")
    (upstream / "toolkit").mkdir()
    (upstream / "toolkit" / "big_module.py").write_text("x = 1\n")
    _git("init", "-q", "-b", "main", cwd=upstream)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A", cwd=upstream)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x", cwd=upstream)

    monkeypatch.setattr(download, "REPO_URL", upstream.as_uri())
    repo_dir = tmp_path / "clone" / "3W"
    download.clone_dataset(repo_dir)

    assert find_dataset_root(repo_dir / "dataset") == (repo_dir / "dataset").resolve()
    assert (repo_dir / "dataset" / "dataset.ini").exists()
    assert len(list((repo_dir / "dataset").glob("*/*.parquet"))) == 4
    assert not (repo_dir / "toolkit").exists()
    download.clone_dataset(repo_dir)  # re-running on an existing clone is safe
