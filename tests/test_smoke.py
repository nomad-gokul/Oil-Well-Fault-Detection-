from pathlib import Path

import owfd
from owfd.config import PROJECT_ROOT, Settings, get_settings


def test_package_imports() -> None:
    assert owfd.__version__


def test_settings_load_from_yaml() -> None:
    settings = get_settings()
    assert settings.random_seed == 42
    assert settings.raw_data_dir == Path("data/raw")


def test_env_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("OWFD_RANDOM_SEED", "7")
    assert Settings().random_seed == 7


def test_resolve_relative_to_project_root() -> None:
    settings = Settings()
    assert settings.resolve(Path("data/raw")) == PROJECT_ROOT / "data/raw"
    assert settings.resolve(Path("/abs/x")) == Path("/abs/x")
