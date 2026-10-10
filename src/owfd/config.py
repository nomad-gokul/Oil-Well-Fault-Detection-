"""Project settings loaded from ``configs/default.yaml`` and ``OWFD_*`` environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Settings(BaseSettings):
    """Paths and global settings. Env vars (``OWFD_*``) take precedence over the YAML file."""

    model_config = SettingsConfigDict(
        env_prefix="OWFD_", env_file=".env", extra="ignore", yaml_file=DEFAULT_CONFIG
    )

    raw_data_dir: Path = Path("data/raw")
    dataset_dir: Path = Path("data/raw/3W/dataset")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    random_seed: int = 42

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init kwargs > env vars > .env file > YAML config > field defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    def resolve(self, path: Path) -> Path:
        """Return ``path`` as absolute, interpreting relative paths from the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Return the cached project settings."""
    return Settings()
