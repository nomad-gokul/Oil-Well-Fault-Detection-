"""Project settings loaded from ``configs/default.yaml`` and ``OWFD_*`` environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def _project_root() -> Path:
    """Folder holding ``configs/`` etc.: ``OWFD_PROJECT_ROOT``, else the source checkout, else
    the working directory (e.g. when the package is installed into a container)."""
    if env := os.environ.get("OWFD_PROJECT_ROOT"):
        return Path(env).resolve()
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "configs").is_dir() else Path.cwd()


PROJECT_ROOT = _project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class PreprocessingConfig(BaseModel):
    """Cleaning and scaling options (see ``configs/default.yaml``)."""

    variables: list[str] = Field(
        default=["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL", "QGL"]
    )
    ffill_limit_s: int = 60
    stuck_after_s: int = 1800
    clip_quantiles: tuple[float, float] = (0.001, 0.999)
    scaler_sample_every_s: int = 10


class WindowConfig(BaseModel):
    """Sliding-window options (see ``configs/default.yaml``)."""

    window_s: int = 300
    stride_s: int = 60
    max_train_windows_per_instance: int = 300


class SplitConfig(BaseModel):
    """Train/val/test split options (see ``configs/default.yaml``)."""

    val_frac: float = 0.15
    test_frac: float = 0.20
    n_trials: int = 20000


class Settings(BaseSettings):
    """Paths and global settings. Env vars (``OWFD_*``) take precedence over the YAML file."""

    model_config = SettingsConfigDict(
        env_prefix="OWFD_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        yaml_file=DEFAULT_CONFIG,
    )

    raw_data_dir: Path = Path("data/raw")
    dataset_dir: Path = Path("data/raw/3W/dataset")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    random_seed: int = 42
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    windows: WindowConfig = Field(default_factory=WindowConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    splits_file: Path = Path("configs/splits.json")
    baseline_s: int = 3600

    # Serving (Tasks 13-14). All can be set with OWFD_* environment variables.
    serving_bundle: Path | None = None  # default: newest folder in artifacts/serving
    api_key: str | None = None  # when set, write endpoints require the X-API-Key header
    rate_limit_per_min: int = 600  # per client IP on POST endpoints; 0 disables
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    database_url: str = "sqlite:///data/owfd.db"
    alert_webhook_url: str | None = None  # Slack/Discord-compatible webhook for new alerts
    alerts_file: Path = Path("configs/alerts.yaml")
    demo_mode: bool = False  # replay the bundled demo wells on startup (public demo)
    demo_speed: int = 60  # demo replay speed (x real time)
    demo_tick_s: int = 5  # the demo sends readings in batches every this many seconds
    demo_dir: Path = Path("demo_data")
    log_level: str = "INFO"

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
