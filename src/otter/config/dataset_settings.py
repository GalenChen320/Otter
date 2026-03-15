from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import (
    ROOT_DIR, tracked_field, untracked_field, coerce_empty_str
    )


class DatasetSettings(BaseSettings):
    """Base class for all dataset settings."""
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore"
    )


class EvalplusSettings(DatasetSettings):
    cache_dir: Path = untracked_field(
        default=ROOT_DIR / "data" / "cache",
        description="Local cache directory for downloaded datasets"
    )


class MbppplusSettings(DatasetSettings):
    cache_dir: Path = untracked_field(
        default=ROOT_DIR / "data" / "cache",
        description="Local cache directory for downloaded datasets"
    )


DATASET_SETTINGS_REGISTRY: dict[str, type[DatasetSettings]] = {
    "evalplus": EvalplusSettings,
    "mbppplus": MbppplusSettings,
}


__all__ = [
    "DatasetSettings",
    "DATASET_SETTINGS_REGISTRY",
    "EvalplusSettings",
    "MbppplusSettings",
]
