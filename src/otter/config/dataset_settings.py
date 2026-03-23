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


class SWECISettings(DatasetSettings):
    splitting: str = tracked_field(
        default="default",
        description="Splitting of the SWE-CI dataset"
    )
    agent_name: str = tracked_field(
        default="opencode",
        description="AI CLI agent to use (claude, codex, opencode, openhands)"
    )
    cache_dir: Path = untracked_field(
        default=ROOT_DIR / "data" / "cache",
        description="Local cache directory for downloaded datasets"
    )


DATASET_SETTINGS_REGISTRY: dict[str, type[DatasetSettings]] = {
    "evalplus": EvalplusSettings,
    "mbppplus": MbppplusSettings,
    "sweci": SWECISettings
}


__all__ = [
    "DatasetSettings",
    "DATASET_SETTINGS_REGISTRY",
    "EvalplusSettings",
    "MbppplusSettings",
]
