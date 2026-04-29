from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import (
    ROOT_DIR, tracked_field, untracked_field
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
    agent_api_key: str = untracked_field(
        default="",
        description="API key for the AI CLI agent"
    )
    agent_model_name: str = tracked_field(
        default="",
        description="Model name for the AI CLI agent"
    )
    agent_base_url: str = tracked_field(
        default="",
        description="Base URL for the AI CLI agent API"
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
