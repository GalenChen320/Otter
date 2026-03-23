from typing import Literal, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import (
    ROOT_DIR, tracked_field, untracked_field, coerce_empty_str
    ) 


class BackendSettings(BaseSettings):
    """Base class for all backend settings."""
    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore"
    )


class DockerSettings(BackendSettings):
    cpus: float | None = tracked_field(
        default=1.0,
        description="CPU cores allocated per container"
    )
    memory: str | None = tracked_field(
        default="4096m",
        description="Memory limit per container"
    )
    memory_swap: str | None = tracked_field(
        default="4096m",
        description="Swap limit per container"
    )
    memory_reservation: str | None = tracked_field(
        default="2048m",
        description="Soft memory limit per container"
    )
    network_mode: str | None = tracked_field(
        default="host",
        description="Container network mode"
    )
    device_read_bps: str | None = tracked_field(
        default="128m",
        description="Device read rate limit"
    )
    device_write_bps: str | None = tracked_field(
        default="128m",
        description="Device write rate limit"
    )
    timeout: int = tracked_field(
        default=10,
        description="Command execution timeout in seconds"
    )

    @field_validator(
        "cpus", "memory", "memory_swap", "memory_reservation",
        "network_mode", "device_read_bps", "device_write_bps",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        return coerce_empty_str(v)
    

class ChatLLMSettings(BackendSettings):
    api_key: str = untracked_field(
        description="API key for the LLM provider"
    )
    base_url: str = tracked_field(
        description="Base URL of the LLM API endpoint"
    )
    model: str = tracked_field(
        description="Model name to use for generation"
    )


BACKEND_SETTINGS_REGISTRY: dict[str, type[BackendSettings]] = {
    "chat_llm": ChatLLMSettings,
    "docker": DockerSettings,
}


__all__ = [
    "BackendSettings",
    "BACKEND_SETTINGS_REGISTRY",
    "DockerSettings",
    "ChatLLMSettings",
]