from pathlib import Path
from pydantic import Field
from typing import Literal, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent.parent.parent

_env_file: str = ".env"


def tracked_field(default=..., **kwargs) -> Any:
    extra = kwargs.pop("json_schema_extra", {})
    return Field(default, json_schema_extra={"core": True, **extra}, **kwargs)


def untracked_field(default=..., **kwargs) -> Any:
    extra = kwargs.pop("json_schema_extra", {})
    return Field(default, json_schema_extra={"core": False, **extra}, **kwargs)


def set_env_file(path: str) -> None:
    global _env_file
    _env_file = path


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM__", 
        extra="ignore"
    )
    api_key: str = untracked_field(
        description="API key for the LLM provider"
    )
    base_url: str = tracked_field(
        description="Base URL of the LLM API endpoint"
    )
    model: str = tracked_field(
        description="Model name to use for generation"
    )
    llm_type: Literal[
        "openai_compatible"
    ] = tracked_field(
        default="openai_compatible",
        description="LLM client type"
    )
    concurrency: int = untracked_field(
        default=1,
        description="Max concurrent LLM requests"
    )
    max_retries: int = tracked_field(
        default=3,
        description="Max retries on API failure"
    )
    retry_base_delay: float = tracked_field(
        default=1.0,
        description="Base delay in seconds for exponential backoff"
    )


class DockerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCKER__",
        extra="ignore"
    )
    cpus: float | None = tracked_field(
        default=1.0,
        description="CPU cores allocated per container"
    )
    memory: str | None = tracked_field(
        default="512m",
        description="Memory limit per container"
    )
    memory_swap: str | None = tracked_field(
        default="512m",
        description="Swap limit per container"
    )
    memory_reservation: str | None = tracked_field(
        default="256m",
        description="Soft memory limit per container"
    )
    network_mode: str | None = tracked_field(
        default="none",
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


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENVIRONMENT__", 
        extra="ignore"
    )
    environment_type: Literal[
        "docker"
    ] = tracked_field(
        default="docker",
        description="Execution environment type"
    )
    docker: DockerSettings = DockerSettings()


class DatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATASET__", 
        extra="ignore"
    )
    cache_dir: Path = untracked_field(
        default=ROOT_DIR / "data" / "cache",
        description="Local cache directory for downloaded datasets"
    )
    dataset_name: Literal[
        "humaneval",
        "apps",
        "mbppplus"
    ] = tracked_field(
        default="mbppplus",
        description="Dataset to evaluate against"
    )


class LoggerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOG__", 
        extra="ignore"
    )
    level: Literal[
        "DEBUG", 
        "INFO", 
        "WARNING", 
        "ERROR"
    ] = untracked_field(
        default="INFO",
        description="Logging verbosity level"
    )
    log_file: Path | None = untracked_field(
        default=None,
        description="Path to log file, stdout only if not set"
    )


class ExperimentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENT__", 
        extra="ignore"
    )
    experiment_id: str = tracked_field(
        default="default",
        description="Unique identifier for this experiment run"
    )
    max_turns: int = tracked_field(
        default=1,
        description="Max feedback iterations per episode"
    )
    samples_per_problem: int = tracked_field(
        default=1,
        description="Independent samples per problem"
    )
    feedback_strategy: Literal[
        "minimal",
        "error_message",
        "progressive"
    ] = tracked_field(
        default="error_message",
        description="Strategy for constructing feedback prompts"
    )
    @property
    def output_dir(self) -> Path:
        return ROOT_DIR / "experiments" / self.experiment_id


class Settings(BaseSettings):
    dataset: DatasetSettings
    llm: LLMSettings
    log: LoggerSettings
    environment: EnvironmentSettings
    experiment: ExperimentSettings


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        raise RuntimeError("Settings not initialized. Call init_settings() first.")
    return _settings


def init_settings() -> Settings:
    global _settings
    _settings = _build_settings()
    return _settings


def _build_settings() -> Settings:
    env = (ROOT_DIR / _env_file).resolve()
    return Settings(
        dataset=DatasetSettings(_env_file=env),
        llm=LLMSettings(_env_file=env),
        log=LoggerSettings(_env_file=env),
        environment=EnvironmentSettings(
            _env_file=env,
            docker=DockerSettings(_env_file=env),
        ),
        experiment=ExperimentSettings(_env_file=env),
    )


def _collect_tracked(obj: BaseSettings, prefix: str = "") -> dict[str, Any]:
    """递归收集标记为 core=True 的字段，返回扁平 dict。"""
    result: dict[str, Any] = {}
    for name, field_info in type(obj).model_fields.items():
        key = f"{prefix}{name}" if prefix else name
        value = getattr(obj, name)
        if isinstance(value, BaseSettings):
            result.update(_collect_tracked(value, f"{key}."))
        elif field_info.json_schema_extra and field_info.json_schema_extra.get("core"):
            result[key] = value
    return result


def get_tracked_config(settings: Settings | None = None) -> dict[str, Any]:
    """提取当前配置中所有 tracked 字段，返回可序列化的扁平 dict。"""
    if settings is None:
        settings = get_settings()
    raw = _collect_tracked(settings)
    # Path 等类型转 str 以便 JSON 序列化
    return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}