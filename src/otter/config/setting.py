from pathlib import Path
from typing import Literal, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent.parent.parent
_REQUIRED = ...

def tracked_field(default=_REQUIRED, **kwargs) -> Any:
    extra = kwargs.pop("json_schema_extra", {})
    return Field(default, json_schema_extra={"core": True, **extra}, **kwargs)


def untracked_field(default=_REQUIRED, **kwargs) -> Any:
    extra = kwargs.pop("json_schema_extra", {})
    return Field(default, json_schema_extra={"core": False, **extra}, **kwargs)


def _coerce_empty_str(v: Any) -> Any:
    """将空字符串转为 None，用于 Optional 字段的 field_validator。"""
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


class ExecutorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXECUTOR__", 
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
    concurrency: int = untracked_field(
        default=1,
        description="Max concurrent executor executions"
    )
    max_retries: int = tracked_field(
        default=3, ge=1,
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

    @field_validator(
        "cpus", "memory", "memory_swap", "memory_reservation",
        "network_mode", "device_read_bps", "device_write_bps",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        return _coerce_empty_str(v)


class EvaluatorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVALUATOR__", 
        extra="ignore"
    )
    concurrency: int = untracked_field(
        default=1,
        description="Max concurrent evaluator executions"
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
        "evalplus",
        "mbppplus"
    ] = tracked_field(
        default="mbppplus",
        description="Dataset to evaluate against"
    )


class ProposerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROPOSER__",
        extra="ignore"
    )
    concurrency: int = untracked_field(
        default=1,
        description="Max concurrent proposer executions"
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

    @field_validator("log_file", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        return _coerce_empty_str(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # ── Experiment (top-level) ──
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

    # ── Component type selectors (None = disabled) ──
    proposer_type: str | None = tracked_field(
        default=None,
        description="Proposer type, None to disable"
    )
    executor_type: str | None = tracked_field(
        default=None,
        description="Executor client type, None to disable"
    )
    evaluator_type: str | None = tracked_field(
        default=None,
        description="Evaluator type, None to disable"
    )

    @field_validator("proposer_type", "executor_type", "evaluator_type", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        return _coerce_empty_str(v)

    @property
    def output_dir(self) -> Path:
        return ROOT_DIR / "experiments" / self.experiment_id

    # ── Component settings ──
    dataset: DatasetSettings
    log: LoggerSettings
    proposer: ProposerSettings | None = None
    executor: ExecutorSettings | None = None
    evaluator: EvaluatorSettings | None = None


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        raise RuntimeError("Settings not initialized. Call init_settings() first.")
    return _settings


def init_settings(env_file: str = ".env") -> Settings:
    global _settings
    _settings = _build_settings(env_file)
    return _settings


def _build_settings(env_file: str) -> Settings:
    env = (ROOT_DIR / env_file).resolve()

    # 先构建顶层 settings 以读取 type 字段
    top = Settings(
        _env_file=env,
        dataset=DatasetSettings(_env_file=env),
        log=LoggerSettings(_env_file=env),
    )

    proposer = (
        ProposerSettings(_env_file=env)
        if top.proposer_type is not None else None
    )
    executor = (
        ExecutorSettings(_env_file=env)
        if top.executor_type is not None else None
    )
    evaluator = (
        EvaluatorSettings(
            _env_file=env,
            docker=DockerSettings(_env_file=env),
        )
        if top.evaluator_type is not None else None
    )

    return Settings(
        _env_file=env,
        dataset=top.dataset,
        log=top.log,
        proposer=proposer,
        executor=executor,
        evaluator=evaluator,
    )


def _collect_tracked(obj: BaseSettings, prefix: str = "") -> dict[str, Any]:
    """递归收集标记为 core=True 的字段，返回扁平 dict。"""
    result: dict[str, Any] = {}
    for name, field_info in type(obj).model_fields.items():
        key = f"{prefix}{name}" if prefix else name
        value = getattr(obj, name)
        if isinstance(value, BaseSettings):
            result.update(_collect_tracked(value, f"{key}."))
        else:
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and extra.get("core"):
                result[key] = value
    return result


def get_tracked_config(settings: Settings | None = None) -> dict[str, Any]:
    """提取当前配置中所有 tracked 字段，返回可序列化的扁平 dict。"""
    if settings is None:
        settings = get_settings()
    raw = _collect_tracked(settings)
    # Path 等类型转 str 以便 JSON 序列化
    return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}
