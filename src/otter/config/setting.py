from pathlib import Path
from typing import Literal, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .backend_settings import *
from .dataset_settings import *
from .utils import (
    ROOT_DIR, tracked_field, untracked_field, coerce_empty_str
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
        return coerce_empty_str(v)


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
    dataset_name: str = tracked_field(
        default="mbppplus",
        description="Dataset name to evaluate against"
    )
    proposer_type: str | None = tracked_field(
        default=None,
        description="Proposer backend type, None to disable"
    )
    executor_type: str | None = tracked_field(
        default=None,
        description="Executor backend type, None to disable"
    )
    evaluator_type: str | None = tracked_field(
        default=None,
        description="Evaluator backend type, None to disable"
    )

    # ── Component concurrency ──
    proposer_concurrency: int = untracked_field(
        default=1,
        description="Max concurrent proposer executions"
    )
    executor_concurrency: int = untracked_field(
        default=1,
        description="Max concurrent executor executions"
    )
    evaluator_concurrency: int = untracked_field(
        default=1,
        description="Max concurrent evaluator executions"
    )

    # ── Component retry ──
    proposer_retry: int = untracked_field(
        default=1, ge=1,
        description="Max retry attempts for proposer"
    )
    executor_retry: int = untracked_field(
        default=1, ge=1,
        description="Max retry attempts for executor"
    )
    evaluator_retry: int = untracked_field(
        default=1, ge=1,
        description="Max retry attempts for evaluator"
    )

    @field_validator("proposer_type", "executor_type", "evaluator_type", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Any:
        return coerce_empty_str(v)

    @property
    def output_dir(self) -> Path:
        return ROOT_DIR / "experiments" / self.experiment_id

    # ── Component settings (dynamically built per type) ──
    log: LoggerSettings
    dataset: DatasetSettings | None = None
    proposer: BackendSettings | None = None
    executor: BackendSettings | None = None
    evaluator: BackendSettings | None = None


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


def _build_backend_settings(
    role_prefix: str,
    backend_type: str | None,
    env_file: Path,
) -> BackendSettings | None:
    """根据 backend_type 和角色前缀动态构建 Backend Settings。"""
    if backend_type is None:
        return None
    cls = BACKEND_SETTINGS_REGISTRY.get(backend_type)
    if cls is None:
        raise ValueError(f"Unknown backend type: {backend_type!r}")
    return cls(_env_prefix=f"{role_prefix}__", _env_file=env_file)


def _build_dataset_settings(
    dataset_name: str,
    env_file: Path,
) -> DatasetSettings:
    """根据 dataset_name 动态构建 Dataset Settings。"""
    cls = DATASET_SETTINGS_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(f"Unknown dataset type: {dataset_name!r}")
    return cls(_env_prefix="DATASET__", _env_file=env_file)


def _build_settings(env_file: str) -> Settings:
    env = (ROOT_DIR / env_file).resolve()

    # 先构建顶层 settings 以读取 type 字段
    top = Settings(
        _env_file=env,
        log=LoggerSettings(_env_file=env),
    )

    return Settings(
        _env_file=env,
        dataset=_build_dataset_settings(top.dataset_name, env),
        log=top.log,
        proposer=_build_backend_settings("PROPOSER", top.proposer_type, env),
        executor=_build_backend_settings("EXECUTOR", top.executor_type, env),
        evaluator=_build_backend_settings("EVALUATOR", top.evaluator_type, env),
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


__all__ = [
    "ROOT_DIR",
    "get_settings",
    "init_settings",
    "get_tracked_config"
]
