from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent.parent.parent

_env_file: str = ".env"


def set_env_file(path: str) -> None:
    global _env_file
    _env_file = path


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM__", 
        extra="ignore"
    )
    api_key: str
    base_url: str
    model: str
    response_format: Literal[
        "openai_compatible"
    ] = "openai_compatible"
    concurrency: int = 1
    samples_per_problem: int = 1
    max_retries: int = 3
    retry_base_delay: float = 1.0


class DockerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCKER__",
        extra="ignore"
    )
    cpus: float = 1.0
    memory: str = "512m"
    memory_swap: str = "512m"
    memory_reservation: str = "256m"
    device_read_bps: str | None = "128m"
    device_write_bps: str | None = "128m"
    timeout: int = 10


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENVIRONMENT__", 
        extra="ignore"
    )
    environment_type: Literal["docker"] = "docker"
    docker: DockerSettings = DockerSettings()


class DatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATASET__", 
        extra="ignore"
    )
    cache_dir: Path = ROOT_DIR / "data" / "cache"
    dataset_name: Literal[
        "humaneval",
        "apps",
        "mbppplus"
    ] = "mbppplus"


class LoggerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOG__", 
        extra="ignore"
    )
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path | None = None


class ExperimentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENT__", 
        extra="ignore"
    )
    experiment_id: str = "default"
    max_turns: int = 1
    feedback_strategy: Literal[
        "minimal",
        "error_message",
        "progressive"
    ] = "error_message"
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