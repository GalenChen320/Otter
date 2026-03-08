from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent.parent.parent


def make_settings_config(**kwargs) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        **kwargs
    )


class LLMSettings(BaseSettings):
    model_config = make_settings_config(env_prefix="LLM__")
    api_key: str
    base_url: str
    model: str
    concurrency: int = 10
    samples_per_problem: int = 1


class ExecutorSettings(BaseSettings):
    model_config = make_settings_config(env_prefix="EXECUTOR__")
    concurrency: int = 5         # Docker 并发数
    timeout: int = 10            # 每个容器最多跑几秒


class DatasetSettings(BaseSettings):
    cache_dir: Path = ROOT_DIR / "data" / "cache"
    dataset_name: Literal[
        "humaneval",
        "apps",
        "mbppplus"
    ] = "mbppplus"
    


class Settings(BaseSettings):
    dataset: DatasetSettings = DatasetSettings()
    llm: LLMSettings = LLMSettings()


settings = Settings()