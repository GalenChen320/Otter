from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from otter.episode import Episode, ExecutionObservation


class BaseDataset(ABC):
    base_dir: Path | None = None
    supported_llms: list[type] = []
    supported_environments: list[type] = []

    @abstractmethod
    def load(self) -> None:
        pass

    @property
    @abstractmethod
    def task_ids(self) -> list[str]:
        pass

    # ── 生命周期 ──

    async def setup(self, output_dir: Path) -> None:
        """Dataset 级别初始化，整个评测开始前调用一次。"""
        self.base_dir = output_dir

    async def teardown(self) -> None:
        """Dataset 级别资源回收，整个评测结束后调用一次。"""
        pass

    async def setup_episode(self, episode: Episode) -> None:
        """Episode 级别初始化，每道题开始前调用。"""
        pass

    async def teardown_episode(self, episode: Episode) -> None:
        """Episode 级别资源回收，每道题结束后调用。"""
        pass

    # ── Pipeline 编排接口 ──

    @abstractmethod
    def prepare_input(self, episode: Episode, llm_type: type) -> Any:
        """往 episode.turns[-1].input_path 写入输入文件，并根据 llm_type 返回 LLM 能消费的格式。"""
        pass

    @abstractmethod
    def prepare_exec(self, episode: Episode, response: Any, env_type: type) -> Any:
        """往 episode.turns[-1].response_path 写入响应文件，并根据 env_type 返回执行规格。"""
        pass

    @abstractmethod
    def write_observation(self, episode: Episode, observation: ExecutionObservation) -> None:
        """往 episode.turns[-1].observation_path 写入观测文件。"""
        pass

    @abstractmethod
    def judge(self, episode: Episode, observation: ExecutionObservation) -> bool:
        """判定当前 turn 是否通过。"""
        pass