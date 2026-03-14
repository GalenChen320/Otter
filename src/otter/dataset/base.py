from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path

from otter.episode import EnvInputManifest, Episode, LLMInputManifest


class BaseDataset(ABC):
    base_dir: Path | None = None

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

    @asynccontextmanager
    async def run_context(self, output_dir: Path):
        """Dataset 级别上下文管理器，包裹 setup/teardown。"""
        await self.setup(output_dir)
        try:
            yield
        finally:
            await self.teardown()

    async def setup_episode(self, episode: Episode) -> None:
        """Episode 级别初始化，每道题开始前调用。"""
        pass

    async def teardown_episode(self, episode: Episode) -> None:
        """Episode 级别资源回收，每道题结束后调用。"""
        pass

    @asynccontextmanager
    async def episode_context(self, episode: Episode):
        """Episode 级别上下文管理器，包裹 setup/teardown。"""
        await self.setup_episode(episode)
        try:
            yield episode
        finally:
            await self.teardown_episode(episode)

    # ── Pipeline 编排接口 ──

    @abstractmethod
    def _prepare_llm_input(self, episode: Episode) -> LLMInputManifest:
        """子类实现：写入 LLM 输入文件，返回 LLMInputManifest。"""
        ...

    @abstractmethod
    def _prepare_env_input(self, episode: Episode) -> EnvInputManifest:
        """子类实现：从 turn.llm_output_manifest 读取响应，写入执行文件，返回 EnvInputManifest。"""
        ...

    def prepare_llm_input(self, episode: Episode) -> None:
        """模板方法：调用子类 _prepare_llm_input，保存 manifest，设置 turn。"""
        manifest = self._prepare_llm_input(episode)
        turn = episode.turns[-1]
        manifest.save(turn.llm_input_path)
        turn.llm_input_manifest = manifest

    def prepare_env_input(self, episode: Episode) -> None:
        """模板方法：调用子类 _prepare_env_input，保存 manifest，设置 turn。"""
        manifest = self._prepare_env_input(episode)
        turn = episode.turns[-1]
        manifest.save(turn.env_input_path)
        turn.env_input_manifest = manifest

    @abstractmethod
    async def _judge(self, episode: Episode) -> None:
        """子类实现：从 turn.env_output_manifest 读取观测，判定是否通过，更新 turn.passed。"""
        pass

    async def make_judgement(self, episode: Episode) -> None:
        """判定 + 保存 meta，标记 turn 完成。"""
        await self._judge(episode)
        episode.turns[-1].save_meta()