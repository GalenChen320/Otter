from abc import ABC, abstractmethod

from otter.episode import Episode, EnvOutputManifest


class BaseEnvironment(ABC):

    async def execute(self, episode: Episode) -> None:
        """模板方法：调用子类 _execute，保存 manifest，设置 turn。"""
        manifest = await self._execute(episode)
        turn = episode.turns[-1]
        manifest.save(turn.env_output_path)
        turn.env_output_manifest = manifest

    @abstractmethod
    async def _execute(self, episode: Episode) -> EnvOutputManifest:
        pass
