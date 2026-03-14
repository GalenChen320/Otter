from abc import ABC, abstractmethod

from otter.episode import Episode, EvalOutputManifest


class BaseEvaluator(ABC):

    async def run(self, episode: Episode) -> None:
        """模板方法：调用子类 _run，保存 manifest，设置 turn。"""
        manifest = await self._run(episode)
        turn = episode.turns[-1]
        manifest.save(turn.eval_output_path)
        turn.eval_output_manifest = manifest

    @abstractmethod
    async def _run(self, episode: Episode) -> EvalOutputManifest:
        pass
