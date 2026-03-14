from abc import ABC, abstractmethod

from otter.episode import Episode, PropOutputManifest


class BaseProposer(ABC):
    async def run(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        manifest = await self._run(episode)
        manifest.save(turn.prop_output_path)
        turn.prop_output_manifest = manifest

    @abstractmethod
    async def _run(self, episode: Episode) -> PropOutputManifest:
        pass