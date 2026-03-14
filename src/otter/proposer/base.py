import asyncio
from abc import ABC, abstractmethod

from otter.episode import Episode, PropOutputManifest


class BaseProposer(ABC):
    async def run(self, episode: Episode) -> None:
        pass

    @abstractmethod
    async def _run(self, episode: Episode) -> PropOutputManifest:
        pass