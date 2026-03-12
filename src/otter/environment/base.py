from abc import ABC, abstractmethod

from otter.episode import Episode


class BaseEnvironment(ABC):

    @abstractmethod
    async def execute(self, episode: Episode) -> None:
        pass
