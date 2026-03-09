from abc import ABC, abstractmethod

from otter.episode import Episode


class BaseDataset(ABC):

    @abstractmethod
    def load(self) -> None:
        pass

    @property
    @abstractmethod
    def task_ids(self) -> list[str]:
        pass

    @abstractmethod
    def make_messages(self, episode: Episode) -> list[dict]:
        pass

