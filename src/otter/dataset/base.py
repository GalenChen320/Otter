from abc import ABC, abstractmethod


class BaseDataset(ABC):

    @abstractmethod
    def load(self) -> None:
        pass

    @property
    @abstractmethod
    def task_ids(self) -> list[str]:
        pass

    @abstractmethod
    def make_prompt(self, task_id: str) -> str:
        pass

