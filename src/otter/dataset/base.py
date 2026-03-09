from abc import ABC, abstractmethod


class BaseDataset(ABC):

    # ── 加载和索引 ──────────────────────────────

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    # ── 核心行为 ────────────────────────────────

    @abstractmethod
    def make_prompt(self, index: int) -> str:
        """给一个索引 → 返回 prompt"""
        pass

