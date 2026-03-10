from abc import ABC, abstractmethod
from dataclasses import dataclass

from otter.episode import ExecutionObservation


@dataclass
class BaseExecSpec:
    """所有 ExecSpec 的基类。"""
    pass


class BaseEnvironment(ABC):

    @classmethod
    @abstractmethod
    async def execute(cls, spec: BaseExecSpec) -> ExecutionObservation:
        pass
