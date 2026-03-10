from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseExecSpec:
    """所有 ExecSpec 的基类。"""
    pass


@dataclass
class ExecutionObservation:
    """环境执行后的原始观测结果。

    不含任何判定逻辑（如 passed），判定由上层负责。
    """
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False


class BaseEnvironment(ABC):

    @classmethod
    @abstractmethod
    async def execute(cls, spec: BaseExecSpec) -> ExecutionObservation:
        pass
