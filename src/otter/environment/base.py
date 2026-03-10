from abc import ABC, abstractmethod

from otter.episode import ExecutionResult


class BaseEnvironment(ABC):

    @abstractmethod
    async def execute(self, code: str, test_code: str) -> ExecutionResult:
        pass
