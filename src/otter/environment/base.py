from abc import ABC, abstractmethod
from typing import Any

from otter.episode import ExecutionObservation


class BaseEnvironment(ABC):

    @abstractmethod
    async def execute(self, exec_input: Any) -> Any:
        pass
