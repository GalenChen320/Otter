import asyncio
from abc import ABC, abstractmethod

from code_eval.config.setting import settings


class BaseLLM(ABC):

    async def generate(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm.max_retries + 1):
            try:
                return await self._generate(prompt)
            except Exception as e:
                last_exc = e
                if attempt < settings.llm.max_retries:
                    delay = settings.llm.retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
        raise last_exc

    @abstractmethod
    async def _generate(self, prompt: str) -> str:
        pass