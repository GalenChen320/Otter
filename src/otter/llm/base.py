import asyncio
from abc import ABC, abstractmethod
from typing import Any

from otter.config.setting import get_settings
from otter.episode import InputManifest
from otter.logger import get_logger


class BaseLLM(ABC):

    async def generate(self, input_manifest: InputManifest) -> Any:
        settings = get_settings()
        logger = get_logger()
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm.max_retries + 1):
            try:
                result = await self._generate(input_manifest)
                logger.info("attempt %d/%d succeeded", attempt, settings.llm.max_retries)
                return result
            except Exception as e:
                last_exc = e
                logger.warning(
                    "attempt %d/%d failed: %s",
                    attempt, settings.llm.max_retries, e,
                )
                if attempt < settings.llm.max_retries:
                    delay = settings.llm.retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
        logger.error("all %d attempts failed: %s", settings.llm.max_retries, last_exc)
        raise last_exc

    @abstractmethod
    async def _generate(self, input_manifest: InputManifest) -> Any:
        pass