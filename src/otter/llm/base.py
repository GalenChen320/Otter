import asyncio
from abc import ABC, abstractmethod
from typing import Any

from otter.config.setting import get_settings
from otter.logger import get_logger


class BaseLLM(ABC):

    async def generate(self, llm_input: Any) -> Any:
        settings = get_settings()
        logger = get_logger()
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm.max_retries + 1):
            try:
                result = await self._generate(llm_input)
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
    async def _generate(self, llm_input: Any) -> Any:
        pass

    async def ping(self) -> bool:
        settings = get_settings()
        logger = get_logger()
        logger.info("pinging LLM: model=%s, base_url=%s", settings.llm.model, settings.llm.base_url)
        try:
            await self._generate([{"role": "user", "content": "hi"}])
            logger.info("ping succeeded")
            return True
        except Exception as e:
            logger.error("ping failed: %s", e)
            return False