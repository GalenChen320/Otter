import asyncio
from abc import ABC, abstractmethod

from otter.config.setting import get_settings
from otter.logger import get_logger


class BaseLLM(ABC):

    async def generate(self, messages: list[dict], *, task_id: str = "") -> str:
        settings = get_settings()
        logger = get_logger()
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm.max_retries + 1):
            try:
                result = await self._generate(messages)
                logger.info("[%s] attempt %d/%d succeeded", task_id, attempt, settings.llm.max_retries)
                return result
            except Exception as e:
                last_exc = e
                logger.warning(
                    "[%s] attempt %d/%d failed: %s",
                    task_id, attempt, settings.llm.max_retries, e,
                )
                if attempt < settings.llm.max_retries:
                    delay = settings.llm.retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
        logger.error("[%s] all %d attempts failed: %s", task_id, settings.llm.max_retries, last_exc)
        raise last_exc

    @abstractmethod
    async def _generate(self, messages: list[dict]) -> str:
        pass

    async def ping(self) -> bool:
        logger = get_logger()
        try:
            await self._generate([{"role": "user", "content": "hi"}])
            logger.info("ping succeeded")
            return True
        except Exception as e:
            logger.error("ping failed: %s", e)
            return False