import asyncio
from abc import ABC, abstractmethod

from otter.config.setting import get_settings
from otter.episode import Episode, LLMOutputManifest
from otter.logger import get_logger


class BaseLLM(ABC):

    async def generate(self, episode: Episode) -> None:
        settings = get_settings()
        logger = get_logger()
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm.max_retries + 1):
            try:
                manifest = await self._generate(episode)
                turn = episode.turns[-1]
                manifest.save(turn.llm_output_path)
                turn.llm_output_manifest = manifest
                logger.info("attempt %d/%d succeeded", attempt, settings.llm.max_retries)
                return
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
        raise RuntimeError(
            f"LLM generation failed after {settings.llm.max_retries} attempts"
        ) from last_exc

    @abstractmethod
    async def _generate(self, episode: Episode) -> LLMOutputManifest:
        pass