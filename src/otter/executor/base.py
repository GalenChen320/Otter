import asyncio
from abc import ABC, abstractmethod

from otter.config.setting import get_settings
from otter.episode import Episode, ExecOutputManifest
from otter.logger import get_logger


class BaseExecutor(ABC):

    async def run(self, episode: Episode) -> None:
        settings = get_settings()
        logger = get_logger()
        last_exc: Exception | None = None
        for attempt in range(1, settings.executor.max_retries + 1):
            try:
                manifest = await self._run(episode)
                turn = episode.turns[-1]
                manifest.save(turn.exec_output_path)
                turn.exec_output_manifest = manifest
                logger.info("attempt %d/%d succeeded", attempt, settings.executor.max_retries)
                return
            except Exception as e:
                last_exc = e
                logger.warning(
                    "attempt %d/%d failed: %s",
                    attempt, settings.executor.max_retries, e,
                )
                if attempt < settings.executor.max_retries:
                    delay = settings.executor.retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
        logger.error("all %d attempts failed: %s", settings.executor.max_retries, last_exc)
        raise RuntimeError(
            f"Run executor failed after {settings.executor.max_retries} attempts"
        ) from last_exc

    @abstractmethod
    async def _run(self, episode: Episode) -> ExecOutputManifest:
        pass