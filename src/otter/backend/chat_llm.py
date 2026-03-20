import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from otter.backend.base import Result

logger = logging.getLogger(__name__)


@dataclass
class ChatLLMRunResult:
    retries: list[Result] = field(default_factory=list)
    products: list[Path | None] = field(default_factory=list)
    error: str = ""


class ChatLLMBackend:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int,
        retry_base_delay: float,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def run(self, messages: list[dict], output_file: Path) -> ChatLLMRunResult:
        retries: list[Result] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                content = response.choices[0].message.content
                output_file.write_text(content, encoding="utf-8")
                retries.append(Result(
                    stdout="", stderr="", returncode=0, timed_out=False,
                ))
                return ChatLLMRunResult(
                    products=[output_file],
                    retries=retries,
                )
            except Exception as e:
                logger.warning(
                    "attempt %d/%d failed: %s", attempt, self.max_retries, e,
                )
                retries.append(Result(
                    stdout="", stderr=str(e), returncode=1, timed_out=False,
                ))
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        error_msg = f"ChatLLMBackend failed after {self.max_retries} attempts"
        logger.error(error_msg)
        return ChatLLMRunResult(
            products=[None],
            retries=retries,
            error=error_msg,
        )

__all__ = [
    "ChatLLMBackend",
    "ChatLLMRunResult",
]