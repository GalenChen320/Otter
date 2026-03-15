import asyncio
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


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

    async def run(self, messages: list[dict]) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_exc = e
                logger.warning(
                    "attempt %d/%d failed: %s", attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
        raise RuntimeError(
            f"ChatLLMBackend failed after {self.max_retries} attempts"
        ) from last_exc

__all__ = [
    "ChatLLMBackend",
]