import asyncio
import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from otter.manifest import InputManifest, Result, OutputManifest, ChatLLMDebugInfo

logger = logging.getLogger(__name__)


class ChatLLMBackend:
    backend_type = "chat_llm"

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

    async def run(self, manifest: InputManifest, output_dir: Path) -> OutputManifest:
        """从 InputManifest 提取参数，调用 LLM。"""
        messages = json.loads(manifest.msg_file.read_text(encoding="utf-8"))
        output_file = output_dir / "response.txt"
        return await self._run(messages=messages, output_file=output_file)

    async def _run(self, messages: list[dict], output_file: Path) -> OutputManifest:
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
                return OutputManifest(
                    backend_type=self.backend_type,
                    products=[output_file],
                    debug_info=ChatLLMDebugInfo(retries=retries)
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
        return OutputManifest(
            backend_type=self.backend_type,
            products=[None], 
            debug_info=ChatLLMDebugInfo(retries=retries)
        )


__all__ = [
    "ChatLLMBackend",
]