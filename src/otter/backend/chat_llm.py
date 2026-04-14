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
    ):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def run(self, manifest: InputManifest, output_dir: Path) -> OutputManifest:
        return await self._run(**manifest.params, output_file=output_dir / "response.txt")

    async def _run(self, messages: list[dict], output_file: Path) -> OutputManifest:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            output_file.write_text(content, encoding="utf-8")
            return OutputManifest(
                backend_type=self.backend_type,
                products=[output_file],
                debug_info=ChatLLMDebugInfo(
                    result=Result(stdout="", stderr="", returncode=0, timed_out=False),
                ),
            )
        except Exception as e:
            logger.warning("chat_llm call failed: %s", e)
            return OutputManifest(
                backend_type=self.backend_type,
                products=[],
                debug_info=ChatLLMDebugInfo(
                    result=Result(stdout="", stderr=str(e), returncode=1, timed_out=False),
                ),
            )


__all__ = [
    "ChatLLMBackend",
]