import json
from typing import Any

from openai import AsyncOpenAI
from .base import BaseLLM

from otter.config.setting import get_settings
from otter.episode import InputManifest


class OpenAICompatibleLLM(BaseLLM):
    """
    兼容 OpenAI 接口的 LLM
    支持 OpenAI / DeepSeek / vLLM / Ollama 等
    """
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

    async def _generate(self, input_manifest: InputManifest) -> Any:
        if not input_manifest.messages_file:
            raise ValueError("OpenAICompatibleLLM requires 'messages_file' in InputManifest")

        messages_path = input_manifest.base_path / input_manifest.messages_file
        messages = json.loads(messages_path.read_text(encoding="utf-8"))

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.llm.model,
            messages=messages,
        )
        return response.choices[0].message.content
