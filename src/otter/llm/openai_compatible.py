from openai import AsyncOpenAI
from .base import BaseLLM

from otter.config.setting import settings, ROOT_DIR


class OpenAICompatibleLLM(BaseLLM):
    """
    兼容 OpenAI 接口的 LLM
    支持 OpenAI / DeepSeek / vLLM / Ollama 等
    """
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

    async def _generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=settings.llm.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
