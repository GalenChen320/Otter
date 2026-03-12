import json

from openai import AsyncOpenAI
from .base import BaseLLM

from otter.config.setting import get_settings
from otter.episode import Episode, ResponseManifest


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

    async def _generate(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        input_manifest = turn.input_manifest

        if not input_manifest or not input_manifest.messages_file:
            raise ValueError("OpenAICompatibleLLM requires 'messages_file' in InputManifest")

        messages_path = input_manifest.base_path / input_manifest.messages_file
        messages = json.loads(messages_path.read_text(encoding="utf-8"))

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.llm.model,
            messages=messages,
        )
        content = response.choices[0].message.content

        # 写 response 文件
        response_dir = turn.response_path
        response_file = "response.txt"
        (response_dir / response_file).write_text(content, encoding="utf-8")

        # 写 manifest.json
        manifest = ResponseManifest(
            base_path=response_dir,
            response_file=response_file,
        )
        (response_dir / "manifest.json").write_text(
            json.dumps({"response_file": response_file}),
            encoding="utf-8",
        )
        turn.response_manifest = manifest
