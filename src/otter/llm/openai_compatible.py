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
        input_manifest = turn.llm_input_manifest

        if not input_manifest or not input_manifest.prompt_file:
            raise ValueError("OpenAICompatibleLLM requires 'prompt_file' in LLMInputManifest")

        # 遍历所有 Turn 构建多轮 messages
        messages = []
        for t in episode.turns:
            # 每轮的 prompt（第一轮是题目，后续是 feedback）
            prompt = t.llm_input_manifest.prompt_file.read_text(encoding="utf-8")
            messages.append({"role": "user", "content": prompt})

            # 历史 Turn 有 response，当前 Turn 还没有
            if t is not turn:
                response_file = t.response_path / "response.txt"
                messages.append({"role": "assistant", "content": response_file.read_text(encoding="utf-8")})

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.llm.model,
            messages=messages,
        )
        content = response.choices[0].message.content

        # 写 response 文件
        response_file = turn.response_path / "response.txt"
        response_file.write_text(content, encoding="utf-8")

        # 写 manifest.json
        manifest = ResponseManifest(response_file=response_file)
        (turn.response_path / "manifest.json").write_text(
            json.dumps({"response_file": str(response_file)}),
            encoding="utf-8",
        )
        turn.response_manifest = manifest
