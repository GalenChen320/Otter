from openai import AsyncOpenAI
from .base import BaseLLM

from otter.config.setting import get_settings
from otter.episode import Episode, LLMOutputManifest


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

    async def _generate(self, episode: Episode) -> LLMOutputManifest:
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

            # 历史 Turn 有 llm_output，当前 Turn 还没有
            if t is not turn:
                messages.append({"role": "assistant", "content": t.llm_output_manifest.llm_output_file.read_text(encoding="utf-8")})

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.llm.model,
            messages=messages,
        )
        content = response.choices[0].message.content

        # 写 llm_output 文件
        llm_output_file = turn.llm_output_path / "response.txt"
        llm_output_file.write_text(content, encoding="utf-8")

        return LLMOutputManifest(llm_output_file=llm_output_file)
