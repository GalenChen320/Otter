from openai import AsyncOpenAI
from .base import BaseExecutor

from otter.config.setting import get_settings
from otter.episode import Episode, ExecOutputManifest


class ChatLLMExecutor(BaseExecutor):
    """
    兼容 OpenAI 接口的 Executor
    支持 OpenAI / DeepSeek / vLLM / Ollama 等
    """
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.executor.api_key,
            base_url=settings.executor.base_url,
        )

    async def _run(self, episode: Episode) -> ExecOutputManifest:
        turn = episode.turns[-1]
        input_manifest = turn.exec_input_manifest

        if not input_manifest or not input_manifest.prompt_file:
            raise ValueError("ChatLLMExecutor requires 'prompt_file' in ExecInputManifest")

        # 遍历所有 Turn 构建多轮 messages
        messages = []
        for t in episode.turns:
            # 每轮的 prompt（第一轮是题目，后续是 feedback）
            prompt = t.exec_input_manifest.prompt_file.read_text(encoding="utf-8")
            messages.append({"role": "user", "content": prompt})

            # 历史 Turn 有 exec_output，当前 Turn 还没有
            if t is not turn:
                messages.append({"role": "assistant", "content": t.exec_output_manifest.exec_output_file.read_text(encoding="utf-8")})

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.executor.model,
            messages=messages,
        )
        content = response.choices[0].message.content

        # 写 exec_output 文件
        exec_output_file = turn.exec_output_path / "response.txt"
        exec_output_file.write_text(content, encoding="utf-8")

        return ExecOutputManifest(exec_output_file=exec_output_file)
