from pathlib import Path
from abc import ABC, abstractmethod


from otter.backend.chat_llm import ChatLLMBackend
from otter.backend.docker import DockerBackend
from otter.episode import Episode, InputManifest, OutputManifest


# ── extract: InputManifest → Backend 入参（读文件 → 原生类型）──────

def extract_for_chat_llm(manifest: InputManifest, episode: Episode, output_dir: Path) -> dict:
    """从 prompt_file 读取 prompt，结合历史 turns 构建 messages。"""
    messages = []

    # 历史轮次的 prompt + response
    for turn in episode.turns[:-1]:
        if turn.exec_input_manifest and turn.exec_input_manifest.prompt_file:
            prompt = turn.exec_input_manifest.prompt_file.read_text(encoding="utf-8")
            messages.append({"role": "user", "content": prompt})
        if turn.exec_output_manifest and turn.exec_output_manifest.exec_output_file:
            response = turn.exec_output_manifest.exec_output_file.read_text(encoding="utf-8")
            messages.append({"role": "assistant", "content": response})

    # 当前轮次的 prompt
    prompt = manifest.prompt_file.read_text(encoding="utf-8")
    messages.append({"role": "user", "content": prompt})

    return {"messages": messages, "output_file": output_dir / "response.txt"}


def extract_for_docker(manifest: InputManifest, episode: Episode, output_dir: Path) -> dict:
    """从 InputManifest 提取 DockerBackend 入参。"""
    params: dict = {
        "image_tag": manifest.image_tag,
        "commands": manifest.commands or [],
    }
    if manifest.script_file is not None:
        # script_file → copy_in: 复制到容器 /tmp 目录
        params["copy_in"] = [(manifest.script_file, "/tmp")]
    if manifest.timeout is not None:
        params["timeout"] = manifest.timeout
    return params



# ── extract/pack 分发 ───────────────────────────────────────────────

EXTRACT_DISPATCH = {
    ChatLLMBackend: extract_for_chat_llm,
    DockerBackend: extract_for_docker,
}

# ── Role 基类 ───────────────────────────────────────────────────────

class BaseRole(ABC):
    """角色基类：持有 Backend，定义 run(episode) 模板。"""

    def __init__(self, backend):
        self.backend = backend
        self._extract = EXTRACT_DISPATCH[type(backend)]

    @abstractmethod
    def _get_input_manifest(self, episode: Episode) -> InputManifest:
        """从 Episode 当前 Turn 读取对应的 InputManifest。"""

    @abstractmethod
    def _get_output_dir(self, episode: Episode) -> Path:
        """返回 OutputManifest 的保存目录。"""

    @abstractmethod
    def _set_output_manifest(self, episode: Episode, manifest: OutputManifest) -> None:
        """将 OutputManifest 赋值给 Turn 的对应字段。"""

    async def run(self, episode: Episode) -> None:
        input_manifest = self._get_input_manifest(episode)
        output_dir = self._get_output_dir(episode)

        params = self._extract(input_manifest, episode, output_dir)
        manifest = await self.backend.run(**params)
        manifest.save(output_dir)
        
        self._set_output_manifest(episode, manifest)


# ── 三个子类 ─────────────────────────────────────────────────────────

class ProposerRole(BaseRole):
    def _get_input_manifest(self, episode: Episode) -> InputManifest:
        return episode.turns[-1].prop_input_manifest

    def _get_output_dir(self, episode: Episode) -> Path:
        return episode.turns[-1].prop_output_path

    def _set_output_manifest(self, episode: Episode, manifest: OutputManifest) -> None:
        episode.turns[-1].prop_output_manifest = manifest


class ExecutorRole(BaseRole):
    def _get_input_manifest(self, episode: Episode) -> InputManifest:
        return episode.turns[-1].exec_input_manifest

    def _get_output_dir(self, episode: Episode) -> Path:
        return episode.turns[-1].exec_output_path

    def _set_output_manifest(self, episode: Episode, manifest: OutputManifest) -> None:
        episode.turns[-1].exec_output_manifest = manifest


class EvaluatorRole(BaseRole):
    def _get_input_manifest(self, episode: Episode) -> InputManifest:
        return episode.turns[-1].eval_input_manifest

    def _get_output_dir(self, episode: Episode) -> Path:
        return episode.turns[-1].eval_output_path

    def _set_output_manifest(self, episode: Episode, manifest: OutputManifest) -> None:
        episode.turns[-1].eval_output_manifest = manifest


__all__ = [
    "BaseRole",
    "ProposerRole",
    "ExecutorRole",
    "EvaluatorRole",
]