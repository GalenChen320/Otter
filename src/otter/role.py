from pathlib import Path
from abc import ABC, abstractmethod

from otter.episode import Episode, InputManifest, OutputManifest


# ── Role 基类 ───────────────────────────────────────────────────────

class BaseRole(ABC):
    """角色基类：持有 Backend，定义 run(episode) 模板。"""

    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def _get_input_manifest(self, episode: Episode) -> InputManifest:
        """从 Episode 当前 Turn 读取对应的 InputManifest。"""

    @abstractmethod
    def _get_output_dir(self, episode: Episode) -> Path:
        """返回 OutputManifest 的保存目录。"""

    @abstractmethod
    def _set_output_manifest(self, episode: Episode, manifest: OutputManifest) -> None:
        """将 OutputManifest 赋值给 Turn 的对应字段。"""

    async def run(self, episode: Episode) -> OutputManifest:
        input_manifest = self._get_input_manifest(episode)
        output_dir = self._get_output_dir(episode)

        manifest = await self.backend.run(input_manifest, output_dir)
        manifest.save(output_dir)

        self._set_output_manifest(episode, manifest)
        return manifest


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
