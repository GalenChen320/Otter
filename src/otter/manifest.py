from typing import Annotated, Literal, Self
from pathlib import Path
from pydantic import BaseModel, Discriminator


class BaseManifest(BaseModel):
    def save(self, directory: Path) -> None:
        """将自身序列化写入 directory/manifest.json。"""
        (directory / "manifest.json").write_text(
            # mode="json" 会把 Path 自动序列化为字符串
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """从 directory/manifest.json 反序列化重建 Manifest。"""
        return cls.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )


class InputManifest(BaseManifest):
    # ChatLLMBackend
    msg_file: Path | None = None

    # DockerBackend
    image_tag: str | None = None
    script_file: Path | None = None
    commands: list[str] | None = None
    command_params: list[dict] | None = None
    timeout: int | None = None


class Result(BaseModel):
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class BaseDebugInfo(BaseModel):
    backend_type: str


class DockerDebugInfo(BaseDebugInfo):
    backend_type: Literal["docker"] = "docker"
    copy_in: list[Result] = []
    commands: list[Result] = []
    copy_out: list[Result] = []


class ChatLLMDebugInfo(BaseDebugInfo):
    backend_type: Literal["chat_llm"] = "chat_llm"
    result: Result


DebugInfo = Annotated[
    DockerDebugInfo | ChatLLMDebugInfo,
    Discriminator("backend_type"),
]


class OutputManifest(BaseManifest):
    backend_type: str
    products: list[Path | None]
    debug_info: DebugInfo | None = None
    unexpected: str = ""


__all__ = [
    "BaseManifest",
    "InputManifest",
    "BaseDebugInfo",
    "DockerDebugInfo",
    "ChatLLMDebugInfo",
    "DebugInfo",
    "OutputManifest",
    "Result",
]