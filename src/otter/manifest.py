import json
from pathlib import Path
from dataclasses import dataclass, fields
from typing import get_type_hints, get_args


def _is_path_field(hint) -> bool:
    """判断类型注解是否包含 Path（支持 Path | None 等联合类型）。"""
    if hint is Path:
        return True
    for arg in get_args(hint):
        if arg is Path:
            return True
    return False


@dataclass
class BaseManifest:
    def to_dict(self) -> dict:
        """序列化为可 JSON 化的 dict。

        当前仅对 Path 类型字段做 Path → str 转换，
        其余字段（str / int / bool / list[str] 等）直接透传。
        与 from_dict 的类型转换逻辑对称，两者需保持一致。
        """
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Path):
                val = str(val)
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "BaseManifest":
        """从 dict 反序列化重建 Manifest。

        当前仅对 Path 类型字段做 str → Path 转换，
        其余字段（str / int / bool / list[str] 等）依赖 JSON 原生类型直接匹配。
        若将来引入 list[Path] 等复合类型，需扩展此方法的类型转换逻辑。
        """
        hints = get_type_hints(cls)
        kwargs = {}
        for key, val in data.items():
            if key not in hints:
                continue
            if val is not None and _is_path_field(hints[key]):
                kwargs[key] = Path(val)
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def save(self, directory: Path) -> None:
        """将自身序列化写入 directory/manifest.json。"""
        (directory / "manifest.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@dataclass
class InputManifest(BaseManifest):  # TODO INPUT侧要不要改呢？
    # ChatLLMBackend
    prompt_file: Path | None = None

    # DockerBackend
    image_tag: str | None = None
    script_file: Path | None = None
    commands: list[str] | None = None
    timeout: int | None = None


@dataclass
class OutputManifest(BaseManifest):
    backend_type: str
    products: list[Path | None]
    debug_info: dict
    unexpected: str = ""


@dataclass
class Result:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool