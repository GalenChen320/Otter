from typing import Any
from pathlib import Path
from pydantic import Field


ROOT_DIR = Path(__file__).parent.parent.parent.parent
_REQUIRED = ...

def tracked_field(default=_REQUIRED, **kwargs) -> Any:
    extra = kwargs.pop("json_schema_extra", {})
    return Field(default, json_schema_extra={"core": True, **extra}, **kwargs)


def untracked_field(default=_REQUIRED, **kwargs) -> Any:
    extra = kwargs.pop("json_schema_extra", {})
    return Field(default, json_schema_extra={"core": False, **extra}, **kwargs)


def coerce_empty_str(v: Any) -> Any:
    """将空字符串转为 None，用于 Optional 字段的 field_validator。"""
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


__all__ = [
    "ROOT_DIR",
    "tracked_field",
    "untracked_field",
    "coerce_empty_str",
]

