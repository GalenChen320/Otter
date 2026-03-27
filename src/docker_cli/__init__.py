from pathlib import Path

from docker_cli.base import BaseAgentDriver
from docker_cli.claude.claude import ClaudeConfig, ClaudeDriver
from docker_cli.codex.codex import CodexConfig, CodexDriver
from docker_cli.opencode.opencode import OpenCodeConfig, OpenCodeDriver
from docker_cli.openhands.openhands import OpenHandsConfig, OpenHandsDriver
from docker_cli.miniswe.miniswe import MiniSWEConfig, MiniSWEDriver


_PKG_DIR = Path(__file__).resolve().parent

# agent_name → (DriverClass, ConfigClass)
AGENT_REGISTRY: dict[str, tuple[type[BaseAgentDriver], type]] = {
    "claude":    (ClaudeDriver, ClaudeConfig),
    "codex":     (CodexDriver, CodexConfig),
    "opencode":  (OpenCodeDriver, OpenCodeConfig),
    "openhands": (OpenHandsDriver, OpenHandsConfig),
    "miniswe": (MiniSWEDriver, MiniSWEConfig),
}

# agent_name → Dockerfile 路径
AGENT_DOCKERFILE_MAP: dict[str, Path] = {
    "claude":    _PKG_DIR / "claude" / "Dockerfile.claude",
    "codex":     _PKG_DIR / "codex" / "Dockerfile.codex",
    "opencode":  _PKG_DIR / "opencode" / "Dockerfile.opencode",
    "openhands": _PKG_DIR / "openhands" / "Dockerfile.openhands",
    "miniswe": _PKG_DIR / "miniswe" / "Dockerfile.miniswe",
}

__all__ = [
    "BaseAgentDriver",
    "ClaudeConfig",
    "ClaudeDriver",
    "CodexConfig",
    "CodexDriver",
    "OpenCodeConfig",
    "OpenCodeDriver",
    "OpenHandsConfig",
    "OpenHandsDriver",
    "MiniSWEConfig",
    "MiniSWEDriver",
    "AGENT_REGISTRY",
    "AGENT_DOCKERFILE_MAP",
]
