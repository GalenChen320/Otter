"""
ClaudeDriver — Anthropic Claude Code 编码智能体的 Driver 实现。

Claude Code 通过 npm 安装 @anthropic-ai/claude-code，
配置通过写入 ~/.claude/settings.json 传入（包含 API Key、Base URL、Model、权限等），
执行结果通过 stdout JSON 输出（--output-format json）。

参考：reference/claude-code/install.sh（配置文件写入）和 start.sh（启动参数）。
"""

import json
import subprocess
from pydantic import BaseModel, field_validator

from docker_cli.base import AgentDriver


class ClaudeConfig(BaseModel):
    """Claude Code 智能体配置。"""

    # ── 运行时配置 ──
    api_key: str
    model_name: str
    base_url: str = ""
    allowed_tools: list[str] = [
        "Bash", "Edit", "Write", "Read", "Glob", "Grep",
        "WebFetch", "NotebookEdit", "NotebookRead",
    ]

    # ── Docker 镜像构建配置 ──
    base_image: str = "ubuntu:22.04"
    node_version: str = "22.11.0"
    agent_npm_pkg: str = "@anthropic-ai/claude-code"
    agent_bin: str = "claude"
    agent_home: str = "/opt/agent"

    @field_validator("base_url")
    @classmethod
    def strip_trailing_v1(cls, v: str) -> str:
        """Claude Code 会自动追加 /v1/messages，去掉用户可能多带的 /v1 后缀。"""
        v = v.rstrip("/")
        if v.endswith("/v1"):
            v = v[:-3]
        return v


class ClaudeDriver(AgentDriver):
    """Claude Code 编码智能体 Driver。"""

    name = "claude"
    cfg: ClaudeConfig

    def __init__(self, cfg: ClaudeConfig) -> None:
        super().__init__(cfg)

    def setup_config(self, container_name: str) -> None:
        """将 ~/.claude/settings.json 写入容器，配置权限、API Key、Base URL 和 Model。"""
        cfg = self.cfg

        env_fields = {
            "ANTHROPIC_API_KEY": cfg.api_key,
            "ANTHROPIC_AUTH_TOKEN": cfg.api_key,
            "ANTHROPIC_MODEL": cfg.model_name,
            "IS_SANDBOX": "1",
        }
        if cfg.base_url:
            env_fields["ANTHROPIC_BASE_URL"] = cfg.base_url

        settings = {
            "$schema": "https://json.schemastore.org/claude-code-settings.json",
            "permissions": {
                "allow": cfg.allowed_tools,
                "defaultMode": "bypassPermissions",
            },
            "env": env_fields,
            "model": cfg.model_name,
        }

        self.write_file_to_container(
            container_name,
            f"{cfg.agent_home}/home/.claude/settings.json",
            json.dumps(settings, indent=4, ensure_ascii=False),
        )

        onboarding_config = {
            "hasCompletedOnboarding": True,
        }
        self.write_file_to_container(
            container_name,
            f"{cfg.agent_home}/home/.claude.json",
            json.dumps(onboarding_config, indent=4, ensure_ascii=False),
        )

    def run(
        self,
        container_name: str,
        prompt: str,
        *,
        work_dir: str = "/app",
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """在容器内执行 Claude Code 任务。"""
        cfg = self.cfg
        allowed_tools_args = ["--allowedTools"] + cfg.allowed_tools

        env_args = [
            "-e", f"ANTHROPIC_API_KEY={cfg.api_key}",
            "-e", f"ANTHROPIC_MODEL={cfg.model_name}",
            "-e", "DISABLE_AUTOUPDATER=1",
            "-e", "DISABLE_SEND_PV=1",
            "-e", "DISABLE_TELEMETRY=1",
            "-e", "IS_SANDBOX=1",
        ]
        if cfg.base_url:
            env_args += ["-e", f"ANTHROPIC_BASE_URL={cfg.base_url}"]

        return subprocess.run(
            [
                "docker", "exec",
                "-w", work_dir,
                *env_args,
                container_name,
                cfg.agent_bin,
                "--verbose",
                "--print",
                "--output-format", "json",
                "--model", cfg.model_name,
                *allowed_tools_args,
                "-p", prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """解析 Claude Code 的 JSON 输出，提取 token 用量、耗时、文本回答和错误信息。"""
        output_data = {}
        if result.stdout and result.stdout.strip():
            try:
                raw = json.loads(result.stdout)
                if isinstance(raw, list):
                    result_objects = [
                        item for item in raw
                        if isinstance(item, dict) and item.get("type") == "result"
                    ]
                    output_data = result_objects[-1] if result_objects else {}
                elif isinstance(raw, dict):
                    output_data = raw
            except Exception:
                pass

        usage = output_data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        duration_ms = output_data.get("duration_ms", 0)
        result_text = output_data.get("result", "")

        # 检查错误情况
        error = ""
        if result.returncode != 0:
            error = result_text or f"stdout: {result.stdout}\nstderr: {result.stderr}"
            error = f"claude exited with non-zero code {result.returncode}. {error}"
        elif not output_data and result.stdout and result.stdout.strip():
            error = f"Failed to parse claude result as JSON. stdout: {result.stdout!r}"
        else:
            subtype = output_data.get("subtype", "")
            is_error = output_data.get("is_error", False)
            if subtype != "success" or is_error:
                error = (
                    f"Claude Code task did not succeed. "
                    f"subtype={subtype!r}, is_error={is_error}. "
                    f"{result_text or f'stdout: {result.stdout}'}"
                )

        return {
            "execution_time": duration_ms / 1000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output": result_text,
            "error": error,
        }


__all__ = [
    "ClaudeConfig",
    "ClaudeDriver",
]