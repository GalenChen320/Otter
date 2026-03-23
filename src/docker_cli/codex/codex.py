"""
CodexDriver — OpenAI Codex CLI 编码智能体的 Driver 实现。

Codex CLI 通过 npm 安装 @openai/codex，
配置通过 CODEX_HOME/auth.json 传入 API Key，
OPENAI_BASE_URL 通过环境变量传入，
使用 codex exec --dangerously-bypass-approvals-and-sandbox 全自动执行任务，
--json 输出 JSONL 格式结构化事件流。

参考：https://developers.openai.com/codex/config-advanced
"""

import json
import shlex

from pydantic import BaseModel

from docker_cli.base import BaseAgentDriver

# Dockerfile 的 wrapper 脚本强制设置 HOME=/opt/agent/home，
# 因此 Codex 的配置目录也应在此目录下
_HOME_DIR = "/opt/agent/home"
_CODEX_HOME = f"{_HOME_DIR}/.codex"


class CodexConfig(BaseModel):
    """OpenAI Codex CLI 智能体配置。"""

    # ── 运行时配置 ──
    api_key: str
    model_name: str
    base_url: str = ""

    # ── Docker 镜像构建配置 ──
    base_image: str = "ubuntu:22.04"
    node_version: str = "22.11.0"
    agent_npm_pkg: str = "@openai/codex"
    agent_bin: str = "codex"
    agent_home: str = "/opt/agent"


class CodexDriver(BaseAgentDriver):
    """OpenAI Codex CLI 编码智能体 Driver。"""

    name = "codex"
    cfg: CodexConfig

    def __init__(self, cfg: CodexConfig) -> None:
        super().__init__(cfg)

    def build_setup_commands(self) -> list[str]:
        """构建 Codex CLI 配置写入命令。"""
        cfg = self.cfg
        model = cfg.model_name.split("/")[-1]

        auth_content = json.dumps({"OPENAI_API_KEY": cfg.api_key}, indent=2)

        config_toml = (
            f'model = "{model}"\n'
            f'model_provider = "Model_Studio_Coding_Plan"\n'
            f'approval_policy = "never"\n'
            f'sandbox_mode = "danger-full-access"\n'
            f'\n'
            f'[model_providers.Model_Studio_Coding_Plan]\n'
            f'name = "Model_Studio_Coding_Plan"\n'
            f'base_url = "{cfg.base_url}"\n'
            f'env_key = "OPENAI_API_KEY"\n'
            f'wire_api = "responses"\n'
        )

        return [
            self._write_file_cmd(f"{_CODEX_HOME}/auth.json", auth_content),
            self._write_file_cmd(f"{_CODEX_HOME}/config.toml", config_toml),
        ]

    def build_command(
        self,
        prompt: str,
        *,
        work_dir: str = "/app",
    ) -> tuple[str, dict]:
        """构建 Codex CLI 执行命令和参数。

        使用 codex exec 子命令以全自动模式运行：
        - --dangerously-bypass-approvals-and-sandbox：跳过沙箱审批，全自动执行
        - --skip-git-repo-check：跳过 git 仓库检查
        - --json：以 JSONL 格式输出结构化事件流
        """
        cfg = self.cfg
        model = cfg.model_name.split("/")[-1]
        escaped_prompt = shlex.quote(prompt)

        env = {
            "HOME": _HOME_DIR,
            "CODEX_HOME": _CODEX_HOME,
            "OPENAI_API_KEY": cfg.api_key,
        }

        cmd = (
            "codex exec "
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {model} "
            "--json "
            f"-- {escaped_prompt}"
        )

        return cmd, {"workdir": work_dir, "environment": env}

    def parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """解析 Codex CLI 的 JSONL 输出，提取 token 用量、耗时、文本回答和错误信息。

        Codex CLI 以 JSONL 格式输出事件流：
        - type=message 事件包含 agent 的文本回答
        - type=turn.completed 事件包含 token 用量
        """
        input_tokens = 0
        output_tokens = 0
        messages = []

        # 逐行解析 JSONL
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type == "turn.completed":
                usage = event.get("usage", {})
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens)
            elif event_type == "message":
                content = event.get("content", "")
                if content:
                    messages.append(content)

        elapsed = 0.0

        error = ""
        if result.returncode != 0:
            error = (
                f"codex exited with non-zero code {result.returncode}.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return {
            "execution_time": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output": "\n".join(messages),
            "error": error,
        }


__all__ = [
    "CodexConfig",
    "CodexDriver",
]
