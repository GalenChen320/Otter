"""
OpenCodeDriver — OpenCode 编码智能体的 Driver 实现。

OpenCode 通过 npm 安装（opencode-ai），
配置文件为 ~/.config/opencode/config.json，
通过 opencode run 命令执行任务，结果输出到 stdout。

参考：https://opencode.ai/docs/cli/
"""

import json
import shlex
import subprocess
from pydantic import BaseModel

from docker_cli.base import BaseAgentDriver

# Dockerfile 的 wrapper 脚本强制设置 HOME=/opt/agent/home，
# 因此 opencode 实际读取的配置路径均在此目录下
_HOME_DIR = "/opt/agent/home"
_AUTH_JSON_PATH = f"{_HOME_DIR}/.local/share/opencode/auth.json"
_CONFIG_JSON_PATH = f"{_HOME_DIR}/.config/opencode/opencode.json"

# 与 reference/opencode/prompt_extra.txt 保持一致，
# 在用户 prompt 前拼接，引导 opencode 直接执行任务而不询问
_PROMPT_EXTRA = """\
## 整体要求
- 准确理解下面的用户需求，如需要执行相关操作或调用相关工具，比如创建文件、修改文件、运行命令等，直接执行即可，不要提示，也不要询问

## 用户需求如下
"""
_PROMPT_EXTRA = ""

class OpenCodeConfig(BaseModel):
    """OpenCode 智能体配置。"""

    # ── 运行时配置 ──
    api_key: str
    model_name: str
    base_url: str = ""

    # ── Docker 镜像构建配置 ──
    base_image: str = "ubuntu:22.04"
    node_version: str = "22.11.0"
    agent_npm_pkg: str = "opencode-ai"
    agent_bin: str = "opencode"
    agent_home: str = "/opt/agent"


class OpenCodeDriver(BaseAgentDriver):
    """OpenCode 编码智能体 Driver。"""

    name = "opencode"
    cfg: OpenCodeConfig

    def __init__(self, cfg: OpenCodeConfig) -> None:
        super().__init__(cfg)

    def build_setup_commands(self) -> list[str]:
        """构建 OpenCode 配置写入命令。"""
        cfg = self.cfg

        auth_payload = json.dumps(
            {
                "custom": {
                    "type": "api",
                    "key": cfg.api_key,
                }
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n"

        opencode_cfg = {
            "$schema": "https://opencode.ai/config.json",
            "permission": "allow",
            "provider": {
                "custom": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "自定义",
                    "options": {
                        "baseURL": cfg.base_url,
                    },
                    "models": {
                        cfg.model_name: {
                            "name": cfg.model_name,
                        }
                    },
                }
            },
        }
        config_payload = json.dumps(opencode_cfg, indent=2, ensure_ascii=False) + "\n"

        return [
            self._write_file_cmd(_AUTH_JSON_PATH, auth_payload),
            self._write_file_cmd(_CONFIG_JSON_PATH, config_payload),
        ]

    def build_command(
        self,
        prompt: str,
        *,
        work_dir: str = "/app",
    ) -> tuple[str, dict]:
        """构建 OpenCode 执行命令和参数。

        使用 opencode run --model {provider}/{model} 的形式传入模型，
        prompt 直接作为位置参数传入。
        """
        cfg = self.cfg
        model_spec = f"custom/{cfg.model_name}"
        full_prompt = _PROMPT_EXTRA + prompt

        env = {
            "HOME": _HOME_DIR,
            "DISABLE_SEND_PV": "1",
        }

        cmd = f"opencode run --model {model_spec} {shlex.quote(full_prompt)}"

        return cmd, {"workdir": work_dir, "environment": env}

    def parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """解析 OpenCode 的执行结果。

        opencode 输出为纯文本，不含结构化 JSON，
        因此仅通过 returncode 判断成功与否，token 计数无法从输出中提取。
        """
        error = ""
        if result.returncode != 0:
            error = (
                f"opencode exited with non-zero code {result.returncode}.\n"
                f"stderr: {result.stderr}"
            )

        return {
            "execution_time": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "output": result.stdout or "",
            "error": error,
        }


__all__ = [
    "OpenCodeConfig",
    "OpenCodeDriver",
]
