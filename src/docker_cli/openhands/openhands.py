"""
OpenHandsDriver — OpenHands 编码智能体的 Driver 实现。

OpenHands 通过 Dockerfile 安装（openhands-ai），安装在 /opt/openhands-venv 虚拟环境中。
配置通过环境变量传入（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL），
直接在容器内激活 venv 后执行 python -m openhands.core.main。

参考：https://github.com/OpenHands/OpenHands
"""

import shlex

from pydantic import BaseModel

from docker_cli.base import BaseAgentDriver


# OpenHands 运行时所需的固定环境变量（对应 start.sh 中的 export 配置）
_OPENHANDS_ENV = {
    "AGENT_ENABLE_PROMPT_EXTENSIONS": "false",
    "AGENT_ENABLE_BROWSING": "false",
    "AGENT_ENABLE_JUPYTER": "false",
    "AGENT_ENABLE_THINK": "false",
    "LLM_FORCE_STRING_SERIALIZER": "true",
    "ENABLE_BROWSER": "false",
    "SANDBOX_ENABLE_AUTO_LINT": "true",
    "SANDBOX_INITIALIZE_PLUGINS": "false",
    "SKIP_DEPENDENCY_CHECK": "1",
    "RUN_AS_OPENHANDS": "false",
    "RUNTIME": "local",
    "LLM_MAX_INPUT_TOKENS": "1000000",
    "LLM_MAX_MESSAGE_CHARS": "1000000",
    "LLM_TEMPERATURE": "0.2",
    "LLM_NUM_RETRIES": "3",
    "LLM_DISABLE_VISION": "true",
    "MAX_ITERATIONS": "150",
    "LITELLM_DROP_PARAMS": "true",
    "LITELLM_SET_VERBOSE": "true",
}


class OpenHandsConfig(BaseModel):
    """OpenHands 智能体配置。"""

    # ── 运行时配置 ──
    api_key: str
    model_name: str
    base_url: str = ""

    # ── Docker 镜像构建配置 ──
    base_image: str = "ubuntu:22.04"
    openhands_version: str = "latest"


class OpenHandsDriver(BaseAgentDriver):
    """OpenHands 编码智能体 Driver。"""

    name = "openhands"
    cfg: OpenHandsConfig

    def __init__(self, cfg: OpenHandsConfig) -> None:
        super().__init__(cfg)

    def build_setup_commands(self) -> list[str]:
        """OpenHands 通过环境变量传递配置，无需写入配置文件。"""
        return []

    def build_command(
        self,
        prompt: str,
        *,
        work_dir: str = "/app",
    ) -> tuple[str, dict]:
        """构建 OpenHands 执行命令和参数。

        激活 /opt/openhands-venv 虚拟环境后执行 python -m openhands.core.main。
        LLM 配置通过环境变量传入，prompt 通过 --task 参数传入。
        """
        cfg = self.cfg

        env = dict(_OPENHANDS_ENV)
        env["LLM_API_KEY"] = cfg.api_key
        env["LLM_BASE_URL"] = cfg.base_url
        env["LLM_MODEL"] = f"hosted_vllm/{cfg.model_name}"
        env["SANDBOX_VOLUMES"] = f"{work_dir}:/workspace:rw"

        cmd = (
            "/opt/openhands-venv/bin/python -m openhands.core.main "
            f"--task {shlex.quote(prompt)}"
        )

        return cmd, {"workdir": work_dir, "environment": env}

    def parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """解析 OpenHands 执行结果。

        OpenHands 输出为纯文本日志，不提供结构化 JSON，
        因此只检查退出码，token 用量无法从输出中提取。
        """
        error = ""
        if result.returncode != 0:
            error = (
                f"openhands exited with non-zero code {result.returncode}.\n"
                f"stderr: {result.stderr}"
            )

        return {
            "execution_time": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "output": result.stdout or "",
            "error": error,
        }


__all__ = [
    "OpenHandsConfig",
    "OpenHandsDriver",
]
