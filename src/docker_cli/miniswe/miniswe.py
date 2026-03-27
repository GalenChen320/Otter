"""
MiniSWEDriver — Mini-SWE-Agent 编码智能体的 Driver 实现。

Mini-SWE-Agent v2 通过 pip 安装（mini-swe-agent），实际可执行命令为 `mini`。
配置通过 build_setup_commands 写入容器内的 .env 和 config.yaml 文件。

模型名格式为 protocol/model_name（如 openai/Qwen3-Coder-Plus），
API Key 根据 protocol 映射到对应的环境变量（OPENAI_API_KEY / ANTHROPIC_API_KEY 等）。
"""

import shlex
import subprocess

from pydantic import BaseModel

from docker_cli.base import BaseAgentDriver

# mini-swe-agent v2 配置文件路径（容器内）
_MSWEA_CONFIG_DIR = "/opt/agent/config"
_MSWEA_ENV_FILE = f"{_MSWEA_CONFIG_DIR}/.env"
_MSWEA_CONFIG_YAML = f"{_MSWEA_CONFIG_DIR}/config.yaml"

# protocol → API Key 环境变量名的映射
_PROTOCOL_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# protocol → Base URL 环境变量名的映射
_PROTOCOL_BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
}


def _build_full_model_name(protocol: str, model_name: str) -> str:
    """将 model_name 拼接为 mini-swe-agent v2 要求的 protocol/model 格式。

    若 model_name 已经包含 '/' 前缀（如已是 openai/xxx），则直接返回。
    """
    if "/" in model_name:
        return model_name
    return f"{protocol}/{model_name}"


class MiniSWEConfig(BaseModel):
    """Mini-SWE-Agent 智能体配置。"""

    api_key: str
    model_name: str
    base_url: str = ""
    protocol: str = "openai"
    cost_limit: str = "12.00"
    step_limit: int = 100
    timeout: int = 30


class MiniSWEDriver(BaseAgentDriver):
    """Mini-SWE-Agent 编码智能体 Driver。"""

    name = "mini_swe"
    cfg: MiniSWEConfig

    def __init__(self, cfg: MiniSWEConfig) -> None:
        super().__init__(cfg)

    def build_setup_commands(self) -> list[str]:
        """构建 mini-swe-agent 配置写入命令。

        写入 .env 和 config.yaml 两个文件，跳过交互式配置向导。
        """
        cfg = self.cfg
        protocol = cfg.protocol
        full_model_name = _build_full_model_name(protocol, cfg.model_name)
        api_key_env_name = _PROTOCOL_API_KEY_ENV.get(protocol, "OPENAI_API_KEY")

        # .env 文件
        env_content = (
            "# mini-swe-agent environment configuration (auto-generated)\n"
            "MSWEA_CONFIGURED=true\n"
            f"MSWEA_MODEL_NAME={full_model_name}\n"
            f"{api_key_env_name}={cfg.api_key}\n"
            "SWE_AGENT_ENV=eval\n"
        )

        # config.yaml 文件
        config_yaml_content = (
            "# mini-SWE-agent configuration file (v2 format, auto-generated)\n"
            "model:\n"
            f'  model_name: "{cfg.model_name}"\n'
            "\n"
            "agent:\n"
            f"  step_limit: {cfg.step_limit}\n"
            f"  cost_limit: {cfg.cost_limit}\n"
            "\n"
            "environment:\n"
            f"  timeout: {cfg.timeout}\n"
            "\n"
            "telemetry:\n"
            "  enabled: false\n"
        )

        return [
            self._write_file_cmd(_MSWEA_ENV_FILE, env_content),
            self._write_file_cmd(_MSWEA_CONFIG_YAML, config_yaml_content),
        ]

    def build_command(
        self,
        prompt: str,
        *,
        work_dir: str = "/app",
    ) -> tuple[str, dict]:
        """构建 mini-swe-agent 执行命令和参数。

        使用 mini -y --exit-immediately -m {protocol/model} --cost-limit {limit} -t {prompt}。
        """
        cfg = self.cfg
        protocol = cfg.protocol
        full_model_name = _build_full_model_name(protocol, cfg.model_name)
        api_key_env_name = _PROTOCOL_API_KEY_ENV.get(protocol, "OPENAI_API_KEY")
        base_url_env_name = _PROTOCOL_BASE_URL_ENV.get(protocol)

        env = {
            "MSWEA_GLOBAL_CONFIG_DIR": _MSWEA_CONFIG_DIR,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_MODEL_NAME": full_model_name,
            "MSWEA_COST_TRACKING": "ignore_errors",
            "TELEMETRY_ENABLED": "0",
            api_key_env_name: cfg.api_key,
        }

        if base_url_env_name and cfg.base_url:
            env[base_url_env_name] = cfg.base_url

        cmd = (
            f"mini -y --exit-immediately "
            f"-m {full_model_name} "
            f"--cost-limit {cfg.cost_limit} "
            f"-t {shlex.quote(prompt)}"
        )

        return cmd, {"workdir": work_dir, "environment": env}

    def parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """解析 Mini-SWE-Agent 的执行结果。

        mini v2 输出为纯文本日志，不提供结构化 JSON，
        因此仅通过 returncode 判断成功与否，token 计数无法从输出中提取。
        """
        error = ""
        if result.returncode != 0:
            error = (
                f"mini-swe-agent exited with non-zero code {result.returncode}.\n"
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
    "MiniSWEConfig",
    "MiniSWEDriver",
]