"""AgentDriver 基类 — 所有编码智能体 Driver 的抽象接口。"""

from __future__ import annotations

import shlex
import subprocess
from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseAgentDriver(ABC):
    """编码智能体 Driver 的抽象基类。

    子类需实现：
    - name: 智能体标识名称
    - build_setup_commands: 构建配置写入命令
    - build_command: 构建执行命令和参数
    - parse_result: 解析执行结果
    """

    name: str = ""

    def __init__(self, cfg: BaseModel) -> None:
        self.cfg = cfg

    # ── 抽象方法 ──────────────────────────────────────────

    @abstractmethod
    def build_setup_commands(self) -> list[str]:
        """构建将配置文件写入容器的 shell 命令列表。

        每条命令会在容器启动后、主命令执行前依次运行。
        """

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        *,
        work_dir: str = "/app",
    ) -> tuple[str, dict]:
        """构建在容器内执行的命令和参数。

        Returns:
            (command, extra_params) 元组，其中：
            - command: 要执行的 shell 命令字符串
            - extra_params: 传给 exec_container 的额外参数，
              如 {"workdir": ..., "environment": {...}}
        """

    @abstractmethod
    def parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """解析执行结果，返回包含以下字段的字典：

        - execution_time: 执行耗时（秒）
        - input_tokens: 输入 token 数
        - output_tokens: 输出 token 数
        - output: agent 的文本回答（成功时）
        - error: 错误信息（失败时，成功时为空字符串）
        """

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _write_file_cmd(path: str, content: str) -> str:
        """生成一条 shell 命令：创建父目录并写入文件内容。"""
        parent_dir = path.rsplit("/", 1)[0]
        escaped = shlex.quote(content)
        return f"mkdir -p {parent_dir} && printf %s {escaped} > {path}"
