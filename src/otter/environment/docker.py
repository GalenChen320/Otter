import tempfile
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass, field

from otter.environment.base import ExecutionObservation
from otter.environment.utils.docker_utils import (
    create_container,
    start_container,
    remove_container,
    exec_container,
    copy_to_container,
    copy_from_container,
)


@dataclass
class DockerExecSpec:
    """DockerEnvironment 的执行规格。

    由 Dataset 构建，描述一次执行需要做什么。
    """
    files_in: list[tuple[str, str]]                     # [(文件内容, 容器内绝对路径), ...]
    commands: list[str]                                  # 按顺序执行的命令
    files_out: list[tuple[str, Path]] = field(default_factory=list)  # [(容器内路径, 本地路径), ...]


class DockerEnvironment:

    def __init__(self, image: str, timeout: int = 10):
        self._image = image
        self._timeout = timeout

    async def setup(self) -> None:
        """全局初始化：确保镜像就绪。"""
        pass

    async def teardown(self) -> None:
        """全局清理。"""
        pass

    async def execute(self, spec: DockerExecSpec) -> ExecutionObservation:
        """创建容器 → 注入文件 → 按顺序执行命令 → 导出文件 → 销毁容器。"""
        container_name = f"otter-{uuid4().hex[:8]}"
        try:
            await create_container(
                self._image, container_name,
                extra_params={"stdin_open": True, "tty": True},
            )
            await start_container(container_name)

            # 注入文件
            for content, container_dst in spec.files_in:
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_path = Path(tmpdir) / Path(container_dst).name
                    local_path.write_text(content, encoding="utf-8")
                    container_dir = str(Path(container_dst).parent)
                    await copy_to_container(container_name, local_path, container_dir)

            # 按顺序执行命令
            last_result = None
            for cmd in spec.commands:
                last_result = await exec_container(container_name, cmd)
                if last_result.returncode != 0:
                    return ExecutionObservation(
                        stdout=last_result.stdout,
                        stderr=last_result.stderr,
                        returncode=last_result.returncode,
                    )

            # 导出文件
            for container_src, local_dst in spec.files_out:
                await copy_from_container(container_name, container_src, local_dst)

            return ExecutionObservation(
                stdout=last_result.stdout if last_result else "",
                stderr=last_result.stderr if last_result else "",
                returncode=last_result.returncode if last_result else 0,
            )

        except Exception as e:
            return ExecutionObservation(
                stdout="",
                stderr=str(e),
                returncode=-1,
            )

        finally:
            await remove_container(container_name, force=True, missing_ok=True)
