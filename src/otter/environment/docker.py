import tempfile
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass, field

from otter.environment.base import BaseExecSpec, BaseEnvironment, ExecutionObservation
from otter.environment.utils.docker_utils import (
    build_image,
    remove_image,
    create_container,
    start_container,
    remove_container,
    exec_container,
    copy_to_container,
    copy_from_container,
)


@dataclass
class DockerExecSpec(BaseExecSpec):
    """DockerEnvironment 的执行规格。

    由 Dataset 构建，描述一次执行需要做什么。
    """
    image: str = ""                                      # 镜像 tag
    files_in: list[tuple[str, str]] = field(default_factory=list)   # [(文件内容, 容器内绝对路径), ...]
    commands: list[str] = field(default_factory=list)    # 按顺序执行的命令
    files_out: list[tuple[str, Path]] = field(default_factory=list) # [(容器内路径, 本地路径), ...]
    timeout: int = 10


class DockerEnvironment(BaseEnvironment):

    @classmethod
    async def build_image(cls, image_tag: str, dockerfile: Path | str, *, exist_ok: bool = True) -> None:
        """构建镜像。"""
        await build_image(image_tag, dockerfile, exist_ok=exist_ok)

    @classmethod
    async def remove_image(cls, image_tag: str, *, missing_ok: bool = True) -> None:
        """删除镜像。"""
        await remove_image(image_tag, missing_ok=missing_ok)

    @classmethod
    async def execute(cls, spec: DockerExecSpec) -> ExecutionObservation:
        """创建容器 → 注入文件 → 按顺序执行命令 → 导出文件 → 销毁容器。"""
        container_name = f"otter-{uuid4().hex[:8]}"
        try:
            await create_container(
                spec.image, container_name,
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
