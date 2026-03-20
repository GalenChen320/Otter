import docker
import asyncio
import logging
from dataclasses import dataclass, field
from uuid import uuid4
from pathlib import Path

from otter.manifest import Result, OutputManifest, DebugInfo
from otter.backend.utils.docker_utils import (
    read_image_tag_from_tar,
    get_docker_storage_device,
    build_image,
    remove_image,
    create_container,
    start_container,
    remove_container,
    exec_container,
    copy_to_container,
    copy_from_container,
)

logger = logging.getLogger(__name__)


@dataclass
class DockerDebugInfo(DebugInfo):
    copy_in: list[Result] = field(default_factory=list)
    commands: list[Result] = field(default_factory=list)
    copy_out: list[Result] = field(default_factory=list)


class DockerBackend:
    backend_type = "docker"

    @staticmethod
    def _parse_size(value: str) -> int:
        """将人类可读的大小字符串转换为字节数，如 '128m' → 134217728。"""
        units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
        value = value.strip().lower()
        if value[-1] in units:
            return int(float(value[:-1]) * units[value[-1]])
        return int(value)

    def __init__(
            self, 
            timeout: int,
            network_mode: str | None = None,
            cpus: float | None = None,
            memory: str | None = None,
            memory_swap: str | None = None,
            memory_reservation: str | None = None,
            device_read_bps: str | None = None,
            device_write_bps: str | None = None,
            ):

        self._timeout = timeout
        self._container_params: dict = {
            "stdin_open": True,
            "tty": True,
        }
        if network_mode is not None:
            self._container_params["network_mode"] = network_mode
        if cpus is not None:
            self._container_params["nano_cpus"] = int(float(cpus) * 1e9)
        if memory is not None:
            self._container_params["mem_limit"] = memory
        if memory_swap is not None:
            self._container_params["memswap_limit"] = memory_swap
        if memory_reservation is not None:
            self._container_params["mem_reservation"] = memory_reservation
        if device_read_bps or device_write_bps:
            device = get_docker_storage_device()
            if device_read_bps:
                self._container_params["device_read_bps"] = [{"Path": device, "Rate": self._parse_size(device_read_bps)}]
            if device_write_bps:
                self._container_params["device_write_bps"] = [{"Path": device, "Rate": self._parse_size(device_write_bps)}]

    @classmethod
    def has_image(cls, image_tag: str) -> bool:
        """检查镜像是否存在。"""
        client = docker.from_env()
        try:
            client.images.get(image_tag)
            return True
        except docker.errors.ImageNotFound:
            return False

    @classmethod
    async def build_image(cls, image_tag: str, dockerfile: Path | str, *,
                          exist_ok: bool = True, extra_params: dict | None = None) -> None:
        """构建镜像。"""
        await build_image(image_tag, dockerfile, exist_ok=exist_ok, extra_params=extra_params)

    @classmethod
    async def load_image(cls, tar_path: Path, *, exist_ok: bool = True) -> str:
        """从 tar 文件加载镜像，返回 image_tag。"""
        if not tar_path.is_file():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        image_tag = read_image_tag_from_tar(tar_path)
        if cls.has_image(image_tag):
            if exist_ok:
                return image_tag
            raise FileExistsError(f"Image already exists: {image_tag}")
        client = docker.from_env()
        with open(tar_path, "rb") as f:
            await asyncio.to_thread(client.images.load, f)
        return image_tag

    @classmethod
    async def remove_image(cls, image_tag: str, *, missing_ok: bool = True) -> None:
        """删除镜像。"""
        await remove_image(image_tag, missing_ok=missing_ok)

    async def run(
        self,
        image_tag: str,
        commands: list[str | tuple[str, dict]],
        *,
        copy_in: list[tuple[Path, str] | tuple[Path, str, str]] | None = None,
        copy_out: list[tuple[str, Path] | tuple[str, Path, str]] | None = None,
        timeout: int | None = None,
    ) -> OutputManifest:
        """在容器中执行命令序列，返回 OutputManifest。

        Args:
            image_tag: 使用的镜像
            commands:  按顺序执行的命令列表
            copy_in:   可选，复制进容器的文件列表 [(本地路径, 容器目标目录, 可选重命名), ...]
            copy_out:  可选，从容器复制出的文件列表 [(容器路径, 本地目标目录, 可选重命名), ...]
            timeout:   单条命令超时（秒），None 则使用构造时的默认值
        """
        timeout = timeout or self._timeout
        container_name = f"otter-{uuid4().hex[:8]}"
        # output = OutputManifest()
        products = []
        debug = DockerDebugInfo()

        try:
            await create_container(
                image_tag, container_name,
                extra_params=self._container_params,
            )
            await start_container(container_name)

            # 复制文件进容器
            for item in (copy_in or []):
                src, dst, *rest = item
                rename = rest[0] if rest else None
                try:
                    await copy_to_container(container_name, src, dst, rename=rename)
                    debug.copy_in.append(
                        Result(stdout="", stderr="", returncode=0, timed_out=False)
                        )
                except Exception as e:
                    logger.error("copy_in failed (%s -> %s): %s", src, dst, e)
                    debug.copy_in.append(
                        Result(
                            stdout="",
                            stderr=f"copy_in failed ({src} -> {dst}): {e}",
                            returncode=-1,
                            timed_out=False,
                        )
                    )

            # 按顺序执行命令
            for item in commands:
                cmd, params = (item, None) if isinstance(item, str) else item
                try:
                    result = await asyncio.wait_for(
                        exec_container(container_name, cmd, extra_params=params),
                        timeout=timeout,
                    )
                    debug.commands.append(
                        Result(
                            stdout=result.stdout,
                            stderr=result.stderr,
                            returncode=result.returncode,
                            timed_out=False,
                        )
                    )
                except asyncio.TimeoutError:
                    logger.warning("command timed out after %ds: %s", timeout, cmd)
                    debug.commands.append(
                        Result(
                            stdout="",
                            stderr=f"Command timed out after {timeout}s: {cmd}",
                            returncode=-1,
                            timed_out=True,
                        )
                    )

            # 从容器复制文件出来
            products: list[Path | None] = []
            for item in (copy_out or []):
                src, dst, *rest = item
                rename = rest[0] if rest else None
                try:
                    await copy_from_container(container_name, src, dst, rename=rename)
                    product_name = rename if rename else Path(src).name
                    products.append(dst / product_name)
                    debug.copy_out.append(
                        Result(stdout="", stderr="", returncode=0, timed_out=False)
                        )
                except Exception as e:
                    logger.error("copy_out failed (%s -> %s): %s", src, dst, e)
                    products.append(None)
                    debug.copy_out.append(
                        Result(
                            stdout="",
                            stderr=f"copy_out failed ({src} -> {dst}): {e}",
                            returncode=-1,
                            timed_out=False,
                        )
                    )

            return OutputManifest(
                backend_type=self.backend_type,
                products=products, 
                debug_info=debug
            )

        except Exception as e:
            logger.error("container execution failed: %s", e)
            OutputManifest(
                backend_type=self.backend_type,
                products=products, 
                debug_info=debug, 
                unexpected=str(e)
            )

        finally:
            await remove_container(container_name, force=True, missing_ok=True)


__all__ = [
    "DockerDebugInfo",
    "DockerBackend",
]