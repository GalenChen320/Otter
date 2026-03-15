import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4
from pathlib import Path


from otter.backend.utils.docker_utils import (
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
class DockerResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class DockerBackend:

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
    async def build_image(cls, image_tag: str, dockerfile: Path | str, *,
                          exist_ok: bool = True, extra_params: dict | None = None) -> None:
        """构建镜像。"""
        await build_image(image_tag, dockerfile, exist_ok=exist_ok, extra_params=extra_params)

    @classmethod
    async def remove_image(cls, image_tag: str, *, missing_ok: bool = True) -> None:
        """删除镜像。"""
        await remove_image(image_tag, missing_ok=missing_ok)

    async def run(
        self,
        image_tag: str,
        commands: list[str],
        *,
        copy_in: list[tuple[Path, str]] | None = None,
        copy_out: list[tuple[str, Path]] | None = None,
        timeout: int | None = None,
    ) -> DockerResult:
        """在容器中执行命令序列，返回 DockerResult。

        Args:
            image_tag: 使用的镜像
            commands:  按顺序执行的命令列表
            copy_in:   可选，复制进容器的文件列表 [(本地路径, 容器目标目录), ...]
            copy_out:  可选，从容器复制出的文件列表 [(容器路径, 本地目标目录), ...]
            timeout:   单条命令超时（秒），None 则使用构造时的默认值
        """
        timeout = timeout or self._timeout
        container_name = f"otter-{uuid4().hex[:8]}"

        try:
            await create_container(
                image_tag, container_name,
                extra_params=self._container_params,
            )
            await start_container(container_name)

            # 复制文件进容器
            for src, dst in (copy_in or []):
                try:
                    await copy_to_container(container_name, src, dst)
                except Exception as e:
                    logger.error("copy_in failed (%s -> %s): %s", src, dst, e)
                    return DockerResult(
                        stdout="",
                        stderr=f"copy_in failed ({src} -> {dst}): {e}",
                        returncode=-1,
                        timed_out=False,
                    )

            # 按顺序执行命令
            last_result = None
            for cmd in commands:
                try:
                    last_result = await asyncio.wait_for(
                        exec_container(container_name, cmd), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("command timed out after %ds: %s", timeout, cmd)
                    return DockerResult(
                        stdout="",
                        stderr=f"Command timed out after {timeout}s: {cmd}",
                        returncode=-1,
                        timed_out=True,
                    )
                if last_result.returncode != 0:
                    return DockerResult(
                        stdout=last_result.stdout,
                        stderr=last_result.stderr,
                        returncode=last_result.returncode,
                        timed_out=False,
                    )

            # 从容器复制文件出来
            for src, dst in (copy_out or []):
                try:
                    await copy_from_container(container_name, src, dst)
                except Exception as e:
                    logger.error("copy_out failed (%s -> %s): %s", src, dst, e)
                    return DockerResult(
                        stdout=last_result.stdout if last_result else "",
                        stderr=f"copy_out failed ({src} -> {dst}): {e}",
                        returncode=-1,
                        timed_out=False,
                    )

            return DockerResult(
                stdout=last_result.stdout if last_result else "",
                stderr=last_result.stderr if last_result else "",
                returncode=last_result.returncode if last_result else 0,
                timed_out=False,
            )

        except Exception as e:
            logger.error("container execution failed: %s", e)
            return DockerResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                timed_out=False,
            )

        finally:
            await remove_container(container_name, force=True, missing_ok=True)

__all__ = [
    "DockerResult",
    "DockerBackend",
]