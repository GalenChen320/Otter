import asyncio
from uuid import uuid4
from pathlib import Path

from otter.config.setting import get_settings
from otter.episode import Episode, EnvOutputManifest
from otter.environment.base import BaseEnvironment
from otter.environment.utils.docker_utils import (
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


class DockerEnvironment(BaseEnvironment):

    @staticmethod
    def _parse_size(value: str) -> int:
        """将人类可读的大小字符串转换为字节数，如 '128m' → 134217728。"""
        units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
        value = value.strip().lower()
        if value[-1] in units:
            return int(float(value[:-1]) * units[value[-1]])
        return int(value)

    def __init__(self):
        docker_cfg = get_settings().environment.docker
        self._timeout = docker_cfg.timeout
        self._container_params: dict = {
            "stdin_open": True,
            "tty": True,
        }
        if docker_cfg.network_mode is not None:
            self._container_params["network_mode"] = docker_cfg.network_mode
        if docker_cfg.cpus is not None:
            self._container_params["nano_cpus"] = int(docker_cfg.cpus * 1e9)
        if docker_cfg.memory is not None:
            self._container_params["mem_limit"] = docker_cfg.memory
        if docker_cfg.memory_swap is not None:
            self._container_params["memswap_limit"] = docker_cfg.memory_swap
        if docker_cfg.memory_reservation is not None:
            self._container_params["mem_reservation"] = docker_cfg.memory_reservation
        if docker_cfg.device_read_bps or docker_cfg.device_write_bps:
            device = get_docker_storage_device()
            if docker_cfg.device_read_bps:
                self._container_params["device_read_bps"] = [{"Path": device, "Rate": self._parse_size(docker_cfg.device_read_bps)}]
            if docker_cfg.device_write_bps:
                self._container_params["device_write_bps"] = [{"Path": device, "Rate": self._parse_size(docker_cfg.device_write_bps)}]

    @classmethod
    async def build_image(cls, image_tag: str, dockerfile: Path | str, *,
                          exist_ok: bool = True, extra_params: dict | None = None) -> None:
        """构建镜像。"""
        await build_image(image_tag, dockerfile, exist_ok=exist_ok, extra_params=extra_params)

    @classmethod
    async def remove_image(cls, image_tag: str, *, missing_ok: bool = True) -> None:
        """删除镜像。"""
        await remove_image(image_tag, missing_ok=missing_ok)

    async def _execute(self, episode: Episode) -> EnvOutputManifest:
        """从 env_input_manifest 读取执行规格，创建容器执行，返回 EnvOutputManifest。"""
        turn = episode.turns[-1]
        manifest = turn.env_input_manifest

        if not manifest:
            raise ValueError("DockerEnvironment requires EnvInputManifest")
        if not manifest.image_tag:
            raise ValueError("DockerEnvironment requires 'image_tag' in EnvInputManifest")

        timeout = manifest.timeout or self._timeout
        container_name = f"otter-{uuid4().hex[:8]}"
        env_output_dir = turn.env_output_path

        try:
            await create_container(
                manifest.image_tag, container_name,
                extra_params=self._container_params,
            )
            await start_container(container_name)

            # 注入脚本文件
            if manifest.script_file:
                await copy_to_container(container_name, manifest.script_file, "/tmp")

            # 按顺序执行命令
            last_result = None
            for cmd in (manifest.commands or []):
                try:
                    last_result = await asyncio.wait_for(
                        exec_container(container_name, cmd), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return self._make_env_output(env_output_dir, "", f"Command timed out after {timeout}s: {cmd}", -1, True)
                if last_result.returncode != 0:
                    return self._make_env_output(env_output_dir, last_result.stdout, last_result.stderr, last_result.returncode, False)

            # 成功
            return self._make_env_output(
                env_output_dir,
                last_result.stdout if last_result else "",
                last_result.stderr if last_result else "",
                last_result.returncode if last_result else 0,
                False,
            )

        except Exception as e:
            return self._make_env_output(env_output_dir, "", str(e), -1, False)

        finally:
            await remove_container(container_name, force=True, missing_ok=True)

    @staticmethod
    def _make_env_output(env_output_dir: Path, stdout: str, stderr: str, returncode: int, timed_out: bool) -> EnvOutputManifest:
        """写入 stdout/stderr 文件，返回 EnvOutputManifest。"""
        stdout_file = env_output_dir / "stdout.txt"
        stderr_file = env_output_dir / "stderr.txt"

        stdout_file.write_text(stdout, encoding="utf-8")
        stderr_file.write_text(stderr, encoding="utf-8")

        return EnvOutputManifest(
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            returncode=returncode,
            timed_out=timed_out,
        )
