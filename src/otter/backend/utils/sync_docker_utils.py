import re
import json
import docker
import platform
import subprocess
import tarfile
import tempfile

from pathlib import Path, PurePosixPath
from subprocess import CompletedProcess



_client = None


def _get_client():
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def is_docker_running() -> bool:
    """
    Check if the Docker daemon is running.

    Returns:
        True if the Docker daemon is running, False otherwise.
    """
    try:
        _get_client().ping()
        return True
    except Exception:
        return False


def get_docker_storage_device() -> str:
    """
    Detect the block device where Docker stores its data.

    Only supports Linux. Returns a path like "/dev/sda" or "/dev/nvme0n1".

    Raises:
        NotImplementedError: If the current platform is not Linux.
        RuntimeError:        If the storage device cannot be determined.
    """
    if platform.system().lower() != "linux":
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")

    # 1. Docker root directory
    try:
        docker_root = subprocess.check_output(
            ["docker", "info", "-f", "{{.DockerRootDir}}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        docker_root = "/var/lib/docker"

    if not Path(docker_root).exists():
        docker_root = "/"

    # 2. Partition backing that path
    try:
        partition = subprocess.check_output(
            ["df", "--output=source", docker_root], text=True,
        ).strip().splitlines()[-1]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Cannot identify Docker storage device: {e}") from e

    # 3. Resolve to parent block device
    try:
        parent = subprocess.check_output(
            ["lsblk", "-no", "pkname", partition], text=True,
        ).strip()
        if parent:
            return f"/dev/{parent}"
    except Exception:
        pass

    # Fallback: strip partition number from device path
    if "nvme" in partition:
        return re.sub(r"p\d+$", "", partition)
    return re.sub(r"\d+$", "", partition)


def read_image_tag_from_tar(tar_path: Path) -> str:
    with tarfile.open(tar_path, "r:gz") as tar:
        manifest_file = tar.extractfile("manifest.json")
        if not manifest_file:
            raise FileNotFoundError(f"manifest.json not found in {tar_path}")
        tag = json.load(manifest_file)[0]["RepoTags"][0]
    return tag


def sync_build_image(
        image_tag: str,
        dockerfile: Path | str,
        *,
        exist_ok: bool = False,
        extra_params: dict | None = None,
    ) -> None:
    """
    Build a Docker image from a Dockerfile.

    Args:
        image_tag:    The tag to assign to the built image, e.g. "myapp:1.0".
        dockerfile:   The Dockerfile to build from.
                      - Path: path to the Dockerfile on the local filesystem.
                      - str:  raw content of the Dockerfile.
        exist_ok:     If False (default), raise an error when an image with the
                      same tag already exists. If True, overwrite it silently.
        extra_params: Additional parameters passed directly to the Docker SDK.
                      Refer to docker.client.images.build() for supported options.

    Raises:
        ValueError:                  If the image already exists and exist_ok is False.
        FileNotFoundError:           If dockerfile is a Path and the file does not exist.
        docker.errors.BuildError:    If the build process fails.
    """
    # Step 1: Check if the image already exists
    try:
        _get_client().images.get(image_tag)
        if not exist_ok:
            raise ValueError(f"Image '{image_tag}' already exists.")
        return
    except docker.errors.ImageNotFound:
        pass

    # Step 2: Build the image from a Path or raw string content
    if isinstance(dockerfile, Path):
        if not dockerfile.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")
        _get_client().images.build(
            path=str(dockerfile.parent),
            dockerfile=str(dockerfile.name),
            tag=image_tag,
            **(extra_params or {}),
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dockerfile = Path(tmpdir) / "Dockerfile"
            tmp_dockerfile.write_text(dockerfile)
            _get_client().images.build(
                path=tmpdir,
                dockerfile="Dockerfile",
                tag=image_tag,
                **(extra_params or {}),
            )


def sync_remove_image(
        image_tag: str, 
        *, 
        missing_ok: bool = False
    ) -> None:
    """
    Remove a Docker image by its tag.

    Args:
        image_tag:  The tag of the image to remove, e.g. "myapp:1.0".
        missing_ok: If False (default), raise an error when the image does not
                    exist. If True, silently ignore it.

    Raises:
        ValueError:             If the image does not exist and missing_ok is False.
        docker.errors.APIError: If the image is currently in use by a container.
    """
    try:
        _get_client().images.remove(image=image_tag, force=False)
    except docker.errors.ImageNotFound:
        if not missing_ok:
            raise ValueError(f"Image '{image_tag}' not found.")


def sync_create_container(
        image_tag: str,
        container_name: str,
        *,
        exist_ok: bool = False,
        extra_params: dict | None = None,
    ) -> None:
    """
    Create a Docker container from an image.

    Args:
        image_tag:      The tag of the image to create the container from, e.g. "myapp:1.0".
        container_name: The name to assign to the container.
        exist_ok:       If False (default), raise an error when a container with the
                        same name already exists. If True, silently ignore it.
        extra_params:   Additional parameters passed directly to the Docker SDK.
                        Refer to docker.client.containers.create() for supported options.

    Raises:
        ValueError:             If the image does not exist, or if the container already
                                exists and exist_ok is False.
        docker.errors.APIError: If the container creation fails.
    """
    # Step 1: Check if the container already exists
    try:
        _get_client().containers.get(container_name)
        if not exist_ok:
            raise ValueError(f"Container '{container_name}' already exists.")
        return
    except docker.errors.NotFound:
        pass

    # Step 2: Check if the image exists
    try:
        _get_client().images.get(image_tag)
    except docker.errors.ImageNotFound:
        raise ValueError(f"Image '{image_tag}' not found.")

    # Step 3: Create the container
    _get_client().containers.create(
        image=image_tag,
        name=container_name,
        **(extra_params or {}),
    )


def sync_remove_container(
        container_name: str,
        *,
        missing_ok: bool = False,
        force: bool = False,
    ) -> None:
    """
    Remove a Docker container.

    Args:
        container_name: The name of the container to remove.
        missing_ok:     If False (default), raise an error when the container does not
                        exist. If True, silently ignore it.
        force:          If True, force remove the container even if it is running.
                        If False (default), raise an error if the container is running.

    Raises:
        ValueError:             If the container does not exist and missing_ok is False.
        docker.errors.APIError: If the container is running and force is False.
    """
    try:
        container = _get_client().containers.get(container_name)
        container.remove(force=force)
    except docker.errors.NotFound:
        if not missing_ok:
            raise ValueError(f"Container '{container_name}' not found.")


def sync_start_container(
        container_name: str
    ) -> None:
    """
    Start a created Docker container in the background.

    Args:
        container_name: The name of the container to start.

    Raises:
        ValueError:             If the container does not exist.
        docker.errors.APIError: If the container fails to start.
    """
    try:
        container = _get_client().containers.get(container_name)
        container.start()
    except docker.errors.NotFound:
        raise ValueError(f"Container '{container_name}' not found.")


def sync_stop_container(
        container_name: str,
        *,
        timeout: int = 10,
    ) -> None:
    """
    Stop a running Docker container.

    Args:
        container_name: The name of the container to stop.
        timeout:        Seconds to wait for the container to stop gracefully
                        before killing it. Defaults to 10.

    Raises:
        ValueError:             If the container does not exist.
        docker.errors.APIError: If the container fails to stop.
    """
    try:
        container = _get_client().containers.get(container_name)
        container.stop(timeout=timeout)
    except docker.errors.NotFound:
        raise ValueError(f"Container '{container_name}' not found.")


def sync_run_container(
        image_tag: str,
        command: str,
        *,
        extra_params: dict | None = None,
    ) -> CompletedProcess:
    """
    Run a one-off command in a new container, then remove it.

    Args:
        image_tag:    The tag of the image to run the container from, e.g. "myapp:1.0".
        command:      The command to run inside the container.
        extra_params: Additional parameters passed directly to the Docker SDK.
                      Refer to docker.client.containers.run() for supported options.

    Returns:
        CompletedProcess with stdout, stderr and returncode.

    Raises:
        ValueError:             If the image does not exist.
        docker.errors.APIError: If the container fails to run.
    """
    # Step 1: Check if the image exists
    try:
        _get_client().images.get(image_tag)
    except docker.errors.ImageNotFound:
        raise ValueError(f"Image '{image_tag}' not found.")

    # Step 2: Run the container and capture output
    container = _get_client().containers.run(
        image=image_tag,
        command=command,
        detach=True,
        **(extra_params or {}),
    )

    # Step 3: Wait for the container to finish
    result = container.wait()
    exit_code = result["StatusCode"]
    stdout = container.logs(stdout=True, stderr=False).decode()
    stderr = container.logs(stdout=False, stderr=True).decode()

    # Step 4: Remove the container
    container.remove()

    return CompletedProcess(
        args=command,
        returncode=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def sync_exec_container(
        container_name: str,
        command: str,
        *,
        extra_params: dict | None = None,
    ) -> CompletedProcess:
    """
    Execute a command in a running container.

    Args:
        container_name: The name of the container to execute the command in.
        command:        The command to execute inside the container.
        extra_params:   Additional parameters passed directly to the Docker SDK.
                        Refer to docker.client.containers.exec_run() for supported options.

    Returns:
        CompletedProcess with stdout, stderr and returncode.

    Raises:
        ValueError:             If the container does not exist or is not running.
        docker.errors.APIError: If the command fails to execute.
    """
    # Step 1: Check if the container exists and is running
    try:
        container = _get_client().containers.get(container_name)
    except docker.errors.NotFound:
        raise ValueError(f"Container '{container_name}' not found.")

    if container.status != "running":
        raise ValueError(f"Container '{container_name}' is not running.")

    # Step 2: Execute the command
    exit_code, output = container.exec_run(
        cmd=command,
        demux=True,
        **(extra_params or {}),
    )
    stdout, stderr = output
    return CompletedProcess(
        args=command,
        returncode=exit_code,
        stdout=(stdout or b"").decode(),
        stderr=(stderr or b"").decode(),
    )


def sync_copy_to_container(
        container_name: str,
        src: Path,
        dst: PurePosixPath | str,
        *,
        rename: str | None = None,
    ) -> None:
    # Step 1: Check if the container exists
    try:
        container = _get_client().containers.get(container_name)
    except docker.errors.NotFound:
        raise ValueError(f"Container '{container_name}' not found.")

    # Step 2: Check if the source exists
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")

    # Step 3: Pack the source into a tar archive and copy it into the container
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "archive.tar"
        with tarfile.open(tmp_path, "w") as tar:
            tar.add(src, arcname=rename or src.name)
        with open(tmp_path, "rb") as f:
            container.put_archive(str(dst), f.read())


def sync_copy_from_container(
        container_name: str,
        src: PurePosixPath,
        dst: Path,
        *,
        rename: str | None = None,
    ) -> None:
    # Step 1: Check if the container exists
    try:
        container = _get_client().containers.get(container_name)
    except docker.errors.NotFound:
        raise ValueError(f"Container '{container_name}' not found.")

    # Step 2: Check if the destination directory exists
    if not dst.exists():
        raise FileNotFoundError(f"Destination path not found: {dst}")

    # Step 3: Get the archive from the container and extract it locally
    bits, _ = container.get_archive(str(src))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "archive.tar"
        with open(tmp_path, "wb") as f:
            for chunk in bits:
                f.write(chunk)
        with tarfile.open(tmp_path, "r") as tar:
            if rename:
                for member in tar.getmembers():
                    slash_idx = member.name.find("/")
                    if slash_idx == -1:
                        # 单个文件，直接替换整个名字
                        member.name = rename
                    else:
                        # 目录，替换顶层名字，保留子路径
                        member.name = rename + member.name[slash_idx:]
            tar.extractall(dst, filter="data")


__all__ = [
    "is_docker_running",
    "get_docker_storage_device",
    "read_image_tag_from_tar",
    "sync_build_image",
    "sync_remove_image",
    "sync_create_container",
    "sync_remove_container",
    "sync_start_container",
    "sync_stop_container",
    "sync_run_container",
    "sync_exec_container",
    "sync_copy_to_container",
    "sync_copy_from_container",
]