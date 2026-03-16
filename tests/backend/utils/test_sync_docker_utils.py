"""Tests for otter.backend.utils.sync_docker_utils module.

All Docker SDK calls are mocked to avoid requiring a running Docker daemon.
"""

import pytest
import docker
import tarfile
import subprocess
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch, call

import otter.backend.utils.sync_docker_utils as mod


# ── Helper: reset module-level _client between tests ──

@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the module-level cached Docker client before each test."""
    mod._client = None
    yield
    mod._client = None


# ── _get_client ──

class TestGetClient:
    def test_returns_docker_client(self, mocker):
        mock_from_env = mocker.patch("otter.backend.utils.sync_docker_utils.docker.from_env")
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        result = mod._get_client()
        assert result is mock_client
        mock_from_env.assert_called_once()

    def test_caches_client_on_second_call(self, mocker):
        mock_from_env = mocker.patch("otter.backend.utils.sync_docker_utils.docker.from_env")
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        first = mod._get_client()
        second = mod._get_client()
        assert first is second
        mock_from_env.assert_called_once()

    def test_propagates_exception_when_docker_unavailable(self, mocker):
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.docker.from_env",
            side_effect=docker.errors.DockerException("not available"),
        )
        with pytest.raises(docker.errors.DockerException):
            mod._get_client()


# ── is_docker_running ──

class TestIsDockerRunning:
    def test_returns_true_when_ping_succeeds(self, mocker):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mocker.patch("otter.backend.utils.sync_docker_utils.docker.from_env", return_value=mock_client)

        assert mod.is_docker_running() is True

    def test_returns_false_when_ping_raises(self, mocker):
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("connection refused")
        mocker.patch("otter.backend.utils.sync_docker_utils.docker.from_env", return_value=mock_client)

        assert mod.is_docker_running() is False

    def test_returns_false_when_client_creation_fails(self, mocker):
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.docker.from_env",
            side_effect=docker.errors.DockerException("not available"),
        )
        assert mod.is_docker_running() is False


# ── get_docker_storage_device ──

class TestGetDockerStorageDevice:
    def test_raises_on_non_linux(self, mocker):
        mocker.patch("otter.backend.utils.sync_docker_utils.platform.system", return_value="Darwin")
        with pytest.raises(NotImplementedError, match="Unsupported platform"):
            mod.get_docker_storage_device()

    def test_returns_parent_device_via_lsblk(self, mocker):
        mocker.patch("otter.backend.utils.sync_docker_utils.platform.system", return_value="Linux")
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.subprocess.check_output",
            side_effect=[
                "/var/lib/docker\n",   # docker info
                "Filesystem\n/dev/sda1\n",  # df
                "sda\n",               # lsblk
            ],
        )
        mocker.patch("otter.backend.utils.sync_docker_utils.Path.exists", return_value=True)
        assert mod.get_docker_storage_device() == "/dev/sda"

    def test_fallback_strips_partition_number(self, mocker):
        mocker.patch("otter.backend.utils.sync_docker_utils.platform.system", return_value="Linux")
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.subprocess.check_output",
            side_effect=[
                "/var/lib/docker\n",
                "Filesystem\n/dev/sda1\n",
                "",  # lsblk returns empty
            ],
        )
        mocker.patch("otter.backend.utils.sync_docker_utils.Path.exists", return_value=True)
        assert mod.get_docker_storage_device() == "/dev/sda"

    def test_fallback_nvme_strips_partition(self, mocker):
        mocker.patch("otter.backend.utils.sync_docker_utils.platform.system", return_value="Linux")
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.subprocess.check_output",
            side_effect=[
                "/var/lib/docker\n",
                "Filesystem\n/dev/nvme0n1p1\n",
                "",  # lsblk returns empty
            ],
        )
        mocker.patch("otter.backend.utils.sync_docker_utils.Path.exists", return_value=True)
        assert mod.get_docker_storage_device() == "/dev/nvme0n1"

    def test_raises_when_df_fails(self, mocker):
        mocker.patch("otter.backend.utils.sync_docker_utils.platform.system", return_value="Linux")
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.subprocess.check_output",
            side_effect=[
                "/var/lib/docker\n",
                subprocess.CalledProcessError(1, "df"),
            ],
        )
        mocker.patch("otter.backend.utils.sync_docker_utils.Path.exists", return_value=True)
        with pytest.raises(RuntimeError, match="Cannot identify Docker storage device"):
            mod.get_docker_storage_device()

    def test_falls_back_to_root_when_docker_root_missing(self, mocker):
        mocker.patch("otter.backend.utils.sync_docker_utils.platform.system", return_value="Linux")
        # docker info returns a non-existent path
        call_count = [0]
        original_exists = Path.exists

        def mock_check_output(cmd, **kwargs):
            if "docker" in cmd:
                return "/nonexistent/docker\n"
            if "df" in cmd:
                return "Filesystem\n/dev/sda1\n"
            if "lsblk" in cmd:
                return "sda\n"
            return ""

        mocker.patch(
            "otter.backend.utils.sync_docker_utils.subprocess.check_output",
            side_effect=mock_check_output,
        )
        # Path("/nonexistent/docker").exists() returns False, Path("/").exists() returns True
        mocker.patch(
            "otter.backend.utils.sync_docker_utils.Path.exists",
            side_effect=[False, True],
        )
        # The function should fall back to "/" and still work
        result = mod.get_docker_storage_device()
        assert result == "/dev/sda"


# ── Shared fixture for Docker SDK client mock ──

@pytest.fixture
def mock_client(mocker):
    """Provide a mocked Docker client injected into the module."""
    client = MagicMock()
    mocker.patch("otter.backend.utils.sync_docker_utils.docker.from_env", return_value=client)
    return client


# ── sync_build_image ──

class TestSyncBuildImage:
    def test_build_from_path(self, mock_client, tmp_path):
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("nope")
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM alpine\n")

        mod.sync_build_image("test:1.0", dockerfile)

        mock_client.images.build.assert_called_once_with(
            path=str(tmp_path),
            dockerfile="Dockerfile",
            tag="test:1.0",
        )

    def test_build_from_string(self, mock_client):
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("nope")

        mod.sync_build_image("test:1.0", "FROM alpine\n")

        mock_client.images.build.assert_called_once()
        call_kwargs = mock_client.images.build.call_args[1]
        assert call_kwargs["tag"] == "test:1.0"
        assert call_kwargs["dockerfile"] == "Dockerfile"

    def test_raises_when_image_exists_and_not_exist_ok(self, mock_client):
        mock_client.images.get.return_value = MagicMock()

        with pytest.raises(ValueError, match="already exists"):
            mod.sync_build_image("test:1.0", "FROM alpine\n", exist_ok=False)

    def test_overwrites_when_exist_ok(self, mock_client):
        mock_client.images.get.return_value = MagicMock()

        mod.sync_build_image("test:1.0", "FROM alpine\n", exist_ok=True)
        mock_client.images.build.assert_called_once()

    def test_raises_file_not_found_for_missing_dockerfile(self, mock_client, tmp_path):
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("nope")
        missing = tmp_path / "NoSuchDockerfile"

        with pytest.raises(FileNotFoundError, match="not found"):
            mod.sync_build_image("test:1.0", missing)

    def test_passes_extra_params(self, mock_client, tmp_path):
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("nope")
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM alpine\n")

        mod.sync_build_image("test:1.0", dockerfile, extra_params={"nocache": True})

        call_kwargs = mock_client.images.build.call_args[1]
        assert call_kwargs["nocache"] is True


# ── sync_remove_image ──

class TestSyncRemoveImage:
    def test_removes_existing_image(self, mock_client):
        mod.sync_remove_image("test:1.0")
        mock_client.images.remove.assert_called_once_with(image="test:1.0", force=False)

    def test_raises_when_image_missing_and_not_missing_ok(self, mock_client):
        mock_client.images.remove.side_effect = docker.errors.ImageNotFound("nope")
        with pytest.raises(ValueError, match="not found"):
            mod.sync_remove_image("test:1.0", missing_ok=False)

    def test_silent_when_image_missing_and_missing_ok(self, mock_client):
        mock_client.images.remove.side_effect = docker.errors.ImageNotFound("nope")
        mod.sync_remove_image("test:1.0", missing_ok=True)  # should not raise


# ── sync_create_container ──

class TestSyncCreateContainer:
    def test_creates_container(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")
        mock_client.images.get.return_value = MagicMock()

        mod.sync_create_container("img:1.0", "my-container")

        mock_client.containers.create.assert_called_once_with(
            image="img:1.0", name="my-container",
        )

    def test_raises_when_container_exists_and_not_exist_ok(self, mock_client):
        mock_client.containers.get.return_value = MagicMock()

        with pytest.raises(ValueError, match="already exists"):
            mod.sync_create_container("img:1.0", "my-container", exist_ok=False)

    def test_returns_silently_when_container_exists_and_exist_ok(self, mock_client):
        mock_client.containers.get.return_value = MagicMock()

        mod.sync_create_container("img:1.0", "my-container", exist_ok=True)
        mock_client.containers.create.assert_not_called()

    def test_raises_when_image_not_found(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("nope")

        with pytest.raises(ValueError, match="Image.*not found"):
            mod.sync_create_container("missing:1.0", "my-container")

    def test_passes_extra_params(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")
        mock_client.images.get.return_value = MagicMock()

        mod.sync_create_container("img:1.0", "my-container", extra_params={"tty": True})

        call_kwargs = mock_client.containers.create.call_args[1]
        assert call_kwargs["tty"] is True


# ── sync_remove_container ──

class TestSyncRemoveContainer:
    def test_removes_existing_container(self, mock_client):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mod.sync_remove_container("my-container")
        mock_container.remove.assert_called_once_with(force=False)

    def test_force_removes_running_container(self, mock_client):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mod.sync_remove_container("my-container", force=True)
        mock_container.remove.assert_called_once_with(force=True)

    def test_raises_when_container_missing_and_not_missing_ok(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")

        with pytest.raises(ValueError, match="not found"):
            mod.sync_remove_container("my-container", missing_ok=False)

    def test_silent_when_container_missing_and_missing_ok(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")

        mod.sync_remove_container("my-container", missing_ok=True)  # should not raise


# ── sync_start_container ──

class TestSyncStartContainer:
    def test_starts_existing_container(self, mock_client):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mod.sync_start_container("my-container")
        mock_container.start.assert_called_once()

    def test_raises_when_container_not_found(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")

        with pytest.raises(ValueError, match="not found"):
            mod.sync_start_container("my-container")


# ── sync_stop_container ──

class TestSyncStopContainer:
    def test_stops_existing_container(self, mock_client):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mod.sync_stop_container("my-container")
        mock_container.stop.assert_called_once_with(timeout=10)

    def test_stops_with_custom_timeout(self, mock_client):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        mod.sync_stop_container("my-container", timeout=30)
        mock_container.stop.assert_called_once_with(timeout=30)

    def test_raises_when_container_not_found(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")

        with pytest.raises(ValueError, match="not found"):
            mod.sync_stop_container("my-container")


# ── sync_run_container ──

class TestSyncRunContainer:
    def test_runs_command_and_returns_completed_process(self, mock_client):
        mock_container = MagicMock()
        mock_client.images.get.return_value = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"hello\n", b""]

        result = mod.sync_run_container("img:1.0", "echo hello")

        assert result.returncode == 0
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.args == "echo hello"
        mock_container.remove.assert_called_once()

    def test_captures_nonzero_exit_code(self, mock_client):
        mock_container = MagicMock()
        mock_client.images.get.return_value = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.side_effect = [b"", b"error\n"]

        result = mod.sync_run_container("img:1.0", "false")

        assert result.returncode == 1
        assert result.stderr == "error\n"

    def test_raises_when_image_not_found(self, mock_client):
        mock_client.images.get.side_effect = docker.errors.ImageNotFound("nope")

        with pytest.raises(ValueError, match="Image.*not found"):
            mod.sync_run_container("missing:1.0", "echo hello")

    def test_passes_extra_params(self, mock_client):
        mock_container = MagicMock()
        mock_client.images.get.return_value = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"", b""]

        mod.sync_run_container("img:1.0", "cmd", extra_params={"network_mode": "none"})

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["network_mode"] == "none"


# ── sync_exec_container ──

class TestSyncExecContainer:
    def test_executes_command_in_running_container(self, mock_client):
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, (b"output\n", b""))
        mock_client.containers.get.return_value = mock_container

        result = mod.sync_exec_container("my-container", "echo output")

        assert result.returncode == 0
        assert result.stdout == "output\n"
        assert result.stderr == ""

    def test_raises_when_container_not_found(self, mock_client):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")

        with pytest.raises(ValueError, match="not found"):
            mod.sync_exec_container("my-container", "echo hello")

    def test_raises_when_container_not_running(self, mock_client):
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_client.containers.get.return_value = mock_container

        with pytest.raises(ValueError, match="not running"):
            mod.sync_exec_container("my-container", "echo hello")

    def test_handles_none_stdout_stderr(self, mock_client):
        """When exec_run returns (None, None) for stdout/stderr, should decode to empty strings."""
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, (None, None))
        mock_client.containers.get.return_value = mock_container

        result = mod.sync_exec_container("my-container", "true")

        assert result.stdout == ""
        assert result.stderr == ""

    def test_passes_extra_params(self, mock_client):
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, (b"", b""))
        mock_client.containers.get.return_value = mock_container

        mod.sync_exec_container("my-container", "cmd", extra_params={"user": "root"})

        call_kwargs = mock_container.exec_run.call_args[1]
        assert call_kwargs["user"] == "root"


# ── sync_copy_to_container ──

class TestSyncCopyToContainer:
    def test_copies_file_to_container(self, mock_client, tmp_path):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        mod.sync_copy_to_container("my-container", src_file, "/app")

        mock_container.put_archive.assert_called_once()
        call_args = mock_container.put_archive.call_args
        assert call_args[0][0] == "/app"

    def test_copies_directory_to_container(self, mock_client, tmp_path):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")

        mod.sync_copy_to_container("my-container", src_dir, "/app")

        mock_container.put_archive.assert_called_once()

    def test_raises_when_container_not_found(self, mock_client, tmp_path):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        with pytest.raises(ValueError, match="not found"):
            mod.sync_copy_to_container("my-container", src_file, "/app")

    def test_raises_when_source_not_found(self, mock_client, tmp_path):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="not found"):
            mod.sync_copy_to_container("my-container", missing, "/app")

    def test_accepts_pure_posix_path_as_dst(self, mock_client, tmp_path):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        mod.sync_copy_to_container("my-container", src_file, PurePosixPath("/app"))
        mock_container.put_archive.assert_called_once()


# ── sync_copy_from_container ──

class TestSyncCopyFromContainer:
    def test_copies_from_container(self, mock_client, tmp_path):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        # Create a real tar archive as mock return value
        src_file = tmp_path / "source.txt"
        src_file.write_text("from container")
        tar_path = tmp_path / "archive.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(src_file, arcname="source.txt")
        with open(tar_path, "rb") as f:
            tar_bytes = f.read()

        mock_container.get_archive.return_value = (iter([tar_bytes]), {})

        dst_dir = tmp_path / "output"
        dst_dir.mkdir()

        mod.sync_copy_from_container("my-container", "/app/source.txt", dst_dir)

        assert (dst_dir / "source.txt").exists()
        assert (dst_dir / "source.txt").read_text() == "from container"

    def test_raises_when_container_not_found(self, mock_client, tmp_path):
        mock_client.containers.get.side_effect = docker.errors.NotFound("nope")
        dst_dir = tmp_path / "output"
        dst_dir.mkdir()

        with pytest.raises(ValueError, match="not found"):
            mod.sync_copy_from_container("my-container", "/app/file.txt", dst_dir)

    def test_raises_when_destination_not_found(self, mock_client, tmp_path):
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        missing_dst = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="not found"):
            mod.sync_copy_from_container("my-container", "/app/file.txt", missing_dst)
