"""Tests for otter.backend.docker module."""

import asyncio

import pytest
from pathlib import Path
from dataclasses import dataclass

from otter.backend.docker import DockerBackend, DockerResult


class TestDockerResult:
    """Test DockerResult dataclass."""

    def test_fields(self):
        r = DockerResult(stdout="out", stderr="err", returncode=0, timed_out=False)
        assert r.stdout == "out"
        assert r.stderr == "err"
        assert r.returncode == 0
        assert r.timed_out is False

    def test_timed_out_result(self):
        r = DockerResult(stdout="", stderr="timeout", returncode=-1, timed_out=True)
        assert r.timed_out is True
        assert r.returncode == -1


class TestDockerBackendParseSize:
    """Test DockerBackend._parse_size static method."""

    def test_bytes(self):
        assert DockerBackend._parse_size("1024b") == 1024

    def test_kilobytes(self):
        assert DockerBackend._parse_size("1k") == 1024

    def test_megabytes(self):
        assert DockerBackend._parse_size("128m") == 128 * 1024 ** 2

    def test_gigabytes(self):
        assert DockerBackend._parse_size("2g") == 2 * 1024 ** 3

    def test_terabytes(self):
        assert DockerBackend._parse_size("1t") == 1024 ** 4

    def test_plain_number(self):
        assert DockerBackend._parse_size("4096") == 4096

    def test_with_whitespace(self):
        assert DockerBackend._parse_size("  512m  ") == 512 * 1024 ** 2

    def test_uppercase_ignored(self):
        # _parse_size lowercases first
        assert DockerBackend._parse_size("1G") == 1024 ** 3


class TestDockerBackendInit:
    """Test DockerBackend.__init__ container params construction."""

    def test_minimal_init(self, mocker):
        mocker.patch("otter.backend.docker.get_docker_storage_device", return_value="/dev/sda")
        backend = DockerBackend(timeout=10)
        assert backend._timeout == 10
        assert backend._container_params["stdin_open"] is True
        assert backend._container_params["tty"] is True

    def test_with_network_mode(self, mocker):
        mocker.patch("otter.backend.docker.get_docker_storage_device", return_value="/dev/sda")
        backend = DockerBackend(timeout=10, network_mode="none")
        assert backend._container_params["network_mode"] == "none"

    def test_with_cpus(self, mocker):
        mocker.patch("otter.backend.docker.get_docker_storage_device", return_value="/dev/sda")
        backend = DockerBackend(timeout=10, cpus=2.0)
        assert backend._container_params["nano_cpus"] == int(2.0 * 1e9)

    def test_with_memory_limits(self, mocker):
        mocker.patch("otter.backend.docker.get_docker_storage_device", return_value="/dev/sda")
        backend = DockerBackend(
            timeout=10,
            memory="512m",
            memory_swap="1g",
            memory_reservation="256m",
        )
        assert backend._container_params["mem_limit"] == "512m"
        assert backend._container_params["memswap_limit"] == "1g"
        assert backend._container_params["mem_reservation"] == "256m"

    def test_with_device_bps(self, mocker):
        mocker.patch("otter.backend.docker.get_docker_storage_device", return_value="/dev/sda")
        backend = DockerBackend(
            timeout=10,
            device_read_bps="128m",
            device_write_bps="64m",
        )
        assert backend._container_params["device_read_bps"] == [
            {"Path": "/dev/sda", "Rate": 128 * 1024 ** 2}
        ]
        assert backend._container_params["device_write_bps"] == [
            {"Path": "/dev/sda", "Rate": 64 * 1024 ** 2}
        ]

    def test_no_device_bps_when_none(self, mocker):
        backend = DockerBackend(timeout=10)
        assert "device_read_bps" not in backend._container_params
        assert "device_write_bps" not in backend._container_params


class TestDockerBackendRun:
    """Test DockerBackend.run method with mocked docker utils."""

    def _make_backend(self):
        return DockerBackend(timeout=10)

    def _mock_docker_utils(self, mocker, exec_results=None):
        """Mock all docker_utils functions used by run()."""
        mocker.patch("otter.backend.docker.create_container", new_callable=mocker.AsyncMock)
        mocker.patch("otter.backend.docker.start_container", new_callable=mocker.AsyncMock)
        mocker.patch("otter.backend.docker.remove_container", new_callable=mocker.AsyncMock)
        mocker.patch("otter.backend.docker.copy_to_container", new_callable=mocker.AsyncMock)
        mocker.patch("otter.backend.docker.copy_from_container", new_callable=mocker.AsyncMock)

        @dataclass
        class ExecResult:
            stdout: str
            stderr: str
            returncode: int

        if exec_results is None:
            exec_results = [ExecResult(stdout="ok", stderr="", returncode=0)]

        mock_exec = mocker.AsyncMock(side_effect=exec_results)
        mocker.patch("otter.backend.docker.exec_container", mock_exec)
        return mock_exec

    async def test_successful_single_command(self, mocker):
        backend = self._make_backend()
        self._mock_docker_utils(mocker)

        result = await backend.run(image_tag="test:v1", commands=["echo hello"])
        assert result.returncode == 0
        assert result.stdout == "ok"
        assert result.timed_out is False

    async def test_empty_commands(self, mocker):
        backend = self._make_backend()
        self._mock_docker_utils(mocker)

        result = await backend.run(image_tag="test:v1", commands=[])
        assert result.returncode == 0
        assert result.timed_out is False

    async def test_command_failure_stops_execution(self, mocker):
        @dataclass
        class ExecResult:
            stdout: str
            stderr: str
            returncode: int

        backend = self._make_backend()
        self._mock_docker_utils(mocker, exec_results=[
            ExecResult(stdout="", stderr="error", returncode=1),
        ])

        result = await backend.run(image_tag="test:v1", commands=["bad_cmd", "never_run"])
        assert result.returncode == 1
        assert result.stderr == "error"
        assert result.timed_out is False

    async def test_copy_in_failure(self, mocker):
        backend = self._make_backend()
        self._mock_docker_utils(mocker)
        mocker.patch(
            "otter.backend.docker.copy_to_container",
            new_callable=mocker.AsyncMock,
            side_effect=Exception("copy failed"),
        )

        result = await backend.run(
            image_tag="test:v1",
            commands=["echo hi"],
            copy_in=[(Path("/local/file"), "/container/dir")],
        )
        assert result.returncode == -1
        assert "copy_in failed" in result.stderr

    async def test_container_removed_on_success(self, mocker):
        backend = self._make_backend()
        self._mock_docker_utils(mocker)
        mock_remove = mocker.patch(
            "otter.backend.docker.remove_container",
            new_callable=mocker.AsyncMock,
        )

        await backend.run(image_tag="test:v1", commands=["echo hi"])
        mock_remove.assert_called_once()

    async def test_container_removed_on_failure(self, mocker):
        backend = self._make_backend()
        self._mock_docker_utils(mocker)
        mocker.patch(
            "otter.backend.docker.create_container",
            new_callable=mocker.AsyncMock,
            side_effect=Exception("create failed"),
        )
        mock_remove = mocker.patch(
            "otter.backend.docker.remove_container",
            new_callable=mocker.AsyncMock,
        )

        result = await backend.run(image_tag="test:v1", commands=["echo hi"])
        assert result.returncode == -1
        mock_remove.assert_called_once()

    async def test_command_timeout(self, mocker):
        backend = self._make_backend()
        self._mock_docker_utils(mocker)
        mocker.patch(
            "otter.backend.docker.exec_container",
            new_callable=mocker.AsyncMock,
            side_effect=asyncio.TimeoutError(),
        )

        result = await backend.run(image_tag="test:v1", commands=["sleep 999"])
        assert result.timed_out is True
        assert result.returncode == -1
