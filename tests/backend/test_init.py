"""Tests for otter.backend.__init__ module (create_backend factory)."""

import pytest

from otter.backend import create_backend
from otter.backend.chat_llm import ChatLLMBackend
from otter.backend.docker import DockerBackend


class TestCreateBackend:
    """Test create_backend factory function."""

    def test_creates_chat_llm_backend(self, mocker):
        mocker.patch("otter.backend.chat_llm.AsyncOpenAI")
        settings = mocker.MagicMock()
        settings.api_key = "test-key"
        settings.base_url = "https://api.example.com"
        settings.model = "gpt-4"
        settings.max_retries = 3
        settings.retry_base_delay = 1.0

        backend = create_backend("chat_llm", settings)
        assert isinstance(backend, ChatLLMBackend)

    def test_creates_docker_backend(self, mocker):
        settings = mocker.MagicMock()
        settings.timeout = 10
        settings.cpus = None
        settings.memory = None
        settings.memory_swap = None
        settings.memory_reservation = None
        settings.network_mode = None
        settings.device_read_bps = None
        settings.device_write_bps = None

        backend = create_backend("docker", settings)
        assert isinstance(backend, DockerBackend)

    def test_unknown_type_raises(self, mocker):
        settings = mocker.MagicMock()
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("unknown_type", settings)
