"""Tests for otter.backend.chat_llm module."""

import pytest
from pathlib import Path

from otter.backend.chat_llm import ChatLLMBackend


class TestChatLLMBackendInit:
    """Test ChatLLMBackend initialization."""

    def test_stores_config(self, mocker):
        mocker.patch("otter.backend.chat_llm.AsyncOpenAI")
        backend = ChatLLMBackend(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
            max_retries=3,
            retry_base_delay=1.0,
        )
        assert backend.model == "gpt-4"
        assert backend.max_retries == 3
        assert backend.retry_base_delay == 1.0

    def test_creates_async_client(self, mocker):
        mock_client_cls = mocker.patch("otter.backend.chat_llm.AsyncOpenAI")
        ChatLLMBackend(
            api_key="key",
            base_url="https://api.example.com",
            model="gpt-4",
            max_retries=1,
            retry_base_delay=0.1,
        )
        mock_client_cls.assert_called_once_with(
            api_key="key",
            base_url="https://api.example.com",
        )


class TestChatLLMBackendRun:
    """Test ChatLLMBackend.run method."""

    def _make_backend(self, mocker):
        mocker.patch("otter.backend.chat_llm.AsyncOpenAI")
        return ChatLLMBackend(
            api_key="key",
            base_url="https://api.example.com",
            model="gpt-4",
            max_retries=2,
            retry_base_delay=0.01,
        )

    async def test_successful_call(self, mocker, tmp_path):
        backend = self._make_backend(mocker)
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Hello world"
        backend.client.chat.completions.create = mocker.AsyncMock(
            return_value=mock_response
        )

        output_file = tmp_path / "response.txt"
        result = await backend.run(
            messages=[{"role": "user", "content": "Hi"}],
            output_file=output_file,
        )
        assert result.error == ""
        assert result.products == [output_file]
        assert output_file.read_text(encoding="utf-8") == "Hello world"
        assert len(result.retries) == 1
        assert result.retries[0].returncode == 0

    async def test_retries_on_failure(self, mocker, tmp_path):
        backend = self._make_backend(mocker)
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "OK"

        backend.client.chat.completions.create = mocker.AsyncMock(
            side_effect=[Exception("API error"), mock_response]
        )
        mocker.patch("otter.backend.chat_llm.asyncio.sleep", new_callable=mocker.AsyncMock)

        output_file = tmp_path / "response.txt"
        result = await backend.run(
            messages=[{"role": "user", "content": "Hi"}],
            output_file=output_file,
        )
        assert result.error == ""
        assert output_file.read_text(encoding="utf-8") == "OK"
        assert len(result.retries) == 2
        assert result.retries[0].returncode == 1
        assert result.retries[1].returncode == 0
        assert backend.client.chat.completions.create.call_count == 2

    async def test_returns_error_after_max_retries(self, mocker, tmp_path):
        backend = self._make_backend(mocker)
        backend.client.chat.completions.create = mocker.AsyncMock(
            side_effect=Exception("persistent error")
        )
        mocker.patch("otter.backend.chat_llm.asyncio.sleep", new_callable=mocker.AsyncMock)

        output_file = tmp_path / "response.txt"
        result = await backend.run(
            messages=[{"role": "user", "content": "Hi"}],
            output_file=output_file,
        )
        assert result.error != ""
        assert result.products == []
        assert len(result.retries) == 2
        assert all(r.returncode == 1 for r in result.retries)
