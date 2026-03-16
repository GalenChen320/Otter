"""Tests for otter.config.backend_settings module."""

import pytest

from otter.config.backend_settings import (
    BackendSettings,
    ChatLLMSettings,
    DockerSettings,
    BACKEND_SETTINGS_REGISTRY,
)


# ── Registry ──

class TestBackendSettingsRegistry:
    def test_registry_contains_chat_llm(self):
        assert "chat_llm" in BACKEND_SETTINGS_REGISTRY

    def test_registry_contains_docker(self):
        assert "docker" in BACKEND_SETTINGS_REGISTRY

    def test_registry_chat_llm_maps_to_correct_class(self):
        assert BACKEND_SETTINGS_REGISTRY["chat_llm"] is ChatLLMSettings

    def test_registry_docker_maps_to_correct_class(self):
        assert BACKEND_SETTINGS_REGISTRY["docker"] is DockerSettings

    def test_all_registry_values_are_backend_settings_subclasses(self):
        for name, cls in BACKEND_SETTINGS_REGISTRY.items():
            assert issubclass(cls, BackendSettings), f"{name} is not a BackendSettings subclass"


# ── DockerSettings ──

class TestDockerSettingsDefaults:
    """Test that DockerSettings has correct default values."""

    def test_defaults(self, tmp_path, monkeypatch):
        """All defaults should match when no env vars are set."""
        env_file = tmp_path / ".env"
        env_file.write_text("")
        monkeypatch.delenv("DOCKER__cpus", raising=False)

        s = DockerSettings(_env_file=env_file, _env_prefix="DOCKER__")
        assert s.cpus == 1.0
        assert s.memory == "512m"
        assert s.memory_swap == "512m"
        assert s.memory_reservation == "256m"
        assert s.network_mode == "none"
        assert s.device_read_bps == "128m"
        assert s.device_write_bps == "128m"
        assert s.timeout == 10


class TestDockerSettingsFromEnv:
    """Test DockerSettings reads from env file with prefix."""

    def test_reads_with_prefix(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EVALUATOR__cpus=2.0\n"
            "EVALUATOR__memory=1g\n"
            "EVALUATOR__timeout=30\n"
        )
        s = DockerSettings(_env_file=env_file, _env_prefix="EVALUATOR__")
        assert s.cpus == 2.0
        assert s.memory == "1g"
        assert s.timeout == 30

    def test_different_prefixes_are_isolated(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EVALUATOR__timeout=30\n"
            "PROPOSER__timeout=60\n"
        )
        eval_s = DockerSettings(_env_file=env_file, _env_prefix="EVALUATOR__")
        prop_s = DockerSettings(_env_file=env_file, _env_prefix="PROPOSER__")
        assert eval_s.timeout == 30
        assert prop_s.timeout == 60


class TestDockerSettingsEmptyStringCoercion:
    """Test that empty strings are coerced to None for Optional fields."""

    def test_empty_cpus_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__cpus=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.cpus is None

    def test_empty_memory_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__memory=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.memory is None

    def test_empty_network_mode_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__network_mode=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.network_mode is None

    def test_whitespace_only_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__memory=   \n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.memory is None

    def test_empty_device_read_bps_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__device_read_bps=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.device_read_bps is None

    def test_empty_device_write_bps_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__device_write_bps=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.device_write_bps is None

    def test_empty_memory_swap_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__memory_swap=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.memory_swap is None

    def test_empty_memory_reservation_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST__memory_reservation=\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.memory_reservation is None


class TestDockerSettingsTracked:
    """Verify tracked/untracked annotations on DockerSettings fields."""

    def test_all_docker_fields_are_tracked(self):
        for name, field_info in DockerSettings.model_fields.items():
            extra = field_info.json_schema_extra
            assert isinstance(extra, dict), f"{name} missing json_schema_extra"
            assert extra.get("core") is True, f"{name} should be tracked"


# ── ChatLLMSettings ──

class TestChatLLMSettingsFromEnv:
    """Test ChatLLMSettings reads required and optional fields from env."""

    def test_reads_all_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EXECUTOR__api_key=sk-test123\n"
            "EXECUTOR__base_url=https://api.example.com\n"
            "EXECUTOR__model=gpt-4o\n"
            "EXECUTOR__max_retries=5\n"
            "EXECUTOR__retry_base_delay=2.5\n"
        )
        s = ChatLLMSettings(_env_file=env_file, _env_prefix="EXECUTOR__")
        assert s.api_key == "sk-test123"
        assert s.base_url == "https://api.example.com"
        assert s.model == "gpt-4o"
        assert s.max_retries == 5
        assert s.retry_base_delay == 2.5

    def test_defaults_for_optional_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST__api_key=sk-xxx\n"
            "TEST__base_url=https://api.example.com\n"
            "TEST__model=test-model\n"
        )
        s = ChatLLMSettings(_env_file=env_file, _env_prefix="TEST__")
        assert s.max_retries == 3
        assert s.retry_base_delay == 1.0


class TestChatLLMSettingsValidation:
    """Test validation constraints on ChatLLMSettings."""

    def test_max_retries_ge_1(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST__api_key=sk-xxx\n"
            "TEST__base_url=https://api.example.com\n"
            "TEST__model=test\n"
            "TEST__max_retries=0\n"
        )
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="TEST__")

    def test_missing_required_api_key_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST__base_url=https://api.example.com\n"
            "TEST__model=test\n"
        )
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="TEST__")

    def test_missing_required_base_url_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST__api_key=sk-xxx\n"
            "TEST__model=test\n"
        )
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="TEST__")

    def test_missing_required_model_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST__api_key=sk-xxx\n"
            "TEST__base_url=https://api.example.com\n"
        )
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="TEST__")


class TestChatLLMSettingsTracked:
    """Verify tracked/untracked annotations on ChatLLMSettings fields."""

    def test_api_key_is_untracked(self):
        extra = ChatLLMSettings.model_fields["api_key"].json_schema_extra
        assert extra["core"] is False

    def test_base_url_is_tracked(self):
        extra = ChatLLMSettings.model_fields["base_url"].json_schema_extra
        assert extra["core"] is True

    def test_model_is_tracked(self):
        extra = ChatLLMSettings.model_fields["model"].json_schema_extra
        assert extra["core"] is True

    def test_max_retries_is_tracked(self):
        extra = ChatLLMSettings.model_fields["max_retries"].json_schema_extra
        assert extra["core"] is True

    def test_retry_base_delay_is_tracked(self):
        extra = ChatLLMSettings.model_fields["retry_base_delay"].json_schema_extra
        assert extra["core"] is True


class TestChatLLMSettingsValidationBoundary:
    """Boundary and edge-case validation for ChatLLMSettings."""

    def _make_env(self, tmp_path, **overrides):
        defaults = {
            "api_key": "sk-xxx",
            "base_url": "https://api.example.com",
            "model": "test",
        }
        defaults.update(overrides)
        env_file = tmp_path / ".env"
        lines = [f"T__{k}={v}\n" for k, v in defaults.items()]
        env_file.write_text("".join(lines))
        return env_file

    def test_max_retries_negative_raises(self, tmp_path):
        env_file = self._make_env(tmp_path, max_retries="-1")
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="T__")

    def test_max_retries_exactly_1_ok(self, tmp_path):
        env_file = self._make_env(tmp_path, max_retries="1")
        s = ChatLLMSettings(_env_file=env_file, _env_prefix="T__")
        assert s.max_retries == 1

    def test_max_retries_large_value(self, tmp_path):
        env_file = self._make_env(tmp_path, max_retries="1000")
        s = ChatLLMSettings(_env_file=env_file, _env_prefix="T__")
        assert s.max_retries == 1000

    def test_retry_base_delay_zero(self, tmp_path):
        env_file = self._make_env(tmp_path, retry_base_delay="0")
        s = ChatLLMSettings(_env_file=env_file, _env_prefix="T__")
        assert s.retry_base_delay == 0.0

    def test_max_retries_non_integer_raises(self, tmp_path):
        env_file = self._make_env(tmp_path, max_retries="abc")
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="T__")

    def test_retry_base_delay_non_numeric_raises(self, tmp_path):
        env_file = self._make_env(tmp_path, retry_base_delay="not_a_number")
        with pytest.raises(Exception):
            ChatLLMSettings(_env_file=env_file, _env_prefix="T__")


class TestDockerSettingsValidationBoundary:
    """Boundary and edge-case validation for DockerSettings."""

    def test_timeout_non_integer_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("T__timeout=abc\n")
        with pytest.raises(Exception):
            DockerSettings(_env_file=env_file, _env_prefix="T__")

    def test_cpus_non_numeric_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("T__cpus=not_a_float\n")
        with pytest.raises(Exception):
            DockerSettings(_env_file=env_file, _env_prefix="T__")

    def test_cpus_zero(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("T__cpus=0\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="T__")
        assert s.cpus == 0.0

    def test_timeout_zero(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("T__timeout=0\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="T__")
        assert s.timeout == 0

    def test_timeout_large_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("T__timeout=999999\n")
        s = DockerSettings(_env_file=env_file, _env_prefix="T__")
        assert s.timeout == 999999
