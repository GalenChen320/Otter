"""Tests for otter.config.setting module."""

import pytest
from pathlib import Path

from otter.config.setting import (
    LoggerSettings,
    Settings,
    _build_backend_settings,
    _build_dataset_settings,
    _build_settings,
    init_settings,
    get_settings,
    get_tracked_config,
    _settings,
)
from otter.config.backend_settings import ChatLLMSettings, DockerSettings
from otter.config.dataset_settings import MbppplusSettings, EvalplusSettings
from otter.config.utils import ROOT_DIR


# ── LoggerSettings ──

class TestLoggerSettings:
    def test_defaults(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        s = LoggerSettings(_env_file=env_file)
        assert s.level == "INFO"
        assert s.log_file is None

    def test_reads_from_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("LOG__level=DEBUG\nLOG__log_file=/tmp/test.log\n")
        s = LoggerSettings(_env_file=env_file)
        assert s.level == "DEBUG"
        assert s.log_file == Path("/tmp/test.log")

    def test_empty_log_file_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("LOG__log_file=\n")
        s = LoggerSettings(_env_file=env_file)
        assert s.log_file is None

    def test_whitespace_log_file_becomes_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("LOG__log_file=   \n")
        s = LoggerSettings(_env_file=env_file)
        assert s.log_file is None

    def test_invalid_level_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("LOG__level=TRACE\n")
        with pytest.raises(Exception):
            LoggerSettings(_env_file=env_file)

    def test_all_valid_levels(self, tmp_path):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            env_file = tmp_path / ".env"
            env_file.write_text(f"LOG__level={level}\n")
            s = LoggerSettings(_env_file=env_file)
            assert s.level == level

    def test_all_fields_are_untracked(self):
        for name, field_info in LoggerSettings.model_fields.items():
            extra = field_info.json_schema_extra
            assert isinstance(extra, dict), f"{name} missing json_schema_extra"
            assert extra.get("core") is False, f"{name} should be untracked"


# ── Settings defaults & fields ──

class TestSettingsDefaults:
    def _make_settings(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        log = LoggerSettings(_env_file=env_file)
        return Settings(_env_file=env_file, log=log)

    def test_experiment_id_default(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.experiment_id == "default"

    def test_max_turns_default(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.max_turns == 1

    def test_samples_per_problem_default(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.samples_per_problem == 1

    def test_dataset_name_default(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.dataset_name == "mbppplus"

    def test_proposer_type_default_none(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.proposer_type is None

    def test_executor_type_default_none(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.executor_type is None

    def test_evaluator_type_default_none(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.evaluator_type is None

    def test_concurrency_defaults(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.proposer_concurrency == 1
        assert s.executor_concurrency == 1
        assert s.evaluator_concurrency == 1

    def test_component_settings_default_none(self, tmp_path):
        s = self._make_settings(tmp_path)
        assert s.dataset is None
        assert s.proposer is None
        assert s.executor is None
        assert s.evaluator is None

    def test_reads_from_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EXPERIMENT_ID=my_exp\n"
            "MAX_TURNS=5\n"
            "SAMPLES_PER_PROBLEM=3\n"
            "DATASET_NAME=evalplus\n"
            "EXECUTOR_TYPE=chat_llm\n"
            "EXECUTOR_CONCURRENCY=10\n"
        )
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        assert s.experiment_id == "my_exp"
        assert s.max_turns == 5
        assert s.samples_per_problem == 3
        assert s.dataset_name == "evalplus"
        assert s.executor_type == "chat_llm"
        assert s.executor_concurrency == 10

    def test_tracked_fields(self):
        tracked_names = {
            "experiment_id", "max_turns", "samples_per_problem",
            "dataset_name", "proposer_type", "executor_type", "evaluator_type",
        }
        for name in tracked_names:
            extra = Settings.model_fields[name].json_schema_extra
            assert extra["core"] is True, f"{name} should be tracked"

    def test_untracked_fields(self):
        untracked_names = {
            "proposer_concurrency", "executor_concurrency", "evaluator_concurrency",
        }
        for name in untracked_names:
            extra = Settings.model_fields[name].json_schema_extra
            assert extra["core"] is False, f"{name} should be untracked"


# ── Settings empty string coercion ──

class TestSettingsEmptyStrCoercion:
    def _make_settings(self, tmp_path, env_content):
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)
        log = LoggerSettings(_env_file=env_file)
        return Settings(_env_file=env_file, log=log)

    def test_empty_proposer_type_becomes_none(self, tmp_path):
        s = self._make_settings(tmp_path, "PROPOSER_TYPE=\n")
        assert s.proposer_type is None

    def test_empty_executor_type_becomes_none(self, tmp_path):
        s = self._make_settings(tmp_path, "EXECUTOR_TYPE=\n")
        assert s.executor_type is None

    def test_empty_evaluator_type_becomes_none(self, tmp_path):
        s = self._make_settings(tmp_path, "EVALUATOR_TYPE=\n")
        assert s.evaluator_type is None

    def test_whitespace_proposer_type_becomes_none(self, tmp_path):
        s = self._make_settings(tmp_path, "PROPOSER_TYPE=   \n")
        assert s.proposer_type is None


# ── Settings.output_dir ──

class TestSettingsOutputDir:
    def test_output_dir_uses_experiment_id(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXPERIMENT_ID=my_exp\n")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        assert s.output_dir == ROOT_DIR / "experiments" / "my_exp"

    def test_output_dir_default(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        assert s.output_dir == ROOT_DIR / "experiments" / "default"

    def test_output_dir_is_path(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        assert isinstance(s.output_dir, Path)


# ── _build_backend_settings ──

class TestBuildBackendSettings:
    def test_none_type_returns_none(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        result = _build_backend_settings("EXECUTOR", None, env_file)
        assert result is None

    def test_chat_llm_returns_correct_type(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EXECUTOR__api_key=sk-test\n"
            "EXECUTOR__base_url=https://api.example.com\n"
            "EXECUTOR__model=gpt-4o\n"
        )
        result = _build_backend_settings("EXECUTOR", "chat_llm", env_file)
        assert isinstance(result, ChatLLMSettings)

    def test_docker_returns_correct_type(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        result = _build_backend_settings("EVALUATOR", "docker", env_file)
        assert isinstance(result, DockerSettings)

    def test_unknown_type_raises_value_error(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        with pytest.raises(ValueError, match="Unknown backend type"):
            _build_backend_settings("EXECUTOR", "nonexistent", env_file)

    def test_prefix_isolation(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "PROPOSER__api_key=sk-proposer\n"
            "PROPOSER__base_url=https://proposer.example.com\n"
            "PROPOSER__model=proposer-model\n"
            "EXECUTOR__api_key=sk-executor\n"
            "EXECUTOR__base_url=https://executor.example.com\n"
            "EXECUTOR__model=executor-model\n"
        )
        proposer = _build_backend_settings("PROPOSER", "chat_llm", env_file)
        executor = _build_backend_settings("EXECUTOR", "chat_llm", env_file)
        assert proposer.api_key == "sk-proposer"
        assert executor.api_key == "sk-executor"
        assert proposer.model == "proposer-model"
        assert executor.model == "executor-model"


# ── _build_dataset_settings ──

class TestBuildDatasetSettings:
    def test_mbppplus_returns_correct_type(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        result = _build_dataset_settings("mbppplus", env_file)
        assert isinstance(result, MbppplusSettings)

    def test_evalplus_returns_correct_type(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        result = _build_dataset_settings("evalplus", env_file)
        assert isinstance(result, EvalplusSettings)

    def test_unknown_name_raises_value_error(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        with pytest.raises(ValueError, match="Unknown dataset type"):
            _build_dataset_settings("nonexistent", env_file)

    def test_reads_cache_dir_from_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DATASET__cache_dir=/custom/cache\n")
        result = _build_dataset_settings("mbppplus", env_file)
        assert result.cache_dir == Path("/custom/cache")


# ── _build_settings (integration) ──

class TestBuildSettings:
    def _write_env(self, content=""):
        """Write a temp .env file in ROOT_DIR and return its relative name."""
        env_file = ROOT_DIR / ".env.test_build"
        env_file.write_text(content)
        return ".env.test_build", env_file

    def test_minimal_defaults(self):
        name, env_file = self._write_env("")
        try:
            s = _build_settings(name)
            assert s.experiment_id == "default"
            assert s.dataset_name == "mbppplus"
            assert isinstance(s.dataset, MbppplusSettings)
            assert s.proposer is None
            assert s.executor is None
            assert s.evaluator is None
            assert s.log.level == "INFO"
        finally:
            env_file.unlink(missing_ok=True)

    def test_with_executor_chat_llm(self):
        name, env_file = self._write_env(
            "EXECUTOR_TYPE=chat_llm\n"
            "EXECUTOR__api_key=sk-test\n"
            "EXECUTOR__base_url=https://api.example.com\n"
            "EXECUTOR__model=gpt-4o\n"
        )
        try:
            s = _build_settings(name)
            assert isinstance(s.executor, ChatLLMSettings)
            assert s.executor.api_key == "sk-test"
            assert s.proposer is None
            assert s.evaluator is None
        finally:
            env_file.unlink(missing_ok=True)

    def test_with_evaluator_docker(self):
        name, env_file = self._write_env(
            "EVALUATOR_TYPE=docker\n"
            "EVALUATOR__timeout=30\n"
        )
        try:
            s = _build_settings(name)
            assert isinstance(s.evaluator, DockerSettings)
            assert s.evaluator.timeout == 30
        finally:
            env_file.unlink(missing_ok=True)

    def test_with_all_components(self):
        name, env_file = self._write_env(
            "DATASET_NAME=evalplus\n"
            "PROPOSER_TYPE=chat_llm\n"
            "PROPOSER__api_key=sk-p\n"
            "PROPOSER__base_url=https://p.example.com\n"
            "PROPOSER__model=p-model\n"
            "EXECUTOR_TYPE=chat_llm\n"
            "EXECUTOR__api_key=sk-e\n"
            "EXECUTOR__base_url=https://e.example.com\n"
            "EXECUTOR__model=e-model\n"
            "EVALUATOR_TYPE=docker\n"
        )
        try:
            s = _build_settings(name)
            assert isinstance(s.dataset, EvalplusSettings)
            assert isinstance(s.proposer, ChatLLMSettings)
            assert isinstance(s.executor, ChatLLMSettings)
            assert isinstance(s.evaluator, DockerSettings)
            assert s.proposer.api_key == "sk-p"
            assert s.executor.api_key == "sk-e"
        finally:
            env_file.unlink(missing_ok=True)

    def test_unknown_backend_type_raises(self):
        name, env_file = self._write_env("EXECUTOR_TYPE=unknown_backend\n")
        try:
            with pytest.raises(ValueError, match="Unknown backend type"):
                _build_settings(name)
        finally:
            env_file.unlink(missing_ok=True)

    def test_unknown_dataset_name_raises(self):
        name, env_file = self._write_env("DATASET_NAME=nonexistent\n")
        try:
            with pytest.raises(ValueError, match="Unknown dataset type"):
                _build_settings(name)
        finally:
            env_file.unlink(missing_ok=True)


# ── init_settings / get_settings ──

class TestInitGetSettings:
    def setup_method(self):
        """Reset global _settings before each test."""
        import otter.config.setting as mod
        mod._settings = None

    def teardown_method(self):
        """Clean up global _settings and temp env file after each test."""
        import otter.config.setting as mod
        mod._settings = None
        env_file = ROOT_DIR / ".env.test_init"
        env_file.unlink(missing_ok=True)

    def test_get_settings_before_init_raises(self):
        with pytest.raises(RuntimeError, match="Settings not initialized"):
            get_settings()

    def test_init_then_get_returns_same_object(self):
        env_file = ROOT_DIR / ".env.test_init"
        env_file.write_text("")
        s1 = init_settings(".env.test_init")
        s2 = get_settings()
        assert s1 is s2

    def test_init_returns_settings_instance(self):
        env_file = ROOT_DIR / ".env.test_init"
        env_file.write_text("")
        s = init_settings(".env.test_init")
        assert isinstance(s, Settings)

    def test_init_overwrites_previous(self):
        env_file = ROOT_DIR / ".env.test_init"
        env_file.write_text("EXPERIMENT_ID=first\n")
        s1 = init_settings(".env.test_init")
        env_file.write_text("EXPERIMENT_ID=second\n")
        s2 = init_settings(".env.test_init")
        assert s1.experiment_id == "first"
        assert s2.experiment_id == "second"
        assert get_settings() is s2


# ── get_tracked_config ──

class TestGetTrackedConfig:
    def test_contains_top_level_tracked_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXPERIMENT_ID=track_test\n")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        tracked = get_tracked_config(s)
        assert tracked["experiment_id"] == "track_test"
        assert tracked["max_turns"] == 1
        assert tracked["samples_per_problem"] == 1
        assert tracked["dataset_name"] == "mbppplus"

    def test_excludes_untracked_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXECUTOR_CONCURRENCY=10\n")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        tracked = get_tracked_config(s)
        assert "proposer_concurrency" not in tracked
        assert "executor_concurrency" not in tracked
        assert "evaluator_concurrency" not in tracked

    def test_excludes_logger_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("LOG__level=DEBUG\n")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        tracked = get_tracked_config(s)
        assert "log.level" not in tracked
        assert "log.log_file" not in tracked

    def test_includes_nested_backend_tracked_fields(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "EXECUTOR__api_key=sk-secret\n"
            "EXECUTOR__base_url=https://api.example.com\n"
            "EXECUTOR__model=gpt-4o\n"
        )
        executor = ChatLLMSettings(_env_file=env_file, _env_prefix="EXECUTOR__")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log, executor=executor)
        tracked = get_tracked_config(s)
        assert tracked["executor.base_url"] == "https://api.example.com"
        assert tracked["executor.model"] == "gpt-4o"
        assert "executor.api_key" not in tracked  # api_key is untracked

    def test_none_components_excluded(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        tracked = get_tracked_config(s)
        # None components should not produce any keys
        assert not any(k.startswith("proposer.") for k in tracked)
        assert not any(k.startswith("executor.") for k in tracked)
        assert not any(k.startswith("evaluator.") for k in tracked)

    def test_all_values_are_json_serializable(self, tmp_path):
        """Path values should be converted to str."""
        import json
        env_file = tmp_path / ".env"
        env_file.write_text("")
        log = LoggerSettings(_env_file=env_file)
        s = Settings(_env_file=env_file, log=log)
        tracked = get_tracked_config(s)
        # Should not raise
        json.dumps(tracked)

    def test_uses_global_settings_when_none_passed(self):
        """When no settings passed, get_tracked_config calls get_settings()."""
        import otter.config.setting as mod
        mod._settings = None
        with pytest.raises(RuntimeError, match="Settings not initialized"):
            get_tracked_config()

