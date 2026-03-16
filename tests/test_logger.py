"""Tests for otter.logger module."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGetLogger:
    """Test get_logger() lazy initialization."""

    def test_returns_logger_instance(self):
        """get_logger should return a logging.Logger instance."""
        import otter.logger as mod
        mod._logger = None  # reset global state

        with patch.object(mod, "_build_logger") as mock_build:
            mock_build.return_value = logging.getLogger("test_get_logger")
            logger = mod.get_logger()
            assert isinstance(logger, logging.Logger)
            mock_build.assert_called_once()

    def test_returns_same_instance_on_second_call(self):
        """get_logger should return cached instance on subsequent calls."""
        import otter.logger as mod
        mod._logger = None

        with patch.object(mod, "_build_logger") as mock_build:
            mock_build.return_value = logging.getLogger("test_cached")
            first = mod.get_logger()
            second = mod.get_logger()
            assert first is second
            mock_build.assert_called_once()


class TestInitLogger:
    """Test init_logger() forced re-initialization."""

    def test_reinitializes_logger(self):
        """init_logger should always call _build_logger, even if already initialized."""
        import otter.logger as mod
        mod._logger = logging.getLogger("old")

        with patch.object(mod, "_build_logger") as mock_build:
            new_logger = logging.getLogger("new")
            mock_build.return_value = new_logger
            result = mod.init_logger()
            assert result is new_logger
            assert mod._logger is new_logger
            mock_build.assert_called_once()


class TestBuildLogger:
    """Test _build_logger() with various configurations."""

    def test_default_when_settings_not_initialized(self):
        """When get_settings raises RuntimeError, should default to WARNING level."""
        import otter.logger as mod

        with patch("otter.logger.get_settings", side_effect=RuntimeError("not init")):
            logger = mod._build_logger()
            assert logger.level == logging.WARNING
            # Should have exactly one handler (stderr console)
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_with_settings_log_level(self):
        """Should use log level from settings."""
        import otter.logger as mod

        mock_settings = type("S", (), {
            "log": type("L", (), {"level": "DEBUG", "log_file": None})()
        })()

        with patch("otter.logger.get_settings", return_value=mock_settings):
            logger = mod._build_logger()
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) == 1

    def test_with_log_file(self, tmp_path):
        """Should add FileHandler when log_file is configured."""
        import otter.logger as mod

        log_file = tmp_path / "logs" / "test.log"
        mock_settings = type("S", (), {
            "log": type("L", (), {"level": "INFO", "log_file": log_file})()
        })()

        with patch("otter.logger.get_settings", return_value=mock_settings):
            logger = mod._build_logger()
            assert logger.level == logging.INFO
            assert len(logger.handlers) == 2
            handler_types = [type(h) for h in logger.handlers]
            assert logging.StreamHandler in handler_types
            assert logging.FileHandler in handler_types
            # log directory should be created
            assert log_file.parent.exists()

    def test_handlers_cleared_on_rebuild(self):
        """Rebuilding logger should replace old handlers, not accumulate."""
        import otter.logger as mod

        with patch("otter.logger.get_settings", side_effect=RuntimeError):
            logger1 = mod._build_logger()
            old_handlers = list(logger1.handlers)
            assert len(old_handlers) == 1

            logger2 = mod._build_logger()
            assert logger1 is logger2  # same "main" logger instance
            assert len(logger2.handlers) == 1
            # handler object should be different (old one was cleared)
            assert logger2.handlers[0] is not old_handlers[0]

    def test_formatter_applied(self):
        """All handlers should have the expected format string."""
        import otter.logger as mod

        with patch("otter.logger.get_settings", side_effect=RuntimeError):
            logger = mod._build_logger()
            for handler in logger.handlers:
                assert handler.formatter is not None
                assert "%(asctime)s" in handler.formatter._fmt
                assert "%(levelname)" in handler.formatter._fmt
                assert "%(name)s" in handler.formatter._fmt

    def test_console_handler_writes_to_stderr(self):
        """Console handler should write to stderr."""
        import sys
        import otter.logger as mod

        with patch("otter.logger.get_settings", side_effect=RuntimeError):
            logger = mod._build_logger()
            console_handler = logger.handlers[0]
            assert console_handler.stream is sys.stderr
