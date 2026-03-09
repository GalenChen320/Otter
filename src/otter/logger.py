import sys
import logging

from otter.config.setting import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = _build_logger()
    return _logger


def init_logger() -> logging.Logger:
    global _logger
    _logger = _build_logger()
    return _logger


def _build_logger() -> logging.Logger:
    settings = get_settings()
    lg = logging.getLogger("main")
    lg.setLevel(settings.log.level)
    lg.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    lg.addHandler(console)

    if settings.log.log_file is not None:
        settings.log.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        lg.addHandler(file_handler)

    return lg
