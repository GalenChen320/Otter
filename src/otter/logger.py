import sys
import logging

from otter.config.setting import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("main")
    logger.setLevel(settings.log.level)
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if settings.log.log_file is not None:
        settings.log.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = _build_logger()
