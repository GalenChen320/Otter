"""Shared test fixtures."""

import os
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Ensure tests are not affected by the project's real .env files
    or lingering environment variables.

    - Changes CWD to a temp directory so pydantic-settings won't
      auto-discover the project root .env.
    - Strips any env vars that could interfere with config tests.
    """
    monkeypatch.chdir(tmp_path)

    prefixes = (
        "EXPERIMENT_ID", "MAX_TURNS", "SAMPLES_PER_PROBLEM",
        "DATASET_NAME", "DATASET__",
        "PROPOSER_TYPE", "PROPOSER__", "PROPOSER_CONCURRENCY",
        "EXECUTOR_TYPE", "EXECUTOR__", "EXECUTOR_CONCURRENCY",
        "EVALUATOR_TYPE", "EVALUATOR__", "EVALUATOR_CONCURRENCY",
        "DOCKER__", "LOG__", "TEST__",
    )
    for key in list(os.environ):
        if key.startswith(prefixes):
            monkeypatch.delenv(key)
