"""Tests for otter.backend.utils.docker_utils module.

This module wraps sync functions with _make_async (asyncio.to_thread).
Tests verify:
1. _make_async correctly delegates to the sync function in a thread.
2. All async wrappers are properly generated and callable.
3. The module's __all__ exports match expectations.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from subprocess import CompletedProcess

from otter.backend.utils import docker_utils as mod
from otter.backend.utils.docker_utils import (
    _make_async,
    build_image,
    remove_image,
    create_container,
    remove_container,
    start_container,
    stop_container,
    run_container,
    exec_container,
    copy_to_container,
    copy_from_container,
    is_docker_running,
    get_docker_storage_device,
)


# ── _make_async ──

class TestMakeAsync:
    async def test_wraps_sync_function_to_async(self):
        def sync_add(a, b):
            return a + b

        async_add = _make_async(sync_add)
        result = await async_add(1, 2)
        assert result == 3

    async def test_preserves_function_name(self):
        def my_function():
            pass

        wrapped = _make_async(my_function)
        assert wrapped.__name__ == "my_function"

    async def test_propagates_exception_from_sync(self):
        def sync_fail():
            raise ValueError("sync error")

        async_fail = _make_async(sync_fail)
        with pytest.raises(ValueError, match="sync error"):
            await async_fail()

    async def test_passes_kwargs(self):
        def sync_fn(*, key="default"):
            return key

        async_fn = _make_async(sync_fn)
        result = await async_fn(key="custom")
        assert result == "custom"


# ── Async wrappers delegation ──

class TestAsyncWrappersDelegation:
    """Verify each async wrapper delegates to its sync counterpart via asyncio.to_thread.

    Because _make_async captures the sync function reference at module load time
    (closure), patching the name in docker_utils has no effect. Instead, we patch
    asyncio.to_thread to intercept the delegation and verify the correct sync
    function and arguments are passed.
    """

    async def test_build_image_delegates(self, mocker):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_build_image
        await build_image("tag", "FROM alpine", exist_ok=True)
        mock_to_thread.assert_called_once_with(sync_build_image, "tag", "FROM alpine", exist_ok=True)

    async def test_remove_image_delegates(self, mocker):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_remove_image
        await remove_image("tag", missing_ok=True)
        mock_to_thread.assert_called_once_with(sync_remove_image, "tag", missing_ok=True)

    async def test_create_container_delegates(self, mocker):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_create_container
        await create_container("img", "name", exist_ok=True)
        mock_to_thread.assert_called_once_with(sync_create_container, "img", "name", exist_ok=True)

    async def test_remove_container_delegates(self, mocker):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_remove_container
        await remove_container("name", force=True, missing_ok=True)
        mock_to_thread.assert_called_once_with(sync_remove_container, "name", force=True, missing_ok=True)

    async def test_start_container_delegates(self, mocker):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_start_container
        await start_container("name")
        mock_to_thread.assert_called_once_with(sync_start_container, "name")

    async def test_stop_container_delegates(self, mocker):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_stop_container
        await stop_container("name", timeout=30)
        mock_to_thread.assert_called_once_with(sync_stop_container, "name", timeout=30)

    async def test_run_container_delegates(self, mocker):
        expected = CompletedProcess("cmd", 0, "out", "err")
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=expected)
        from otter.backend.utils.sync_docker_utils import sync_run_container
        result = await run_container("img", "cmd")
        mock_to_thread.assert_called_once_with(sync_run_container, "img", "cmd")
        assert result is expected

    async def test_exec_container_delegates(self, mocker):
        expected = CompletedProcess("cmd", 0, "out", "err")
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=expected)
        from otter.backend.utils.sync_docker_utils import sync_exec_container
        result = await exec_container("name", "cmd")
        mock_to_thread.assert_called_once_with(sync_exec_container, "name", "cmd")
        assert result is expected

    async def test_copy_to_container_delegates(self, mocker, tmp_path):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_copy_to_container
        src = tmp_path / "f.txt"
        src.write_text("x")
        await copy_to_container("name", src, "/dst")
        mock_to_thread.assert_called_once_with(sync_copy_to_container, "name", src, "/dst")

    async def test_copy_from_container_delegates(self, mocker, tmp_path):
        mock_to_thread = mocker.patch("asyncio.to_thread", return_value=None)
        from otter.backend.utils.sync_docker_utils import sync_copy_from_container
        await copy_from_container("name", "/src", tmp_path)
        mock_to_thread.assert_called_once_with(sync_copy_from_container, "name", "/src", tmp_path)


# ── Module exports ──

class TestModuleExports:
    def test_all_contains_expected_names(self):
        expected = {
            "is_docker_running",
            "get_docker_storage_device",
            "build_image",
            "remove_image",
            "create_container",
            "remove_container",
            "start_container",
            "stop_container",
            "run_container",
            "exec_container",
            "copy_to_container",
            "copy_from_container",
        }
        assert set(mod.__all__) == expected

    def test_sync_functions_not_in_all(self):
        """sync_ prefixed functions should not be exported."""
        for name in mod.__all__:
            assert not name.startswith("sync_"), f"{name} should not be in __all__"

    def test_is_docker_running_is_not_async(self):
        """is_docker_running is re-exported directly, not wrapped."""
        assert not asyncio.iscoroutinefunction(is_docker_running)

    def test_get_docker_storage_device_is_not_async(self):
        """get_docker_storage_device is re-exported directly, not wrapped."""
        assert not asyncio.iscoroutinefunction(get_docker_storage_device)

    def test_all_async_wrappers_are_coroutine_functions(self):
        """All _make_async-wrapped functions should be coroutine functions."""
        async_names = [
            "build_image", "remove_image",
            "create_container", "remove_container",
            "start_container", "stop_container",
            "run_container", "exec_container",
            "copy_to_container", "copy_from_container",
        ]
        for name in async_names:
            fn = getattr(mod, name)
            assert asyncio.iscoroutinefunction(fn), f"{name} should be async"
