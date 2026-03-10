import asyncio
import functools

from typing import ParamSpec, TypeVar, Callable, Coroutine, Any
from .sync_docker_utils import (
    is_docker_running,
    sync_build_image,
    sync_remove_image,
    sync_create_container,
    sync_remove_container,
    sync_start_container,
    sync_stop_container,
    sync_run_container,
    sync_exec_container,
    sync_copy_to_container,
    sync_copy_from_container,
)


P = ParamSpec("P")
R = TypeVar("R")


def _make_async(fn: Callable[P, R]) -> Callable[P, Coroutine[Any, Any, R]]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(fn, *args, **kwargs)
    return wrapper


build_image         = _make_async(sync_build_image)
remove_image        = _make_async(sync_remove_image)
create_container    = _make_async(sync_create_container)
remove_container    = _make_async(sync_remove_container)
start_container     = _make_async(sync_start_container)
stop_container      = _make_async(sync_stop_container)
run_container       = _make_async(sync_run_container)
exec_container      = _make_async(sync_exec_container)
copy_to_container   = _make_async(sync_copy_to_container)
copy_from_container = _make_async(sync_copy_from_container)


__all__ = [
    "is_docker_running",
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
]
