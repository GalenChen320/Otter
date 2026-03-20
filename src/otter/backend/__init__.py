from otter.backend.chat_llm import ChatLLMDebugInfo, ChatLLMBackend
from otter.backend.docker import DockerDebugInfo, DockerBackend


def create_backend(backend_type: str, settings):
    """根据 type 字符串和对应的 settings 实例化 Backend。"""
    match backend_type:
        case "chat_llm":
            return ChatLLMBackend(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.model,
                max_retries=settings.max_retries,
                retry_base_delay=settings.retry_base_delay,
            )
        case "docker":
            return DockerBackend(
                timeout=settings.timeout,
                cpus=settings.cpus,
                memory=settings.memory,
                memory_swap=settings.memory_swap,
                memory_reservation=settings.memory_reservation,
                network_mode=settings.network_mode,
                device_read_bps=settings.device_read_bps,
                device_write_bps=settings.device_write_bps,
            )
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")


__all__ = [
    "ChatLLMDebugInfo",
    "ChatLLMBackend",
    "DockerDebugInfo",
    "DockerBackend",
    "create_backend",
]
