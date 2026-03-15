from otter.backend.chat_llm import ChatLLMBackend
from otter.backend.docker import DockerBackend, DockerResult


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
            docker = settings.docker
            return DockerBackend(
                timeout=docker.timeout,
                cpus=docker.cpus,
                memory=docker.memory,
                memory_swap=docker.memory_swap,
                memory_reservation=docker.memory_reservation,
                network_mode=docker.network_mode,
                device_read_bps=docker.device_read_bps,
                device_write_bps=docker.device_write_bps,
            )
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")


__all__ = [
    "ChatLLMBackend",
    "DockerBackend",
    "DockerResult",
    "create_backend",
]
