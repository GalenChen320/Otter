import asyncio

from docker_cli.base import BaseAgentDriver
from .docker import DockerBackend


class AgentDockerBackend(DockerBackend):

    def __init__(self, driver: BaseAgentDriver, **kwargs) -> None:
        super().__init__(**kwargs)
        self._driver = driver

    async def _on_container_started(self, container_name: str) -> None:
        await asyncio.to_thread(self._driver.setup_config, container_name)