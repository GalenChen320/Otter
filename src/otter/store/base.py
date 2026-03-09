from abc import ABC, abstractmethod

from otter.episode import Episode, Turn


class BaseStore(ABC):

    @abstractmethod
    def load_episodes(self) -> dict[str, Episode]:
        """加载所有已有的 Episode，返回 {episode.eid: Episode}。

        包含已完成和部分完成的 Episode，用于断点续跑时恢复进度。
        """
        pass

    @abstractmethod
    async def save_turn(self, episode: Episode, turn: Turn) -> None:
        """保存一个刚完成的 Turn。

        每完成一个 Turn 就调用一次，确保进度实时持久化。
        """
        pass
