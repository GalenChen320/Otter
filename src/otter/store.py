import json
from pathlib import Path

from otter.episode import Episode, Turn
from otter.logger import get_logger


class Store:
    """统一的目录结构 Store。

    目录布局：
        {output_dir}/
        └── {eid}/                      # Episode 目录
            ├── turn_1/                 # Turn 目录
            │   ├── input/              # 输入目录
            │   ├── response/           # 响应目录
            │   ├── observation/        # 观测目录
            │   └── meta.json           # Turn 元信息 (passed 等)
            ├── turn_2/
            │   └── ...
            └── ...
    """

    META_FILENAME = "meta.json"

    def __init__(self, output_dir: Path):
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _episode_dir(self, eid: str) -> Path:
        return self._dir / eid

    def _turn_dir(self, eid: str, turn_index: int) -> Path:
        return self._episode_dir(eid) / f"turn_{turn_index + 1}"

    def allocate_turn(self, episode: Episode) -> Turn:
        """为 Episode 分配下一个 Turn，创建目录结构，返回带路径的 Turn。"""
        turn_index = len(episode.turns)
        turn_dir = self._turn_dir(episode.eid, turn_index)

        input_dir = turn_dir / "input"
        response_dir = turn_dir / "response"
        observation_dir = turn_dir / "observation"

        input_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)
        observation_dir.mkdir(parents=True, exist_ok=True)

        return Turn(
            input_path=input_dir,
            response_path=response_dir,
            observation_path=observation_dir,
        )

    def save_meta(self, episode: Episode, turn_index: int) -> None:
        """保存指定 Turn 的元信息。"""
        turn = episode.turns[turn_index]
        turn_dir = self._turn_dir(episode.eid, turn_index)
        turn_dir.mkdir(parents=True, exist_ok=True)

        meta = {"passed": turn.passed}
        meta_path = turn_dir / self.META_FILENAME
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8",
        )

    def load_episodes(self) -> dict[str, Episode]:
        """扫描目录结构，重建所有 Episode 和 Turn。"""
        logger = get_logger()
        episodes: dict[str, Episode] = {}

        for ep_dir in sorted(self._dir.iterdir()):
            if not ep_dir.is_dir() or "#" not in ep_dir.name:
                continue

            eid = ep_dir.name
            task_id, sample_id = eid.rsplit("#", 1)

            turn_dirs = sorted(
                [d for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith("turn_")],
                key=lambda d: int(d.name.split("_")[1]),
            )

            turns: list[Turn] = []
            for turn_dir in turn_dirs:
                input_dir = turn_dir / "input"
                response_dir = turn_dir / "response"
                observation_dir = turn_dir / "observation"

                passed = None
                meta_path = turn_dir / self.META_FILENAME
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    passed = meta.get("passed")

                turns.append(Turn(
                    input_path=input_dir if input_dir.exists() else None,
                    response_path=response_dir if response_dir.exists() else None,
                    observation_path=observation_dir if observation_dir.exists() else None,
                    passed=passed,
                ))

            episodes[eid] = Episode(
                task_id=task_id,
                sample_id=int(sample_id),
                turns=turns,
            )

        logger.info("loaded %d existing episodes from %s", len(episodes), self._dir)
        return episodes
