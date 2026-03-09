import json
from pathlib import Path

from otter.episode import Episode, Turn
from otter.store.base import BaseStore


class LineStore(BaseStore):
    """每个 Episode 一个 .jsonl 文件，每行一个 Turn。

    文件结构：
        {output_dir}/
        ├── {eid}.jsonl
        ├── {eid}.jsonl
        └── ...
    """

    def __init__(self, output_dir: Path):
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _episode_path(self, eid: str) -> Path:
        return self._dir / f"{eid}.jsonl"

    def load_episodes(self) -> dict[str, Episode]:
        episodes: dict[str, Episode] = {}
        for path in self._dir.glob("*.jsonl"):
            eid = path.stem
            task_id, sample_id = eid.rsplit("#", 1)
            turns = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                turns.append(Turn.from_dict(json.loads(line)))
            episodes[eid] = Episode(
                task_id=task_id,
                sample_id=int(sample_id),
                turns=turns,
            )
        return episodes

    async def save_turn(self, episode: Episode, turn: Turn) -> None:
        path = self._episode_path(episode.eid)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")
