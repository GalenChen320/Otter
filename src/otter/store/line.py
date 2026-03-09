import json
from dataclasses import asdict
from pathlib import Path

from otter.episode import Episode, Turn, ExecutionResult
from otter.store.base import BaseStore


class LineStore(BaseStore):
    """每个 Episode 一个 .jsonl 文件，每行一个 Turn。

    文件结构：
        {output_dir}/
        ├── {eid}.jsonl
        ├── {eid}.jsonl
        └── ...
    """

    def __init__(self, output_dir: Path, max_turns: int):
        self._dir = output_dir
        self._max_turns = max_turns
        self._dir.mkdir(parents=True, exist_ok=True)

    def _episode_path(self, eid: str) -> Path:
        return self._dir / f"{eid}.jsonl"

    def load_episodes(self) -> dict[str, Episode]:
        episodes: dict[str, Episode] = {}
        for path in self._dir.glob("*.jsonl"):
            eid = path.stem
            turns = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                turns.append(self._parse_turn(json.loads(line)))
            episodes[eid] = Episode(
                eid=eid,
                max_turns=self._max_turns,
                turns=turns,
            )
        return episodes

    async def save_turn(self, episode: Episode, turn: Turn) -> None:
        path = self._episode_path(episode.eid)
        data = self._serialize_turn(turn)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    @staticmethod
    def _serialize_turn(turn: Turn) -> dict:
        d = asdict(turn)
        d.pop("execution_result", None)
        if turn.execution_result is not None:
            d["passed"] = turn.execution_result.passed
            d["stdout"] = turn.execution_result.stdout
            d["stderr"] = turn.execution_result.stderr
            d["timed_out"] = turn.execution_result.timed_out
        else:
            d["passed"] = None
        return d

    @staticmethod
    def _parse_turn(d: dict) -> Turn:
        er = None
        if d.get("passed") is not None:
            er = ExecutionResult(
                passed=d["passed"],
                stdout=d.get("stdout", ""),
                stderr=d.get("stderr", ""),
                timed_out=d.get("timed_out", False),
            )
        return Turn(
            turn_number=d["turn_number"],
            prompt=d["prompt"],
            response=d.get("response", ""),
            code=d.get("code", ""),
            execution_result=er,
        )
