from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil


@dataclass
class InputManifest:
    """LLM 输入侧的句柄。Dataset 写入，LLM 读取。"""
    base_path: Path | None = None
    messages_file: str | None = None


@dataclass
class ResponseManifest:
    """LLM 输出侧的句柄。LLM 写入，Dataset 读取。"""
    base_path: Path | None = None
    response_file: str | None = None


@dataclass
class ExecManifest:
    """执行侧的句柄。Dataset 写入，Environment 读取。"""
    base_path: Path | None = None
    image_tag: str | None = None
    script_file: str | None = None
    commands: list[str] | None = None
    timeout: int | None = None


@dataclass
class ObservationManifest:
    """观测侧的句柄。Environment 写入，Dataset 读取。"""
    base_path: Path | None = None
    stdout_file: str | None = None
    stderr_file: str | None = None
    returncode: int | None = None
    timed_out: bool | None = None


META_FILENAME = "meta.json"


@dataclass
class Turn:
    input_path: Path | None = None
    response_path: Path | None = None
    observation_path: Path | None = None
    passed: bool | None = None
    input_manifest: InputManifest | None = None
    response_manifest: ResponseManifest | None = None
    exec_manifest: ExecManifest | None = None
    observation_manifest: ObservationManifest | None = None

    @property
    def turn_dir(self) -> Path | None:
        if self.input_path:
            return self.input_path.parent
        return None

    def save_meta(self) -> None:
        """写入 meta.json，标记 turn 完成。"""
        turn_dir = self.turn_dir
        if not turn_dir:
            raise ValueError("Turn has no directory")
        meta = {"passed": self.passed}
        (turn_dir / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8",
        )


@dataclass
class Episode:
    task_id: str
    sample_id: int
    turns: list[Turn] = field(default_factory=list)
    base_dir: Path | None = None

    @staticmethod
    def make_eid(task_id: str, sample_id: int) -> str:
        return f"{task_id}#{sample_id}"

    @property
    def eid(self) -> str:
        return Episode.make_eid(self.task_id, self.sample_id)

    @property
    def resolved(self) -> bool:
        return len(self.turns) > 0 and self.turns[-1].passed is True

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    def exhausted(self, max_turns: int) -> bool:
        return self.total_turns >= max_turns

    def next_turn(self) -> None:
        """创建下一个 Turn，建立目录结构，append 到 turns。"""
        turn_dir = self.base_dir / f"turn_{len(self.turns) + 1}"
        input_dir = turn_dir / "input"
        response_dir = turn_dir / "response"
        observation_dir = turn_dir / "observation"

        input_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)
        observation_dir.mkdir(parents=True, exist_ok=True)

        self.turns.append(Turn(
            input_path=input_dir,
            response_path=response_dir,
            observation_path=observation_dir,
        ))

    @classmethod
    def sync_all(cls, output_dir: Path) -> dict[str, "Episode"]:
        """扫描目录结构，清理未完成的 Turn，重建所有 Episode。"""
        from otter.logger import get_logger
        logger = get_logger()
        episodes: dict[str, Episode] = {}

        if not output_dir.exists():
            return episodes

        for ep_dir in sorted(output_dir.iterdir()):
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
                meta_path = turn_dir / META_FILENAME
                if not meta_path.exists():
                    shutil.rmtree(turn_dir)
                    logger.info("cleaned incomplete turn: %s", turn_dir)
                    continue

                input_dir = turn_dir / "input"
                response_dir = turn_dir / "response"
                observation_dir = turn_dir / "observation"

                meta = json.loads(meta_path.read_text(encoding="utf-8"))

                turns.append(Turn(
                    input_path=input_dir if input_dir.exists() else None,
                    response_path=response_dir if response_dir.exists() else None,
                    observation_path=observation_dir if observation_dir.exists() else None,
                    passed=meta.get("passed"),
                ))

            episodes[eid] = Episode(
                task_id=task_id,
                sample_id=int(sample_id),
                turns=turns,
                base_dir=ep_dir,
            )

        logger.info("synced %d episodes from %s", len(episodes), output_dir)
        return episodes
