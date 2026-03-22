import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field

from otter.config.setting import get_settings
from otter.manifest import InputManifest, OutputManifest


META_FILENAME = "meta.json"
EXPERIMENT_META = "experiment.json"


@dataclass
class Turn:
    turn_dir: Path
    passed: bool | None = None
    prop_input_path: Path | None = None
    prop_output_path: Path | None = None
    exec_input_path: Path | None = None
    exec_output_path: Path | None = None
    eval_input_path: Path | None = None
    eval_output_path: Path | None = None
    prop_input_manifest: InputManifest | None = None
    prop_output_manifest: OutputManifest | None = None
    exec_input_manifest: InputManifest | None = None
    exec_output_manifest: OutputManifest | None = None
    eval_input_manifest: InputManifest | None = None
    eval_output_manifest: OutputManifest | None = None

    def setup_dirs(self) -> None:
        """根据 settings 按需创建子目录，并设置对应的 path 字段。"""
        settings = get_settings()
        self.turn_dir.mkdir(parents=True, exist_ok=True)

        if settings.proposer is not None:
            self.prop_input_path = self.turn_dir / "prop_input"
            self.prop_output_path = self.turn_dir / "prop_output"
            self.prop_input_path.mkdir(exist_ok=True)
            self.prop_output_path.mkdir(exist_ok=True)

        if settings.executor is not None:
            self.exec_input_path = self.turn_dir / "exec_input"
            self.exec_output_path = self.turn_dir / "exec_output"
            self.exec_input_path.mkdir(exist_ok=True)
            self.exec_output_path.mkdir(exist_ok=True)

        if settings.evaluator is not None:
            self.eval_input_path = self.turn_dir / "eval_input"
            self.eval_output_path = self.turn_dir / "eval_output"
            self.eval_input_path.mkdir(exist_ok=True)
            self.eval_output_path.mkdir(exist_ok=True)

    def archive_output(self, role: str, suffix: str) -> None:
        """将指定角色的 output 目录归档为带后缀的目录，然后重建空目录。"""
        match role:
            case "proposer":
                output_dir = self.prop_output_path
                self.prop_output_manifest = None
            case "executor":
                output_dir = self.exec_output_path
                self.exec_output_manifest = None
            case "evaluator":
                output_dir = self.eval_output_path
                self.eval_output_manifest = None
            case _:
                raise ValueError(f"unknown role: {role}")
        if output_dir and output_dir.exists():
            archive_dir = output_dir.with_name(f"{output_dir.name}.{suffix}")
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            output_dir.rename(archive_dir)
            output_dir.mkdir(exist_ok=True)

    def save_meta(self) -> None:
        """写入 meta.json，标记 turn 完成。"""
        meta = {"passed": self.passed}
        (self.turn_dir / META_FILENAME).write_text(
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

    def archive_last_output(self, suffix: str) -> None:
        """归档最后一个 turn 中最末端角色的 output。"""
        turn = self.turns[-1]
        if turn.eval_output_manifest is not None:
            turn.archive_output("evaluator", suffix)
        elif turn.exec_output_manifest is not None:
            turn.archive_output("executor", suffix)
        elif turn.prop_output_manifest is not None:
            turn.archive_output("proposer", suffix)
        else:
            raise RuntimeError(f"[{self.eid}] no output to archive in last turn")

    def next_turn(self) -> None:
        """创建下一个 Turn，建立目录结构，append 到 turns。"""
        turn_dir = self.base_dir / f"turn_{len(self.turns) + 1}"
        turn = Turn(turn_dir=turn_dir)
        turn.setup_dirs()
        self.turns.append(turn)

    @classmethod
    def sync_all(cls, output_dir: Path) -> dict[str, "Episode"]:
        """扫描目录结构，清理未完成的 Turn，重建所有 Episode。"""
        from otter.logger import get_logger
        logger = get_logger()
        episodes: dict[str, Episode] = {}

        if not output_dir.exists():
            return episodes

        def _load_manifest(directory: Path, manifest_cls):
            mf = directory / "manifest.json"
            if mf.exists():
                return manifest_cls.load(directory)
            return None

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

                prop_input_dir = turn_dir / "prop_input"
                prop_output_dir = turn_dir / "prop_output"
                exec_input_dir = turn_dir / "exec_input"
                exec_output_dir = turn_dir / "exec_output"
                eval_input_dir = turn_dir / "eval_input"
                eval_output_dir = turn_dir / "eval_output"

                meta = json.loads(meta_path.read_text(encoding="utf-8"))

                turns.append(Turn(
                    turn_dir=turn_dir,
                    passed=meta.get("passed"),
                    prop_input_path=prop_input_dir if prop_input_dir.exists() else None,
                    prop_output_path=prop_output_dir if prop_output_dir.exists() else None,
                    exec_input_path=exec_input_dir if exec_input_dir.exists() else None,
                    exec_output_path=exec_output_dir if exec_output_dir.exists() else None,
                    eval_input_path=eval_input_dir if eval_input_dir.exists() else None,
                    eval_output_path=eval_output_dir if eval_output_dir.exists() else None,
                    prop_input_manifest=_load_manifest(prop_input_dir, InputManifest),
                    prop_output_manifest=_load_manifest(prop_output_dir, OutputManifest),
                    exec_input_manifest=_load_manifest(exec_input_dir, InputManifest),
                    exec_output_manifest=_load_manifest(exec_output_dir, OutputManifest),
                    eval_input_manifest=_load_manifest(eval_input_dir, InputManifest),
                    eval_output_manifest=_load_manifest(eval_output_dir, OutputManifest),
                ))

            episodes[eid] = Episode(
                task_id=task_id,
                sample_id=int(sample_id),
                turns=turns,
                base_dir=ep_dir,
            )

        logger.info("synced %d episodes from %s", len(episodes), output_dir)
        return episodes


__all__ = [
    "Episode",
    "InputManifest",
    "OutputManifest",
    "EXPERIMENT_META",
]
