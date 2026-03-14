from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import get_type_hints, get_args
import json
import shutil


def _is_path_field(hint) -> bool:
    """判断类型注解是否包含 Path（支持 Path | None 等联合类型）。"""
    if hint is Path:
        return True
    for arg in get_args(hint):
        if arg is Path:
            return True
    return False


@dataclass
class BaseManifest:
    def to_dict(self) -> dict:
        """序列化为可 JSON 化的 dict。

        当前仅对 Path 类型字段做 Path → str 转换，
        其余字段（str / int / bool / list[str] 等）直接透传。
        与 from_dict 的类型转换逻辑对称，两者需保持一致。
        """
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Path):
                val = str(val)
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "BaseManifest":
        """从 dict 反序列化重建 Manifest。

        当前仅对 Path 类型字段做 str → Path 转换，
        其余字段（str / int / bool / list[str] 等）依赖 JSON 原生类型直接匹配。
        若将来引入 list[Path] 等复合类型，需扩展此方法的类型转换逻辑。
        """
        hints = get_type_hints(cls)
        kwargs = {}
        for key, val in data.items():
            if key not in hints:
                continue
            if val is not None and _is_path_field(hints[key]):
                kwargs[key] = Path(val)
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def save(self, directory: Path) -> None:
        """将自身序列化写入 directory/manifest.json。"""
        (directory / "manifest.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

@dataclass
class ExecInputManifest(BaseManifest):
    """Executor 输入侧的句柄。Dataset 写入，Executor 读取。"""
    prompt_file: Path | None = None


@dataclass
class ExecOutputManifest(BaseManifest):
    """Executor 输出侧的句柄。Executor 写入，Dataset 读取。"""
    exec_output_file: Path | None = None


@dataclass
class EvalInputManifest(BaseManifest):
    """执行侧的句柄。Dataset 写入，Evaluator 读取。"""
    image_tag: str | None = None
    script_file: Path | None = None
    commands: list[str] | None = None
    timeout: int | None = None


@dataclass
class EvalOutputManifest(BaseManifest):
    """评估器输出侧的句柄。Evaluator 写入，Dataset 读取。"""
    stdout_file: Path | None = None
    stderr_file: Path | None = None
    returncode: int | None = None
    timed_out: bool | None = None


META_FILENAME = "meta.json"
EXPERIMENT_META = "experiment.json"


@dataclass
class Turn:
    exec_input_path: Path | None = None
    exec_output_path: Path | None = None
    eval_input_path: Path | None = None
    eval_output_path: Path | None = None
    passed: bool | None = None
    exec_input_manifest: ExecInputManifest | None = None
    exec_output_manifest: ExecOutputManifest | None = None
    eval_input_manifest: EvalInputManifest | None = None
    eval_output_manifest: EvalOutputManifest | None = None

    @property
    def turn_dir(self) -> Path | None:
        if self.exec_input_path:
            return self.exec_input_path.parent
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
        exec_input_dir = turn_dir / "exec_input"
        exec_output_dir = turn_dir / "exec_output"
        eval_input_dir = turn_dir / "eval_input"
        eval_output_dir = turn_dir / "eval_output"

        exec_input_dir.mkdir(parents=True, exist_ok=True)
        exec_output_dir.mkdir(parents=True, exist_ok=True)
        eval_input_dir.mkdir(parents=True, exist_ok=True)
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        self.turns.append(Turn(
            exec_input_path=exec_input_dir,
            exec_output_path=exec_output_dir,
            eval_input_path=eval_input_dir,
            eval_output_path=eval_output_dir,
        ))

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
                return manifest_cls.from_dict(json.loads(mf.read_text(encoding="utf-8")))
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

                exec_input_dir = turn_dir / "exec_input"
                exec_output_dir = turn_dir / "exec_output"
                eval_input_dir = turn_dir / "eval_input"
                eval_output_dir = turn_dir / "eval_output"

                meta = json.loads(meta_path.read_text(encoding="utf-8"))

                turns.append(Turn(
                    exec_input_path=exec_input_dir if exec_input_dir.exists() else None,
                    exec_output_path=exec_output_dir if exec_output_dir.exists() else None,
                    eval_input_path=eval_input_dir if eval_input_dir.exists() else None,
                    eval_output_path=eval_output_dir if eval_output_dir.exists() else None,
                    passed=meta.get("passed"),
                    exec_input_manifest=_load_manifest(exec_input_dir, ExecInputManifest),
                    exec_output_manifest=_load_manifest(exec_output_dir, ExecOutputManifest),
                    eval_input_manifest=_load_manifest(eval_input_dir, EvalInputManifest),
                    eval_output_manifest=_load_manifest(eval_output_dir, EvalOutputManifest),
                ))

            episodes[eid] = Episode(
                task_id=task_id,
                sample_id=int(sample_id),
                turns=turns,
                base_dir=ep_dir,
            )

        logger.info("synced %d episodes from %s", len(episodes), output_dir)
        return episodes
