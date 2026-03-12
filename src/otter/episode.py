from dataclasses import dataclass, field
from pathlib import Path


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
