from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionObservation:
    """环境执行后的原始观测结果。
    """
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False


@dataclass
class Turn:
    input_path: Path | None = None
    response_path: Path | None = None
    observation_path: Path | None = None
    passed: bool | None = None


@dataclass
class Episode:
    task_id: str
    sample_id: int
    turns: list[Turn] = field(default_factory=list)

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
