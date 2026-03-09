from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass
class Turn:
    turn_number: int
    prompt: str
    response: str = ""
    code: str = ""
    execution_result: ExecutionResult | None = None

    @property
    def passed(self) -> bool:
        return self.execution_result is not None and self.execution_result.passed


@dataclass
class Episode:
    task_id: str
    max_turns: int
    turns: list[Turn] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return len(self.turns) > 0 and self.turns[-1].passed

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def exhausted(self) -> bool:
        return self.total_turns >= self.max_turns
