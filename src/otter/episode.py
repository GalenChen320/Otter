from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionResult":
        return cls(
            passed=d["passed"],
            stdout=d.get("stdout", ""),
            stderr=d.get("stderr", ""),
            timed_out=d.get("timed_out", False),
        )


@dataclass
class Turn:
    turn_number: int
    prompt: str
    response: str = ""
    execution_result: ExecutionResult | None = None

    @property
    def passed(self) -> bool:
        return self.execution_result is not None and self.execution_result.passed

    def to_dict(self) -> dict:
        d: dict = {
            "turn_number": self.turn_number,
            "prompt": self.prompt,
            "response": self.response,
        }
        if self.execution_result is not None:
            d.update(self.execution_result.to_dict())
        else:
            d["passed"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        er = None
        if d.get("passed") is not None:
            er = ExecutionResult.from_dict(d)
        return cls(
            turn_number=d["turn_number"],
            prompt=d["prompt"],
            response=d.get("response", ""),
            execution_result=er,
        )


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
        return len(self.turns) > 0 and self.turns[-1].passed

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    def exhausted(self, max_turns: int) -> bool:
        return self.total_turns >= max_turns
