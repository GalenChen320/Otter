from dataclasses import dataclass


@dataclass
class Result:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


__all__ = [
    "Result",
]
