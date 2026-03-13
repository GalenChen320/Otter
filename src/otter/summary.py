import json
from dataclasses import dataclass, field
from pathlib import Path


EXPERIMENT_META = "experiment.json"


@dataclass
class EpisodeRecord:
    """单个 episode 的轮次记录。"""
    task_id: str
    sample_id: int
    turns: list[bool | None] = field(default_factory=list)  # 每轮的 passed 值

    @property
    def eid(self) -> str:
        return f"{self.task_id}#{self.sample_id}"

    @property
    def resolved(self) -> bool:
        return any(p is True for p in self.turns)

    @property
    def resolved_at(self) -> int | None:
        """在第几轮通过的（1-based），未通过返回 None。"""
        for i, p in enumerate(self.turns):
            if p is True:
                return i + 1
        return None

    @property
    def total_turns(self) -> int:
        return len(self.turns)


@dataclass
class TurnStats:
    """截止到第 k 轮的累积统计。"""
    turn: int
    total: int
    passed: int
    completed: int  # 有最终结论的：通过了，或用完了所有轮次
    pending: int    # 未完成的

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def completed_rate(self) -> float:
        return self.completed / self.total if self.total else 0.0

    @property
    def pending_rate(self) -> float:
        return self.pending / self.total if self.total else 0.0


@dataclass
class SampleSummary:
    """单个 sample_id 的统计结果。"""
    sample_id: int
    episodes: list[EpisodeRecord]
    turn_stats: list[TurnStats]  # 从第 1 轮到第 max_turns 轮


@dataclass
class ExperimentSummary:
    """整个实验的统计结果。"""
    experiment_id: str
    config: dict | None
    max_turns: int
    samples: list[SampleSummary]


def _load_episodes(experiment_dir: Path) -> list[EpisodeRecord]:
    """从实验目录读取所有 episode 的轮次数据。"""
    episodes: list[EpisodeRecord] = []

    for ep_dir in sorted(experiment_dir.iterdir()):
        if not ep_dir.is_dir() or "#" not in ep_dir.name:
            continue

        task_id, sample_id_str = ep_dir.name.rsplit("#", 1)
        sample_id = int(sample_id_str)

        turns: list[bool | None] = []
        turn_idx = 1
        while True:
            meta_path = ep_dir / f"turn_{turn_idx}" / "meta.json"
            if not meta_path.exists():
                break
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            turns.append(meta.get("passed"))
            turn_idx += 1

        if turns:
            episodes.append(EpisodeRecord(
                task_id=task_id,
                sample_id=sample_id,
                turns=turns,
            ))

    return episodes


def _compute_turn_stats(
    episodes: list[EpisodeRecord],
    max_turns: int,
) -> list[TurnStats]:
    """计算截止到每一轮的累积统计。"""
    total = len(episodes)
    stats: list[TurnStats] = []

    for k in range(1, max_turns + 1):
        passed = 0
        completed = 0

        for ep in episodes:
            # 截止到第 k 轮是否通过（累积）
            resolved_at = ep.resolved_at
            if resolved_at is not None and resolved_at <= k:
                passed += 1
                completed += 1
            elif ep.total_turns <= k:
                # 已经用完了所有轮次（跑了 <= k 轮且未通过）
                completed += 1

        stats.append(TurnStats(
            turn=k,
            total=total,
            passed=passed,
            completed=completed,
            pending=total - completed,
        ))

    return stats


def summarize(experiment_dir: Path) -> ExperimentSummary:
    """从实验目录读取数据，生成统计摘要。"""
    experiment_id = experiment_dir.name

    # 读取实验配置（可选）
    config_path = experiment_dir / EXPERIMENT_META
    config = None
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    episodes = _load_episodes(experiment_dir)

    # 推断 max_turns：从 config 读取，或从数据中推断
    if config and "experiment.max_turns" in config:
        max_turns = config["experiment.max_turns"]
    else:
        max_turns = max((ep.total_turns for ep in episodes), default=1)

    # 按 sample_id 分组
    sample_ids = sorted(set(ep.sample_id for ep in episodes))
    samples: list[SampleSummary] = []

    for sid in sample_ids:
        group = [ep for ep in episodes if ep.sample_id == sid]
        turn_stats = _compute_turn_stats(group, max_turns)
        samples.append(SampleSummary(
            sample_id=sid,
            episodes=group,
            turn_stats=turn_stats,
        ))

    return ExperimentSummary(
        experiment_id=experiment_id,
        config=config,
        max_turns=max_turns,
        samples=samples,
    )


def show_summary(result: ExperimentSummary) -> None:
    """用 rich 输出 ExperimentSummary。"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box as rich_box

    console = Console()

    # 实验信息
    header = Text()
    header.append("Experiment: ", style="bold")
    header.append(f"{result.experiment_id}\n")
    header.append("Max turns:  ", style="bold")
    header.append(f"{result.max_turns}")
    console.print(Panel(header, border_style="blue", expand=False))

    for sample in result.samples:
        if len(result.samples) > 1:
            console.print()
            console.print(f"[bold cyan]── Sample {sample.sample_id} ──[/]")

        table = Table(show_header=True, header_style="bold", box=rich_box.ROUNDED, padding=(0, 2))
        table.add_column("Turn",      justify="center", style="bold")
        table.add_column("Passed",    justify="right",  style="green")
        table.add_column("Completed", justify="right",  style="blue")
        table.add_column("Pending",   justify="right")  # 行内动态控制样式

        for ts in sample.turn_stats:
            passed    = f"{ts.passed}/{ts.total} ({ts.pass_rate:.2%})"
            completed = f"{ts.completed}/{ts.total} ({ts.completed_rate:.2%})"
            pending   = f"{ts.pending}/{ts.total} ({ts.pending_rate:.2%})"

            pending_style = "dim" if ts.pending == 0 else "yellow"
            table.add_row(
                str(ts.turn),
                passed,
                completed,
                f"[{pending_style}]{pending}[/]",
            )

        console.print(table)
        console.print()
