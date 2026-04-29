import json
from pathlib import Path

from otter.episode import Episode, EXPERIMENT_META
from otter.config.setting import get_settings


def _load_episodes(experiment_dir: Path) -> list[dict]:
    """从实验目录读取所有 episode 的轮次数据。"""
    synced = Episode.sync_all(experiment_dir)
    episodes = []
    for eid in sorted(synced):
        ep = synced[eid]
        turns = [t.is_solved for t in ep.turns]
        if turns:
            episodes.append({
                "task_id": ep.task_id,
                "sample_id": ep.sample_id,
                "turns": turns,
            })
    return episodes


def _compute_turn_stats(episodes: list[dict], max_turns: int) -> list[dict]:
    """计算截止到每一轮的累积统计。"""
    total = len(episodes)
    stats = []

    for k in range(1, max_turns + 1):
        passed = 0
        completed = 0

        for ep in episodes:
            turns = ep["turns"]
            resolved_at = None
            for i, p in enumerate(turns):
                if p is True:
                    resolved_at = i + 1
                    break

            if resolved_at is not None and resolved_at <= k:
                passed += 1
                completed += 1
            elif len(turns) <= k:
                completed += 1

        pending = total - completed
        stats.append({
            "turn": k,
            "total": total,
            "passed": passed,
            "completed": completed,
            "pending": pending,
        })

    return stats


def show_evalplus_summary():
    """读取 settings，加载数据，用 rich 展示 evalplus / mbppplus 的实验结果。"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box as rich_box

    settings = get_settings()
    dataset_name = settings.dataset_name
    exp_id = settings.experiment_id
    max_turns = settings.max_turns
    num_sample = settings.samples_per_problem

    experiment_dir = Path("experiments") / exp_id

    episodes = _load_episodes(experiment_dir)

    # 按 sample_id 分组
    sample_ids = sorted(set(ep["sample_id"] for ep in episodes))

    console = Console()

    # Meta 信息
    meta_text = Text()
    meta_text.append("exp_id: ", style="bold cyan")
    meta_text.append(f"{exp_id}\n")
    meta_text.append("dataset: ", style="bold cyan")
    meta_text.append(f"{dataset_name}\n")
    meta_text.append("max_turns: ", style="bold cyan")
    meta_text.append(f"{max_turns}\n")
    meta_text.append("num_sample: ", style="bold cyan")
    meta_text.append(f"{num_sample}\n")
    meta_text.append("episodes: ", style="bold cyan")
    meta_text.append(f"{len(episodes)}")
    console.print(Panel(meta_text, title="Meta", border_style="blue"))

    # 总体统计
    total = len(episodes)
    resolved = sum(1 for ep in episodes if any(p is True for p in ep["turns"]))
    stat_text = Text()
    stat_text.append("total: ", style="bold cyan")
    stat_text.append(f"{total}\n")
    stat_text.append("resolved: ", style="bold cyan")
    stat_text.append(f"{resolved}\n")
    stat_text.append("pass_rate: ", style="bold cyan")
    stat_text.append(f"{resolved / total:.4f}" if total else "N/A")
    console.print(Panel(stat_text, title="Stat", border_style="green"))

    # 每个 sample 的 Turn Stat 表格
    for sid in sample_ids:
        group = [ep for ep in episodes if ep["sample_id"] == sid]
        turn_stats = _compute_turn_stats(group, max_turns)

        title = f"Turn Stat (sample {sid})" if len(sample_ids) > 1 else "Turn Stat"
        table = Table(title=title, show_lines=True, border_style="magenta", box=rich_box.ROUNDED)
        table.add_column("Turn", justify="center", style="bold")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Completed", justify="right", style="blue")
        table.add_column("Pending", justify="right")

        for ts in turn_stats:
            total_n = ts["total"]
            pass_rate = ts["passed"] / total_n if total_n else 0
            comp_rate = ts["completed"] / total_n if total_n else 0
            pend_rate = ts["pending"] / total_n if total_n else 0

            passed_str = f"{ts['passed']}/{total_n} ({pass_rate:.2%})"
            completed_str = f"{ts['completed']}/{total_n} ({comp_rate:.2%})"
            pending_str = f"{ts['pending']}/{total_n} ({pend_rate:.2%})"

            pending_style = "dim" if ts["pending"] == 0 else "yellow"
            table.add_row(
                str(ts["turn"]),
                passed_str,
                completed_str,
                f"[{pending_style}]{pending_str}[/]",
            )

        console.print(table)