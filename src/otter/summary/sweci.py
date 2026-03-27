import json
import pandas as pd
from pathlib import Path
from itertools import chain
from otter.episode import Episode
from collections import Counter, defaultdict

# ── Rich 可视化 ──
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from otter.config.setting import get_settings



def evoscore(relative_changes, gamma):
    assert gamma > 0, "gamma must be a positive number."
    weights = []
    weighted_rc = []
    w = 1.
    for rc in relative_changes:
        weights.append(w)
        weighted_rc.append(w * rc)
        w *= gamma
    return sum(weighted_rc) / sum(weights)


def load_episode_result(ep_dir: Path):
    # Step 1 验证合规性
    if not ep_dir.is_dir():
        raise NotADirectoryError(f"{ep_dir} is not a directory.")
    meta = json.loads((ep_dir / "meta.json").read_text(encoding='utf-8'))
    turn_number = sorted([int(d.stem[5:]) for d in ep_dir.iterdir() if d.is_dir()])
    if not turn_number:
        meta["turns"] = []
        return meta
    is_consecutive = turn_number == list(range(turn_number[0], turn_number[-1] + 1))
    if not is_consecutive:
        raise RuntimeError("Turn directories are not consecutive.")
    turns = []
    for n in turn_number[:-1]:
        conclu_path = ep_dir / f"turn_{n}" / "conclusion.json"
        turns.append(json.loads(conclu_path.read_text(encoding='utf-8')))
    last_conclu = ep_dir / f"turn_{turn_number[-1]}" / "conclusion.json"
    if last_conclu.is_file():
        turns.append(json.loads(last_conclu.read_text(encoding='utf-8')))
    meta["turns"] = turns
    return meta


def parse_results(results: dict, max_turns: int):
    assert max_turns > 0, "max_turns must be positive."
    n_turns = len(results["turns"])

    pass_seq = [t["num_passed"] for t in results["turns"]]
    coll_seq = [t["is_collapsed"] for t in results["turns"]]

    if n_turns == 0:
        progress = 0
    elif (n_turns == max_turns) or results["turns"][-1]["is_solved"]:
        progress = 1
    else:
        progress = n_turns / max_turns

    len_pad = max_turns - n_turns
    pad_pass = pass_seq[-1] if n_turns > 0 else results["base_passed"]
    pass_seq = pass_seq[:max_turns] + [pad_pass] * len_pad
    coll_seq = coll_seq[:max_turns] + [False] * len_pad

    prev_idx_list = []
    last_success = -1
    for i in range(max_turns):
        prev_idx_list.append(last_success)
        if coll_seq[i] is False:
            last_success = i

    num_regress = 0
    regress_time = {}
    regress_amplitudes = []

    for idx, prev_idx in enumerate(prev_idx_list):
        prev_pass = results["base_passed"] if prev_idx == -1 else pass_seq[prev_idx]
        if pass_seq[idx] < prev_pass:
            num_regress += 1
            regress_time[idx] = 1
            amplitude = (prev_pass - pass_seq[idx]) / prev_pass if prev_pass != 0 else 0
            regress_amplitudes.append(amplitude) 

    pass_seq = [min(max(0, p), results["target_passed"]) for p in pass_seq]

    rela_changes = []
    for n_pass in pass_seq:
        if n_pass >= results["base_passed"]:
            target_gap = results["target_passed"] - results["base_passed"]
            if target_gap != 0:
                rc = (n_pass - results["base_passed"]) / target_gap
            else:
                rc = 0
        else:
            rc = (n_pass - results["base_passed"]) / results["base_passed"] if results["base_passed"] else -1.0
        rela_changes.append(rc)
    
    if rela_changes:
        max_rc_sofar = [rela_changes[0]]
        for rc in rela_changes[1:]:
            max_rc_sofar.append(max(max_rc_sofar[-1], rc))
    else:
        max_rc_sofar = []


    return {
        "n_turns": n_turns,
        "progress": progress,
        "solved": (n_turns > 0) and (pass_seq[-1] == results["target_passed"]),

        "relative_changes": rela_changes,
        "max_rc_sofar":max_rc_sofar,

        "num_regress": num_regress,
        "regress_count": regress_time,
        "regress_amplitudes": regress_amplitudes,

        "evoscore_0.8": evoscore(rela_changes, 0.8),
        "evoscore_0.9": evoscore(rela_changes, 0.9),
        "evoscore_1.0": evoscore(rela_changes, 1.0),
        "evoscore_1.1": evoscore(rela_changes, 1.1),
        "evoscore_1.2": evoscore(rela_changes, 1.2),
    }


def average_result(results_list:list[dict]) -> dict:
    
    def _list_avg(lst: list[list[float]]) -> list[float]:
        if not lst:
            return []
        return [sum(col) / len(col) for col in zip(*lst)]

    lens = len(results_list)
    lst = results_list

    return {
        "n_turns": sum(r["n_turns"] for r in lst) / lens,
        "progress": sum(r["progress"] for r in lst) / lens,
        "solved": sum(int(r["solved"]) for r in lst) / lens,

        "relative_changes": _list_avg([r["relative_changes"] for r in results_list]),
        "max_rc_sofar": _list_avg([r["max_rc_sofar"] for r in results_list]),

        "num_regress": sum(r["num_regress"] for r in lst),
        "regress_count": list(chain.from_iterable(r["regress_count"] for r in lst)),
        "regress_amplitudes": list(chain.from_iterable(r["regress_amplitudes"] for r in lst)),

        "evoscore_0.8": sum(r["evoscore_0.8"] for r in lst) / lens,
        "evoscore_0.9": sum(r["evoscore_0.9"] for r in lst) / lens,
        "evoscore_1.0": sum(r["evoscore_1.0"] for r in lst) / lens,
        "evoscore_1.1": sum(r["evoscore_1.1"] for r in lst) / lens,
        "evoscore_1.2": sum(r["evoscore_1.2"] for r in lst) / lens,
    }


def show_sweci_summary():

    def _count_cumulative(A: list[int], max_val: int) -> list[int]:
        counter = Counter(A)
        result = []
        cumsum = 0
        for i in range(max_val, 0, -1):
            cumsum += counter.get(i, 0)
            result.append(cumsum)
        result.reverse()
        return result

    settings = get_settings()
    assert settings.dataset_name == "sweci"
    
    exp_id = settings.experiment_id
    dataset = settings.dataset_name
    splitting = settings.dataset.splitting
    agent = settings.dataset.agent_name
    llm = settings.dataset.agent_model_name
    num_sample = settings.samples_per_problem
    max_turns = settings.max_turns

    meta_csv = settings.dataset.cache_dir / "metadata" / f"{splitting}.csv"
    task_ids = pd.read_csv(meta_csv, usecols=[0], header=0).iloc[:, 0].tolist()
    exp_dir = Path("experiments") / exp_id

    results = []
    for task_id in task_ids:
        for i in range(num_sample):
            eid = Episode.make_eid(task_id, i)
            try: 
                raw = load_episode_result(exp_dir / eid)
                result = parse_results(raw, max_turns)
                results.append(result)
            except Exception as e:
                print(str(e))

    avg = average_result(results)

    # 每一轮的平均回退率
    all_turns = [r["n_turns"] for r in results]
    active_counts = _count_cumulative(all_turns, max_turns)
    regress_count_by_turn = Counter(avg["regress_count"])
    regress_counts_aligned = [regress_count_by_turn.get(i, 0) for i in range(max_turns)]
    avg_regress_rate_by_turn = [
        cnt / active if active > 0 else 0
        for cnt, active in zip(regress_counts_aligned, active_counts)
    ]

    # 计算每一轮回退的回退幅度

    regress_amplitude_by_turn = defaultdict(list)
    for turn, amplitude in zip(avg["regress_count"], avg["regress_amplitudes"]):
        regress_amplitude_by_turn[turn].append(amplitude)
    avg_regress_amplitude_by_turn = [
        sum(regress_amplitude_by_turn[i]) / len(regress_amplitude_by_turn[i]) if regress_amplitude_by_turn[i] else 0
        for i in range(max_turns)
    ]
                
    meta = {
        "exp_id": exp_id,
        "dataset": dataset,
        "splitting": splitting,
        "agent": agent,
        "llm": llm,
        "num_sample": num_sample,
    }

    progress = avg["progress"]

    stat = {
        "n_turns": avg["n_turns"],
        "solved": avg["solved"],
        "evoscore_0.8": avg["evoscore_0.8"],
        "evoscore_0.9": avg["evoscore_0.9"],
        "evoscore_1.0": avg["evoscore_1.0"],
        "evoscore_1.1": avg["evoscore_1.1"],
        "evoscore_1.2": avg["evoscore_1.2"],                      
    }

    turn_stat = {
        "regress_rate": avg_regress_rate_by_turn,
        "regress_amplitude": avg_regress_amplitude_by_turn,
        "relative_changes": avg["relative_changes"],
        "max_RC": avg["max_rc_sofar"]
    }

    console = Console()

    # Progress 进度（单独醒目展示）
    console.print()
    console.print(f"  [bold]Experiment progress:[/bold] [bold yellow]{progress:.2%}[/bold yellow]")
    console.print()

    # Meta 信息
    meta_text = Text()
    for k, v in meta.items():
        meta_text.append(f"{k}: ", style="bold cyan")
        meta_text.append(f"{v}\n")
    console.print(Panel(meta_text, title="Meta", border_style="blue"))

    # Stat 统计
    stat_text = Text()
    for k, v in stat.items():
        stat_text.append(f"{k}: ", style="bold cyan")
        stat_text.append(f"{v:.4f}\n")
    console.print(Panel(stat_text, title="Stat", border_style="green"))

    # Turn Stat 表格
    table = Table(title="Turn Stat", show_lines=True, border_style="magenta")
    table.add_column("Turn", justify="center", style="bold")
    table.add_column("Regress Rate", justify="right")
    table.add_column("Regress Amp", justify="right")
    table.add_column("Relative Change", justify="right")
    table.add_column("Max RC", justify="right")

    for i in range(max_turns):
        table.add_row(
            str(i + 1),
            f"{turn_stat['regress_rate'][i]:.4f}",
            f"{turn_stat['regress_amplitude'][i]:.4f}",
            f"{turn_stat['relative_changes'][i]:.4f}",
            f"{turn_stat['max_RC'][i]:.4f}",
        )

    console.print(table)


__all__ = [
    "show_sweci_summary"
]