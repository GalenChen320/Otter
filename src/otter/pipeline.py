import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from otter import dataset
from otter import executor
from otter import evaluator
from otter.config.setting import get_settings, get_tracked_config
from otter.episode import Episode, EXPERIMENT_META
from otter.logger import get_logger


def create_dataset() -> dataset.BaseDataset:
    settings = get_settings()
    output_dir = settings.experiment.output_dir
    match settings.dataset.dataset_name:
        case "evalplus": return dataset.EvalPlusDataset(output_dir)
        case "mbppplus": return dataset.MBPPPlusDataset(output_dir)
        case _:
            raise ValueError(f"unknown dataset: {settings.dataset.dataset_name}")


def create_executor() -> executor.BaseExecutor:
    settings = get_settings()
    match settings.executor.executor_type:
        case "chat_llm":
            return executor.ChatLLMExecutor()
        case _:
            raise ValueError(f"unknown executor_type: {settings.executor.executor_type}")


def create_evaluator() -> evaluator.BaseEvaluator:
    settings = get_settings()
    match settings.evaluator.evaluator_type:
        case "docker":
            return evaluator.DockerEvaluator()
        case _:
            raise ValueError(f"unknown evaluator_type: {settings.evaluator.evaluator_type}")


def get_pending_episodes(ds: dataset.BaseDataset) -> list[Episode]:
    """筛选未完成的 Episode。

    将每道题展开 samples_per_problem 份，每份是独立的 Episode。
    已完成的（resolved 或 exhausted）跳过，部分完成的继续。
    """
    settings = get_settings()
    output_dir = settings.experiment.output_dir
    existing = Episode.sync_all(output_dir)
    episodes: list[Episode] = []

    for task_id in ds.task_ids:
        for k in range(settings.experiment.samples_per_problem):
            eid = Episode.make_eid(task_id, k)
            ep_dir = output_dir / eid
            if eid in existing:
                ep = existing[eid]
                if ep.resolved or ep.exhausted(settings.experiment.max_turns):
                    continue
                episodes.append(ep)
            else:
                episodes.append(Episode(task_id=task_id, sample_id=k, base_dir=ep_dir))

    return episodes


async def run_turn(
    ds: dataset.BaseDataset,
    exec_client: executor.BaseExecutor,
    eval_client: evaluator.BaseEvaluator,
    ep: Episode,
    exec_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
) -> None:
    """执行单轮：创建 Turn → prepare → generate → prepare → execute → judge。"""
    logger = get_logger()

    # Step 1: make new turn
    ep.next_turn()
    logger.info("[%s] turn %d queued", ep.eid, ep.total_turns)

    # Step 2: Prpopser run
    # ds.prepare_prop_input(ep)
    # async with prop_semaphore:
    #     logger.info("[%s] turn %d proposing...", ep.eid, ep.total_turns)
    #     await prop_client.run(ep)

    # Step 3: Executor run
    ds.prepare_exec_input(ep)
    async with exec_semaphore:
        logger.info("[%s] turn %d executing...", ep.eid, ep.total_turns)
        await exec_client.run(ep)

    # Step 4: Evaluator run
    ds.prepare_eval_input(ep)
    async with eval_semaphore:
        logger.info("[%s] turn %d evaluating...", ep.eid, ep.total_turns)
        await eval_client.run(ep)

    # Step 5: make judgement
    await ds.make_judgement(ep)
    logger.info("[%s] turn %d completed (passed=%s)",
                ep.eid, ep.total_turns, ep.turns[-1].passed)


async def run(
    ds: dataset.BaseDataset,
    exec_client: executor.BaseExecutor,
    eval_client: evaluator.BaseEvaluator,
) -> list[Episode]:
    settings = get_settings()
    logger = get_logger()
    exec_semaphore = asyncio.Semaphore(settings.executor.concurrency)
    eval_semaphore = asyncio.Semaphore(settings.evaluator.concurrency)
    max_turns = settings.experiment.max_turns
    episodes = get_pending_episodes(ds)
    logger.info("starting run: %d episodes to process", len(episodes))

    async def process(ep: Episode):
        try:
            async with ds.episode_context(ep):
                while not ep.resolved and not ep.exhausted(max_turns):
                    await run_turn(ds, exec_client, eval_client, ep, exec_semaphore, eval_semaphore)
        except Exception as e:
            logger.error("[%s] episode failed: %s", ep.eid, e)

    await asyncio.gather(*[process(ep) for ep in episodes])
    resolved = sum(1 for ep in episodes if ep.resolved)
    logger.info("run finished: %d/%d episodes resolved", resolved, len(episodes))
    return episodes


def verify_or_create_experiment_meta(output_dir) -> None:
    """首次运行写入 experiment.json，续跑时校验 tracked 参数一致性。"""
    logger = get_logger()
    meta_path = output_dir / EXPERIMENT_META
    current = get_tracked_config()

    if not meta_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("created experiment meta: %s", meta_path)
        return

    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    diffs: list[str] = []
    all_keys = set(saved) | set(current)
    for key in sorted(all_keys):
        old_val = saved.get(key)
        new_val = current.get(key)
        if old_val != new_val:
            diffs.append(f"  {key}: {old_val!r} → {new_val!r}")

    if diffs:
        console = Console(stderr=True)
        console.print(
            Panel(
                "\n".join(diffs),
                title="[bold yellow]Config Mismatch[/bold yellow]",
                subtitle=str(meta_path),
                border_style="yellow",
            )
        )
        if Confirm.ask(
            "[yellow]Override experiment meta with current config?[/yellow]",
            default=False,
            console=console,
        ):
            meta_path.write_text(
                json.dumps(current, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("experiment meta overridden: %s", meta_path)
        else:
            raise SystemExit(1)

    logger.info("experiment meta verified: %s", meta_path)


async def main():
    settings = get_settings()
    logger = get_logger()

    verify_or_create_experiment_meta(settings.experiment.output_dir)

    ds = create_dataset()
    exec_client = create_executor()
    eval_client = create_evaluator()

    async with ds.run_context():
        episodes = await run(ds, exec_client, eval_client)
        logger.info("done: %d episodes processed", len(episodes))