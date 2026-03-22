import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from otter import dataset
from otter.backend import create_backend
from otter.role import BaseRole, ProposerRole, ExecutorRole, EvaluatorRole
from otter.config.setting import get_settings, get_tracked_config
from otter.episode import Episode, EXPERIMENT_META
from otter.logger import get_logger


def create_dataset() -> dataset.BaseDataset:
    settings = get_settings()
    output_dir = settings.output_dir
    match settings.dataset_name:
        case "evalplus": return dataset.EvalPlusDataset(output_dir)
        case "mbppplus": return dataset.MBPPPlusDataset(output_dir)
        case "sweci": return dataset.SWECIDataset(output_dir)
        case _:
            raise ValueError(f"unknown dataset: {settings.dataset_name}")


def create_role(
    role_cls: type[BaseRole],
    backend_type: str | None,
    backend_settings,
) -> BaseRole | None:
    """根据 type 和 settings 创建 Role，type 为 None 时返回 None。"""
    if backend_type is None:
        return None
    backend = create_backend(backend_type, backend_settings)
    return role_cls(backend)


def get_pending_episodes(ds: dataset.BaseDataset) -> list[Episode]:
    """筛选未完成的 Episode。

    将每道题展开 samples_per_problem 份，每份是独立的 Episode。
    已完成的（resolved 或 exhausted）跳过，部分完成的继续。
    """
    settings = get_settings()
    output_dir = settings.output_dir
    existing = Episode.sync_all(output_dir)
    episodes: list[Episode] = []

    for task_id in ds.task_ids:
        for k in range(settings.samples_per_problem):
            eid = Episode.make_eid(task_id, k)
            ep_dir = output_dir / eid
            if eid in existing:
                ep = existing[eid]
                if ep.resolved or ep.exhausted(settings.max_turns):
                    continue
                episodes.append(ep)
            else:
                episodes.append(Episode(task_id=task_id, sample_id=k, base_dir=ep_dir))

    return episodes


async def run_turn(
    ds: dataset.BaseDataset,
    ep: Episode,
    prop_client: ProposerRole | None = None,
    exec_client: ExecutorRole | None = None,
    eval_client: EvaluatorRole | None = None,
    prop_semaphore: asyncio.Semaphore | None = None,
    exec_semaphore: asyncio.Semaphore | None = None,
    eval_semaphore: asyncio.Semaphore | None = None,
) -> None:
    """执行单轮：创建 Turn → [propose] → [generate] → [execute] → judge。"""
    logger = get_logger()
    settings = get_settings()

    # Step 1: make new turn
    ep.next_turn()
    logger.info("[%s] turn %d queued", ep.eid, ep.total_turns)

    # Step 2: Proposer run (optional)
    if prop_client is not None:
        ds.prepare_prop_input(ep)
        for attempt in range(1, settings.proposer_retry + 1):
            async with prop_semaphore:
                logger.info("[%s] turn %d proposing (attempt %d/%d)...",
                            ep.eid, ep.total_turns, attempt, settings.proposer_retry)
                manifest = await prop_client.run(ep)
            if ds.validate_prop_output(ep, manifest):
                break
            logger.warning("[%s] turn %d proposer output rejected (attempt %d/%d)",
                           ep.eid, ep.total_turns, attempt, settings.proposer_retry)
            ep.archive_last_output("attempt_{attempt}")
        else:
            raise RuntimeError(f"[{ep.eid}] proposer failed after {settings.proposer_retry} attempts")

    # Step 3: Executor run (optional)
    if exec_client is not None:
        ds.prepare_exec_input(ep)
        for attempt in range(1, settings.executor_retry + 1):
            async with exec_semaphore:
                logger.info("[%s] turn %d executing (attempt %d/%d)...",
                            ep.eid, ep.total_turns, attempt, settings.executor_retry)
                manifest = await exec_client.run(ep)
            if ds.validate_exec_output(ep, manifest):
                break
            logger.warning("[%s] turn %d executor output rejected (attempt %d/%d)",
                           ep.eid, ep.total_turns, attempt, settings.executor_retry)
            ep.archive_last_output("attempt_{attempt}")
        else:
            raise RuntimeError(f"[{ep.eid}] executor failed after {settings.executor_retry} attempts")

    # Step 4: Evaluator run (optional)
    if eval_client is not None:
        ds.prepare_eval_input(ep)
        for attempt in range(1, settings.evaluator_retry + 1):
            async with eval_semaphore:
                logger.info("[%s] turn %d evaluating (attempt %d/%d)...",
                            ep.eid, ep.total_turns, attempt, settings.evaluator_retry)
                manifest = await eval_client.run(ep)
            if ds.validate_eval_output(ep, manifest):
                break
            logger.warning("[%s] turn %d evaluator output rejected (attempt %d/%d)",
                           ep.eid, ep.total_turns, attempt, settings.evaluator_retry)
            ep.archive_last_output("attempt_{attempt}")
        else:
            raise RuntimeError(f"[{ep.eid}] evaluator failed after {settings.evaluator_retry} attempts")

    # Step 5: make judgement
    await ds.make_judgement(ep)
    logger.info("[%s] turn %d completed (passed=%s)",
                ep.eid, ep.total_turns, ep.turns[-1].passed)


async def run(
    ds: dataset.BaseDataset,
    prop_client: ProposerRole | None = None,
    exec_client: ExecutorRole | None = None,
    eval_client: EvaluatorRole | None = None,
) -> list[Episode]:
    settings = get_settings()
    logger = get_logger()

    prop_semaphore = (
        asyncio.Semaphore(settings.proposer_concurrency)
        if settings.proposer_type is not None else None
    )
    exec_semaphore = (
        asyncio.Semaphore(settings.executor_concurrency)
        if settings.executor_type is not None else None
    )
    eval_semaphore = (
        asyncio.Semaphore(settings.evaluator_concurrency)
        if settings.evaluator_type is not None else None
    )

    max_turns = settings.max_turns
    episodes = get_pending_episodes(ds)
    logger.info("starting run: %d episodes to process", len(episodes))

    async def process(ep: Episode):
        try:
            async with ds.episode_context(ep):
                while not ep.resolved and not ep.exhausted(max_turns):
                    await run_turn(
                        ds, ep,
                        prop_client, exec_client, eval_client,
                        prop_semaphore, exec_semaphore, eval_semaphore,
                    )
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

    verify_or_create_experiment_meta(settings.output_dir)

    ds = create_dataset()
    prop_client = create_role(ProposerRole, settings.proposer_type, settings.proposer)
    exec_client = create_role(ExecutorRole, settings.executor_type, settings.executor)
    eval_client = create_role(EvaluatorRole, settings.evaluator_type, settings.evaluator)

    async with ds.run_context():
        episodes = await run(ds, prop_client, exec_client, eval_client)
        logger.info("done: %d episodes processed", len(episodes))


__all__ = ["main"]