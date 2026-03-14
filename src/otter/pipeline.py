import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from otter import dataset
from otter import llm
from otter import environment
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


def create_llm() -> llm.BaseLLM:
    settings = get_settings()
    match settings.llm.llm_type:
        case "openai_compatible":
            return llm.OpenAICompatibleLLM()
        case _:
            raise ValueError(f"unknown llm_type: {settings.llm.llm_type}")


def create_environment() -> environment.BaseEnvironment:
    settings = get_settings()
    match settings.environment.environment_type:
        case "docker":
            return environment.DockerEnvironment()
        case _:
            raise ValueError(f"unknown environment_type: {settings.environment.environment_type}")


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
    llm_client: llm.BaseLLM,
    env_client: environment.BaseEnvironment,
    ep: Episode,
    llm_semaphore: asyncio.Semaphore,
    env_semaphore: asyncio.Semaphore,
) -> None:
    """执行单轮：创建 Turn → prepare → generate → prepare → execute → judge。"""
    logger = get_logger()
    ep.next_turn()

    ds.prepare_llm_input(ep)

    logger.info("[%s] turn %d queued", ep.eid, ep.total_turns)
    async with llm_semaphore:
        logger.info("[%s] turn %d generating", ep.eid, ep.total_turns)
        await llm_client.generate(ep)

    ds.prepare_env_input(ep)
    async with env_semaphore:
        await env_client.execute(ep)
    await ds.make_judgement(ep)

    logger.info("[%s] turn %d completed (passed=%s)",
                ep.eid, ep.total_turns, ep.turns[-1].passed)


async def run(
    ds: dataset.BaseDataset,
    llm_client: llm.BaseLLM,
    env_client: environment.BaseEnvironment,
) -> list[Episode]:
    settings = get_settings()
    logger = get_logger()
    llm_semaphore = asyncio.Semaphore(settings.llm.concurrency)
    env_semaphore = asyncio.Semaphore(settings.environment.concurrency)
    max_turns = settings.experiment.max_turns
    episodes = get_pending_episodes(ds)
    logger.info("starting run: %d episodes to process", len(episodes))

    async def process(ep: Episode):
        try:
            async with ds.episode_context(ep):
                while not ep.resolved and not ep.exhausted(max_turns):
                    await run_turn(ds, llm_client, env_client, ep, llm_semaphore, env_semaphore)
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
    llm_client = create_llm()
    env_client = create_environment()

    async with ds.run_context():
        episodes = await run(ds, llm_client, env_client)
        logger.info("done: %d episodes processed", len(episodes))