import asyncio

from otter import dataset
from otter import llm
from otter.config.setting import get_settings
from otter.episode import Episode
from otter.environment.docker import DockerEnvironment
from otter.logger import get_logger
from otter.store import Store


def create_dataset() -> dataset.BaseDataset:
    settings = get_settings()
    match settings.dataset.dataset_name:
        case "humaneval": return dataset.HumanEvalDataset()
        case "apps": return dataset.APPSDataset()
        case "mbppplus": return dataset.MBPPPlusDataset()
        case _:
            raise ValueError(f"unknown dataset: {settings.dataset.dataset_name}")


def create_llm() -> llm.BaseLLM:
    settings = get_settings()
    match settings.llm.response_format:
        case "openai_compatible":
            return llm.OpenAICompatibleLLM()
        case _:
            raise ValueError(f"unknown response_format: {settings.llm.response_format}")


def create_store() -> Store:
    settings = get_settings()
    return Store(output_dir=settings.experiment.output_dir)


def get_pending_episodes(ds: dataset.BaseDataset, store: Store) -> list[Episode]:
    """筛选未完成的 Episode。

    将每道题展开 samples_per_problem 份，每份是独立的 Episode。
    已完成的（resolved 或 exhausted）跳过，部分完成的继续。
    """
    settings = get_settings()
    existing = store.sync_episodes()
    episodes: list[Episode] = []

    for task_id in ds.task_ids:
        for k in range(settings.llm.samples_per_problem):
            eid = Episode.make_eid(task_id, k)
            ep_dir = ds.base_dir / eid
            if eid in existing:
                ep = existing[eid]
                if ep.resolved or ep.exhausted(settings.experiment.max_turns):
                    continue
                episodes.append(ep)
            else:
                episodes.append(Episode(task_id=task_id, sample_id=k, base_dir=ep_dir))

    return episodes


async def run(
    ds: dataset.BaseDataset,
    llm_client: llm.BaseLLM,
    store: Store,
) -> list[Episode]:
    settings = get_settings()
    logger = get_logger()
    gen_semaphore = asyncio.Semaphore(settings.llm.concurrency)
    episodes = get_pending_episodes(ds, store)
    logger.info("starting run: %d episodes to process", len(episodes))

    async def process(ep: Episode):
        async with gen_semaphore:
            logger.info("[%s] processing (turn %d)", ep.eid, ep.total_turns + 1)

            await ds.setup_episode(ep)

            # 1. 创建新 Turn
            ep.next_turn()

            # 2. Dataset 写 input
            ds.write_input(ep)

            # 3. Dataset 构建 LLM 输入
            llm_input = ds.prepare_llm_input(ep)

            # 4. LLM 生成
            response = await llm_client.generate(llm_input, eid=ep.eid)

            # 5. Dataset 写 response
            ds.write_response(ep, response)

            # 6. Dataset 构建 ExecSpec
            spec = ds.to_exec_spec(ep)

            # 7. Environment 执行
            observation = await DockerEnvironment.execute(spec)

            # 8. Dataset 写 observation
            ds.write_observation(ep, observation)

            # 9. Dataset 判定
            ep.turns[-1].passed = ds.judge(ep, observation)

            # 10. 保存 meta（标记 turn 完成）
            store.save_meta(ep)

            await ds.teardown_episode(ep)

            logger.info("[%s] turn %d completed (passed=%s)",
                        ep.eid, ep.total_turns, ep.turns[-1].passed)

    await asyncio.gather(*[process(ep) for ep in episodes])
    resolved = sum(1 for ep in episodes if ep.resolved)
    logger.info("run finished: %d/%d episodes resolved", resolved, len(episodes))
    return episodes


async def main():
    settings = get_settings()
    logger = get_logger()
    ds = create_dataset()
    ds.load()

    llm_client = create_llm()
    if not await llm_client.ping():
        raise SystemExit("LLM ping failed, check your config")

    store = create_store()

    await ds.setup(settings.experiment.output_dir)
    try:
        episodes = await run(ds, llm_client, store)
        logger.info("done: %d episodes processed", len(episodes))
    finally:
        await ds.teardown()