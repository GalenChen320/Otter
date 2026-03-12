import asyncio

from otter import dataset
from otter import llm
from otter import environment
from otter.config.setting import get_settings
from otter.episode import Episode
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


def create_environment() -> environment.BaseEnvironment:
    settings = get_settings()
    match settings.environment.environment_type:
        case "docker":
            return environment.DockerEnvironment()
        case _:
            raise ValueError(f"unknown environment_type: {settings.environment.environment_type}")


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
    env_client: environment.BaseEnvironment,
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

            async with ds.episode_context(ep):
                # 1. 创建新 Turn
                ep.next_turn()

                # Step1. 我希望这里准备的输入是跟llm_client的类别无关，
                # 特别的，prepare_input会在input文件夹里面写一个json文件和，其他文件也烦在文件夹里。
                # 不同的llm_client，他们会需要不同的字段，比如对于一般的mbppplus和目前有的llmclient来说，json里面只需要一个messages_file字段/
                # ds.prepare_input(ep)

                # Step2. 这里
                # 2. Dataset 准备 input（写文件 + 设置 turn.input_manifest）
                ds.prepare_input(ep)

                # 3. LLM 生成
                turn = ep.turns[-1]
                response = await llm_client.generate(turn.input_manifest)

                # 4. Dataset 写 response 并构建 ExecSpec
                spec = ds.prepare_exec(ep, response, type(env_client))

                # 5. Environment 执行
                observation = await env_client.execute(spec)

                # 6. Dataset 判定
                await ds.make_judgement(ep, observation)

                # 7. 保存 meta（标记 turn 完成）
                store.save_meta(ep)

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

    env_client = create_environment()
    store = create_store()

    async with ds.run_context(settings.experiment.output_dir):
        episodes = await run(ds, llm_client, env_client, store)
        logger.info("done: %d episodes processed", len(episodes))