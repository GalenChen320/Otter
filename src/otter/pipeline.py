import asyncio

from otter import dataset
from otter import llm
from otter.config.setting import get_settings
from otter.episode import Episode, Turn
from otter.store import BaseStore, LineStore


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


def create_store() -> BaseStore:
    settings = get_settings()
    match settings.dataset.resolved_store_type:
        case "line":
            return LineStore(
                output_dir=settings.experiment.output_dir,
            )
        case "dir":
            raise NotImplementedError("DirStore is not implemented yet")
        case _:
            raise ValueError(f"unknown store_type: {settings.dataset.resolved_store_type}")


def build_episodes(ds: dataset.BaseDataset, store: BaseStore) -> list[Episode]:
    """构建待处理的 Episode 列表。

    将每道题展开 samples_per_problem 份，每份是独立的 Episode。
    已完成的（resolved 或 exhausted）跳过，部分完成的继续。
    """
    settings = get_settings()
    existing = store.load_episodes()
    episodes: list[Episode] = []

    for task_id in ds.task_ids:
        for k in range(settings.llm.samples_per_problem):
            eid = Episode.make_eid(task_id, k)
            if eid in existing:
                ep = existing[eid]
                if ep.resolved or ep.exhausted(settings.experiment.max_turns):
                    continue
                episodes.append(ep)
            else:
                episodes.append(Episode(task_id=task_id, sample_id=k))

    return episodes


async def run(
    ds: dataset.BaseDataset,
    llm_client: llm.BaseLLM,
    store: BaseStore,
) -> list[Episode]:
    settings = get_settings()
    gen_semaphore = asyncio.Semaphore(settings.llm.concurrency)
    episodes = build_episodes(ds, store)

    async def process(ep: Episode):
        async with gen_semaphore:
            messages = ds.make_messages(ep)
            response = await llm_client.generate(messages, task_id=ep.task_id)
            turn = Turn(
                turn_number=ep.total_turns + 1,
                prompt=messages[-1]["content"],
                response=response,
            )
            ep.turns.append(turn)
            await store.save_turn(ep, turn)

    await asyncio.gather(*[process(ep) for ep in episodes])
    return episodes


async def main():
    ds = create_dataset()
    ds.load()

    llm_client = create_llm()
    if not await llm_client.ping():
        raise SystemExit("LLM ping failed, check your config")

    store = create_store()
    episodes = await run(ds, llm_client, store)
    print(f"processed {len(episodes)} episodes")