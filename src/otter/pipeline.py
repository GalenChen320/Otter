import asyncio

from otter import dataset
from otter import llm
from otter.config.setting import get_settings
from otter.episode import Episode
from otter.store import BaseStore


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
    # TODO: 根据 settings 创建具体的 Store 实现
    raise NotImplementedError


def build_episodes(ds: dataset.BaseDataset, store: BaseStore) -> list[Episode]:
    """构建待处理的 Episode 列表。

    将每道题展开 samples_per_problem 份，每份是独立的 Episode。
    eid 格式："{原始task_id}#{sample序号}"。
    已完成的（resolved 或 exhausted）跳过，部分完成的继续。
    """
    settings = get_settings()
    existing = store.load_episodes()
    episodes: list[Episode] = []

    for i in range(len(ds)):
        problem = ds[i]
        for k in range(settings.llm.samples_per_problem):
            eid = f"{problem.task_id}#{k}"
            if eid in existing:
                ep = existing[eid]
                if ep.resolved or ep.exhausted:
                    continue
                episodes.append(ep)
            else:
                episodes.append(Episode(
                    eid=eid,
                    max_turns=settings.experiment.max_turns,
                ))

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
            # TODO: 多轮循环（生成 → 执行 → feedback），每完成一个 Turn 调用 store.save_turn()
            pass

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