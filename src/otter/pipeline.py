import asyncio

from otter import datasets
from otter import llm
from otter.config.setting import settings


def create_dataset() -> datasets.BaseDataset:
    match settings.dataset.dataset_name:
        case "humaneval": return datasets.HumanEvalDataset()
        case "apps": return datasets.APPSDataset()
        case "mbppplus": return datasets.MBPPPlusDataset()
        case _:
            raise ValueError(f"unknown dataset: {settings.dataset.dataset_name}")


def create_llm() -> llm.BaseLLM:
    match settings.llm.llm_type:
        case "openai_compatible":
            return llm.OpenAICompatibleLLM()
        case _:
            raise ValueError(f"unknown llm: {settings.llm.llm_type}")


async def run(dataset: datasets.BaseDataset, llm_client: llm.BaseLLM) -> list[dict]:

    results: list[dict] = []
    gen_semaphore = asyncio.Semaphore(settings.llm.concurrency)

    async def process(index: int, sample_idx: int):
        problem = dataset[index]

        async with gen_semaphore:
            prompt = dataset.make_prompt(index)
            solution = await llm_client.generate(prompt, task_id=f"{problem.task_id}#{sample_idx}")

        results.append({
            "task_id": problem.task_id,
            "sample_idx": sample_idx,
            "solution": solution,
        })

    tasks = [
        process(i, k)
        for i in range(len(dataset))
        for k in range(settings.llm.samples_per_problem)
    ]
    await asyncio.gather(*tasks)
    return results


async def main():
    dataset = create_dataset()
    dataset.load()

    llm_client = create_llm()
    if not await llm_client.ping():
        raise SystemExit("LLM ping failed, check your config")

    results = await run(dataset, llm_client)
    print(f"generated {len(results)} solutions")


if __name__ == "__main__":
    asyncio.run(main())
