import json
import asyncio
from pathlib import Path

from code_eval import datasets
from code_eval import llm
from code_eval.config.setting import settings


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


async def run(dataset: datasets.BaseDataset, llm_client: llm.BaseLLM, result_file: Path):

    finished: set[tuple[str, int]] = set()
    if result_file.exists():
        for line in result_file.read_text().splitlines():
            row = json.loads(line)
            finished.add((row["task_id"], row["sample_idx"]))

    gen_semaphore = asyncio.Semaphore(settings.llm.concurrency)

    async def process(index: int, sample_idx: int):
        problem = dataset[index]

        if (problem.task_id, sample_idx) in finished:
            print(f"skip  task_id={problem.task_id} sample={sample_idx}")
            return

        async with gen_semaphore:
            prompt = dataset.make_prompt(index)
            solution = await llm_client.generate(prompt)

        row = {
            "task_id": problem.task_id,
            "sample_idx": sample_idx,
            "solution": solution,
            "status": "pending",
        }
        with result_file.open("a") as f:
            f.write(json.dumps(row) + "\n")

        print(f"done  task_id={problem.task_id} sample={sample_idx}")

    tasks = [
        process(i, k)
        for i in range(len(dataset))
        for k in range(settings.llm.samples_per_problem)
    ]
    await asyncio.gather(*tasks)


async def main():
    dataset = create_dataset()
    dataset.load()

    llm = create_llm()

    result_file = settings.result.result_dir / f"{settings.dataset.dataset_name.replace('/', '_')}.jsonl"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    await run(dataset, llm, result_file)


if __name__ == "__main__":
    asyncio.run(main())
