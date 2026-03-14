from dataclasses import dataclass
from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.dataset.utils import extract_code
from otter.config.setting import get_settings
from otter.episode import Episode, EnvInputManifest, LLMInputManifest
from otter.environment.docker import DockerEnvironment
from otter.logger import get_logger


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    entry_point: str
    test: str
    canonical_solution: str


class EvalPlusDataset(BaseDataset):

    IMAGE_TAG = "otter-evalplus:latest"

    async def setup(self) -> None:
        settings = get_settings()
        logger = get_logger()
        raw = load_dataset(
            "evalplus/humanevalplus",
            cache_dir=str(settings.dataset.cache_dir),
        )
        self._problems: dict[str, HumanEvalProblem] = {}
        for row in raw["test"]:
            p = HumanEvalProblem(
                task_id=row["task_id"],
                prompt=row["prompt"],
                entry_point=row["entry_point"],
                test=row["test"],
                canonical_solution=row["canonical_solution"],
            )
            self._problems[p.task_id] = p
        logger.info("loaded dataset evalplus: %d problems", len(self._problems))

        await DockerEnvironment.build_image(
            self.IMAGE_TAG,
            "FROM python:3.11-slim\n"
            "RUN pip install uv && uv pip install --system numpy==2.2.3\n",
        )

    async def teardown(self) -> None:
        await DockerEnvironment.remove_image(self.IMAGE_TAG)

    @property
    def task_ids(self) -> list[str]:
        return list(self._problems.keys())

    def _format_prompt(self, task_id: str) -> str:
        problem = self._problems[task_id]
        return (
            f"Complete the following Python function:\n\n"
            f"```python\n{problem.prompt}```\n\n"
            f"Return the complete function implementation (including the signature). "
            f"Wrap your code in a ```python``` code block."
        )

    def _prepare_llm_input(self, episode: Episode) -> LLMInputManifest:
        turn = episode.turns[-1]

        if len(episode.turns) == 1:
            prompt = self._format_prompt(episode.task_id)
        else:
            prompt = "Your code is incorrect. Please try again."

        prompt_file = turn.llm_input_path / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        return LLMInputManifest(prompt_file=prompt_file)
