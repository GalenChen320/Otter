from dataclasses import dataclass
from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings
from otter.episode import Episode
from otter.logger import get_logger


@dataclass
class MBPPPlusProblem:
    task_id: str
    prompt: str
    sample_tests: list[str]
    official_tests: str
    extra_imports: list[str]
    canonical_solution: str


class MBPPPlusDataset(BaseDataset):

    def load(self):
        settings = get_settings()
        logger = get_logger()
        raw = load_dataset(
            'evalplus/mbppplus',
            cache_dir=str(settings.dataset.cache_dir)
        )
        self._problems: dict[str, MBPPPlusProblem] = {}
        for row in raw["test"]:
            p = self._parse(row)
            self._problems[p.task_id] = p
        logger.info("loaded dataset mbppplus: %d problems", len(self._problems))

    def _parse(self, row: dict) -> MBPPPlusProblem:
        return MBPPPlusProblem(
            task_id=str(row["task_id"]),
            prompt=row["prompt"],
            sample_tests=row["test_list"],
            official_tests=row["test"],
            extra_imports=row["test_imports"],
            canonical_solution=row["code"],
        )

    @property
    def task_ids(self) -> list[str]:
        return list(self._problems.keys())

    def _format_prompt(self, task_id: str) -> str:
        problem = self._problems[task_id]
        sample = "\n".join(problem.sample_tests)
        return (
            f"{problem.prompt}\n\n"
            f"Your code should pass the following tests:\n"
            f"{sample}"
        )

    def make_messages(self, episode: Episode) -> list[dict]:
        messages = [{"role": "user", "content": self._format_prompt(episode.task_id)}]
        for turn in episode.turns:
            messages.append({"role": "assistant", "content": turn.response})
            messages.append({"role": "user", "content": "Your code is incorrect. Please try again."})
        return messages

    
if __name__ == "__main__":
    mbpp = MBPPPlusDataset()
    mbpp.load()
    print(mbpp.task_ids[0])
    print(mbpp._format_prompt(mbpp.task_ids[0]))