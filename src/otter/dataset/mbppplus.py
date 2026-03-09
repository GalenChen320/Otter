from dataclasses import dataclass
from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings


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
        raw = load_dataset(
            'evalplus/mbppplus',
            cache_dir=str(settings.dataset.cache_dir)
        )
        self._problems: list[MBPPPlusProblem] = [
            self._parse(row) for row in raw["test"]
        ]

    def _parse(self, row: dict) -> MBPPPlusProblem:
        return MBPPPlusProblem(
            task_id=str(row["task_id"]),
            prompt=row["prompt"],
            sample_tests=row["test_list"],
            official_tests=row["test"],
            extra_imports=row["test_imports"],
            canonical_solution=row["code"],
        )

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, index: int) -> MBPPPlusProblem:
        return self._problems[index]

    def make_prompt(self, index: int) -> str:
        problem = self._problems[index]
        sample = "\n".join(problem.sample_tests)
        return (
            f"{problem.prompt}\n\n"
            f"Your code should pass the following tests:\n"
            f"{sample}"
        )

    
if __name__ == "__main__":
    mbpp = MBPPPlusDataset()
    mbpp.load()
    print(mbpp[0])
    print(mbpp.make_prompt(0))