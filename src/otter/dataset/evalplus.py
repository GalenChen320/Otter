from dataclasses import dataclass
from datasets import load_dataset
import json

from otter.dataset.base import BaseDataset
from otter.dataset.utils import extract_code, build_messages
from otter.config.setting import get_settings
from otter.episode import Episode, InputManifest, OutputManifest
from otter.backend.docker import DockerBackend
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
                task_id=row["task_id"].replace("/", "_"),
                prompt=row["prompt"],
                entry_point=row["entry_point"],
                test=row["test"],
                canonical_solution=row["canonical_solution"],
            )
            self._problems[p.task_id] = p
        logger.info("loaded dataset evalplus: %d problems", len(self._problems))

        await DockerBackend.build_image(
            self.IMAGE_TAG,
            "FROM python:3.11-slim\n"
            "RUN pip install uv && uv pip install --system numpy==2.2.3\n",
        )

    async def teardown(self) -> None:
        await DockerBackend.remove_image(self.IMAGE_TAG)

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

    def _prepare_prop_input(self, episode: Episode) -> InputManifest:
        ...

    def _prepare_exec_input(self, episode: Episode) -> InputManifest:
        turn = episode.turns[-1]

        if len(episode.turns) == 1:
            prompt = self._format_prompt(episode.task_id)
        else:
            prompt = "Your code is incorrect. Please try again."

        messages = build_messages(episode, prompt)
        msg_file = turn.exec_input_path / "messages.json"
        msg_file.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")

        return InputManifest(params={
            "messages": messages,
        })

    def _prepare_eval_input(self, episode: Episode) -> InputManifest:
        turn = episode.turns[-1]
        exec_output_manifest = turn.exec_output_manifest

        # TODO 实际上这里应该判断backend的类型，根据类型判断怎么取。
        if not exec_output_manifest.products or not exec_output_manifest.products[0]:
            raise ValueError("EvalPlusDataset requires 'products[0]' in OutputManifest")

        response = exec_output_manifest.products[0].read_text(encoding="utf-8")

        code = extract_code(response)
        problem = self._problems[episode.task_id]
        full_code = (
            f"{code}\n\n"
            f"{problem.test}\n\n"
            f"check({problem.entry_point})\n"
        )

        script_file = turn.eval_input_path / "solution.py"
        script_file.write_text(full_code, encoding="utf-8")

        return InputManifest(params={
            "image_tag": self.IMAGE_TAG,
            "commands": ["python /tmp/solution.py"],
            "copy_in": [(str(script_file), "/tmp")],
        })

    async def _judge(self, episode: Episode) -> bool:
        turn = episode.turns[-1]
        eval_result = turn.eval_output_manifest.debug_info.commands[0]
        return eval_result.returncode == 0 and not eval_result.timed_out

    def validate_prop_output(self, manifest: OutputManifest) -> bool:
        return True
    
    def validate_exec_output(self, manifest: OutputManifest) -> bool:
        return True

    def validate_eval_output(self, manifest: OutputManifest) -> bool:
        return True

__all__ = [
    "EvalPlusDataset",
]