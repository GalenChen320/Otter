import json
import re
from dataclasses import dataclass
from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings
from otter.episode import Episode, ExecManifest, InputManifest
from otter.environment.docker import DockerEnvironment
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

    IMAGE_TAG = "python:3.11-slim"

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

    async def setup(self, output_dir) -> None:
        await super().setup(output_dir)
        await DockerEnvironment.build_image(self.IMAGE_TAG, "FROM python:3.11-slim\n")

    async def teardown(self) -> None:
        await DockerEnvironment.remove_image(self.IMAGE_TAG)

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

    def _extract_code(self, response: str) -> str:
        """从 LLM response 中提取 Python 代码块。"""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def prepare_input(self, episode: Episode) -> None:
        turn = episode.turns[-1]

        # 写入 prompt 文件
        prompt = self._format_prompt(episode.task_id)
        (turn.input_path / "prompt.txt").write_text(prompt, encoding="utf-8")

        # 多轮：拼接历史 response 和 feedback
        messages = [{"role": "user", "content": prompt}]
        for prev_turn in episode.turns[:-1]:
            response_file = prev_turn.response_path / "response.txt"
            prev_response = response_file.read_text(encoding="utf-8")
            messages.append({"role": "assistant", "content": prev_response})
            messages.append({"role": "user", "content": "Your code is incorrect. Please try again."})

        # 写入 messages 文件
        (turn.input_path / "messages.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        # 写入 manifest 并设置句柄
        manifest = InputManifest(
            base_path=turn.input_path,
            messages_file="messages.json",
        )
        manifest_dict = {"messages_file": manifest.messages_file}
        (turn.input_path / "manifest.json").write_text(
            json.dumps(manifest_dict, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        turn.input_manifest = manifest

    def prepare_exec(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        response_manifest = turn.response_manifest

        if not response_manifest or not response_manifest.response_file:
            raise ValueError("MBPPPlusDataset requires 'response_file' in ResponseManifest")

        # 从 response manifest 读取 response
        response_path = response_manifest.base_path / response_manifest.response_file
        response = response_path.read_text(encoding="utf-8")

        # 提取代码，拼接测试，写入脚本文件
        code = self._extract_code(response)
        problem = self._problems[episode.task_id]
        imports = "\n".join(problem.extra_imports)
        full_code = f"{imports}\n\n{code}\n\n{problem.official_tests}"

        script_file = "solution.py"
        (turn.response_path / script_file).write_text(full_code, encoding="utf-8")

        # 写入 manifest 并设置句柄
        manifest = ExecManifest(
            base_path=turn.response_path,
            image_tag=self.IMAGE_TAG,
            script_file=script_file,
            commands=["python /tmp/solution.py"],
        )
        (turn.response_path / "exec_manifest.json").write_text(
            json.dumps({
                "image_tag": manifest.image_tag,
                "script_file": manifest.script_file,
                "commands": manifest.commands,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        turn.exec_manifest = manifest

    async def make_judgement(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        obs = turn.observation_manifest

        if not obs:
            raise ValueError("MBPPPlusDataset requires ObservationManifest")

        turn.passed = obs.returncode == 0 and not obs.timed_out
