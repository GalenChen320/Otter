import json
import re
from dataclasses import dataclass
from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings
from otter.episode import Episode, EnvInputManifest, LLMInputManifest
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

    def prepare_llm_input(self, episode: Episode) -> None:
        turn = episode.turns[-1]

        # 第一轮：写题目 prompt；后续轮次：写 feedback
        if len(episode.turns) == 1:
            prompt = self._format_prompt(episode.task_id)
        else:
            prompt = "Your code is incorrect. Please try again."

        prompt_file = turn.llm_input_path / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        # 写入 manifest 并设置句柄
        manifest = LLMInputManifest(prompt_file=prompt_file)
        (turn.llm_input_path / "manifest.json").write_text(
            json.dumps({"prompt_file": str(prompt_file)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        turn.llm_input_manifest = manifest

    def prepare_env_input(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        llm_output_manifest = turn.llm_output_manifest

        if not llm_output_manifest or not llm_output_manifest.llm_output_file:
            raise ValueError("MBPPPlusDataset requires 'llm_output_file' in LLMOutputManifest")

        # 从 llm_output manifest 读取 response
        response = llm_output_manifest.llm_output_file.read_text(encoding="utf-8")

        # 提取代码，拼接测试，写入脚本文件
        code = self._extract_code(response)
        problem = self._problems[episode.task_id]
        imports = "\n".join(problem.extra_imports)
        full_code = f"{imports}\n\n{code}\n\n{problem.official_tests}"

        script_file = turn.env_input_path / "solution.py"
        script_file.write_text(full_code, encoding="utf-8")

        # 写入 manifest 并设置句柄
        manifest = EnvInputManifest(
            image_tag=self.IMAGE_TAG,
            script_file=script_file,
            commands=["python /tmp/solution.py"],
        )
        (turn.env_input_path / "env_input_manifest.json").write_text(
            json.dumps({
                "image_tag": manifest.image_tag,
                "script_file": str(manifest.script_file),
                "commands": manifest.commands,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        turn.env_input_manifest = manifest

    async def _judge(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        env_output = turn.env_output_manifest

        if not env_output:
            raise ValueError("MBPPPlusDataset requires EnvOutputManifest")

        turn.passed = env_output.returncode == 0 and not env_output.timed_out
