"""Tests for otter.dataset.mbppplus module."""

import pytest
from pathlib import Path

from otter.dataset.mbppplus import MBPPPlusDataset, MBPPPlusProblem
from otter.episode import Episode, Turn, InputManifest, OutputManifest


class TestMBPPPlusProblem:
    """Test MBPPPlusProblem dataclass."""

    def test_fields(self):
        p = MBPPPlusProblem(
            task_id="11",
            prompt="Write a function to remove all odd numbers.",
            sample_tests=["assert remove_odd([1,2,3]) == [2]"],
            official_tests="assert remove_odd([1,2,3]) == [2]",
            extra_imports=[],
            canonical_solution="def remove_odd(l): return [x for x in l if x%2==0]",
        )
        assert p.task_id == "11"
        assert len(p.sample_tests) == 1


class TestMBPPPlusDatasetParse:
    """Test MBPPPlusDataset._parse method."""

    def test_parse_row(self, tmp_path):
        ds = MBPPPlusDataset(tmp_path)
        row = {
            "task_id": 11,
            "prompt": "Write a function.",
            "test_list": ["assert foo() == 1"],
            "test": "assert foo() == 1\nassert foo() == 1",
            "test_imports": ["import math"],
            "code": "def foo(): return 1",
        }
        p = ds._parse(row)
        assert p.task_id == "11"
        assert p.prompt == "Write a function."
        assert p.sample_tests == ["assert foo() == 1"]
        assert p.extra_imports == ["import math"]


class TestMBPPPlusDatasetMethods:
    """Test MBPPPlusDataset methods with pre-populated problems."""

    @pytest.fixture
    def dataset(self, tmp_path):
        ds = MBPPPlusDataset(tmp_path)
        ds._problems = {
            "11": MBPPPlusProblem(
                task_id="11",
                prompt="Write a function to remove all odd numbers from a list.",
                sample_tests=["assert remove_odd([1,2,3]) == [2]"],
                official_tests="assert remove_odd([1,2,3]) == [2]\nassert remove_odd([]) == []",
                extra_imports=["import math"],
                canonical_solution="def remove_odd(l): return [x for x in l if x%2==0]",
            ),
        }
        return ds

    def _make_episode(self, tmp_path, task_id="11", num_turns=1):
        turns = []
        for i in range(num_turns):
            turn_dir = tmp_path / f"ep_dir/turn_{i+1}"
            exec_in = turn_dir / "exec_input"
            exec_out = turn_dir / "exec_output"
            eval_in = turn_dir / "eval_input"
            eval_out = turn_dir / "eval_output"
            for d in [turn_dir, exec_in, exec_out, eval_in, eval_out]:
                d.mkdir(parents=True, exist_ok=True)
            turns.append(Turn(
                turn_dir=turn_dir,
                exec_input_path=exec_in,
                exec_output_path=exec_out,
                eval_input_path=eval_in,
                eval_output_path=eval_out,
            ))
        return Episode(
            task_id=task_id, sample_id=0,
            turns=turns, base_dir=tmp_path / "ep_dir",
        )

    def test_task_ids(self, dataset):
        assert dataset.task_ids == ["11"]

    def test_format_prompt(self, dataset):
        prompt = dataset._format_prompt("11")
        assert "remove_odd" in prompt
        assert "DO NOT" in prompt

    def test_prepare_exec_input_first_turn(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        manifest = dataset._prepare_exec_input(ep)
        assert manifest.prompt_file is not None
        content = manifest.prompt_file.read_text(encoding="utf-8")
        assert "remove_odd" in content

    def test_prepare_exec_input_later_turn(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path, num_turns=2)
        manifest = dataset._prepare_exec_input(ep)
        content = manifest.prompt_file.read_text(encoding="utf-8")
        assert "incorrect" in content.lower()

    def test_prepare_eval_input(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        exec_out_file = ep.turns[-1].exec_output_path / "response.txt"
        exec_out_file.write_text(
            '```python\ndef remove_odd(l):\n    return [x for x in l if x%2==0]\n```',
            encoding="utf-8",
        )
        ep.turns[-1].exec_output_manifest = OutputManifest(
            exec_output_file=exec_out_file,
        )
        manifest = dataset._prepare_eval_input(ep)
        assert manifest.image_tag == MBPPPlusDataset.IMAGE_TAG
        assert manifest.commands == ["python /tmp/solution.py"]
        script = manifest.script_file.read_text(encoding="utf-8")
        assert "import math" in script
        assert "remove_odd" in script

    def test_prepare_eval_input_raises_without_exec_output(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        ep.turns[-1].exec_output_manifest = None
        with pytest.raises(ValueError, match="exec_output_file"):
            dataset._prepare_eval_input(ep)

    async def test_judge_passed(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        ep.turns[-1].eval_output_manifest = OutputManifest(
            returncode=0, timed_out=False,
        )
        await dataset._judge(ep)
        assert ep.turns[-1].passed is True

    async def test_judge_failed(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        ep.turns[-1].eval_output_manifest = OutputManifest(
            returncode=1, timed_out=False,
        )
        await dataset._judge(ep)
        assert ep.turns[-1].passed is False

    async def test_judge_raises_without_manifest(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        ep.turns[-1].eval_output_manifest = None
        with pytest.raises(ValueError, match="OutputManifest"):
            await dataset._judge(ep)
