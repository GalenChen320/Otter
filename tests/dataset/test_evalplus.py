"""Tests for otter.dataset.evalplus module."""

import pytest

from otter.dataset.evalplus import EvalPlusDataset, HumanEvalProblem
from otter.episode import Episode, Turn, OutputManifest


class TestHumanEvalProblem:
    """Test HumanEvalProblem dataclass."""

    def test_fields(self):
        p = HumanEvalProblem(
            task_id="HumanEval_0",
            prompt="def foo():\n",
            entry_point="foo",
            test="assert foo() == 1",
            canonical_solution="    return 1\n",
        )
        assert p.task_id == "HumanEval_0"
        assert p.entry_point == "foo"


class TestEvalPlusDatasetMethods:
    """Test EvalPlusDataset methods with pre-populated problems."""

    @pytest.fixture
    def dataset(self, tmp_path):
        ds = EvalPlusDataset(tmp_path)
        ds._problems = {
            "HumanEval_0": HumanEvalProblem(
                task_id="HumanEval_0",
                prompt="def has_close_elements(numbers, threshold):\n",
                entry_point="has_close_elements",
                test="def check(candidate):\n    assert candidate([1.0, 2.0], 0.5) == False\n",
                canonical_solution="    return False\n",
            ),
        }
        return ds

    def _make_episode(self, tmp_path, task_id="HumanEval_0", num_turns=1):
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
        assert dataset.task_ids == ["HumanEval_0"]

    def test_format_prompt(self, dataset):
        prompt = dataset._format_prompt("HumanEval_0")
        assert "has_close_elements" in prompt
        assert "```python" in prompt

    def test_prepare_exec_input_first_turn(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        manifest = dataset._prepare_exec_input(ep)
        assert manifest.prompt_file is not None
        content = manifest.prompt_file.read_text(encoding="utf-8")
        assert "has_close_elements" in content

    def test_prepare_exec_input_later_turn(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path, num_turns=2)
        manifest = dataset._prepare_exec_input(ep)
        content = manifest.prompt_file.read_text(encoding="utf-8")
        assert "incorrect" in content.lower()

    def test_prepare_eval_input(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        # Simulate exec output
        exec_out_file = ep.turns[-1].exec_output_path / "response.txt"
        exec_out_file.write_text(
            '```python\ndef has_close_elements(numbers, threshold):\n    return False\n```',
            encoding="utf-8",
        )
        ep.turns[-1].exec_output_manifest = OutputManifest(
            exec_output_file=exec_out_file,
        )
        manifest = dataset._prepare_eval_input(ep)
        assert manifest.image_tag == EvalPlusDataset.IMAGE_TAG
        assert manifest.commands == ["python /tmp/solution.py"]
        script_content = manifest.script_file.read_text(encoding="utf-8")
        assert "has_close_elements" in script_content
        assert "check(has_close_elements)" in script_content

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

    async def test_judge_timed_out(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        ep.turns[-1].eval_output_manifest = OutputManifest(
            returncode=0, timed_out=True,
        )
        await dataset._judge(ep)
        assert ep.turns[-1].passed is False

    async def test_judge_raises_without_manifest(self, dataset, tmp_path):
        ep = self._make_episode(tmp_path)
        ep.turns[-1].eval_output_manifest = None
        with pytest.raises(ValueError, match="OutputManifest"):
            await dataset._judge(ep)
