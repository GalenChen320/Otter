"""Tests for otter.dataset.base module."""

import json
from pathlib import Path

import pytest

from otter.dataset.base import BaseDataset
from otter.episode import Episode, InputManifest, Turn, META_FILENAME


class ConcreteDataset(BaseDataset):
    """Concrete implementation of BaseDataset for testing."""

    def __init__(self, base_dir: Path, task_id_list=None):
        super().__init__(base_dir)
        self._task_ids = task_id_list or ["task_1", "task_2"]

    @property
    def task_ids(self) -> list[str]:
        return self._task_ids

    def _prepare_exec_input(self, episode: Episode) -> InputManifest:
        turn = episode.turns[-1]
        prompt_file = turn.exec_input_path / "prompt.txt"
        prompt_file.write_text("test prompt", encoding="utf-8")
        return InputManifest(prompt_file=prompt_file)

    def _prepare_eval_input(self, episode: Episode) -> InputManifest:
        turn = episode.turns[-1]
        script_file = turn.eval_input_path / "script.py"
        script_file.write_text("print('test')", encoding="utf-8")
        return InputManifest(image_tag="test:v1", script_file=script_file)

    async def _judge(self, episode: Episode) -> None:
        turn = episode.turns[-1]
        turn.passed = True


class TestBaseDatasetInit:
    """Test BaseDataset initialization."""

    def test_stores_base_dir(self, tmp_path):
        ds = ConcreteDataset(tmp_path)
        assert ds.base_dir == tmp_path

    def test_task_ids(self, tmp_path):
        ds = ConcreteDataset(tmp_path, task_id_list=["a", "b", "c"])
        assert ds.task_ids == ["a", "b", "c"]


class TestBaseDatasetLifecycle:
    """Test BaseDataset lifecycle context managers."""

    async def test_run_context(self, tmp_path):
        ds = ConcreteDataset(tmp_path)
        async with ds.run_context():
            pass  # should not raise

    async def test_episode_context(self, tmp_path):
        ds = ConcreteDataset(tmp_path)
        ep = Episode(task_id="t1", sample_id=0, base_dir=tmp_path)
        async with ds.episode_context(ep) as yielded:
            assert yielded is ep

    async def test_setup_teardown_called(self, tmp_path, mocker):
        ds = ConcreteDataset(tmp_path)
        setup_spy = mocker.patch.object(ds, "setup", new_callable=mocker.AsyncMock)
        teardown_spy = mocker.patch.object(ds, "teardown", new_callable=mocker.AsyncMock)

        async with ds.run_context():
            setup_spy.assert_called_once()
            teardown_spy.assert_not_called()
        teardown_spy.assert_called_once()

    async def test_episode_setup_teardown_called(self, tmp_path, mocker):
        ds = ConcreteDataset(tmp_path)
        ep = Episode(task_id="t1", sample_id=0, base_dir=tmp_path)
        setup_spy = mocker.patch.object(ds, "setup_episode", new_callable=mocker.AsyncMock)
        teardown_spy = mocker.patch.object(ds, "teardown_episode", new_callable=mocker.AsyncMock)

        async with ds.episode_context(ep):
            setup_spy.assert_called_once_with(ep)
            teardown_spy.assert_not_called()
        teardown_spy.assert_called_once_with(ep)

    async def test_teardown_called_on_exception(self, tmp_path, mocker):
        ds = ConcreteDataset(tmp_path)
        teardown_spy = mocker.patch.object(ds, "teardown", new_callable=mocker.AsyncMock)

        with pytest.raises(RuntimeError):
            async with ds.run_context():
                raise RuntimeError("test error")
        teardown_spy.assert_called_once()


class TestBaseDatasetTemplateMethods:
    """Test prepare_exec_input, prepare_eval_input, make_judgement template methods."""

    def _make_episode_with_turn(self, tmp_path):
        turn_dir = tmp_path / "turn_1"
        exec_in = turn_dir / "exec_input"
        exec_out = turn_dir / "exec_output"
        eval_in = turn_dir / "eval_input"
        eval_out = turn_dir / "eval_output"
        for d in [turn_dir, exec_in, exec_out, eval_in, eval_out]:
            d.mkdir(parents=True, exist_ok=True)

        turn = Turn(
            turn_dir=turn_dir,
            exec_input_path=exec_in,
            exec_output_path=exec_out,
            eval_input_path=eval_in,
            eval_output_path=eval_out,
        )
        ep = Episode(task_id="t1", sample_id=0, turns=[turn], base_dir=tmp_path)
        return ep

    def test_prepare_exec_input_saves_manifest(self, tmp_path):
        ds = ConcreteDataset(tmp_path)
        ep = self._make_episode_with_turn(tmp_path)

        ds.prepare_exec_input(ep)

        turn = ep.turns[-1]
        assert turn.exec_input_manifest is not None
        assert turn.exec_input_manifest.prompt_file is not None
        # Manifest should be saved to disk with correct content
        manifest_path = turn.exec_input_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["prompt_file"] == str(turn.exec_input_manifest.prompt_file)

    def test_prepare_eval_input_saves_manifest(self, tmp_path):
        ds = ConcreteDataset(tmp_path)
        ep = self._make_episode_with_turn(tmp_path)

        ds.prepare_eval_input(ep)

        turn = ep.turns[-1]
        assert turn.eval_input_manifest is not None
        assert turn.eval_input_manifest.image_tag == "test:v1"
        manifest_path = turn.eval_input_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["image_tag"] == "test:v1"
        assert data["script_file"] == str(turn.eval_input_manifest.script_file)

    async def test_make_judgement_updates_passed_and_saves_meta(self, tmp_path):
        ds = ConcreteDataset(tmp_path)
        ep = self._make_episode_with_turn(tmp_path)

        await ds.make_judgement(ep)

        turn = ep.turns[-1]
        assert turn.passed is True
        meta_path = turn.turn_dir / META_FILENAME
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["passed"] is True
