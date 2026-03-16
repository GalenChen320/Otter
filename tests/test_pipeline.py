"""Tests for otter.pipeline module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from otter.episode import Episode, Turn, EXPERIMENT_META
from otter.pipeline import (
    create_dataset,
    create_role,
    get_pending_episodes,
    verify_or_create_experiment_meta,
)


class TestCreateDataset:
    """Test create_dataset factory function."""

    def test_creates_evalplus(self, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.dataset_name = "evalplus"
        mock_settings.output_dir = Path("/tmp/out")
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)

        ds = create_dataset()
        from otter.dataset.evalplus import EvalPlusDataset
        assert isinstance(ds, EvalPlusDataset)

    def test_creates_mbppplus(self, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.dataset_name = "mbppplus"
        mock_settings.output_dir = Path("/tmp/out")
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)

        ds = create_dataset()
        from otter.dataset.mbppplus import MBPPPlusDataset
        assert isinstance(ds, MBPPPlusDataset)

    def test_unknown_dataset_raises(self, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.dataset_name = "nonexistent"
        mock_settings.output_dir = Path("/tmp/out")
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)

        with pytest.raises(ValueError, match="unknown dataset"):
            create_dataset()


class TestCreateRole:
    """Test create_role factory function."""

    def test_returns_none_when_type_is_none(self, mocker):
        from otter.role import ExecutorRole
        result = create_role(ExecutorRole, None, None)
        assert result is None

    def test_creates_role_calls_create_backend(self, mocker):
        from otter.role import ExecutorRole
        mock_create = mocker.patch("otter.pipeline.create_backend")
        mock_settings = mocker.MagicMock()

        # Patch EXTRACT_DISPATCH and PACK_DISPATCH to accept any type
        mocker.patch.dict(
            "otter.role.EXTRACT_DISPATCH",
            {type(mock_create.return_value): lambda m, e: {}},
        )
        mocker.patch.dict(
            "otter.role.PACK_DISPATCH",
            {type(mock_create.return_value): lambda r, d: None},
        )

        result = create_role(ExecutorRole, "docker", mock_settings)
        mock_create.assert_called_once_with("docker", mock_settings)
        assert isinstance(result, ExecutorRole)


class TestGetPendingEpisodes:
    """Test get_pending_episodes filtering logic."""

    def test_all_new_episodes(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.output_dir = tmp_path
        mock_settings.samples_per_problem = 1
        mock_settings.max_turns = 3
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)
        mocker.patch("otter.pipeline.Episode.sync_all", return_value={})

        mock_ds = mocker.MagicMock()
        mock_ds.task_ids = ["t1", "t2"]

        episodes = get_pending_episodes(mock_ds)
        assert len(episodes) == 2
        assert episodes[0].task_id == "t1"
        assert episodes[1].task_id == "t2"

    def test_skips_resolved_episodes(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.output_dir = tmp_path
        mock_settings.samples_per_problem = 1
        mock_settings.max_turns = 3
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)

        resolved_ep = Episode(
            task_id="t1", sample_id=0,
            turns=[Turn(turn_dir=tmp_path / "t", passed=True)],
            base_dir=tmp_path / "t1#0",
        )
        mocker.patch(
            "otter.pipeline.Episode.sync_all",
            return_value={"t1#0": resolved_ep},
        )

        mock_ds = mocker.MagicMock()
        mock_ds.task_ids = ["t1", "t2"]

        episodes = get_pending_episodes(mock_ds)
        assert len(episodes) == 1
        assert episodes[0].task_id == "t2"

    def test_skips_exhausted_episodes(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.output_dir = tmp_path
        mock_settings.samples_per_problem = 1
        mock_settings.max_turns = 1
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)

        exhausted_ep = Episode(
            task_id="t1", sample_id=0,
            turns=[Turn(turn_dir=tmp_path / "t", passed=False)],
            base_dir=tmp_path / "t1#0",
        )
        mocker.patch(
            "otter.pipeline.Episode.sync_all",
            return_value={"t1#0": exhausted_ep},
        )

        mock_ds = mocker.MagicMock()
        mock_ds.task_ids = ["t1"]

        episodes = get_pending_episodes(mock_ds)
        assert len(episodes) == 0

    def test_continues_partial_episodes(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.output_dir = tmp_path
        mock_settings.samples_per_problem = 1
        mock_settings.max_turns = 3
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)

        partial_ep = Episode(
            task_id="t1", sample_id=0,
            turns=[Turn(turn_dir=tmp_path / "t", passed=False)],
            base_dir=tmp_path / "t1#0",
        )
        mocker.patch(
            "otter.pipeline.Episode.sync_all",
            return_value={"t1#0": partial_ep},
        )

        mock_ds = mocker.MagicMock()
        mock_ds.task_ids = ["t1"]

        episodes = get_pending_episodes(mock_ds)
        assert len(episodes) == 1
        assert episodes[0] is partial_ep

    def test_multiple_samples(self, tmp_path, mocker):
        mock_settings = mocker.MagicMock()
        mock_settings.output_dir = tmp_path
        mock_settings.samples_per_problem = 3
        mock_settings.max_turns = 2
        mocker.patch("otter.pipeline.get_settings", return_value=mock_settings)
        mocker.patch("otter.pipeline.Episode.sync_all", return_value={})

        mock_ds = mocker.MagicMock()
        mock_ds.task_ids = ["t1"]

        episodes = get_pending_episodes(mock_ds)
        assert len(episodes) == 3
        sample_ids = [ep.sample_id for ep in episodes]
        assert sample_ids == [0, 1, 2]


class TestVerifyOrCreateExperimentMeta:
    """Test verify_or_create_experiment_meta function."""

    def test_creates_meta_on_first_run(self, tmp_path, mocker):
        mocker.patch("otter.pipeline.get_logger")
        mocker.patch(
            "otter.pipeline.get_tracked_config",
            return_value={"model": "gpt-4", "max_turns": 3},
        )

        output_dir = tmp_path / "experiment"
        verify_or_create_experiment_meta(output_dir)

        meta_path = output_dir / EXPERIMENT_META
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["model"] == "gpt-4"
        assert data["max_turns"] == 3

    def test_passes_when_config_matches(self, tmp_path, mocker):
        mocker.patch("otter.pipeline.get_logger")
        config = {"model": "gpt-4", "max_turns": 3}
        mocker.patch("otter.pipeline.get_tracked_config", return_value=config)

        output_dir = tmp_path / "experiment"
        output_dir.mkdir()
        (output_dir / EXPERIMENT_META).write_text(
            json.dumps(config), encoding="utf-8"
        )

        # Should not raise
        verify_or_create_experiment_meta(output_dir)

    def test_raises_on_mismatch_when_user_declines(self, tmp_path, mocker):
        mocker.patch("otter.pipeline.get_logger")
        old_config = {"model": "gpt-3.5", "max_turns": 1}
        new_config = {"model": "gpt-4", "max_turns": 3}
        mocker.patch("otter.pipeline.get_tracked_config", return_value=new_config)
        mocker.patch("otter.pipeline.Confirm.ask", return_value=False)

        output_dir = tmp_path / "experiment"
        output_dir.mkdir()
        (output_dir / EXPERIMENT_META).write_text(
            json.dumps(old_config), encoding="utf-8"
        )

        with pytest.raises(SystemExit):
            verify_or_create_experiment_meta(output_dir)

    def test_overrides_on_mismatch_when_user_confirms(self, tmp_path, mocker):
        mocker.patch("otter.pipeline.get_logger")
        old_config = {"model": "gpt-3.5"}
        new_config = {"model": "gpt-4"}
        mocker.patch("otter.pipeline.get_tracked_config", return_value=new_config)
        mocker.patch("otter.pipeline.Confirm.ask", return_value=True)

        output_dir = tmp_path / "experiment"
        output_dir.mkdir()
        (output_dir / EXPERIMENT_META).write_text(
            json.dumps(old_config), encoding="utf-8"
        )

        verify_or_create_experiment_meta(output_dir)

        data = json.loads(
            (output_dir / EXPERIMENT_META).read_text(encoding="utf-8")
        )
        assert data["model"] == "gpt-4"


class TestRunTurn:
    """Test run_turn orchestration."""

    async def test_run_turn_executor_only(self, tmp_path, mocker):
        """run_turn with only executor should call prepare, run, judge."""
        from otter.pipeline import run_turn
        import asyncio

        mocker.patch("otter.pipeline.get_logger")

        # Mock settings for Turn.setup_dirs
        mock_settings = mocker.MagicMock()
        mock_settings.proposer = None
        mock_settings.executor = mocker.MagicMock()
        mock_settings.evaluator = None
        mocker.patch("otter.episode.get_settings", return_value=mock_settings)

        ep = Episode(task_id="t1", sample_id=0, base_dir=tmp_path)

        mock_ds = mocker.MagicMock()
        mock_ds.make_judgement = mocker.AsyncMock()

        mock_exec = mocker.MagicMock()
        mock_exec.run = mocker.AsyncMock()

        sem = asyncio.Semaphore(1)

        await run_turn(
            mock_ds, ep,
            exec_client=mock_exec,
            exec_semaphore=sem,
        )

        assert ep.total_turns == 1
        # Verify call order: prepare → run → judge
        expected_calls = [
            mocker.call.prepare_exec_input(ep),
            mocker.call.make_judgement(ep),
        ]
        mock_ds.assert_has_calls(expected_calls, any_order=False)
        mock_exec.run.assert_called_once_with(ep)

    async def test_run_turn_no_clients(self, tmp_path, mocker):
        """run_turn with no clients should still create turn and judge."""
        from otter.pipeline import run_turn

        mocker.patch("otter.pipeline.get_logger")

        mock_settings = mocker.MagicMock()
        mock_settings.proposer = None
        mock_settings.executor = None
        mock_settings.evaluator = None
        mocker.patch("otter.episode.get_settings", return_value=mock_settings)

        ep = Episode(task_id="t1", sample_id=0, base_dir=tmp_path)

        mock_ds = mocker.MagicMock()
        mock_ds.make_judgement = mocker.AsyncMock()

        await run_turn(mock_ds, ep)

        assert ep.total_turns == 1
        mock_ds.make_judgement.assert_called_once()
