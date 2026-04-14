"""Tests for otter.summary module."""

import json

import pytest

from otter.summary import (
    EpisodeRecord,
    TurnStats,
    SampleSummary,
    ExperimentSummary,
    _compute_turn_stats,
    summarize,
    show_summary,
)
from otter.episode import META_FILENAME, EXPERIMENT_META


class TestEpisodeRecord:
    """Test EpisodeRecord dataclass properties."""

    def test_eid(self):
        r = EpisodeRecord(task_id="task_1", sample_id=0, turns=[True, False])
        assert r.eid == "task_1#0"

    def test_resolved_true(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[False, True])
        assert r.resolved is True

    def test_resolved_false(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[False, False])
        assert r.resolved is False

    def test_resolved_empty(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[])
        assert r.resolved is False

    def test_resolved_at_first_turn(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[True])
        assert r.resolved_at == 1

    def test_resolved_at_second_turn(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[False, True])
        assert r.resolved_at == 2

    def test_resolved_at_none(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[False, False])
        assert r.resolved_at is None

    def test_total_turns(self):
        r = EpisodeRecord(task_id="t", sample_id=0, turns=[False, True, False])
        assert r.total_turns == 3


class TestTurnStats:
    """Test TurnStats computed properties."""

    def test_pass_rate(self):
        ts = TurnStats(turn=1, total=10, passed=3, completed=5, pending=5)
        assert ts.pass_rate == pytest.approx(0.3)

    def test_completed_rate(self):
        ts = TurnStats(turn=1, total=10, passed=3, completed=5, pending=5)
        assert ts.completed_rate == pytest.approx(0.5)

    def test_pending_rate(self):
        ts = TurnStats(turn=1, total=10, passed=3, completed=5, pending=5)
        assert ts.pending_rate == pytest.approx(0.5)

    def test_zero_total(self):
        ts = TurnStats(turn=1, total=0, passed=0, completed=0, pending=0)
        assert ts.pass_rate == 0.0
        assert ts.completed_rate == 0.0
        assert ts.pending_rate == 0.0


class TestComputeTurnStats:
    """Test _compute_turn_stats function."""

    def test_single_episode_resolved_at_turn_1(self):
        episodes = [EpisodeRecord(task_id="t1", sample_id=0, turns=[True])]
        stats = _compute_turn_stats(episodes, max_turns=3)
        assert len(stats) == 3
        # Turn 1: passed=1, completed=1
        assert stats[0].passed == 1
        assert stats[0].completed == 1
        assert stats[0].pending == 0
        # Turn 2: still passed=1 (cumulative)
        assert stats[1].passed == 1

    def test_single_episode_resolved_at_turn_2(self):
        episodes = [EpisodeRecord(task_id="t1", sample_id=0, turns=[False, True])]
        stats = _compute_turn_stats(episodes, max_turns=3)
        assert stats[0].passed == 0
        assert stats[0].completed == 0  # not exhausted at turn 1 (has 2 turns)
        assert stats[1].passed == 1
        assert stats[1].completed == 1

    def test_never_resolved(self):
        episodes = [EpisodeRecord(task_id="t1", sample_id=0, turns=[False, False])]
        stats = _compute_turn_stats(episodes, max_turns=2)
        assert stats[0].passed == 0
        assert stats[1].passed == 0
        assert stats[1].completed == 1  # exhausted at turn 2

    def test_multiple_episodes(self):
        episodes = [
            EpisodeRecord(task_id="t1", sample_id=0, turns=[True]),
            EpisodeRecord(task_id="t2", sample_id=0, turns=[False, True]),
            EpisodeRecord(task_id="t3", sample_id=0, turns=[False, False]),
        ]
        stats = _compute_turn_stats(episodes, max_turns=2)
        assert stats[0].total == 3
        assert stats[0].passed == 1
        assert stats[1].passed == 2
        assert stats[1].completed == 3

    def test_empty_episodes(self):
        stats = _compute_turn_stats([], max_turns=2)
        assert len(stats) == 2
        assert stats[0].total == 0


class TestSummarize:
    """Test summarize function with real directory structures."""

    def _make_episode(self, output_dir, task_id, sample_id, turn_results):
        """Helper: create episode directory with turns."""
        ep_dir = output_dir / f"{task_id}#{sample_id}"
        ep_dir.mkdir(parents=True)
        for i, passed in enumerate(turn_results, 1):
            turn_dir = ep_dir / f"turn_{i}"
            turn_dir.mkdir()
            (turn_dir / META_FILENAME).write_text(
                json.dumps({"passed": passed}), encoding="utf-8"
            )

    def test_basic_summarize(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        exp_dir = tmp_path / "my_experiment"
        exp_dir.mkdir()
        self._make_episode(exp_dir, "t1", 0, [False, True])
        self._make_episode(exp_dir, "t2", 0, [False, False])

        result = summarize(exp_dir)
        assert isinstance(result, ExperimentSummary)
        assert result.experiment_id == "my_experiment"
        assert result.max_turns == 2
        assert len(result.samples) == 1  # only sample_id=0
        assert len(result.samples[0].episodes) == 2

    def test_with_experiment_meta(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        config = {"experiment.max_turns": 5, "model": "gpt-4"}
        (exp_dir / EXPERIMENT_META).write_text(json.dumps(config), encoding="utf-8")
        self._make_episode(exp_dir, "t1", 0, [True])

        result = summarize(exp_dir)
        assert result.config == config
        assert result.max_turns == 5

    def test_empty_experiment(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        exp_dir = tmp_path / "empty_exp"
        exp_dir.mkdir()

        result = summarize(exp_dir)
        assert result.max_turns == 1  # default
        assert len(result.samples) == 0

    def test_episode_without_turns_skipped(self, tmp_path, mocker):
        """Episode dirs that exist but have no completed turns should be excluded."""
        mocker.patch("otter.logger.get_logger")
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        # Create episode dir with no turn subdirs
        ep_dir = exp_dir / "t1#0"
        ep_dir.mkdir()

        result = summarize(exp_dir)
        assert len(result.samples) == 0

    def test_multiple_samples(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        self._make_episode(exp_dir, "t1", 0, [True])
        self._make_episode(exp_dir, "t1", 1, [False])

        result = summarize(exp_dir)
        assert len(result.samples) == 2


class TestShowSummary:
    """Test show_summary output content."""

    def test_show_summary_contains_experiment_id(self, capsys):
        """show_summary output should contain the experiment ID."""
        result = ExperimentSummary(
            experiment_id="test_exp",
            config={"model": "gpt-4"},
            max_turns=2,
            samples=[
                SampleSummary(
                    sample_id=0,
                    episodes=[
                        EpisodeRecord(task_id="t1", sample_id=0, turns=[True]),
                    ],
                    turn_stats=[
                        TurnStats(turn=1, total=1, passed=1, completed=1, pending=0),
                        TurnStats(turn=2, total=1, passed=1, completed=1, pending=0),
                    ],
                ),
            ],
        )
        show_summary(result)
        captured = capsys.readouterr().out
        assert "test_exp" in captured

    def test_show_summary_contains_pass_stats(self, capsys):
        """show_summary should display pass/total counts."""
        result = ExperimentSummary(
            experiment_id="exp",
            config=None,
            max_turns=1,
            samples=[
                SampleSummary(
                    sample_id=0,
                    episodes=[],
                    turn_stats=[
                        TurnStats(turn=1, total=5, passed=2, completed=3, pending=2),
                    ],
                ),
            ],
        )
        show_summary(result)
        captured = capsys.readouterr().out
        assert "2/5" in captured  # passed count
        assert "3/5" in captured  # completed count
