"""Tests for otter.analysis.trend — ExperimentTrendReport."""

import json
import pytest
from pathlib import Path
from otter.analysis.trend import (
    ExperimentSnapshot,
    TrendPoint,
    ExperimentTrendReport,
    analyze_experiment_trend,
    _ols_slope,
    _direction,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episode(ep_dir: Path, evoscore: float, is_solved: bool, num_regress: int) -> None:
    """Create a minimal episode directory with one turn conclusion."""
    turn_dir = ep_dir / "turn_0"
    turn_dir.mkdir(parents=True)
    conclusion = {
        "evoscore_1.0": evoscore,
        "is_solved": is_solved,
        "num_regress": num_regress,
    }
    (turn_dir / "conclusion.json").write_text(json.dumps(conclusion))


def _make_experiment(base: Path, exp_id: str, episodes: list[dict]) -> Path:
    """Create a minimal experiment directory."""
    exp_dir = base / exp_id
    exp_dir.mkdir(parents=True)
    manifest = {"experiment_id": exp_id}
    (exp_dir / "experiment.json").write_text(json.dumps(manifest))
    for i, ep in enumerate(episodes):
        _make_episode(exp_dir / f"ep_{i}", **ep)
    return exp_dir


# ── OLS slope ─────────────────────────────────────────────────────────────────

class TestOlsSlope:
    def test_increasing(self):
        assert _ols_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_decreasing(self):
        assert _ols_slope([3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_flat(self):
        assert _ols_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_single_value_returns_zero(self):
        assert _ols_slope([1.0]) == 0.0

    def test_empty_returns_zero(self):
        assert _ols_slope([]) == 0.0


# ── Direction ─────────────────────────────────────────────────────────────────

class TestDirection:
    def test_improving(self):
        assert _direction(0.05) == "improving"

    def test_worsening(self):
        assert _direction(-0.05) == "worsening"

    def test_stable(self):
        assert _direction(0.0) == "stable"
        assert _direction(0.0005) == "stable"


# ── analyze_experiment_trend ──────────────────────────────────────────────────

class TestAnalyzeExperimentTrend:
    def test_basic_improving(self, tmp_path):
        e1 = _make_experiment(tmp_path, "v1", [
            {"evoscore": 0.5, "is_solved": False, "num_regress": 2},
            {"evoscore": 0.6, "is_solved": False, "num_regress": 1},
        ])
        e2 = _make_experiment(tmp_path, "v2", [
            {"evoscore": 0.7, "is_solved": True, "num_regress": 0},
            {"evoscore": 0.8, "is_solved": True, "num_regress": 0},
        ])
        report = analyze_experiment_trend([e1, e2])
        assert report.evoscore_direction == "improving"
        assert not report.any_regression
        assert len(report.snapshots) == 2
        assert len(report.trend_points) == 1
        assert report.trend_points[0].evoscore_delta > 0

    def test_regression_detected(self, tmp_path):
        e1 = _make_experiment(tmp_path, "v1", [
            {"evoscore": 0.8, "is_solved": True, "num_regress": 0},
        ])
        e2 = _make_experiment(tmp_path, "v2", [
            {"evoscore": 0.5, "is_solved": False, "num_regress": 3},
        ])
        report = analyze_experiment_trend([e1, e2])
        assert report.any_regression
        assert len(report.regressions) == 1
        assert report.evoscore_direction == "worsening"

    def test_three_experiments_slope(self, tmp_path):
        dirs = []
        for i, score in enumerate([0.5, 0.65, 0.8]):
            d = _make_experiment(tmp_path, f"v{i+1}", [
                {"evoscore": score, "is_solved": score >= 0.7, "num_regress": 0},
            ])
            dirs.append(d)
        report = analyze_experiment_trend(dirs)
        assert report.evoscore_slope > 0
        assert report.evoscore_direction == "improving"
        assert not report.any_regression

    def test_requires_two_valid_dirs(self, tmp_path):
        # Only one valid directory
        e1 = _make_experiment(tmp_path, "v1", [
            {"evoscore": 0.6, "is_solved": True, "num_regress": 0},
        ])
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="at least 2"):
            analyze_experiment_trend([e1, empty])

    def test_missing_dir_skipped(self, tmp_path):
        e1 = _make_experiment(tmp_path, "v1", [
            {"evoscore": 0.5, "is_solved": False, "num_regress": 1},
        ])
        e2 = _make_experiment(tmp_path, "v2", [
            {"evoscore": 0.7, "is_solved": True, "num_regress": 0},
        ])
        nonexistent = tmp_path / "does_not_exist"
        report = analyze_experiment_trend([nonexistent, e1, e2])
        assert len(report.snapshots) == 2

    def test_regression_threshold_respected(self, tmp_path):
        # Delta of 0.01 should NOT trigger regression at default threshold 0.02
        e1 = _make_experiment(tmp_path, "v1", [
            {"evoscore": 0.80, "is_solved": True, "num_regress": 0},
        ])
        e2 = _make_experiment(tmp_path, "v2", [
            {"evoscore": 0.79, "is_solved": True, "num_regress": 0},
        ])
        report = analyze_experiment_trend([e1, e2], regression_threshold=0.02)
        assert not report.any_regression

    def test_experiment_id_from_manifest(self, tmp_path):
        e1 = _make_experiment(tmp_path, "run_001", [
            {"evoscore": 0.6, "is_solved": False, "num_regress": 0},
        ])
        e2 = _make_experiment(tmp_path, "run_002", [
            {"evoscore": 0.7, "is_solved": True, "num_regress": 0},
        ])
        report = analyze_experiment_trend([e1, e2])
        assert report.snapshots[0].experiment_id == "run_001"
        assert report.trend_points[0].from_id == "run_001"
        assert report.trend_points[0].to_id == "run_002"

    def test_report_attributes(self, tmp_path):
        e1 = _make_experiment(tmp_path, "a", [
            {"evoscore": 0.5, "is_solved": False, "num_regress": 2},
        ])
        e2 = _make_experiment(tmp_path, "b", [
            {"evoscore": 0.6, "is_solved": True, "num_regress": 0},
        ])
        report = analyze_experiment_trend([e1, e2])
        assert isinstance(report, ExperimentTrendReport)
        assert isinstance(report.snapshots[0], ExperimentSnapshot)
        assert isinstance(report.trend_points[0], TrendPoint)
        assert report.regression_threshold == 0.02
