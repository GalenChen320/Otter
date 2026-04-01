"""Cross-experiment behavioral trend analysis for Otter.

Provides :class:`ExperimentTrendReport` which detects whether an agent's
evoscore (and related metrics) is improving or degrading across a sequence of
experiment runs.

Usage::

    from pathlib import Path
    from otter.analysis.trend import analyze_experiment_trend

    report = analyze_experiment_trend(
        [Path("experiments/v1"), Path("experiments/v2"), Path("experiments/v3")]
    )
    if report.any_regression:
        print("Agent performance is degrading:", report.regressions)

CLI equivalent (see ``otter trend --help``)::

    otter trend experiments/v1 experiments/v2 experiments/v3

Reference: PDR in Production v2.5 — DOI 10.5281/zenodo.19362461
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class ExperimentSnapshot:
    """Aggregated statistics for a single experiment directory.

    Populated by reading the ``experiment.json`` manifest and the per-episode
    ``conclusion.json`` files that Otter produces during a run.

    Attributes:
        experiment_dir: Absolute path to the experiment directory.
        experiment_id: Value of the ``experiment_id`` field in
            ``experiment.json``, or the directory name when absent.
        avg_evoscore: Mean evoscore_1.0 across all episodes.
        solved_rate: Fraction of episodes that were solved (0.0–1.0).
        zero_regress_rate: Fraction of episodes with zero regressions.
        episode_count: Number of episodes that were successfully parsed.
    """

    experiment_dir: Path
    experiment_id: str
    avg_evoscore: float
    solved_rate: float
    zero_regress_rate: float
    episode_count: int


@dataclass
class TrendPoint:
    """Per-metric pairwise delta between two consecutive snapshots.

    Attributes:
        from_id: ``experiment_id`` of the earlier snapshot.
        to_id: ``experiment_id`` of the later snapshot.
        evoscore_delta: Change in ``avg_evoscore`` (positive = improvement).
        solved_rate_delta: Change in ``solved_rate``.
        zero_regress_rate_delta: Change in ``zero_regress_rate``.
        regressed: ``True`` when *any* metric moved in the adverse direction
            by more than *regression_threshold*.
    """

    from_id: str
    to_id: str
    evoscore_delta: float
    solved_rate_delta: float
    zero_regress_rate_delta: float
    regressed: bool


@dataclass
class ExperimentTrendReport:
    """Cross-experiment trend analysis result.

    Attributes:
        snapshots: Ordered list of per-experiment statistics.
        trend_points: Pairwise deltas between consecutive snapshots.
        evoscore_slope: OLS slope of ``avg_evoscore`` over snapshot index
            (positive = improving over time).
        solved_rate_slope: OLS slope of ``solved_rate``.
        zero_regress_rate_slope: OLS slope of ``zero_regress_rate``.
        evoscore_direction: ``"improving"``, ``"worsening"``, or ``"stable"``.
        any_regression: ``True`` if any consecutive pair regressed on at least
            one metric beyond *regression_threshold*.
        regressions: Subset of *trend_points* where regression was detected.
        regression_threshold: The threshold used when computing this report.
    """

    snapshots: list[ExperimentSnapshot]
    trend_points: list[TrendPoint]
    evoscore_slope: float
    solved_rate_slope: float
    zero_regress_rate_slope: float
    evoscore_direction: str  # "improving" | "worsening" | "stable"
    any_regression: bool
    regressions: list[TrendPoint] = field(default_factory=list)
    regression_threshold: float = 0.02


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ols_slope(values: list[float]) -> float:
    """Return the OLS slope of *values* over index 0, 1, …, n-1.

    Returns 0.0 for fewer than two data points.
    Uses :func:`statistics.linear_regression` (Python ≥ 3.10, stdlib only).
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    slope, _ = statistics.linear_regression(xs, values)
    return slope


def _direction(slope: float, threshold: float = 0.001) -> str:
    if slope > threshold:
        return "improving"
    if slope < -threshold:
        return "worsening"
    return "stable"


def _load_snapshot(exp_dir: Path) -> Optional[ExperimentSnapshot]:
    """Parse a single experiment directory into an :class:`ExperimentSnapshot`.

    Reads ``experiment.json`` for metadata, then walks every episode
    subdirectory for per-episode ``conclusion.json`` files.

    Returns ``None`` if the directory is missing or contains no parseable
    episodes.
    """
    if not exp_dir.is_dir():
        return None

    # Read experiment manifest
    manifest_path = exp_dir / "experiment.json"
    experiment_id = exp_dir.name
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment_id = manifest.get("experiment_id", experiment_id)
        except (json.JSONDecodeError, OSError):
            pass

    # Walk episode directories: any subdirectory that has at least one
    # turn_N/conclusion.json file is an episode.
    evoscores: list[float] = []
    solved_flags: list[float] = []
    zero_regress_flags: list[float] = []

    for ep_dir in sorted(exp_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        # Collect conclusion files from turn sub-directories
        turn_conclusions = sorted(ep_dir.glob("turn_*/conclusion.json"))
        if not turn_conclusions:
            continue
        try:
            # Use the last turn's conclusion to get episode-level solved state
            last_conclu = json.loads(
                turn_conclusions[-1].read_text(encoding="utf-8")
            )
            # evoscore_1.0 is the neutral-gamma variant used in Otter's own
            # summary output — prefer it, fall back to evoscore if present.
            evoscore_val = last_conclu.get(
                "evoscore_1.0", last_conclu.get("evoscore", None)
            )
            if evoscore_val is None:
                continue
            evoscores.append(float(evoscore_val))
            solved_flags.append(1.0 if last_conclu.get("is_solved", False) else 0.0)
            zero_regress_flags.append(
                1.0 if last_conclu.get("num_regress", 1) == 0 else 0.0
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            continue

    if not evoscores:
        return None

    n = len(evoscores)
    return ExperimentSnapshot(
        experiment_dir=exp_dir.resolve(),
        experiment_id=experiment_id,
        avg_evoscore=sum(evoscores) / n,
        solved_rate=sum(solved_flags) / n,
        zero_regress_rate=sum(zero_regress_flags) / n,
        episode_count=n,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def analyze_experiment_trend(
    experiment_dirs: list[Path],
    *,
    regression_threshold: float = 0.02,
) -> ExperimentTrendReport:
    """Compute a cross-experiment behavioral trend report.

    Args:
        experiment_dirs: Ordered list of experiment directories to compare.
            The order determines the temporal sequence; typically oldest first.
        regression_threshold: Minimum absolute change in a metric that
            constitutes a regression.  Defaults to 0.02 (2 percentage points).

    Returns:
        An :class:`ExperimentTrendReport` summarising slopes, directions, and
        per-pair regression flags.

    Raises:
        ValueError: If fewer than two parseable experiment directories are
            provided.

    Example::

        from pathlib import Path
        from otter.analysis.trend import analyze_experiment_trend

        report = analyze_experiment_trend(
            [Path("experiments/claude-v1"), Path("experiments/claude-v2")],
            regression_threshold=0.02,
        )
        print(report.evoscore_direction)   # "improving" / "worsening" / "stable"
        print(report.any_regression)       # True / False
    """
    snapshots: list[ExperimentSnapshot] = []
    for d in experiment_dirs:
        snap = _load_snapshot(Path(d))
        if snap is not None:
            snapshots.append(snap)

    if len(snapshots) < 2:
        raise ValueError(
            f"Need at least 2 parseable experiment directories; "
            f"got {len(snapshots)} from {len(experiment_dirs)} provided paths."
        )

    # Compute pairwise deltas
    trend_points: list[TrendPoint] = []
    for prev, curr in zip(snapshots, snapshots[1:]):
        es_delta = curr.avg_evoscore - prev.avg_evoscore
        sr_delta = curr.solved_rate - prev.solved_rate
        zr_delta = curr.zero_regress_rate - prev.zero_regress_rate
        # Adverse direction: evoscore ↓, solved_rate ↓, zero_regress_rate ↓
        regressed = (
            es_delta < -regression_threshold
            or sr_delta < -regression_threshold
            or zr_delta < -regression_threshold
        )
        trend_points.append(
            TrendPoint(
                from_id=prev.experiment_id,
                to_id=curr.experiment_id,
                evoscore_delta=es_delta,
                solved_rate_delta=sr_delta,
                zero_regress_rate_delta=zr_delta,
                regressed=regressed,
            )
        )

    es_slope = _ols_slope([s.avg_evoscore for s in snapshots])
    sr_slope = _ols_slope([s.solved_rate for s in snapshots])
    zr_slope = _ols_slope([s.zero_regress_rate for s in snapshots])

    regressions = [tp for tp in trend_points if tp.regressed]

    return ExperimentTrendReport(
        snapshots=snapshots,
        trend_points=trend_points,
        evoscore_slope=es_slope,
        solved_rate_slope=sr_slope,
        zero_regress_rate_slope=zr_slope,
        evoscore_direction=_direction(es_slope),
        any_regression=bool(regressions),
        regressions=regressions,
        regression_threshold=regression_threshold,
    )
