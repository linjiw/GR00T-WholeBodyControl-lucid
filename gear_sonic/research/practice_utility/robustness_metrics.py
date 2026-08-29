"""Capability metrics: the difficulty curve, its AUC, tail risk, and retention.

A single headline number cannot describe a generalist controller. Two policies
scoring (100, 100, 40) and (80, 80, 80) across three difficulties have almost the
same mean and completely different capability, and a robot that is excellent on
average but falls over in some physics configurations is not robust. So this
module reports four things that a mean hides:

* the **difficulty curve** ``S(lambda)`` over a fixed grid, reported cell by cell;
* the **capability AUC**, the integral of that curve, normalised so it reads on
  the success scale -- how much usable capability exists across the whole
  difficulty continuum rather than at one point;
* **tail risk**, as the worst difficulty cell and as ``CVaR`` over the worst
  fraction of individual environments;
* the **retention matrix**, checkpoint by difficulty, which is the only view in
  which catastrophic forgetting and capability accumulation look different: a
  curriculum whose easy column falls 95 -> 74 while its hard column climbs is
  trading, not accumulating.

Everything is computed from per-episode arrays that SONIC's evaluation already
writes. ``metrics_eval.json`` carries ``eval/all_metrics_dict`` with
``terminated`` and ``progress`` per scored episode, so no new instrumentation is
needed -- verified against a real receipt, where the arrays reproduce the
recorded ``success_rate`` and ``progress_rate`` exactly.

Two deliberate refusals. ``CVaR`` is computed on **progress** and never on binary
success: once the failure rate exceeds the tail fraction, every episode in the
tail is a zero and the statistic degenerates to "0.0" for policies that differ
enormously. And a cell whose per-episode arrays disagree with the aggregate the
same file reports is an error, not a datum -- silently trusting one of the two
would put an unfalsifiable number in a table.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gear_sonic.research.practice_utility.motion_paired import auc_weights

#: Per-episode quantities CVaR may be computed on. Binary success is excluded on
#: purpose: see the module docstring.
CVAR_ALLOWED = ("progress",)


@dataclass(frozen=True)
class CellScores:
    """One evaluated (checkpoint, preset) cell, as per-episode arrays."""

    preset: str
    difficulty: float | None
    num_episodes: int
    success: tuple[int, ...]
    progress: tuple[float, ...]
    motion_keys_sha256: str
    source: str

    @property
    def success_rate(self) -> float:
        return statistics.fmean(self.success)

    @property
    def progress_rate(self) -> float:
        return statistics.fmean(self.progress)


def read_cell(
    metrics_path: str | Path, preset: str, difficulty: float | None = None
) -> CellScores:
    """Load one evaluation cell's per-episode outcomes, checked against itself.

    The per-episode arrays and the aggregate rates in the same file are two
    views of one tensor. If they disagree, something upstream changed shape and
    neither can be trusted, so this raises rather than picking one.
    """
    path = Path(metrics_path)
    payload = json.loads(path.read_text())
    bundle = payload.get("eval/all_metrics_dict") or {}
    terminated = bundle.get("terminated")
    progress = bundle.get("progress")
    keys = bundle.get("motion_keys")
    if terminated is None or progress is None:
        raise ValueError(f"{path} has no per-episode terminated/progress arrays")
    if not (len(terminated) == len(progress) == len(keys or terminated)):
        raise ValueError(
            f"{path}: per-episode arrays disagree in length "
            f"({len(terminated)}, {len(progress)}, {len(keys or [])})"
        )
    success = tuple(0 if bool(t) else 1 for t in terminated)
    values = tuple(float(g) for g in progress)
    for name, derived, recorded in (
        ("success_rate", statistics.fmean(success), payload.get("eval/success/success_rate")),
        ("progress_rate", statistics.fmean(values), payload.get("eval/success/progress_rate")),
    ):
        if recorded is not None and abs(derived - float(recorded)) > 1e-9:
            raise ValueError(
                f"{path}: per-episode arrays give {name}={derived!r} but the file records "
                f"{recorded!r}; the two views of the same tensor disagree"
            )
    if any(not 0.0 <= g <= 1.0 for g in values):
        raise ValueError(f"{path}: progress outside [0, 1]")
    for succeeded, g in zip(success, values, strict=True):
        if bool(succeeded) != (g >= 1.0):
            raise ValueError(
                f"{path}: an episode is marked {'succeeded' if succeeded else 'failed'} "
                f"with progress {g}; success must mean progress == 1"
            )
    digest = hashlib.sha256(("\n".join(keys or []) + "\n").encode()).hexdigest()
    return CellScores(
        preset=preset,
        difficulty=difficulty,
        num_episodes=len(values),
        success=success,
        progress=values,
        motion_keys_sha256=digest,
        source=str(path),
    )


def _ordered(cells: Mapping[str, CellScores], grid: Mapping[str, float]) -> list[CellScores]:
    missing = sorted(set(grid) - set(cells))
    if missing:
        raise ValueError(f"difficulty grid is incomplete; missing cells: {missing}")
    digests = {cells[p].motion_keys_sha256 for p in grid}
    if len(digests) != 1:
        raise ValueError("cells were scored on different panels; they cannot be combined")
    return [cells[p] for p in sorted(grid, key=lambda p: grid[p])]


@dataclass(frozen=True)
class CurveResult:
    """The difficulty curve and everything read off it."""

    grid: dict[str, float]
    score: str
    per_cell: dict[str, float]
    capability_auc: float
    weights: dict[str, float]
    worst_cell: str
    worst_value: float
    num_episodes_per_cell: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "score": self.score,
            "per_cell": self.per_cell,
            "capability_auc": self.capability_auc,
            "auc_weights": self.weights,
            "worst_cell": {"preset": self.worst_cell, "value": self.worst_value},
            "num_episodes_per_cell": self.num_episodes_per_cell,
        }


def difficulty_curve(
    cells: Mapping[str, CellScores], grid: Mapping[str, float], score: str = "progress"
) -> CurveResult:
    """S(lambda) over the grid, its normalised AUC, and the worst cell.

    The weights are trapezoidal and sum to one, so the AUC is on the same scale
    as the cells it summarises -- an average capability over the difficulty
    continuum, not an area in arbitrary units.
    """
    if score not in ("progress", "success"):
        raise ValueError(f"score must be 'progress' or 'success', got {score!r}")
    ordered = _ordered(cells, grid)
    names = sorted(grid, key=lambda p: grid[p])
    values = [c.progress_rate if score == "progress" else c.success_rate for c in ordered]
    weights = auc_weights([grid[p] for p in names])
    per_cell = dict(zip(names, values, strict=True))
    worst = min(per_cell, key=lambda p: per_cell[p])
    sizes = {c.num_episodes for c in ordered}
    if len(sizes) != 1:
        raise ValueError(f"cells have different episode counts: {sorted(sizes)}")
    return CurveResult(
        grid=dict(grid),
        score=score,
        per_cell=per_cell,
        capability_auc=sum(w * v for w, v in zip(weights, values, strict=True)),
        weights=dict(zip(names, weights, strict=True)),
        worst_cell=worst,
        worst_value=per_cell[worst],
        num_episodes_per_cell=sizes.pop(),
    )


def cvar(values: Sequence[float], alpha: float = 0.2, score: str = "progress") -> dict[str, Any]:
    """Mean over the worst ``alpha`` fraction of environments.

    Fractional tails are interpolated: with n = 512 and alpha = 0.2 the tail is
    102.4 episodes, so the 103rd contributes its remaining 0.4 weight. Rounding
    instead would make the statistic jump discontinuously with n.
    """
    if score not in CVAR_ALLOWED:
        raise ValueError(
            f"CVaR is only defined here on {CVAR_ALLOWED}; binary success degenerates to 0 "
            "whenever the failure rate exceeds the tail fraction"
        )
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("CVaR of an empty sample is undefined")
    k = alpha * len(ordered)
    whole = int(math.floor(k))
    total = sum(ordered[:whole])
    weight = float(whole)
    remainder = k - whole
    if remainder > 0 and whole < len(ordered):
        total += remainder * ordered[whole]
        weight += remainder
    return {
        "alpha": alpha,
        "score": score,
        "cvar": total / weight,
        "n": len(ordered),
        "tail_size": k,
        "var_threshold": ordered[min(whole, len(ordered) - 1)],
        "mean": statistics.fmean(ordered),
    }


@dataclass(frozen=True)
class RetentionResult:
    """Checkpoint x difficulty, and the scalars that read it."""

    rows: list[str]
    columns: list[str]
    matrix: dict[str, dict[str, float]]
    per_column_peak: dict[str, float]
    per_column_final: dict[str, float]
    per_column_drop: dict[str, float]
    forgetting: float
    forgetting_column: str
    accumulating: bool
    tolerance_pts: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "matrix": self.matrix,
            "per_column": {
                c: {
                    "peak": self.per_column_peak[c],
                    "final": self.per_column_final[c],
                    "drop_from_peak": self.per_column_drop[c],
                }
                for c in self.columns
            },
            "forgetting": self.forgetting,
            "forgetting_column": self.forgetting_column,
            "accumulating": self.accumulating,
            "tolerance_pts": self.tolerance_pts,
            "notes": self.notes,
        }


def retention_matrix(
    cells: Mapping[str, Mapping[str, CellScores]],
    grid: Mapping[str, float],
    row_order: Sequence[str],
    score: str = "progress",
    tolerance_pts: float = 5.0,
) -> RetentionResult:
    """Capability at every checkpoint against every difficulty.

    ``forgetting`` is the largest drop, in points, from any column's peak to its
    value at the final checkpoint. It is the scalar that separates a policy which
    *accumulated* capability from one which merely *traded* it: a curriculum
    whose easy column peaks at 95 and ends at 74 has forgetting 21, whatever its
    hard column did. ``accumulating`` is true when no column gives up more than
    ``tolerance_pts`` from its peak.
    """
    missing_rows = [r for r in row_order if r not in cells]
    if missing_rows:
        raise ValueError(f"no cells for checkpoints: {missing_rows}")
    columns = sorted(grid, key=lambda p: grid[p])
    matrix: dict[str, dict[str, float]] = {}
    for row in row_order:
        row_cells = cells[row]
        absent = [c for c in columns if c not in row_cells]
        if absent:
            raise ValueError(f"checkpoint {row!r} is missing difficulty cells: {absent}")
        matrix[row] = {
            c: (row_cells[c].progress_rate if score == "progress" else row_cells[c].success_rate)
            for c in columns
        }
    peak = {c: max(matrix[r][c] for r in row_order) for c in columns}
    final = {c: matrix[row_order[-1]][c] for c in columns}
    drop = {c: 100.0 * (peak[c] - final[c]) for c in columns}
    worst_column = max(columns, key=lambda c: drop[c])
    notes = []
    if len(row_order) < 2:
        notes.append("a single checkpoint cannot show retention; forgetting is 0 by construction")
    return RetentionResult(
        rows=list(row_order),
        columns=columns,
        matrix=matrix,
        per_column_peak=peak,
        per_column_final=final,
        per_column_drop=drop,
        forgetting=drop[worst_column],
        forgetting_column=worst_column,
        accumulating=all(d <= tolerance_pts for d in drop.values()),
        tolerance_pts=tolerance_pts,
        notes=notes,
    )


def target_expectation(
    cells: Mapping[str, CellScores], p_target: Mapping[str, float], score: str = "progress"
) -> dict[str, Any]:
    """E[S(phi)] under a target distribution over difficulty, fixed in advance.

    The weights are the *deployment* prior, not the trapezoid rule: this answers
    "if the robot meets the whole expected dynamics distribution, which policy is
    better", which is a different question from the area under the curve.
    """
    if abs(sum(p_target.values()) - 1.0) > 1e-9:
        raise ValueError(f"p_target must sum to 1, got {sum(p_target.values())}")
    missing = sorted(set(p_target) - set(cells))
    if missing:
        raise ValueError(f"p_target names cells that were not evaluated: {missing}")
    per_cell = {
        p: (cells[p].progress_rate if score == "progress" else cells[p].success_rate)
        for p in p_target
    }
    return {
        "score": score,
        "p_target": dict(p_target),
        "per_cell": per_cell,
        "expectation": sum(p_target[p] * per_cell[p] for p in p_target),
    }


def pooled_tail(cells: Iterable[CellScores], alpha: float = 0.2) -> dict[str, Any]:
    """CVaR over every environment in every cell, pooled.

    The per-cell CVaR answers "how bad is the tail at this difficulty"; this
    answers "how bad is the tail across the whole distribution the robot will
    meet", which is the quantity a deployment cares about.
    """
    pool: list[float] = []
    presets: list[str] = []
    for cell in cells:
        pool.extend(cell.progress)
        presets.append(cell.preset)
    out = cvar(pool, alpha=alpha)
    out["pooled_over"] = sorted(presets)
    return out
