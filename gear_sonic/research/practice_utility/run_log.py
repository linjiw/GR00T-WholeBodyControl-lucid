"""Parse SONIC training logs into comparable per-iteration metric series.

Two verifications depend on being able to compare two runs numerically:

**No-op parity** -- a run with the research callbacks disabled must reproduce a
native run. Without this, every "research off" baseline in the programme is
unaudited.

**Resume equivalence** -- N iterations uninterrupted must match N/2, capsule,
resume, N/2. Without this, a paired continuation from a capsule is not the same
experiment as one that never stopped.

The trainer prints a Rich table that is re-rendered many times per iteration, so
the same iteration appears repeatedly with identical values. Parsing keeps the
*last* occurrence of each iteration, which is the completed one.

Nothing here writes files or touches a GPU; it reads logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Strip terminal control sequences before matching.
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

ITERATION = re.compile(r"Learning iteration\s+(\d+)")
THROUGHPUT = re.compile(r"Computation:\s*([0-9.]+)\s*steps/s")
COLLECTION = re.compile(r"Collection:\s*([0-9.]+)s")
LEARNING = re.compile(r"Learning\s+([0-9.]+)s")

#: ``label: value`` rows inside the table. Labels may contain slashes and dots.
METRIC = re.compile(r"([A-Za-z][\w/. ()-]*?)\s*:\s*(-?[0-9]+\.?[0-9]*(?:[eE][-+]?\d+)?)\s*$")

#: Metrics whose equality across two runs is the parity claim.
PARITY_KEYS = (
    "Mean rewards",
    "Mean length",
    "Mean entropy",
    "Mean action noise std",
)


@dataclass
class Iteration:
    """One completed learning iteration."""

    index: int
    metrics: dict[str, float] = field(default_factory=dict)
    steps_per_second: float | None = None
    collection_seconds: float | None = None
    learning_seconds: float | None = None

    @property
    def wall_seconds(self) -> float | None:
        if self.collection_seconds is None or self.learning_seconds is None:
            return None
        return self.collection_seconds + self.learning_seconds


@dataclass
class RunLog:
    """Every completed iteration parsed from one training log."""

    path: str
    iterations: list[Iteration]

    @property
    def indices(self) -> list[int]:
        return [it.index for it in self.iterations]

    def series(self, key: str) -> dict[int, float]:
        return {it.index: it.metrics[key] for it in self.iterations if key in it.metrics}

    def median_steps_per_second(self, skip_first: int = 1) -> float | None:
        """Median throughput, skipping warm-up iterations.

        The first iteration carries CUDA graph capture and allocator warm-up, so
        including it understates steady-state throughput.
        """
        rates = [
            it.steps_per_second
            for it in self.iterations[skip_first:]
            if it.steps_per_second is not None
        ]
        if not rates:
            return None
        rates.sort()
        middle = len(rates) // 2
        return rates[middle] if len(rates) % 2 else 0.5 * (rates[middle - 1] + rates[middle])

    def median_iteration_seconds(self, skip_first: int = 1) -> float | None:
        values = [
            it.wall_seconds
            for it in self.iterations[skip_first:]
            if it.wall_seconds is not None
        ]
        if not values:
            return None
        values.sort()
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])


def parse_run_log(path: str | Path) -> RunLog:
    """Parse a training log, keeping the last render of each iteration."""
    text = ANSI.sub("", Path(path).read_text(errors="replace")).replace("\r", "\n")
    by_index: dict[int, Iteration] = {}
    current: Iteration | None = None

    for raw in text.split("\n"):
        line = raw.strip().strip("│").strip()
        if not line:
            continue

        found = ITERATION.search(line)
        if found:
            # A later render of the same iteration supersedes the earlier one.
            current = Iteration(index=int(found.group(1)))
            by_index[current.index] = current
            continue
        if current is None:
            continue

        rate = THROUGHPUT.search(line)
        if rate:
            current.steps_per_second = float(rate.group(1))
        collection = COLLECTION.search(line)
        if collection:
            current.collection_seconds = float(collection.group(1))
        learning = LEARNING.search(line)
        if learning:
            current.learning_seconds = float(learning.group(1))

        metric = METRIC.match(line)
        if metric:
            key = metric.group(1).strip()
            if key.lower().startswith(("computation", "learning iteration")):
                continue
            current.metrics[key] = float(metric.group(2))

    return RunLog(path=str(path), iterations=[by_index[i] for i in sorted(by_index)])


@dataclass
class ComparisonResult:
    """Outcome of comparing two runs on the parity metrics."""

    shared_iterations: list[int]
    max_abs_difference: dict[str, float]
    matched: dict[str, bool]
    tolerance: float
    missing_keys: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return bool(self.matched) and all(self.matched.values()) and not self.missing_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_iterations": self.shared_iterations,
            "tolerance": self.tolerance,
            "max_abs_difference": self.max_abs_difference,
            "matched": self.matched,
            "missing_keys": self.missing_keys,
            "passes": self.passes,
        }


def compare_runs(
    left: RunLog,
    right: RunLog,
    keys: tuple[str, ...] = PARITY_KEYS,
    tolerance: float = 0.0,
    skip_iterations: int = 0,
) -> ComparisonResult:
    """Compare two runs on ``keys`` over the iterations they share.

    ``tolerance = 0.0`` demands exact equality, which is the right default for a
    no-op parity claim on CPU-reduced scalars. GPU physics is not bitwise
    reproducible, so a nonzero tolerance is an honest choice for cross-run
    comparisons -- but it must be stated, not assumed, hence it is explicit here
    and recorded in the result.
    """
    shared = sorted(set(left.indices) & set(right.indices))
    shared = [i for i in shared if i >= skip_iterations]
    if not shared:
        return ComparisonResult([], {}, {}, tolerance, missing_keys=["no shared iterations"])

    differences: dict[str, float] = {}
    matched: dict[str, bool] = {}
    missing: list[str] = []
    for key in keys:
        a, b = left.series(key), right.series(key)
        usable = [i for i in shared if i in a and i in b]
        if not usable:
            missing.append(key)
            continue
        worst = max(abs(a[i] - b[i]) for i in usable)
        differences[key] = worst
        matched[key] = worst <= tolerance

    return ComparisonResult(shared, differences, matched, tolerance, missing)


def throughput_report(log: RunLog, num_envs: int, steps_per_env: int = 24) -> dict[str, Any]:
    """Per-iteration cost and what it implies for a branch and a campaign."""
    seconds = log.median_iteration_seconds()
    rate = log.median_steps_per_second()
    report: dict[str, Any] = {
        "iterations_parsed": len(log.iterations),
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "transitions_per_iteration": num_envs * steps_per_env,
        "median_steps_per_second": rate,
        "median_iteration_seconds": seconds,
    }
    if seconds:
        report["env_steps_per_second"] = num_envs * steps_per_env / seconds
        report["hours_per_128_iteration_branch"] = 128 * seconds / 3600.0
    return report
