"""Per-motion paired comparison between two evaluated arms.

Three seeds is thin, and a table of three-seed means throws away almost all the
information the evaluation actually produced: every run scores the *same* frozen
102-motion panel, motion by motion, and SONIC's evaluation callback records
exactly which motions failed (``eval/failed_metrics_dict.failed_idxes``). Two
arms evaluated at the same seed on the same panel are therefore **paired at the
motion level**, and the difference between them can be estimated with 102 x 3
paired observations instead of 3.

This adds no hypothesis. The preregistered decision rules are scored, unchanged,
by ``analyze_lucid_s.py`` from the same three-seed means they were written
against. What lives here is *precision* on those same estimands -- a confidence
interval and a paired discordance count for a difference that was going to be
reported either way.

Two guards, because a paired analysis is worthless if the pairing is wrong:

* the panel is only comparable if both runs evaluated the same motions in the
  same order. Every recorded ``failed_idxes`` comes with the matching
  ``failed_keys``, so any index the two runs both failed must name the same
  motion. A mismatch raises rather than quietly comparing different robots.
* a bootstrap over motions alone would understate uncertainty across seeds, so
  the default resamples seeds *and* motions -- the hierarchy the data has.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class MotionOutcome:
    """One evaluated run, as a per-motion success vector."""

    seed: int
    mode: str
    preset: str
    motion_count: int
    failed_indices: frozenset[int]
    #: index -> motion key, for the failed motions only. That is all SONIC
    #: records, and it is enough to prove two panels line up.
    failed_keys: dict[int, str]

    @property
    def success_rate(self) -> float:
        return (self.motion_count - len(self.failed_indices)) / self.motion_count

    def successes(self) -> list[int]:
        """1/0 per motion, in panel order."""
        return [0 if i in self.failed_indices else 1 for i in range(self.motion_count)]


def read_outcome(metrics_path: str | Path, seed: int, mode: str, preset: str) -> MotionOutcome:
    """Load one ``metrics_eval.json`` into a per-motion success vector."""
    payload = json.loads(Path(metrics_path).read_text())
    failed = payload.get("eval/failed_metrics_dict") or {}
    indices = [int(i) for i in failed.get("failed_idxes", [])]
    keys = list(failed.get("failed_keys", []))
    rate = float(payload["eval/success/success_rate"])
    count = _motion_count(rate, len(indices))
    if len(keys) != len(indices):
        keys = [""] * len(indices)
    return MotionOutcome(
        seed=seed,
        mode=mode,
        preset=preset,
        motion_count=count,
        failed_indices=frozenset(indices),
        failed_keys=dict(zip(indices, keys, strict=True)),
    )


def _motion_count(success_rate: float, num_failed: int) -> int:
    """Recover the panel size from the reported rate and the failure count.

    ``success_rate = (n - failed) / n``, so ``n = failed / (1 - rate)``. Done in
    integers and checked, because a panel size that does not reproduce the
    reported rate means the two numbers came from different runs.
    """
    if num_failed == 0:
        raise ValueError("cannot infer the panel size from a run with no failures")
    if not 0.0 <= success_rate < 1.0:
        raise ValueError(f"success rate {success_rate} is not in [0, 1)")
    count = round(num_failed / (1.0 - success_rate))
    if count <= 0 or abs((count - num_failed) / count - success_rate) > 1e-6:
        raise ValueError(
            f"{num_failed} failures and a rate of {success_rate} do not describe one panel"
        )
    return count


def assert_comparable(left: MotionOutcome, right: MotionOutcome) -> None:
    """Refuse to pair two runs whose panels are not demonstrably the same."""
    if left.motion_count != right.motion_count:
        raise ValueError(
            f"panel sizes differ: {left.motion_count} vs {right.motion_count}"
        )
    if left.seed != right.seed:
        raise ValueError(f"paired runs must share a checkpoint seed: {left.seed} vs {right.seed}")
    if left.preset != right.preset:
        raise ValueError(f"paired runs must share a preset: {left.preset} vs {right.preset}")
    shared = set(left.failed_keys) & set(right.failed_keys)
    disagreeing = [
        index
        for index in sorted(shared)
        if left.failed_keys[index] and right.failed_keys[index]
        and left.failed_keys[index] != right.failed_keys[index]
    ]
    if disagreeing:
        index = disagreeing[0]
        raise ValueError(
            "panel order differs between runs: index "
            f"{index} is {left.failed_keys[index]!r} in one and "
            f"{right.failed_keys[index]!r} in the other"
        )


@dataclass(frozen=True)
class PairedResult:
    """A treatment-minus-reference difference, in success-rate points."""

    delta_pts: float
    ci_low_pts: float
    ci_high_pts: float
    treatment_only_wins: int
    reference_only_wins: int
    both_agree: int
    num_motions: int
    num_seeds: int
    per_seed_delta_pts: dict[int, float]
    bootstrap_samples: int
    excludes_zero: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_pts": self.delta_pts,
            "ci95_pts": [self.ci_low_pts, self.ci_high_pts],
            "excludes_zero": self.excludes_zero,
            "discordant_motions": {
                "treatment_only_success": self.treatment_only_wins,
                "reference_only_success": self.reference_only_wins,
                "agree": self.both_agree,
            },
            "num_motions": self.num_motions,
            "num_seeds": self.num_seeds,
            "per_seed_delta_pts": self.per_seed_delta_pts,
            "bootstrap_samples": self.bootstrap_samples,
        }


def auc_weights(grid: Sequence[float]) -> list[float]:
    """Trapezoid weights that turn per-cell rates into the normalised profile AUC.

    The AUC is a *fixed* linear functional of the per-cell success rates, so a
    motion's contribution to it is a fixed weighted sum of that motion's
    outcomes across cells -- which makes the AUC difference bootstrappable at
    the motion level exactly like any single cell.
    """
    scales = list(grid)
    if len(scales) < 2 or any(b <= a for a, b in zip(scales, scales[1:])):
        raise ValueError(f"grid must be strictly increasing with >= 2 points, got {scales}")
    width = scales[-1] - scales[0]
    weights = [0.0] * len(scales)
    for i in range(len(scales) - 1):
        span = 0.5 * (scales[i + 1] - scales[i]) / width
        weights[i] += span
        weights[i + 1] += span
    return weights


def auc_scores(
    by_preset: dict[str, Sequence[MotionOutcome]], grid: dict[str, float]
) -> dict[int, list[float]]:
    """Per-seed, per-motion contribution to the profile AUC, in [0, 1].

    Every cell of the profile must be present for a seed, or that seed is
    dropped: a profile with a hole in it is not a profile, and filling the hole
    with a mean would invent the very number the interval is meant to bound.
    """
    ordered = sorted(grid.items(), key=lambda kv: kv[1])
    weights = auc_weights([s for _, s in ordered])
    per_seed: dict[int, list[list[float]]] = {}
    for (preset, _), weight in zip(ordered, weights, strict=True):
        runs = {run.seed: run for run in by_preset.get(preset, [])}
        for checkpoint_seed, run in runs.items():
            per_seed.setdefault(checkpoint_seed, []).append(
                [weight * value for value in run.successes()]
            )
    complete = len(ordered)
    return {
        checkpoint_seed: [sum(cell) for cell in zip(*columns, strict=True)]
        for checkpoint_seed, columns in per_seed.items()
        if len(columns) == complete
    }


def paired_scores(
    treatment: dict[int, Sequence[float]],
    reference: dict[int, Sequence[float]],
    samples: int = 10000,
    seed: int = 20260828,
) -> PairedResult:
    """Hierarchical paired bootstrap over real-valued per-motion scores."""
    seeds = sorted(set(treatment) & set(reference))
    if not seeds:
        raise ValueError("no checkpoint seed is present in both arms")
    per_seed_pairs = {
        s: list(zip(treatment[s], reference[s], strict=True)) for s in seeds
    }
    return _bootstrap(per_seed_pairs, seeds, samples, seed)


def paired_difference(
    treatment: Sequence[MotionOutcome],
    reference: Sequence[MotionOutcome],
    samples: int = 10000,
    seed: int = 20260828,
) -> PairedResult:
    """Hierarchical paired bootstrap of treatment minus reference.

    Resamples seeds with replacement and, within each drawn seed, motions with
    replacement -- so the interval reflects both the motion panel and the small
    number of training seeds, rather than pretending 102 motions are 102
    independent experiments.
    """
    by_seed_t = {run.seed: run for run in treatment}
    by_seed_r = {run.seed: run for run in reference}
    seeds = sorted(set(by_seed_t) & set(by_seed_r))
    if not seeds:
        raise ValueError("no checkpoint seed is present in both arms")

    per_seed_pairs: dict[int, list[tuple[float, float]]] = {}
    for checkpoint_seed in seeds:
        left, right = by_seed_t[checkpoint_seed], by_seed_r[checkpoint_seed]
        assert_comparable(left, right)
        per_seed_pairs[checkpoint_seed] = list(
            zip(left.successes(), right.successes(), strict=True)
        )
    return _bootstrap(per_seed_pairs, seeds, samples, seed)


def _bootstrap(
    per_seed_pairs: dict[int, list[tuple[float, float]]],
    seeds: list[int],
    samples: int,
    seed: int,
) -> PairedResult:
    per_seed_delta = {
        s: 100.0 * sum(t - r for t, r in pairs) / len(pairs)
        for s, pairs in per_seed_pairs.items()
    }
    total = [pair for pairs in per_seed_pairs.values() for pair in pairs]
    delta = 100.0 * sum(t - r for t, r in total) / len(total)

    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        picked_seeds = [rng.choice(seeds) for _ in seeds]
        values = []
        for checkpoint_seed in picked_seeds:
            pairs = per_seed_pairs[checkpoint_seed]
            values.extend(rng.choice(pairs) for _ in pairs)
        draws.append(100.0 * sum(t - r for t, r in values) / len(values))
    draws.sort()
    low = draws[int(0.025 * (samples - 1))]
    high = draws[int(0.975 * (samples - 1))]

    return PairedResult(
        delta_pts=delta,
        ci_low_pts=low,
        ci_high_pts=high,
        treatment_only_wins=sum(1 for t, r in total if t > r),
        reference_only_wins=sum(1 for t, r in total if r > t),
        both_agree=sum(1 for t, r in total if t == r),
        num_motions=len(per_seed_pairs[seeds[0]]),
        num_seeds=len(seeds),
        per_seed_delta_pts=per_seed_delta,
        bootstrap_samples=samples,
        excludes_zero=(low > 0.0) or (high < 0.0),
    )


def outcomes_from_receipt(
    receipt: dict[str, Any], presets: Iterable[str] | None = None
) -> dict[tuple[str, str], list[MotionOutcome]]:
    """Every run in an evaluation receipt, keyed by ``(mode, preset)``.

    Runs whose ``metrics_eval.json`` is missing, unreadable, or has no failures
    to infer a panel size from are skipped -- a perfect arm is real but cannot
    be paired by this route, and silently treating it as absent is wrong, so it
    is reported by its absence from the returned mapping rather than by a zero.
    """
    wanted = set(presets) if presets else None
    out: dict[tuple[str, str], list[MotionOutcome]] = {}
    for run in receipt.get("runs", {}).values():
        preset = run.get("preset")
        if wanted is not None and preset not in wanted:
            continue
        path = run.get("metrics_path")
        if not path or not Path(path).is_file():
            continue
        try:
            outcome = read_outcome(path, int(run["checkpoint_seed"]), run["mode"], preset)
        except (ValueError, KeyError):
            continue
        out.setdefault((run["mode"], preset), []).append(outcome)
    return out
