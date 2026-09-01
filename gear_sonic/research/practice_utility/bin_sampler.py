"""Difficulty-bin samplers for expanding-support curricula, with the safeguards.

The scalar-lambda curriculum this programme started from *moves* a point: at
epoch k every environment trains at intensity lambda_k, and the easy end of the
range is left behind. That is the design the literature says loses to direct
mixed training, and the design our own 128-iteration horizon study watched
collapse. The alternative here keeps the whole range in the mixture and advances
only its upper edge:

    d ~ P over bins in [0, d_max],   only d_max advances

Three samplers, in increasing order of what they assume:

:class:`UniformExpandingSampler`     uniform over the active bins.
:class:`ErrorWeightedSampler`        weights active bins by *lagged, frozen*
                                     per-bin failure statistics, subject to a
                                     floor on the aggregate easy-bin
                                     probability.
:class:`FixedMixtureSampler`         the target mixture, ignoring d_max. This is
                                     the direct-mixed baseline and the
                                     consolidation phase, and it exists here so
                                     that both are the *same* code path as the
                                     curriculum rather than a different one.

Why each safeguard exists
-------------------------
**Easy-bin floor.** An error-weighted sampler will drive probability to whatever
it is currently bad at, which is the hard end, and easy bins can be sampled to
near zero. That is how an expanding-support curriculum quietly becomes a moving
point again, and how easy-bin competence is lost without anything in the config
changing. The floor is a *preregistered aggregate* minimum over the easy bins,
enforced after weighting, and it is recorded in the receipt.

**Lagged, frozen statistics.** Weighting on the statistics of the batch you are
about to draw couples the sampler to its own noise and makes the realised
distribution unreproducible. Failure statistics are therefore snapshotted, held
for a frozen number of updates, and only then used.

**Coverage, fail-closed.** A PPO update whose batch contains no samples from an
active bin is not training on the distribution the receipt claims. Coverage is
checked against a required minimum per active bin and *raises* rather than
warning, because the alternative is a run that looks fine and means something
else.

**Effective bin count.** Per-bin counts alone hide the case where a bin is
nominally present but dominated by a handful of environments. The inverse
Herfindahl count over realised bin occupancy is recorded alongside the raw
counts: it equals the number of bins for a balanced batch and tends to one as a
single bin dominates.

**Deterministic resume.** The sampler owns its RNG and serialises it, so a
resumed run continues the same draw sequence rather than restarting it.

Nothing here touches the live curriculum callback. These are pure, seeded,
testable objects; wiring them into a training arm is a separate, later step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class DifficultyBins:
    """A fixed, ordered partition of normalised difficulty in [0, 1].

    ``centres`` are what the environment is actually configured to; the edges
    exist so that "which bin is this" has one answer. Bins are frozen for a
    campaign: a curriculum that redefines its own bins mid-run cannot be
    compared to anything.
    """

    centres: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.centres) < 2:
            raise ValueError(f"need at least two bins, got {self.centres}")
        if any(b <= a for a, b in zip(self.centres, self.centres[1:])):
            raise ValueError(f"bin centres must strictly increase, got {self.centres}")
        if not (0.0 <= self.centres[0] and self.centres[-1] <= 1.0):
            raise ValueError(f"bin centres must lie in [0, 1], got {self.centres}")

    @classmethod
    def uniform(cls, count: int) -> "DifficultyBins":
        """``count`` bins with centres spread over [0, 1] inclusive of both ends."""
        if count < 2:
            raise ValueError(f"need at least two bins, got {count}")
        return cls(tuple(i / (count - 1) for i in range(count)))

    def __len__(self) -> int:
        return len(self.centres)

    def active(self, d_max: float) -> int:
        """How many bins are inside the current support.

        The lowest bin is always active: a support that excludes the nominal
        distribution is not an expanding support, it is a moving one.
        """
        if not 0.0 <= d_max <= 1.0:
            raise ValueError(f"d_max must be in [0, 1], got {d_max}")
        return max(1, sum(1 for centre in self.centres if centre <= d_max + 1e-12))

    def easy_indices(self, easy_fraction: float) -> tuple[int, ...]:
        """The leading bins the easy-bin floor protects.

        Defined over the *whole* bin set, not the active prefix, so the set a
        floor protects does not change as the support expands.
        """
        if not 0.0 < easy_fraction <= 1.0:
            raise ValueError(f"easy_fraction must be in (0, 1], got {easy_fraction}")
        count = max(1, int(round(easy_fraction * len(self.centres))))
        return tuple(range(count))


def _normalise(weights: Sequence[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(weights)] * len(weights)
    return [float(w) / total for w in weights]


def apply_easy_floor(
    probabilities: Sequence[float], easy_indices: Sequence[int], floor: float
) -> list[float]:
    """Guarantee the easy bins at least ``floor`` aggregate probability.

    The deficit is taken from the non-easy bins in proportion to their current
    mass, so their *relative* ordering -- which is what the error weighting was
    computing -- survives the correction. If every active bin is an easy bin the
    floor is already satisfied and nothing moves.
    """
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}")
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("probabilities must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"probabilities must be finite and non-negative, got {values}")
    # Callers currently pass normalised probabilities, but normalising first is
    # what makes the public helper's floor invariant true for every valid input.
    values = _normalise(values)
    easy = sorted({i for i in easy_indices if 0 <= i < len(values)})
    if not easy:
        return _normalise(values)
    easy_mass = sum(values[i] for i in easy)
    if easy_mass >= floor or len(easy) == len(values):
        return _normalise(values)
    hard = [i for i in range(len(values)) if i not in set(easy)]
    hard_mass = sum(values[i] for i in hard)
    scale_easy = floor / easy_mass if easy_mass > 0 else 0.0
    for i in easy:
        values[i] = values[i] * scale_easy if easy_mass > 0 else floor / len(easy)
    if hard_mass > 0:
        scale_hard = (1.0 - floor) / hard_mass
        for i in hard:
            values[i] *= scale_hard
    else:
        for i in hard:
            values[i] = (1.0 - floor) / len(hard)
    return _normalise(values)


def effective_bin_count(counts: Sequence[int]) -> float:
    """Inverse-Herfindahl effective count over realised bin occupancy.

    This is ``1 / sum(p_k**2)`` for empirical bin probabilities ``p_k``. It is
    the number of occupied bins for a balanced batch and tends to one as a
    single bin dominates. Calling this a sample count would be dimensionally
    wrong: the maximum is the number of bins, not the number of environments.
    """
    values = [float(c) for c in counts if c > 0]
    if not values:
        return 0.0
    total = sum(values)
    return total * total / sum(v * v for v in values)


class CoverageError(RuntimeError):
    """An active bin was absent from a batch that claimed to train on it."""


@dataclass
class BinTelemetry:
    """Per-update accounting, for the receipt."""

    updates: int = 0
    counts: list[int] = field(default_factory=list)
    last_probabilities: list[float] = field(default_factory=list)
    min_effective_bins: float = math.inf
    coverage_failures: int = 0

    def observe(self, counts: Sequence[int], probabilities: Sequence[float]) -> None:
        if not self.counts:
            self.counts = [0] * len(counts)
        for index, value in enumerate(counts):
            self.counts[index] += int(value)
        self.updates += 1
        self.last_probabilities = list(probabilities)
        self.min_effective_bins = min(self.min_effective_bins, effective_bin_count(counts))

    def to_dict(self) -> dict[str, Any]:
        total = sum(self.counts)
        return {
            "updates": self.updates,
            "cumulative_counts": list(self.counts),
            "cumulative_fractions": ([c / total for c in self.counts] if total else []),
            "last_probabilities": list(self.last_probabilities),
            "min_effective_bin_count": (
                None if self.min_effective_bins == math.inf else self.min_effective_bins
            ),
            "coverage_failures": self.coverage_failures,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], num_bins: int) -> "BinTelemetry":
        """Validate and restore receipt accounting without resetting it."""
        if not isinstance(state, dict):
            raise ValueError("telemetry state must be an object")
        try:
            updates = int(state["updates"])
            counts = [int(value) for value in state["cumulative_counts"]]
            probabilities = [float(value) for value in state["last_probabilities"]]
            coverage_failures = int(state["coverage_failures"])
            raw_minimum = state["min_effective_bin_count"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed telemetry state") from exc
        if updates < 0 or coverage_failures < 0 or coverage_failures > updates:
            raise ValueError(
                "telemetry updates and coverage_failures must satisfy "
                f"0 <= failures <= updates, got {coverage_failures}/{updates}"
            )
        if counts and len(counts) != num_bins:
            raise ValueError(f"telemetry has {len(counts)} bin counts for a {num_bins}-bin sampler")
        if any(value < 0 for value in counts):
            raise ValueError(f"telemetry counts must be non-negative, got {counts}")
        if probabilities:
            if len(probabilities) != num_bins:
                raise ValueError(
                    f"telemetry has {len(probabilities)} probabilities for {num_bins} bins"
                )
            if any(
                not math.isfinite(value) or value < 0.0 for value in probabilities
            ) or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "telemetry probabilities must be finite, non-negative, and sum to one"
                )
        if updates == 0:
            if counts or probabilities or raw_minimum is not None or coverage_failures:
                raise ValueError("zero-update telemetry contains observations")
            minimum = math.inf
        else:
            if len(counts) != num_bins or len(probabilities) != num_bins:
                raise ValueError("observed telemetry is missing counts or probabilities")
            try:
                minimum = float(raw_minimum)
            except (TypeError, ValueError) as exc:
                raise ValueError("telemetry minimum effective bin count is invalid") from exc
            if not math.isfinite(minimum) or not 0.0 <= minimum <= num_bins + 1e-9:
                raise ValueError(
                    f"telemetry minimum effective bin count is out of range: {minimum}"
                )
        fractions = state.get("cumulative_fractions", [])
        if counts and fractions:
            try:
                recorded = [float(value) for value in fractions]
            except (TypeError, ValueError) as exc:
                raise ValueError("telemetry cumulative fractions are malformed") from exc
            total = sum(counts)
            expected = [value / total for value in counts] if total else [0.0] * num_bins
            if len(recorded) != num_bins or any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(recorded, expected, strict=True)
            ):
                raise ValueError("telemetry cumulative fractions do not match counts")
        return cls(
            updates=updates,
            counts=counts,
            last_probabilities=probabilities,
            min_effective_bins=minimum,
            coverage_failures=coverage_failures,
        )


class ExpandingSupportSampler:
    """Base: draw a difficulty bin per environment over the active support."""

    STATE_SCHEMA_VERSION = 2

    def __init__(
        self,
        bins: DifficultyBins,
        seed: int,
        d_max: float = 0.0,
        min_samples_per_active_bin: int = 1,
    ) -> None:
        self.bins = bins
        self.seed = int(seed)
        self._d_max = 0.0
        self.d_max = d_max
        self.min_samples_per_active_bin = int(min_samples_per_active_bin)
        if self.min_samples_per_active_bin < 1:
            raise ValueError(
                "min_samples_per_active_bin must be >= 1, got " f"{self.min_samples_per_active_bin}"
            )
        self.generator = torch.Generator().manual_seed(int(seed))
        self.telemetry = BinTelemetry()

    # ------------------------------------------------------------- support --

    @property
    def d_max(self) -> float:
        return self._d_max

    @d_max.setter
    def d_max(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"d_max must be in [0, 1], got {value}")
        if value < self._d_max - 1e-12:
            raise ValueError(
                f"support may only expand: d_max {self._d_max} -> {value}. "
                "Shrinking it is the moving-point curriculum this design exists "
                "to avoid; back off by other means if a guard demands it."
            )
        self._d_max = value

    @property
    def num_active(self) -> int:
        return self.bins.active(self._d_max)

    def probabilities(self) -> list[float]:
        """Uniform over the active prefix; zero elsewhere."""
        active = self.num_active
        return [1.0 / active if i < active else 0.0 for i in range(len(self.bins))]

    # -------------------------------------------------------------- drawing --

    def draw(self, count: int) -> torch.Tensor:
        """Bin index per environment, drawn from the current probabilities."""
        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        weights = torch.tensor(self.probabilities(), dtype=torch.double)
        return torch.multinomial(weights, count, replacement=True, generator=self.generator)

    def difficulties(self, assignment: torch.Tensor) -> torch.Tensor:
        centres = torch.tensor(self.bins.centres, dtype=torch.double)
        return centres[assignment.long()]

    # ------------------------------------------------------------- coverage --

    def bin_counts(self, assignment: torch.Tensor) -> list[int]:
        counts = [0] * len(self.bins)
        for index in assignment.tolist():
            counts[int(index)] += 1
        return counts

    def check_coverage(self, assignment: torch.Tensor, record: bool = True) -> list[int]:
        """Fail closed if any active bin is under-represented in this batch.

        Raises rather than warns. A batch missing an active bin trains on a
        different distribution than the receipt describes, and that difference
        is invisible in every aggregate the run reports.
        """
        counts = self.bin_counts(assignment)
        if record:
            self.telemetry.observe(counts, self.probabilities())
        active = self.num_active
        short = [
            (i, counts[i]) for i in range(active) if counts[i] < self.min_samples_per_active_bin
        ]
        if short:
            if record:
                self.telemetry.coverage_failures += 1
            raise CoverageError(
                f"active bins {[i for i, _ in short]} have "
                f"{[c for _, c in short]} samples, below the required "
                f"{self.min_samples_per_active_bin}; batch of {int(assignment.numel())} "
                f"across {active} active bins"
            )
        return counts

    # ---------------------------------------------------------- persistence --

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "schema_version": self.STATE_SCHEMA_VERSION,
            "bins": list(self.bins.centres),
            "seed": self.seed,
            "d_max": self._d_max,
            "min_samples_per_active_bin": self.min_samples_per_active_bin,
            "generator_state": self.generator.get_state().tolist(),
            "telemetry": self.telemetry.to_dict(),
        }

    def _validate_state_config(self, state: dict[str, Any]) -> None:
        """Reject a resume whose frozen construction differs from this sampler."""
        try:
            saved_seed = int(state["seed"])
            saved_minimum = int(state["min_samples_per_active_bin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sampler state has malformed base configuration") from exc
        mismatches = []
        if saved_seed != self.seed:
            mismatches.append(f"seed {saved_seed} != {self.seed}")
        if saved_minimum != self.min_samples_per_active_bin:
            mismatches.append(
                "min_samples_per_active_bin "
                f"{saved_minimum} != {self.min_samples_per_active_bin}"
            )
        if mismatches:
            raise ValueError(
                "refusing to resume into a different sampler configuration: "
                + "; ".join(mismatches)
            )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("sampler state must be an object")
        if state.get("schema_version") != self.STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported sampler state schema: "
                f"{state.get('schema_version')!r}; expected {self.STATE_SCHEMA_VERSION}"
            )
        expected_kind = type(self).__name__
        if state.get("kind") != expected_kind:
            raise ValueError(f"refusing to load {state.get('kind')!r} state into {expected_kind}")
        try:
            saved_bins = [float(value) for value in state["bins"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sampler state has malformed bin centres") from exc
        if saved_bins != list(self.bins.centres):
            raise ValueError(
                "refusing to resume into a different bin definition: "
                f"{saved_bins} != {list(self.bins.centres)}"
            )
        self._validate_state_config(state)
        try:
            d_max = float(state["d_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sampler state has an invalid d_max") from exc
        # ``active`` performs the same finite/range validation as live updates.
        self.bins.active(d_max)
        telemetry = BinTelemetry.from_dict(state.get("telemetry"), len(self.bins))
        try:
            generator_state = torch.tensor(state["generator_state"], dtype=torch.uint8)
            candidate_generator = torch.Generator()
            candidate_generator.set_state(generator_state)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("sampler state has an invalid RNG state") from exc

        # Mutate only after every field has passed validation, so a rejected
        # resume leaves the live sampler untouched.
        self._d_max = d_max
        self.generator.set_state(generator_state)
        self.telemetry = telemetry

    def receipt(self) -> dict[str, Any]:
        return {
            "sampler": type(self).__name__,
            "bins": list(self.bins.centres),
            "seed": self.seed,
            "d_max": self._d_max,
            "num_active_bins": self.num_active,
            "probabilities": self.probabilities(),
            "min_samples_per_active_bin": self.min_samples_per_active_bin,
            "telemetry": self.telemetry.to_dict(),
        }


class UniformExpandingSampler(ExpandingSupportSampler):
    """Uniform over the active support. The plain expanding-support arm."""


class FixedMixtureSampler(ExpandingSupportSampler):
    """The full target mixture from step zero, whatever ``d_max`` says.

    This is the direct-mixed baseline *and* the final consolidation phase. Both
    go through the same object as the curriculum so that "the baseline" and
    "the curriculum's finish" are provably the same distribution rather than two
    implementations that are meant to agree.
    """

    def probabilities(self) -> list[float]:
        return [1.0 / len(self.bins)] * len(self.bins)

    @property
    def num_active(self) -> int:
        return len(self.bins)


class ErrorWeightedSampler(ExpandingSupportSampler):
    """Weight active bins by lagged failure statistics, above an easy-bin floor.

    ``lag`` updates of statistics are held before any of them are used, and the
    weights are recomputed only after every ``update_every`` *released* lagged
    statistics, so the realised distribution over a batch does not depend on
    that batch. Both are frozen before the run, not tuned against it.
    """

    def __init__(
        self,
        bins: DifficultyBins,
        seed: int,
        d_max: float = 0.0,
        min_samples_per_active_bin: int = 1,
        easy_fraction: float = 0.4,
        easy_floor: float = 0.15,
        lag: int = 1,
        update_every: int = 1,
        smoothing: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(bins, seed, d_max, min_samples_per_active_bin)
        if not 0.10 <= easy_floor <= 0.20:
            raise ValueError(
                f"easy_floor must be in [0.10, 0.20] -- the preregistered band -- got {easy_floor}"
            )
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Validate at construction rather than waiting until the first weighted
        # probability query to discover a bad frozen configuration.
        bins.easy_indices(float(easy_fraction))
        self.easy_fraction = float(easy_fraction)
        self.easy_floor = float(easy_floor)
        self.lag = int(lag)
        self.update_every = int(update_every)
        self.smoothing = float(smoothing)
        self.temperature = float(temperature)
        self._pending: deque[list[float]] = deque(maxlen=self.lag)
        self._smoothed: list[float] | None = None
        self._live: list[float] | None = None
        self._observations = 0
        self._released_statistics = 0

    def observe_failure_rates(self, rates: Sequence[float]) -> None:
        """Record per-bin failure rates. They become usable ``lag`` calls later."""
        values = [float(r) for r in rates]
        if len(values) != len(self.bins):
            raise ValueError(f"expected {len(self.bins)} per-bin failure rates, got {len(values)}")
        if any(not math.isfinite(v) or v < 0.0 for v in values):
            raise ValueError(f"failure rates must be finite and non-negative, got {values}")
        self._observations += 1
        released = self._pending[0] if len(self._pending) == self.lag else None
        self._pending.append(values)
        if released is None:
            return
        self._released_statistics += 1
        if self._smoothed is None:
            self._smoothed = list(released)
        else:
            self._smoothed = [
                (1.0 - self.smoothing) * old + self.smoothing * new
                for old, new in zip(self._smoothed, released, strict=True)
            ]
        if self._released_statistics % self.update_every == 0:
            self._live = list(self._smoothed)

    def probabilities(self) -> list[float]:
        active = self.num_active
        if self._live is None:
            return super().probabilities()
        weights = [
            max(self._live[i], 0.0) ** (1.0 / self.temperature) if i < active else 0.0
            for i in range(len(self.bins))
        ]
        if sum(weights[:active]) <= 0.0:
            return super().probabilities()
        probabilities = _normalise(weights)
        easy = [i for i in self.bins.easy_indices(self.easy_fraction) if i < active]
        return apply_easy_floor(probabilities, easy, self.easy_floor)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "easy_fraction": self.easy_fraction,
                "easy_floor": self.easy_floor,
                "lag": self.lag,
                "update_every": self.update_every,
                "smoothing": self.smoothing,
                "temperature": self.temperature,
                "pending": [list(v) for v in self._pending],
                "smoothed": None if self._smoothed is None else list(self._smoothed),
                "live": None if self._live is None else list(self._live),
                "observations": self._observations,
                "released_statistics": self._released_statistics,
            }
        )
        return state

    def _validate_state_config(self, state: dict[str, Any]) -> None:
        super()._validate_state_config(state)
        try:
            saved = {
                "easy_fraction": float(state["easy_fraction"]),
                "easy_floor": float(state["easy_floor"]),
                "lag": int(state["lag"]),
                "update_every": int(state["update_every"]),
                "smoothing": float(state["smoothing"]),
                "temperature": float(state["temperature"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sampler state has malformed error-weighting configuration") from exc
        if not 0.0 < saved["easy_fraction"] <= 1.0:
            raise ValueError("saved easy_fraction must be in (0, 1]")
        if not 0.10 <= saved["easy_floor"] <= 0.20:
            raise ValueError("saved easy_floor is outside the preregistered band")
        if saved["lag"] < 1 or saved["update_every"] < 1:
            raise ValueError("saved lag and update_every must be >= 1")
        if not 0.0 < saved["smoothing"] <= 1.0 or saved["temperature"] <= 0.0:
            raise ValueError("saved smoothing or temperature is outside its valid range")
        current = {
            "easy_fraction": self.easy_fraction,
            "easy_floor": self.easy_floor,
            "lag": self.lag,
            "update_every": self.update_every,
            "smoothing": self.smoothing,
            "temperature": self.temperature,
        }
        mismatches = []
        for name, value in saved.items():
            expected = current[name]
            equal = (
                value == expected
                if isinstance(value, int)
                else math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
            )
            if not equal:
                mismatches.append(f"{name} {value} != {expected}")
        if mismatches:
            raise ValueError(
                "refusing to resume into a different sampler configuration: "
                + "; ".join(mismatches)
            )

    def _validate_dynamic_state(
        self, state: dict[str, Any]
    ) -> tuple[list[list[float]], list[float] | None, list[float] | None, int, int]:
        def vector(value: Any, name: str, *, optional: bool) -> list[float] | None:
            if value is None and optional:
                return None
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"saved {name} must be a vector")
            try:
                values = [float(item) for item in value]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"saved {name} is malformed") from exc
            if len(values) != len(self.bins) or any(
                not math.isfinite(item) or item < 0.0 for item in values
            ):
                raise ValueError(
                    f"saved {name} must contain {len(self.bins)} finite non-negative values"
                )
            return values

        try:
            observations = int(state["observations"])
            released = int(state["released_statistics"])
            raw_pending = state["pending"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sampler state has malformed cadence counters") from exc
        if observations < 0 or released < 0:
            raise ValueError("saved cadence counters must be non-negative")
        if not isinstance(raw_pending, (list, tuple)):
            raise ValueError("saved pending lag queue must be a list")
        pending = [
            vector(value, f"pending[{index}]", optional=False)
            for index, value in enumerate(raw_pending)
        ]
        expected_pending = min(observations, self.lag)
        expected_released = max(0, observations - self.lag)
        if len(pending) != expected_pending or released != expected_released:
            raise ValueError(
                "saved lag queue/counters are inconsistent: "
                f"observations={observations}, released={released}, "
                f"pending={len(pending)}, lag={self.lag}"
            )
        smoothed = vector(state.get("smoothed"), "smoothed", optional=True)
        live = vector(state.get("live"), "live", optional=True)
        if (released == 0) != (smoothed is None):
            raise ValueError("saved smoothed statistics disagree with the release counter")
        should_have_live = released >= self.update_every
        if should_have_live != (live is not None):
            raise ValueError("saved live weights disagree with the frozen update cadence")
        if (
            live is not None
            and smoothed is not None
            and released % self.update_every == 0
            and live != smoothed
        ):
            raise ValueError("saved live weights do not match the latest refresh")
        return pending, smoothed, live, observations, released

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Validate subclass state before the base loader mutates RNG/telemetry.
        self._validate_state_config(state)
        pending, smoothed, live, observations, released = self._validate_dynamic_state(state)
        super().load_state_dict(state)
        self._pending = deque((list(value) for value in pending), maxlen=self.lag)
        self._smoothed = None if smoothed is None else list(smoothed)
        self._live = None if live is None else list(live)
        self._observations = observations
        self._released_statistics = released

    def receipt(self) -> dict[str, Any]:
        out = super().receipt()
        out.update(
            {
                "easy_fraction": self.easy_fraction,
                "easy_floor": self.easy_floor,
                "easy_bin_indices": list(self.bins.easy_indices(self.easy_fraction)),
                "lag": self.lag,
                "update_every": self.update_every,
                "smoothing": self.smoothing,
                "temperature": self.temperature,
                "observations": self._observations,
                "released_statistics": self._released_statistics,
                "weighting_active": self._live is not None,
                "realised_easy_mass": sum(
                    self.probabilities()[i] for i in self.bins.easy_indices(self.easy_fraction)
                ),
            }
        )
        return out
