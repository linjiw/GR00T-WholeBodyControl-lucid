"""Select the contexts a probe campaign will measure, and freeze the plan.

Study 0 measures the utility of practising particular contexts. *Which*
contexts are chosen decides what the study can conclude, so the selection is
made and frozen before any branch runs.

The central rule: **do not select only the hardest contexts.** Sampling the top
of the failure distribution would quietly convert the study into a hard-example
study, and it would make the headline question unanswerable -- if every probed
context is difficult, no evidence can show that difficulty and utility come
apart. :func:`stratified_select` therefore balances across failure quartiles,
motion family, and contact regime, and :func:`validate_manifest` refuses a plan
whose coverage collapsed.

Freezing matters for the same reason. Choosing contexts after seeing which ones
produced interesting branches would turn a measurement into a search over
20-odd chances to find a reversal.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from gear_sonic.research.practice_utility.schema import ContextKey, sha256_of

#: Stratification axes. Failure quartile is mandatory -- it is the axis that
#: keeps the study from becoming a hard-example study.
DEFAULT_STRATA: tuple[str, ...] = ("failure_quartile", "family", "contact_regime")

#: A campaign must probe contexts from every failure quartile.
REQUIRED_QUARTILES = (0, 1, 2, 3)


class ManifestError(RuntimeError):
    """Raised when a probe plan is unusable as designed."""


@dataclass
class ContextCandidate:
    """A context that could be probed, with the features used to stratify."""

    context: ContextKey
    failure_rate: float
    sampling_probability: float
    family: str
    contact_regime: str = "unknown"
    reference_speed: float = 0.0
    partition: str = "adaptation"
    failure_quartile: int = -1
    extras: dict[str, float] = field(default_factory=dict)

    def stratum(self, axes: Sequence[str]) -> tuple[Any, ...]:
        return tuple(getattr(self, axis) for axis in axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "context_id": self.context.context_id,
            "failure_rate": self.failure_rate,
            "failure_quartile": self.failure_quartile,
            "sampling_probability": self.sampling_probability,
            "family": self.family,
            "contact_regime": self.contact_regime,
            "reference_speed": self.reference_speed,
            "partition": self.partition,
            "extras": self.extras,
        }


@dataclass
class ProbeManifest:
    """A frozen campaign: which contexts, at which stages, with what dose."""

    campaign_id: str
    stages: list[str]
    contexts_per_stage: dict[str, list[ContextCandidate]]
    seeds: list[int]
    epsilon: float
    kernel_radius_bins: int
    horizons: dict[str, int]
    pool_sha256: str
    split_sha256: str
    strata: tuple[str, ...] = DEFAULT_STRATA
    notes: str = ""

    @property
    def num_branches(self) -> int:
        """Intervention branches. Controls are shared per (stage, seed)."""
        return sum(len(v) for v in self.contexts_per_stage.values()) * len(self.seeds)

    @property
    def num_control_branches(self) -> int:
        return len(self.stages) * len(self.seeds)

    @property
    def manifest_sha256(self) -> str:
        return sha256_of(
            {
                "campaign_id": self.campaign_id,
                "contexts": {
                    stage: sorted(c.context.context_id for c in candidates)
                    for stage, candidates in sorted(self.contexts_per_stage.items())
                },
                "seeds": sorted(self.seeds),
                "epsilon": self.epsilon,
                "kernel_radius_bins": self.kernel_radius_bins,
                "horizons": self.horizons,
                "pool_sha256": self.pool_sha256,
                "split_sha256": self.split_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "practice_utility_probe_manifest",
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "manifest_sha256": self.manifest_sha256,
            "stages": self.stages,
            "seeds": self.seeds,
            "epsilon": self.epsilon,
            "kernel_radius_bins": self.kernel_radius_bins,
            "horizons": self.horizons,
            "strata": list(self.strata),
            "pool_sha256": self.pool_sha256,
            "split_sha256": self.split_sha256,
            "num_intervention_branches": self.num_branches,
            "num_control_branches": self.num_control_branches,
            "coverage": self.coverage(),
            "notes": self.notes,
            "contexts_per_stage": {
                stage: [c.to_dict() for c in candidates]
                for stage, candidates in sorted(self.contexts_per_stage.items())
            },
        }

    def coverage(self) -> dict[str, Any]:
        """Stratum coverage per stage, for auditing the plan before it runs."""
        report: dict[str, Any] = {}
        for stage, candidates in sorted(self.contexts_per_stage.items()):
            quartiles: dict[int, int] = defaultdict(int)
            families: dict[str, int] = defaultdict(int)
            regimes: dict[str, int] = defaultdict(int)
            for candidate in candidates:
                quartiles[candidate.failure_quartile] += 1
                families[candidate.family] += 1
                regimes[candidate.contact_regime] += 1
            rates = [c.failure_rate for c in candidates]
            report[stage] = {
                "num_contexts": len(candidates),
                "failure_quartiles": dict(sorted(quartiles.items())),
                "families": dict(sorted(families.items())),
                "contact_regimes": dict(sorted(regimes.items())),
                "failure_rate_min": min(rates) if rates else 0.0,
                "failure_rate_max": max(rates) if rates else 0.0,
                "failure_rate_median": statistics.median(rates) if rates else 0.0,
            }
        return report


def assign_failure_quartiles(candidates: Iterable[ContextCandidate]) -> list[ContextCandidate]:
    """Label each candidate with its failure-rate quartile within the pool.

    Quartiles are assigned by rank, not by fixed thresholds, so the strata stay
    populated whatever the shape of the failure distribution -- which changes a
    great deal between an early and a late checkpoint.

    Raises if every candidate shares one failure rate. That is what a snapshot
    taken before any episode looks like: SONIC seeds ``adp_samp_num_episodes``
    and ``adp_samp_num_failures`` with the same ``init_num_failures``, so every
    rate reads exactly 1.0. Ranking ties would manufacture four strata out of
    noise and the campaign would have no difficulty axis at all, so this fails
    loudly instead.
    """
    candidates = list(candidates)
    if not candidates:
        return []
    rates = {round(c.failure_rate, 9) for c in candidates}
    if len(rates) == 1 and len(candidates) > 1:
        raise ManifestError(
            f"all {len(candidates)} candidates share failure rate "
            f"{next(iter(rates))}; the sampler snapshot carries no statistics. "
            "Take it after warm-up (PracticeContextCallback.snapshot_at_step > 0), "
            "not at install."
        )
    order = sorted(range(len(candidates)), key=lambda i: candidates[i].failure_rate)
    for rank, index in enumerate(order):
        quartile = min(3, (rank * 4) // len(order))
        candidates[index].failure_quartile = quartile
    return candidates


def stratified_select(
    candidates: Sequence[ContextCandidate],
    num_contexts: int,
    seed: int = 0,
    strata: Sequence[str] = DEFAULT_STRATA,
) -> list[ContextCandidate]:
    """Pick ``num_contexts`` spread as evenly as possible across strata.

    Round-robin over strata rather than proportional sampling: proportional
    allocation would leave rare strata -- exactly the ones most likely to behave
    differently -- with no representative at all in a 24-context budget.
    """
    if num_contexts <= 0:
        raise ManifestError(f"num_contexts must be positive, got {num_contexts}")
    if not candidates:
        raise ManifestError("no candidates to select from")

    buckets: dict[tuple, list[ContextCandidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.stratum(strata)].append(candidate)

    # Deterministic, seed-dependent order inside and across strata.
    for members in buckets.values():
        members.sort(key=lambda c: sha256_of({"seed": seed, "id": c.context.context_id}))
    stratum_order = sorted(
        buckets, key=lambda key: sha256_of({"seed": seed, "stratum": [str(k) for k in key]})
    )

    selected: list[ContextCandidate] = []
    cursor = 0
    while len(selected) < num_contexts:
        progressed = False
        for key in stratum_order:
            if cursor < len(buckets[key]):
                selected.append(buckets[key][cursor])
                progressed = True
                if len(selected) == num_contexts:
                    break
        if not progressed:
            break                      # every stratum exhausted
        cursor += 1

    if len(selected) < num_contexts:
        raise ManifestError(
            f"only {len(selected)} distinct contexts available for a request of "
            f"{num_contexts}; widen the candidate pool or reduce the budget"
        )
    return selected


def build_probe_manifest(
    campaign_id: str,
    candidates_per_stage: dict[str, Sequence[ContextCandidate]],
    num_contexts: int,
    seeds: Sequence[int],
    horizons: dict[str, int],
    pool_sha256: str,
    split_sha256: str,
    epsilon: float = 0.10,
    kernel_radius_bins: int = 1,
    selection_seed: int = 20260818,
    strata: Sequence[str] = DEFAULT_STRATA,
    notes: str = "",
) -> ProbeManifest:
    """Build and validate a frozen probe campaign."""
    if not candidates_per_stage:
        raise ManifestError("no policy stages supplied")
    if not seeds:
        raise ManifestError("at least one seed is required")
    if not horizons:
        raise ManifestError("at least one horizon is required")
    if not 0.0 <= epsilon <= 1.0:
        raise ManifestError(f"epsilon must be in [0, 1], got {epsilon}")

    contexts_per_stage: dict[str, list[ContextCandidate]] = {}
    for stage, candidates in candidates_per_stage.items():
        labelled = assign_failure_quartiles(candidates)
        contexts_per_stage[stage] = stratified_select(
            labelled, num_contexts, seed=selection_seed, strata=strata
        )

    manifest = ProbeManifest(
        campaign_id=campaign_id,
        stages=sorted(contexts_per_stage),
        contexts_per_stage=contexts_per_stage,
        seeds=list(seeds),
        epsilon=epsilon,
        kernel_radius_bins=kernel_radius_bins,
        horizons=dict(horizons),
        pool_sha256=pool_sha256,
        split_sha256=split_sha256,
        strata=tuple(strata),
        notes=notes,
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: ProbeManifest, min_families: int = 3) -> None:
    """Refuse a plan whose stratification collapsed.

    A campaign that probes only difficult contexts, or only one motion family,
    cannot answer the question it was built for, and finding that out after the
    GPU time is spent is expensive.
    """
    if not manifest.stages:
        raise ManifestError("manifest has no stages")

    for stage, candidates in manifest.contexts_per_stage.items():
        if not candidates:
            raise ManifestError(f"stage {stage!r} has no contexts")

        ids = [c.context.context_id for c in candidates]
        if len(set(ids)) != len(ids):
            raise ManifestError(f"stage {stage!r} contains duplicate contexts")

        quartiles = {c.failure_quartile for c in candidates}
        missing = [q for q in REQUIRED_QUARTILES if q not in quartiles]
        if missing:
            raise ManifestError(
                f"stage {stage!r} covers failure quartiles {sorted(quartiles)} but misses "
                f"{missing}. Probing only part of the difficulty range turns this into a "
                "hard-example study and makes difficulty/utility divergence unobservable."
            )

        families = {c.family for c in candidates}
        if len(families) < min_families:
            raise ManifestError(
                f"stage {stage!r} covers only {len(families)} motion families "
                f"({sorted(families)}); at least {min_families} are required for the "
                "result to say anything about motion structure"
            )

        foreign = sorted({c.partition for c in candidates} - {"adaptation"})
        if foreign:
            raise ManifestError(
                f"stage {stage!r} probes contexts from partitions {foreign}; probes must "
                "come from the adaptation split or the evaluation is contaminated"
            )
