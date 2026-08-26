"""Leakage-free directional calibration for the frozen LUCID latent gap.

``latent_gap_p90`` is non-negative, while counterfactual practice utility is
signed.  Its raw sign therefore cannot answer whether practising a context
helps or harms.  This module implements the deliberately small, CPU-only test
used to answer that question: an affine ridge calibration evaluated by nested,
context-grouped out-of-fold prediction.

The public algorithm artifact is both fully specified and self-hashed.  The
claim path accepts exactly that artifact, so folds, model selection, bootstrap,
and pass thresholds cannot be changed after outcomes are observed.  No learned
utility estimator or allocator lives here; the one-dimensional calibrator is a
statistical test of the already-frozen proxy.
"""

# ruff: noqa: I001  # repository isort and Ruff force-sort rules conflict

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from gear_sonic.research.practice_utility import proxy_audit as PA
from gear_sonic.research.practice_utility.schema import sha256_of

ALGORITHM_ARTIFACT_KIND = "practice_utility_latent_directional_calibration"
ALGORITHM_SCHEMA_VERSION = 1
LATENT_PROXY = "latent_gap_p90"


def default_algorithm_artifact() -> dict[str, Any]:
    """Return the one supported, immutable directional-test specification."""
    artifact: dict[str, Any] = {
        "kind": ALGORITHM_ARTIFACT_KIND,
        "schema_version": ALGORITHM_SCHEMA_VERSION,
        "proxy": LATENT_PROXY,
        "target": "signed_dose_normalized_practice_utility_H_l",
        "raw_proxy_sign_allowed": False,
        "model": {
            "class": "univariate_affine_ridge",
            "features": [LATENT_PROXY],
            "fit_intercept": True,
            "penalize_intercept": False,
            "standardization": "training_fold_mean_and_population_sd",
            "objective": "mean_squared_error_plus_alpha_times_slope_squared",
            "ridge_alpha_grid": [0.0, 0.1, 1.0],
            "selection_metric": "context_macro_mean_squared_error",
            "selection_tie_break": "smallest_alpha",
        },
        "cross_validation": {
            "method": "nested_grouped_out_of_fold",
            "outer_folds": 5,
            "inner_folds": 4,
            "split_unit": "motion_family",
            "context_leakage_guard": "context_id",
            "assignment": "hash_tied_greedy_context_count_balance",
            "fold_seed": 20260826,
            "same_motion_family_must_share_fold": True,
            "same_context_across_seeds_or_stages_must_share_fold": True,
        },
        "support": {
            "min_records": 20,
            "min_contexts": 10,
            "min_motion_families": 8,
            "min_rankable_motion_families": 4,
            "min_contexts_for_family_rank": 2,
            "min_scorable_contexts_outside_deadband": 8,
            "min_unique_proxy_values": 3,
        },
        "deadband": {
            "source": "same_estimand_noise_floor_abs_quantile",
            "quantile": 0.95,
            "units": "must_equal_practice_utility_units",
            "comparison": "exclude_abs_true_utility_less_than_or_equal_to_deadband",
        },
        "diagnostics": {
            "unit": "context_mean_across_replicate_seeds",
            "rank_grouping": "macro_average_within_motion_family",
            "metrics": [
                "outer_oof_sign_accuracy",
                "outer_oof_spearman",
                "outer_oof_pairwise_accuracy",
            ],
        },
        "bootstrap": {
            "method": "hierarchical_motion_family_then_context_block",
            "replicates": 1000,
            "seed": 20260827,
            "confidence": 0.95,
            "interval": "equal_tailed_empirical_nearest_rank",
            "resample_seed_replicates_within_context": False,
        },
        "latent_proxy_pass_rule": {
            "all_conditions_required": True,
            "min_outer_oof_sign_accuracy": PA.SUFFICIENCY["min_sign_accuracy"],
            "min_outer_oof_spearman": PA.SUFFICIENCY["min_abs_spearman"],
            "min_outer_oof_pairwise_accuracy": PA.SUFFICIENCY["min_pairwise_accuracy"],
            "min_bootstrap_sign_accuracy_lower": 0.5,
            "min_bootstrap_spearman_lower": 0.0,
            "min_bootstrap_pairwise_accuracy_lower": 0.5,
            "bootstrap_lower_comparison": "strictly_greater_than",
            "negative_motion_family_directions_allowed": 0,
        },
        "decision_semantics": {
            "pass_field": "supports_latent_proxy_claim",
            "separate_from_inverse_estimator_authorization": True,
            "failure_does_not_by_itself_authorize_an_estimator": True,
            "gate_a_must_pass_before_labels_are_read": True,
        },
    }
    artifact["algorithm_sha256"] = sha256_of(artifact)
    return artifact


def validate_algorithm_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that an embedded preregistration exactly matches the implementation."""
    if not isinstance(artifact, Mapping):
        raise ValueError("directional_calibration must be an algorithm artifact object")
    actual = dict(artifact)
    recorded = actual.pop("algorithm_sha256", None)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("directional calibration requires a 64-character algorithm_sha256")
    computed = sha256_of(actual)
    if recorded != computed:
        raise ValueError(
            "directional calibration algorithm_sha256 does not match its canonical content"
        )
    expected = default_algorithm_artifact()
    if dict(artifact) != expected:
        raise ValueError(
            "directional calibration artifact differs from the only implemented, "
            "preregistered algorithm"
        )
    return expected


@dataclass(frozen=True)
class CalibrationRow:
    """One label/proxy observation; repeated seeds share ``context_id``."""

    sample_id: str
    context_id: str
    motion_family: str
    latent_gap_p90: float
    utility: float

    def __post_init__(self) -> None:
        if not self.sample_id or not self.context_id or not self.motion_family:
            raise ValueError("calibration rows require sample, context, and motion-family IDs")
        if not math.isfinite(self.latent_gap_p90) or self.latent_gap_p90 < 0.0:
            raise ValueError("latent_gap_p90 must be finite and non-negative")
        if not math.isfinite(self.utility):
            raise ValueError("practice utility must be finite")


@dataclass(frozen=True)
class CalibrationDesignRow:
    """Outcome-free row used to freeze and audit fold feasibility."""

    sample_id: str
    context_id: str
    motion_family: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.context_id or not self.motion_family:
            raise ValueError("design rows require sample, context, and motion-family IDs")


class _GroupedRow(Protocol):
    context_id: str
    motion_family: str


@dataclass(frozen=True)
class _AffineRidge:
    mean_x: float
    scale_x: float
    intercept: float
    slope: float
    alpha: float

    def predict(self, value: float) -> float:
        return self.intercept + self.slope * ((value - self.mean_x) / self.scale_x)


@dataclass(frozen=True)
class _OOFPrediction:
    row: CalibrationRow
    prediction: float
    outer_fold: int


@dataclass(frozen=True)
class _ContextPoint:
    context_id: str
    motion_family: str
    prediction: float
    utility: float
    num_records: int


@dataclass
class DirectionalCalibrationResult:
    """Machine-readable result of the directional latent-proxy test."""

    status: str
    algorithm_sha256: str
    gate_a_prerequisite_passed: bool
    utility_units: str
    deadband: float | None = None
    supports_latent_proxy_claim: bool = False
    decision_complete: bool = False
    blockers: list[dict[str, str]] = field(default_factory=list)
    support: dict[str, Any] = field(default_factory=dict)
    folds: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    bootstrap: dict[str, Any] | None = None
    pass_rule: dict[str, Any] = field(default_factory=dict)
    pass_conditions: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _blocked(
    algorithm: Mapping[str, Any],
    utility_units: str,
    gate_a_passed: bool,
    code: str,
    message: str,
    *,
    support: Mapping[str, Any] | None = None,
) -> DirectionalCalibrationResult:
    return DirectionalCalibrationResult(
        status="blocked",
        algorithm_sha256=str(algorithm["algorithm_sha256"]),
        gate_a_prerequisite_passed=gate_a_passed,
        utility_units=utility_units,
        blockers=[{"code": code, "message": message}],
        support=dict(support or {}),
        pass_rule=dict(algorithm["latent_proxy_pass_rule"]),
    )


def noise_deadband(noise_values: Sequence[float], quantile: float = 0.95) -> float:
    """Nearest-rank absolute-noise quantile, in the input utility units."""
    values = sorted(abs(float(value)) for value in noise_values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("noise-floor values must be a non-empty finite sequence")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("deadband quantile must lie in (0, 1]")
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _hash_order(seed: int, *parts: str) -> int:
    raw = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _family_group_folds(rows: Sequence[_GroupedRow], num_folds: int, seed: int) -> dict[str, int]:
    families_by_context: dict[str, str] = {}
    for row in rows:
        previous = families_by_context.setdefault(row.context_id, row.motion_family)
        if previous != row.motion_family:
            raise ValueError(f"context {row.context_id!r} appears in multiple motion families")
    contexts_by_family: dict[str, list[str]] = defaultdict(list)
    for context_id, family in families_by_context.items():
        contexts_by_family[family].append(context_id)

    assignment: dict[str, int] = {}
    fold_context_counts = [0] * num_folds
    fold_family_counts = [0] * num_folds
    ordered_families = sorted(
        contexts_by_family,
        key=lambda family: (
            -len(contexts_by_family[family]),
            _hash_order(seed, family),
            family,
        ),
    )
    for family in ordered_families:
        fold = min(
            range(num_folds),
            key=lambda index: (
                fold_context_counts[index],
                fold_family_counts[index],
                index,
            ),
        )
        for context_id in contexts_by_family[family]:
            assignment[context_id] = fold
        fold_context_counts[fold] += len(contexts_by_family[family])
        fold_family_counts[fold] += 1
    return assignment


def _fit_affine_ridge(rows: Sequence[CalibrationRow], alpha: float) -> _AffineRidge:
    x_values = [row.latent_gap_p90 for row in rows]
    y_values = [row.utility for row in rows]
    mean_x = statistics.fmean(x_values)
    scale_x = statistics.pstdev(x_values)
    if scale_x <= 0.0:
        scale_x = 1.0
    z_values = [(value - mean_x) / scale_x for value in x_values]
    mean_y = statistics.fmean(y_values)
    covariance = statistics.fmean(z * (value - mean_y) for z, value in zip(z_values, y_values))
    variance = statistics.fmean(z * z for z in z_values)
    slope = covariance / (variance + alpha) if variance + alpha > 0.0 else 0.0
    return _AffineRidge(mean_x, scale_x, mean_y, slope, alpha)


def _context_macro_mse(predictions: Sequence[tuple[CalibrationRow, float]]) -> float:
    errors: dict[str, list[float]] = defaultdict(list)
    for row, prediction in predictions:
        errors[row.context_id].append((prediction - row.utility) ** 2)
    return statistics.fmean(statistics.fmean(values) for values in errors.values())


def _select_alpha(
    rows: Sequence[CalibrationRow], outer_fold: int, algorithm: Mapping[str, Any]
) -> tuple[float, dict[str, float]]:
    cv = algorithm["cross_validation"]
    inner_folds = int(cv["inner_folds"])
    inner_assignment = _family_group_folds(
        rows, inner_folds, int(cv["fold_seed"]) + 1009 * (outer_fold + 1)
    )
    scores: dict[float, float] = {}
    for alpha_value in algorithm["model"]["ridge_alpha_grid"]:
        alpha = float(alpha_value)
        predictions: list[tuple[CalibrationRow, float]] = []
        for fold in range(inner_folds):
            train = [row for row in rows if inner_assignment[row.context_id] != fold]
            valid = [row for row in rows if inner_assignment[row.context_id] == fold]
            if not train or not valid:
                raise ValueError("inner grouped CV produced an empty train or validation fold")
            if len({row.latent_gap_p90 for row in train}) < 2:
                raise ValueError("an inner training fold has no latent-gap variation")
            model = _fit_affine_ridge(train, alpha)
            predictions.extend((row, model.predict(row.latent_gap_p90)) for row in valid)
        scores[alpha] = _context_macro_mse(predictions)
    selected = min(scores, key=lambda alpha: (scores[alpha], alpha))
    return selected, {str(alpha): scores[alpha] for alpha in sorted(scores)}


def _context_points(predictions: Sequence[_OOFPrediction]) -> list[_ContextPoint]:
    grouped: dict[str, list[_OOFPrediction]] = defaultdict(list)
    for item in predictions:
        grouped[item.row.context_id].append(item)
    points = []
    for context_id, members in sorted(grouped.items()):
        families = {member.row.motion_family for member in members}
        if len(families) != 1:
            raise ValueError(f"context {context_id!r} crosses motion-family groups")
        points.append(
            _ContextPoint(
                context_id=context_id,
                motion_family=next(iter(families)),
                prediction=statistics.fmean(member.prediction for member in members),
                utility=statistics.fmean(member.row.utility for member in members),
                num_records=len(members),
            )
        )
    return points


def _metric_summary(points: Sequence[_ContextPoint], deadband: float) -> dict[str, Any]:
    scorable = [point for point in points if abs(point.utility) > deadband]
    sign = (
        PA.sign_accuracy(
            [point.prediction for point in scorable],
            [point.utility for point in scorable],
        )
        if scorable
        else float("nan")
    )
    by_family: dict[str, list[_ContextPoint]] = defaultdict(list)
    for point in points:
        by_family[point.motion_family].append(point)
    rankable = {
        family: members
        for family, members in by_family.items()
        if len(members) >= 2
        and len({point.prediction for point in members}) >= 2
        and len({point.utility for point in members}) >= 2
    }
    family_spearman = {
        family: PA.spearman(
            [point.prediction for point in members], [point.utility for point in members]
        )
        for family, members in sorted(rankable.items())
    }
    family_pairwise = {
        family: PA.pairwise_ranking_accuracy(
            [point.prediction for point in members], [point.utility for point in members]
        )
        for family, members in sorted(rankable.items())
    }
    spearman = statistics.fmean(family_spearman.values()) if family_spearman else float("nan")
    pairwise = statistics.fmean(family_pairwise.values()) if family_pairwise else float("nan")
    negative = sorted(family for family, value in family_spearman.items() if value < -0.1)
    return {
        "outer_oof_sign_accuracy": sign,
        "outer_oof_spearman": spearman,
        "outer_oof_pairwise_accuracy": pairwise,
        "num_scorable_contexts": len(scorable),
        "num_rankable_motion_families": len(rankable),
        "rank_excluded_unsupported_motion_families": sorted(set(by_family) - set(rankable)),
        "per_motion_family_spearman": family_spearman,
        "per_motion_family_pairwise_accuracy": family_pairwise,
        "negative_direction_motion_families": negative,
    }


def _nearest_rank_interval(values: Sequence[float], confidence: float) -> list[float]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return [float("nan"), float("nan")]
    tail = (1.0 - confidence) / 2.0

    def at(quantile: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
        return ordered[index]

    return [at(tail), at(1.0 - tail)]


def _hierarchical_bootstrap(
    points: Sequence[_ContextPoint], deadband: float, algorithm: Mapping[str, Any]
) -> dict[str, Any]:
    spec = algorithm["bootstrap"]
    rng = random.Random(int(spec["seed"]))
    by_family: dict[str, list[_ContextPoint]] = defaultdict(list)
    for point in points:
        by_family[point.motion_family].append(point)
    families = sorted(by_family)
    samples: dict[str, list[float]] = {
        "outer_oof_sign_accuracy": [],
        "outer_oof_spearman": [],
        "outer_oof_pairwise_accuracy": [],
    }
    for _ in range(int(spec["replicates"])):
        draw: list[_ContextPoint] = []
        for family_occurrence in range(len(families)):
            family = rng.choice(families)
            members = by_family[family]
            for context_occurrence in range(len(members)):
                point = rng.choice(members)
                draw.append(
                    dataclasses.replace(
                        point,
                        context_id=(
                            f"bootstrap_family_{family_occurrence}_context_{context_occurrence}"
                        ),
                        motion_family=f"bootstrap_family_{family_occurrence}_{family}",
                    )
                )
        metrics = _metric_summary(draw, deadband)
        for name in samples:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    confidence = float(spec["confidence"])
    return {
        "method": spec["method"],
        "replicates_requested": int(spec["replicates"]),
        "seed": int(spec["seed"]),
        "confidence": confidence,
        "interval": spec["interval"],
        "intervals": {
            name: {
                "lower": interval[0],
                "upper": interval[1],
                "valid_replicates": len(samples[name]),
            }
            for name, interval in (
                (name, _nearest_rank_interval(values, confidence))
                for name, values in samples.items()
            )
        },
    }


def _design_support_summary(rows: Sequence[_GroupedRow]) -> dict[str, Any]:
    contexts_by_family: dict[str, set[str]] = defaultdict(set)
    family_by_context: dict[str, str] = {}
    for row in rows:
        previous = family_by_context.setdefault(row.context_id, row.motion_family)
        if previous != row.motion_family:
            raise ValueError(f"context {row.context_id!r} appears in multiple motion families")
        contexts_by_family[row.motion_family].add(row.context_id)
    contexts_per_family = {
        family: len(contexts) for family, contexts in sorted(contexts_by_family.items())
    }
    return {
        "num_records": len(rows),
        "num_contexts": len(family_by_context),
        "num_motion_families": len(contexts_by_family),
        "num_rankable_motion_families": sum(count >= 2 for count in contexts_per_family.values()),
        "contexts_per_motion_family": contexts_per_family,
    }


def validate_design_support(
    rows: Iterable[CalibrationDesignRow], algorithm_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit nested-fold feasibility using only IDs/groups, never labels or proxy values."""
    algorithm = validate_algorithm_artifact(algorithm_artifact)
    materialized = list(rows)
    duplicates = [
        sample_id
        for sample_id, count in _counts(row.sample_id for row in materialized).items()
        if count > 1
    ]
    blockers: list[dict[str, str]] = []
    if duplicates:
        blockers.append(
            {
                "code": "directional_design_sample_ids_not_unique",
                "message": f"duplicate design sample IDs: {sorted(duplicates)[:8]}",
            }
        )
    try:
        support = _design_support_summary(materialized)
    except ValueError as error:
        return {
            "status": "blocked",
            "algorithm_sha256": algorithm["algorithm_sha256"],
            "outcomes_read": False,
            "support": {},
            "folds": [],
            "blockers": [{"code": "directional_design_grouping_invalid", "message": str(error)}],
        }

    limits = algorithm["support"]
    for code, field_name, minimum in (
        ("directional_records_insufficient", "num_records", limits["min_records"]),
        ("directional_contexts_insufficient", "num_contexts", limits["min_contexts"]),
        (
            "directional_motion_families_insufficient",
            "num_motion_families",
            limits["min_motion_families"],
        ),
        (
            "directional_rankable_families_insufficient",
            "num_rankable_motion_families",
            limits["min_rankable_motion_families"],
        ),
    ):
        if int(support[field_name]) < int(minimum):
            blockers.append(
                {
                    "code": code,
                    "message": (
                        f"{field_name}={support[field_name]} is below the preregistered minimum "
                        f"{minimum}"
                    ),
                }
            )

    folds = []
    cv = algorithm["cross_validation"]
    outer_folds = int(cv["outer_folds"])
    inner_folds = int(cv["inner_folds"])
    if not blockers:
        outer = _family_group_folds(materialized, outer_folds, int(cv["fold_seed"]))
        for fold in range(outer_folds):
            train = [row for row in materialized if outer[row.context_id] != fold]
            test = [row for row in materialized if outer[row.context_id] == fold]
            train_families = {row.motion_family for row in train}
            test_families = {row.motion_family for row in test}
            if not train or not test or train_families & test_families:
                blockers.append(
                    {
                        "code": "directional_outer_fold_support_insufficient",
                        "message": f"outer fold {fold} is empty or leaks a motion family",
                    }
                )
                continue
            inner = _family_group_folds(
                train, inner_folds, int(cv["fold_seed"]) + 1009 * (fold + 1)
            )
            inner_feasible = all(
                any(inner[row.context_id] == inner_fold for row in train)
                and any(inner[row.context_id] != inner_fold for row in train)
                for inner_fold in range(inner_folds)
            )
            if not inner_feasible:
                blockers.append(
                    {
                        "code": "directional_inner_fold_support_insufficient",
                        "message": f"outer fold {fold} cannot support {inner_folds} inner folds",
                    }
                )
            folds.append(
                {
                    "outer_fold": fold,
                    "train_motion_families": sorted(train_families),
                    "test_motion_families": sorted(test_families),
                    "train_context_ids": sorted({row.context_id for row in train}),
                    "test_context_ids": sorted({row.context_id for row in test}),
                    "inner_folds_feasible": inner_feasible,
                }
            )
    return {
        "status": "ready" if not blockers else "blocked",
        "algorithm_sha256": algorithm["algorithm_sha256"],
        "outcomes_read": False,
        "support": support,
        "folds": folds,
        "blockers": blockers,
    }


def _support_summary(rows: Sequence[CalibrationRow], deadband: float) -> dict[str, Any]:
    support = _design_support_summary(rows)
    context_utilities: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        context_utilities[row.context_id].append(row.utility)
    support.update(
        {
            "num_scorable_contexts_outside_deadband": sum(
                abs(statistics.fmean(values)) > deadband for values in context_utilities.values()
            ),
            "num_unique_proxy_values": len({row.latent_gap_p90 for row in rows}),
        }
    )
    return support


def _support_blockers(
    rows: Sequence[CalibrationRow], deadband: float, algorithm: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    support = _support_summary(rows, deadband)
    limits = algorithm["support"]
    blockers = []

    def require(code: str, field: str, minimum: int) -> None:
        value = int(support[field])
        if value < minimum:
            blockers.append(
                {
                    "code": code,
                    "message": f"{field}={value} is below the preregistered minimum {minimum}",
                }
            )

    require("directional_records_insufficient", "num_records", int(limits["min_records"]))
    require("directional_contexts_insufficient", "num_contexts", int(limits["min_contexts"]))
    require(
        "directional_motion_families_insufficient",
        "num_motion_families",
        int(limits["min_motion_families"]),
    )
    require(
        "directional_rankable_families_insufficient",
        "num_rankable_motion_families",
        int(limits["min_rankable_motion_families"]),
    )
    require(
        "directional_sign_support_insufficient",
        "num_scorable_contexts_outside_deadband",
        int(limits["min_scorable_contexts_outside_deadband"]),
    )
    require(
        "directional_proxy_variation_insufficient",
        "num_unique_proxy_values",
        int(limits["min_unique_proxy_values"]),
    )
    return support, blockers


def run_directional_test(
    rows: Iterable[CalibrationRow],
    *,
    gate_a_passed: bool,
    noise_floor_values: Sequence[float],
    utility_units: str,
    algorithm_artifact: Mapping[str, Any],
) -> DirectionalCalibrationResult:
    """Run the preregistered test, reading labels only after Gate A passes.

    ``rows`` may be a lazy iterable.  The Gate-A check occurs before it is
    iterated, which gives callers a testable guarantee that failed/unrun Gate A
    cannot leak labels into calibration or model selection.
    """
    algorithm = validate_algorithm_artifact(algorithm_artifact)
    if not isinstance(utility_units, str) or not utility_units:
        raise ValueError("directional calibration requires explicit practice-utility units")
    if gate_a_passed is not True:
        return _blocked(
            algorithm,
            utility_units,
            False,
            "gate_a_prerequisite_not_passed",
            "directional calibration may not read utility labels until Gate A passes",
        )

    materialized = list(rows)
    duplicate_ids = [
        sample_id
        for sample_id, count in _counts(row.sample_id for row in materialized).items()
        if count > 1
    ]
    if duplicate_ids:
        return _blocked(
            algorithm,
            utility_units,
            True,
            "directional_sample_ids_not_unique",
            f"duplicate calibration sample IDs: {sorted(duplicate_ids)[:8]}",
        )
    try:
        deadband = noise_deadband(noise_floor_values, float(algorithm["deadband"]["quantile"]))
        support, blockers = _support_blockers(materialized, deadband, algorithm)
        if blockers:
            result = _blocked(
                algorithm,
                utility_units,
                True,
                blockers[0]["code"],
                blockers[0]["message"],
                support=support,
            )
            result.blockers = blockers
            result.deadband = deadband
            return result

        cv = algorithm["cross_validation"]
        outer_folds = int(cv["outer_folds"])
        assignment = _family_group_folds(materialized, outer_folds, int(cv["fold_seed"]))
        if set(assignment.values()) != set(range(outer_folds)):
            return _blocked(
                algorithm,
                utility_units,
                True,
                "directional_outer_fold_support_insufficient",
                "deterministic grouped assignment produced an empty outer fold",
                support=support,
            )

        oof: list[_OOFPrediction] = []
        folds = []
        for fold in range(outer_folds):
            train = [row for row in materialized if assignment[row.context_id] != fold]
            test = [row for row in materialized if assignment[row.context_id] == fold]
            selected_alpha, inner_scores = _select_alpha(train, fold, algorithm)
            model = _fit_affine_ridge(train, selected_alpha)
            oof.extend(_OOFPrediction(row, model.predict(row.latent_gap_p90), fold) for row in test)
            folds.append(
                {
                    "outer_fold": fold,
                    "train_motion_families": sorted({row.motion_family for row in train}),
                    "test_motion_families": sorted({row.motion_family for row in test}),
                    "train_context_ids": sorted({row.context_id for row in train}),
                    "test_context_ids": sorted({row.context_id for row in test}),
                    "selected_alpha": selected_alpha,
                    "inner_context_macro_mse_by_alpha": inner_scores,
                    "fitted_model": dataclasses.asdict(model),
                }
            )
    except ValueError as error:
        result = _blocked(
            algorithm,
            utility_units,
            True,
            "directional_fold_or_group_support_invalid",
            str(error),
            support=_support_summary(materialized, deadband if "deadband" in locals() else 0.0),
        )
        result.deadband = deadband if "deadband" in locals() else None
        return result

    points = _context_points(oof)
    diagnostics = _metric_summary(points, deadband)
    bootstrap = _hierarchical_bootstrap(points, deadband, algorithm)
    pass_rule = algorithm["latent_proxy_pass_rule"]
    intervals = bootstrap["intervals"]
    conditions = {
        "outer_oof_sign_accuracy": diagnostics["outer_oof_sign_accuracy"]
        >= float(pass_rule["min_outer_oof_sign_accuracy"]),
        "outer_oof_spearman": diagnostics["outer_oof_spearman"]
        >= float(pass_rule["min_outer_oof_spearman"]),
        "outer_oof_pairwise_accuracy": diagnostics["outer_oof_pairwise_accuracy"]
        >= float(pass_rule["min_outer_oof_pairwise_accuracy"]),
        "bootstrap_sign_accuracy_lower": intervals["outer_oof_sign_accuracy"]["lower"]
        > float(pass_rule["min_bootstrap_sign_accuracy_lower"]),
        "bootstrap_spearman_lower": intervals["outer_oof_spearman"]["lower"]
        > float(pass_rule["min_bootstrap_spearman_lower"]),
        "bootstrap_pairwise_accuracy_lower": intervals["outer_oof_pairwise_accuracy"]["lower"]
        > float(pass_rule["min_bootstrap_pairwise_accuracy_lower"]),
        "negative_motion_family_directions": len(diagnostics["negative_direction_motion_families"])
        <= int(pass_rule["negative_motion_family_directions_allowed"]),
    }
    passed = all(conditions.values())
    return DirectionalCalibrationResult(
        status="pass" if passed else "fail",
        algorithm_sha256=str(algorithm["algorithm_sha256"]),
        gate_a_prerequisite_passed=True,
        utility_units=utility_units,
        deadband=deadband,
        supports_latent_proxy_claim=passed,
        decision_complete=True,
        support=support,
        folds=folds,
        diagnostics=diagnostics,
        bootstrap=bootstrap,
        pass_rule=dict(pass_rule),
        pass_conditions=conditions,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(counts)
