"""Turn paired branch evaluations into utility labels, and decide Gate A.

A label is the difference between two branches that started from the same
checkpoint, received the same compute, and differed only in how much training
mass went to one context::

    U_H(x) = [ J_eff(theta_H^x) - J_eff(theta_H^0) ] / realized extra dose

Three choices here are what separate a usable label from a plausible-looking
number.

**The comparison is against a continued-training control, never the source
checkpoint.** Continuing to train helps (or, on a small pool, hurts) regardless
of which context was emphasized. Differencing against the origin would fold
that generic effect into every context's "utility" and make them all look
similar and all wrong.

**The denominator is realized dose, not the nominal ``epsilon * H``.** Episodes
terminate early, motions differ in length, and resampling is stochastic, so the
intervention branch rarely receives the dose it was asked for -- least of all
on the hard contexts where the difference matters most.

**Efficacy alone cannot label a context positive.** A context that raises
success while worsening slip, contact, action smoothness, or actuator load has
bought the metric, not the capability. Labels are therefore three-way:
``safe_positive`` requires efficacy *and* clean non-inferiority *and* every harm
gate.

Gate A asks whether any of this is measurable at all: if repeated pairs on the
*same* context disagree as much as different contexts do, then context-level
utility is not identifiable at this granularity, and no estimator can fix it.
:func:`assess_identifiability` reports that comparison rather than assuming it.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    DoseReport,
    HarmVector,
    UtilityRecord,
)

#: Harm gates, in the units the quality metrics report. A positive entry is the
#: largest tolerated *worsening* relative to the control branch.
DEFAULT_HARM_GATES: dict[str, float] = {
    "action_rate": 0.05,
    "slip": 0.02,
    "contact_impulse": 25.0,
    "torque_saturation": 0.02,
    "clean_noninferiority": 0.02,
}

#: Efficacy below this magnitude is treated as no effect rather than a small one.
DEFAULT_EFFICACY_DEADBAND = 1e-4


@dataclass
class BranchEvaluation:
    """One branch's measured outcome at one horizon."""

    branch_id: str
    role: str
    horizon_label: str
    j_eff: float
    clean_j_eff: float
    action_rate: float
    foot_slip: float
    contact_impulse: float
    torque_saturation: float
    extras: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ("control", "intervention"):
            raise ValueError(f"unknown branch role: {self.role!r}")


@dataclass
class IdentifiabilityReport:
    """Gate A: is context-level utility measurable above branch noise?"""

    num_contexts: int
    num_replicates: int
    within_context_sd: float
    between_context_sd: float
    noise_floor_sd: float | None
    variance_ratio: float
    intraclass_correlation: float
    passes: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_contexts": self.num_contexts,
            "num_replicates": self.num_replicates,
            "within_context_sd": self.within_context_sd,
            "between_context_sd": self.between_context_sd,
            "noise_floor_sd": self.noise_floor_sd,
            "variance_ratio": self.variance_ratio,
            "intraclass_correlation": self.intraclass_correlation,
            "passes": self.passes,
            "reasons": self.reasons,
        }


def build_harm_vector(
    control: BranchEvaluation, intervention: BranchEvaluation
) -> HarmVector:
    """Paired quality deltas, signed so that positive means worse.

    ``clean_delta`` is the exception and follows efficacy convention, because it
    gates non-inferiority on nominal control rather than measuring harm.
    """
    if control.horizon_label != intervention.horizon_label:
        raise ValueError(
            f"horizon mismatch: {control.horizon_label!r} vs {intervention.horizon_label!r}"
        )
    return HarmVector(
        clean_delta=intervention.clean_j_eff - control.clean_j_eff,
        action_rate_delta=intervention.action_rate - control.action_rate,
        slip_delta=intervention.foot_slip - control.foot_slip,
        contact_impulse_delta=intervention.contact_impulse - control.contact_impulse,
        torque_saturation_delta=intervention.torque_saturation - control.torque_saturation,
    )


def classify_context(
    efficacy_delta: float,
    harm: HarmVector,
    gates: dict[str, float] | None = None,
    deadband: float = DEFAULT_EFFICACY_DEADBAND,
) -> tuple[str, list[str]]:
    """Label a context ``safe_positive`` / ``neutral`` / ``harmful``.

    Returns the label and the reasons behind it. A positive efficacy that
    breaches any harm gate is ``harmful``, not a smaller positive: the point of
    the gates is that they cannot be traded against success.
    """
    gates = dict(DEFAULT_HARM_GATES if gates is None else gates)
    breached = harm.exceeds(gates)
    if breached:
        return "harmful", [f"gate:{name}" for name in breached]
    if efficacy_delta > deadband:
        return "safe_positive", ["efficacy_positive"]
    if efficacy_delta < -deadband:
        return "harmful", ["efficacy_negative"]
    return "neutral", ["within_deadband"]


def build_utility_record(
    *,
    branch_pair_id: str,
    context: ContextKey,
    policy_stage: str,
    seed: int,
    horizons: dict[str, int],
    control_dose: DoseReport,
    intervention_dose: DoseReport,
    control_evaluations: Sequence[BranchEvaluation],
    intervention_evaluations: Sequence[BranchEvaluation],
    epsilon: float,
    kernel_radius_bins: int,
    base_distribution_sha256: str,
    intervention_distribution_sha256: str,
    proxy_features: dict[str, float] | None = None,
    gates: dict[str, float] | None = None,
) -> UtilityRecord:
    """Assemble one paired measurement into a :class:`UtilityRecord`."""
    control_by_horizon = {e.horizon_label: e for e in control_evaluations}
    intervention_by_horizon = {e.horizon_label: e for e in intervention_evaluations}

    missing = sorted(set(horizons) - set(control_by_horizon))
    if missing:
        raise ValueError(f"control branch is missing horizons {missing}")
    missing = sorted(set(horizons) - set(intervention_by_horizon))
    if missing:
        raise ValueError(f"intervention branch is missing horizons {missing}")

    record = UtilityRecord(
        branch_pair_id=branch_pair_id,
        context=context,
        policy_stage=policy_stage,
        seed=seed,
        horizons=dict(horizons),
        base_distribution_sha256=base_distribution_sha256,
        intervention_distribution_sha256=intervention_distribution_sha256,
        epsilon=epsilon,
        kernel_radius_bins=kernel_radius_bins,
        control_dose=control_dose,
        intervention_dose=intervention_dose,
        proxy_features=dict(proxy_features or {}),
    )

    for label in horizons:
        control = control_by_horizon[label]
        treated = intervention_by_horizon[label]
        delta = treated.j_eff - control.j_eff
        harm = build_harm_vector(control, treated)
        record.efficacy_delta[label] = delta
        record.harm[label] = harm
        classification, _ = classify_context(delta, harm, gates)
        record.safety_label[label] = classification  # type: ignore[assignment]

    # utility_at raises when the dose was never delivered; such a pair carries no
    # information about the context and must not silently become a label.
    for label in horizons:
        try:
            record.utility[label] = record.utility_at(label)
        except ValueError:
            record.utility.clear()
            break

    return record


def is_usable(record: UtilityRecord) -> bool:
    """Whether a record delivered enough dose to carry a label."""
    return bool(record.utility)


def horizon_reversals(record: UtilityRecord, short: str, long: str) -> str | None:
    """Name the short-to-long-horizon pattern, if this record shows one.

    These are the curriculum failure cases the whole programme is looking for.
    ``reversal_harmful`` is the most important: a short-horizon proxy would
    reward the context while it causes interference or forgetting later.
    """
    if short not in record.utility or long not in record.utility:
        return None
    early, late = record.utility[short], record.utility[long]
    deadband = DEFAULT_EFFICACY_DEADBAND
    if early > deadband and late < -deadband:
        return "reversal_harmful"
    if early <= deadband and late > deadband:
        return "delayed_useful"
    if early > deadband and abs(late) <= deadband:
        return "immediate_only"
    return None


def assess_identifiability(
    records: Iterable[UtilityRecord],
    horizon_label: str,
    noise_floor: Sequence[float] | None = None,
    min_variance_ratio: float = 2.0,
    min_icc: float = 0.4,
) -> IdentifiabilityReport:
    """Gate A: compare between-context variation against paired branch noise.

    Args:
        noise_floor: utilities measured on ``epsilon = 0`` pairs, where the true
            effect is zero by construction. Their spread is the cleanest
            available estimate of branch noise; without it the within-context
            replicate spread is used instead.
        min_variance_ratio: required ratio of between-context to within-context
            variance.
        min_icc: required intraclass correlation.

    Failing this gate is a real result, not a setback: it says context-level
    utility is not measurable at this granularity, and the honest responses are
    coarser context groups or dropping context-level claims -- not a larger
    estimator fitted to noise.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if horizon_label in record.utility:
            grouped[record.context.context_id].append(record.utility[horizon_label])

    num_contexts = len(grouped)
    replicated = {k: v for k, v in grouped.items() if len(v) > 1}
    reasons: list[str] = []

    if num_contexts < 2:
        return IdentifiabilityReport(
            num_contexts=num_contexts, num_replicates=0, within_context_sd=0.0,
            between_context_sd=0.0, noise_floor_sd=None, variance_ratio=0.0,
            intraclass_correlation=0.0, passes=False,
            reasons=["fewer than two contexts carry a label"],
        )

    context_means = [statistics.fmean(v) for v in grouped.values()]
    between_var = statistics.pvariance(context_means) if len(context_means) > 1 else 0.0

    within_var = 0.0
    num_replicates = 0
    if replicated:
        residuals: list[float] = []
        for values in replicated.values():
            mean = statistics.fmean(values)
            residuals.extend(value - mean for value in values)
            num_replicates += len(values)
        within_var = statistics.pvariance(residuals) if len(residuals) > 1 else 0.0

    noise_sd = None
    if noise_floor:
        noise_values = list(noise_floor)
        if len(noise_values) > 1:
            noise_sd = statistics.stdev(noise_values)
            # The epsilon=0 spread is the more trustworthy noise estimate: its
            # true effect is zero by construction, so nothing real inflates it.
            within_var = max(within_var, noise_sd**2)
            reasons.append("noise floor taken from epsilon=0 pairs")

    if within_var <= 0:
        reasons.append("no replicate or epsilon=0 variance available; cannot separate signal")
        return IdentifiabilityReport(
            num_contexts=num_contexts, num_replicates=num_replicates,
            within_context_sd=0.0, between_context_sd=math.sqrt(between_var),
            noise_floor_sd=noise_sd, variance_ratio=0.0, intraclass_correlation=0.0,
            passes=False, reasons=reasons,
        )

    ratio = between_var / within_var
    icc = between_var / (between_var + within_var)

    if ratio < min_variance_ratio:
        reasons.append(
            f"between/within variance {ratio:.2f} < {min_variance_ratio}: contexts differ "
            "no more than repeated measurements of the same context"
        )
    if icc < min_icc:
        reasons.append(f"intraclass correlation {icc:.2f} < {min_icc}")
    if not replicated and noise_sd is None:
        reasons.append("no replicated contexts and no epsilon=0 pairs")

    passes = ratio >= min_variance_ratio and icc >= min_icc
    if passes and not reasons:
        reasons.append("between-context variation exceeds branch noise")

    return IdentifiabilityReport(
        num_contexts=num_contexts,
        num_replicates=num_replicates,
        within_context_sd=math.sqrt(within_var),
        between_context_sd=math.sqrt(between_var),
        noise_floor_sd=noise_sd,
        variance_ratio=ratio,
        intraclass_correlation=icc,
        passes=passes,
        reasons=reasons,
    )


def summarize_labels(
    records: Sequence[UtilityRecord], horizon_label: str
) -> dict[str, Any]:
    """Aggregate a label set for reporting."""
    usable = [r for r in records if horizon_label in r.utility]
    if not usable:
        return {"num_records": len(records), "num_usable": 0}

    utilities = [r.utility[horizon_label] for r in usable]
    labels: dict[str, int] = defaultdict(int)
    for record in usable:
        labels[record.safety_label.get(horizon_label, "unknown")] += 1

    dropped = len(records) - len(usable)
    return {
        "num_records": len(records),
        "num_usable": len(usable),
        "num_dropped_no_dose": dropped,
        "utility_mean": statistics.fmean(utilities),
        "utility_median": statistics.median(utilities),
        "utility_sd": statistics.stdev(utilities) if len(utilities) > 1 else 0.0,
        "utility_min": min(utilities),
        "utility_max": max(utilities),
        "positive_fraction": sum(u > 0 for u in utilities) / len(utilities),
        "safety_labels": dict(sorted(labels.items())),
    }
