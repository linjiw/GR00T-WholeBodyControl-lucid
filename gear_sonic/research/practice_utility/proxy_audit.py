"""Audit whether a difficulty proxy predicts measured practice utility.

This is the decisive analysis, and it runs *before* any estimator is trained.
The programme's premise is that current difficulty -- failure rate, tracking
error, latent gap, TD error, learning progress -- describes where a policy is
weak now, not what it gains from practising there. That premise is testable and
must be tested first, because the alternative outcome is also a good result: if
a simple proxy already predicts long-horizon utility, the right paper says so
and no learned curriculum is needed.

Four measures, because they fail differently:

``spearman``          monotone association overall.
``sign_accuracy``     does the proxy know help from harm? A proxy can rank
                      contexts well and still put the zero crossing in the
                      wrong place, which is what a curriculum actually acts on.
``pairwise_accuracy`` given two contexts, is the better one identified? This is
                      the operation an allocator performs.
``calibration_error`` are predicted magnitudes usable, or only their order?

Correlations are computed *within* group and then averaged, never pooled across
groups. Pooling lets a between-family difference masquerade as within-family
predictive power: a proxy that merely knows "crawling is hard" would score well
pooled while being useless for choosing among crawling contexts.

Gate B combines these into one decision, and it is deliberately hard to pass in
the direction of more machinery: a learned estimator is authorized only if no
simple proxy suffices.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from gear_sonic.research.practice_utility.schema import UtilityRecord
from gear_sonic.research.practice_utility.utility_label import horizon_reversals

#: Proxies audited by default. Each is a key in ``UtilityRecord.proxy_features``.
DEFAULT_PROXIES: tuple[str, ...] = (
    "native_failure_rate",
    "latent_gap_p90",
    "latent_gap_median",
    "raw_mismatch_p90",
    "mpjpe",
    "td_error",
    "value_loss",
    "advantage_abs",
    "learning_progress",
    "sampling_probability",
)

#: A proxy meeting all of these is treated as sufficient on its own.
SUFFICIENCY = {
    "min_abs_spearman": 0.5,
    "min_sign_accuracy": 0.75,
    "min_pairwise_accuracy": 0.70,
}


@dataclass
class ProxyResult:
    """How well one proxy predicts utility at one horizon."""

    proxy: str
    horizon_label: str
    num_samples: int
    num_groups: int
    spearman: float
    sign_accuracy: float
    pairwise_accuracy: float
    calibration_error: float
    per_group_spearman: dict[str, float] = field(default_factory=dict)

    @property
    def sign_flips_across_groups(self) -> bool:
        """True when the proxy's direction reverses between groups.

        A proxy that predicts utility positively for one motion family and
        negatively for another has no single usable interpretation, however good
        its pooled correlation looks.
        """
        signs = {
            math.copysign(1.0, v)
            for v in self.per_group_spearman.values()
            if abs(v) > 0.1
        }
        return len(signs) > 1

    @property
    def is_sufficient(self) -> bool:
        return (
            abs(self.spearman) >= SUFFICIENCY["min_abs_spearman"]
            and self.sign_accuracy >= SUFFICIENCY["min_sign_accuracy"]
            and self.pairwise_accuracy >= SUFFICIENCY["min_pairwise_accuracy"]
            and not self.sign_flips_across_groups
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proxy": self.proxy,
            "horizon_label": self.horizon_label,
            "num_samples": self.num_samples,
            "num_groups": self.num_groups,
            "spearman": self.spearman,
            "sign_accuracy": self.sign_accuracy,
            "pairwise_accuracy": self.pairwise_accuracy,
            "calibration_error": self.calibration_error,
            "sign_flips_across_groups": self.sign_flips_across_groups,
            "is_sufficient": self.is_sufficient,
            "per_group_spearman": self.per_group_spearman,
        }


@dataclass
class GateBReport:
    """Is a learned utility estimator warranted?"""

    horizon_label: str
    best_proxy: str | None
    best_spearman: float
    sufficient_proxies: list[str]
    unstable_proxies: list[str]
    num_reversals: int
    reversal_fraction: float
    authorizes_estimator: bool
    reasons: list[str] = field(default_factory=list)
    proxy_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_label": self.horizon_label,
            "best_proxy": self.best_proxy,
            "best_spearman": self.best_spearman,
            "sufficient_proxies": self.sufficient_proxies,
            "unstable_proxies": self.unstable_proxies,
            "num_reversals": self.num_reversals,
            "reversal_fraction": self.reversal_fraction,
            "authorizes_estimator": self.authorizes_estimator,
            "reasons": self.reasons,
            "proxy_results": self.proxy_results,
        }


# ------------------------------------------------------------ statistics ----


def rank_data(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged, as Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; 0.0 when either input has no variance."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return 0.0
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    dx = [a - mean_x for a in x]
    dy = [b - mean_y for b in y]
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denominator <= 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, tolerant of ties."""
    if len(x) < 2:
        return 0.0
    return pearson(rank_data(x), rank_data(y))


def sign_accuracy(predicted: Sequence[float], actual: Sequence[float],
                  deadband: float = 0.0) -> float:
    """Fraction of contexts whose *sign* the proxy gets right.

    Contexts whose true utility sits inside the deadband are excluded: their
    sign is not meaningfully defined, and scoring them would reward guessing.
    """
    pairs = [(p, a) for p, a in zip(predicted, actual) if abs(a) > deadband]
    if not pairs:
        return 0.0
    correct = sum(1 for p, a in pairs if math.copysign(1.0, p) == math.copysign(1.0, a))
    return correct / len(pairs)


def pairwise_ranking_accuracy(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Fraction of context pairs ordered correctly -- the allocator's operation."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must align")
    concordant = comparable = 0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            if actual[i] == actual[j]:
                continue
            comparable += 1
            if (predicted[i] - predicted[j]) * (actual[i] - actual[j]) > 0:
                concordant += 1
    return concordant / comparable if comparable else 0.0


def calibration_error(predicted: Sequence[float], actual: Sequence[float],
                      num_bins: int = 5) -> float:
    """Mean absolute gap between predicted and actual means, per quantile bin.

    Both series are standardized first, so this measures whether the *shape* of
    the relationship is usable rather than penalizing a proxy for living on a
    different scale than utility.
    """
    if len(predicted) < num_bins or len(predicted) != len(actual):
        return float("nan")
    p = _standardize(predicted)
    a = _standardize(actual)
    order = sorted(range(len(p)), key=lambda i: p[i])
    errors = []
    for bin_index in range(num_bins):
        start = bin_index * len(order) // num_bins
        stop = (bin_index + 1) * len(order) // num_bins
        members = order[start:stop]
        if not members:
            continue
        errors.append(
            abs(statistics.fmean(p[i] for i in members) - statistics.fmean(a[i] for i in members))
        )
    return statistics.fmean(errors) if errors else float("nan")


# ---------------------------------------------------------------- audits ----


def audit_proxy(
    records: Sequence[UtilityRecord],
    proxy: str,
    horizon_label: str,
    group_by: Callable[[UtilityRecord], str] | None = None,
    sign_deadband: float = 0.0,
) -> ProxyResult:
    """Score one proxy against measured utility, grouped to avoid confounds.

    ``group_by`` defaults to policy stage. Grouping matters: a proxy evaluated
    across pooled stages can look predictive purely because both it and utility
    drift with training progress.
    """
    group_by = group_by or (lambda record: record.policy_stage)
    usable = [
        r for r in records
        if horizon_label in r.utility and proxy in r.proxy_features
    ]
    if len(usable) < 2:
        return ProxyResult(proxy, horizon_label, len(usable), 0, 0.0, 0.0, 0.0, float("nan"))

    grouped: dict[str, list[UtilityRecord]] = defaultdict(list)
    for record in usable:
        grouped[group_by(record)].append(record)

    per_group: dict[str, float] = {}
    spearmans, signs, pairwise, calibrations = [], [], [], []
    for name, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        proxy_values = [m.proxy_features[proxy] for m in members]
        utilities = [m.utility[horizon_label] for m in members]
        rho = spearman(proxy_values, utilities)
        per_group[name] = rho
        spearmans.append(rho)
        signs.append(sign_accuracy(proxy_values, utilities, sign_deadband))
        pairwise.append(pairwise_ranking_accuracy(proxy_values, utilities))
        calibration = calibration_error(proxy_values, utilities)
        if not math.isnan(calibration):
            calibrations.append(calibration)

    return ProxyResult(
        proxy=proxy,
        horizon_label=horizon_label,
        num_samples=len(usable),
        num_groups=len(per_group),
        spearman=statistics.fmean(spearmans) if spearmans else 0.0,
        sign_accuracy=statistics.fmean(signs) if signs else 0.0,
        pairwise_accuracy=statistics.fmean(pairwise) if pairwise else 0.0,
        calibration_error=statistics.fmean(calibrations) if calibrations else float("nan"),
        per_group_spearman=per_group,
    )


def audit_all_proxies(
    records: Sequence[UtilityRecord],
    horizon_label: str,
    proxies: Iterable[str] = DEFAULT_PROXIES,
    group_by: Callable[[UtilityRecord], str] | None = None,
) -> dict[str, ProxyResult]:
    """Audit every proxy present in the records."""
    available = {name for record in records for name in record.proxy_features}
    return {
        proxy: audit_proxy(records, proxy, horizon_label, group_by)
        for proxy in proxies
        if proxy in available
    }


def count_reversals(records: Sequence[UtilityRecord], short: str, long: str) -> dict[str, int]:
    """Tally short-to-long-horizon utility patterns across a label set."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        pattern = horizon_reversals(record, short, long)
        if pattern:
            counts[pattern] += 1
    return dict(sorted(counts.items()))


def assess_sufficiency(
    records: Sequence[UtilityRecord],
    horizon_label: str,
    short_horizon: str | None = None,
    proxies: Iterable[str] = DEFAULT_PROXIES,
    group_by: Callable[[UtilityRecord], str] | None = None,
) -> GateBReport:
    """Gate B: decide whether a learned estimator is warranted.

    Authorized only if **no** simple proxy is sufficient. Any sufficient proxy
    blocks the estimator, and that outcome is a publishable result in its own
    right -- a calibrated mapping from an existing signal to a curriculum
    action, and a stronger justification for the sampler that already exists.
    """
    results = audit_all_proxies(records, horizon_label, proxies, group_by)
    reasons: list[str] = []

    if not results:
        return GateBReport(
            horizon_label=horizon_label, best_proxy=None, best_spearman=0.0,
            sufficient_proxies=[], unstable_proxies=[], num_reversals=0,
            reversal_fraction=0.0, authorizes_estimator=False,
            reasons=["no proxy features recorded; cannot audit"],
        )

    sufficient = sorted(name for name, r in results.items() if r.is_sufficient)
    unstable = sorted(name for name, r in results.items() if r.sign_flips_across_groups)
    best_name = max(results, key=lambda name: abs(results[name].spearman))
    best = results[best_name]

    reversals = 0
    reversal_fraction = 0.0
    if short_horizon:
        counts = count_reversals(records, short_horizon, horizon_label)
        reversals = counts.get("reversal_harmful", 0)
        labelled = sum(1 for r in records if horizon_label in r.utility)
        reversal_fraction = reversals / labelled if labelled else 0.0
        if reversals:
            reasons.append(
                f"{reversals} contexts reverse sign between {short_horizon} and "
                f"{horizon_label}: a short-horizon proxy would reward them"
            )

    if sufficient:
        reasons.append(
            f"proxies {sufficient} already predict {horizon_label} utility; "
            "a learned estimator is not warranted"
        )
    else:
        reasons.append(
            f"no proxy reached sufficiency (best {best_name}, "
            f"spearman {best.spearman:+.2f}, sign {best.sign_accuracy:.2f}, "
            f"pairwise {best.pairwise_accuracy:.2f})"
        )
    if unstable:
        reasons.append(f"proxies {unstable} reverse direction between groups")

    return GateBReport(
        horizon_label=horizon_label,
        best_proxy=best_name,
        best_spearman=best.spearman,
        sufficient_proxies=sufficient,
        unstable_proxies=unstable,
        num_reversals=reversals,
        reversal_fraction=reversal_fraction,
        authorizes_estimator=not sufficient,
        reasons=reasons,
        proxy_results={name: result.to_dict() for name, result in sorted(results.items())},
    )


def _standardize(values: Sequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    return [0.0] * len(values) if sd <= 0 else [(v - mean) / sd for v in values]
