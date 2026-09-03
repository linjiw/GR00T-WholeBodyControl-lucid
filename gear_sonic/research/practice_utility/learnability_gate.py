"""Gate A: is there a hard bin that direct mixed training cannot already learn?

A curriculum is only *necessary* when the target distribution contains something
that equal-budget direct mixed training does not reach -- because reward is
sparse there, or exploration stalls. If direct mixed already learns the hardest
informative bin, a curriculum is at best neutral and at worst adds forgetting,
and the honest conclusion is that it is unnecessary here. This module decides
that question, and it is written so the decision cannot be moved afterwards.

Two things have to be pinned before any outcome is read.

**Which bin is "hardest informative".** Not the one where the treatment looks
best, and not a saturated one. :func:`select_hard_bin` takes the *reference*
arms only -- the untrained origin -- ranks the candidate bins by the origin's
success there, and returns the hardest that is still rankable. Saturated bins
are excluded by measurement, not by taste: a bin that reads the same for every
policy on the panel cannot separate two policies, whatever its mean is. The
60 ms latency cell is additionally banned by name, because it is a measured
floor for every policy ever run here including the untrained one, and a floor
is a failure bound rather than a ranking endpoint.

**What counts as "already learns it".** :func:`score_gate_a` compares direct
mixed against the untrained origin on that bin, and against the best curriculum
arm offered. The thresholds live in :class:`GateAThresholds` and are meant to be
frozen from the noise floor before the confirmatory arms exist.

The verdict is deliberately three-valued. ``curriculum_unnecessary`` is a real
finding and is reported as such; it is not a failure of the experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

#: Presets that may never be used to rank policies, whatever they measure.
#: ``latency_60ms`` and ``lat_60ms`` read 0.00% for every arm ever evaluated in
#: this programme, the untrained origin included. They are retained as failure
#: bounds and excluded from selection by name so that no later edit can quietly
#: promote a floor to an endpoint.
BANNED_RANKING_PRESETS = frozenset({"latency_60ms", "lat_60ms"})

#: A bin whose spread across arms is below this cannot separate policies.
MIN_RANKABLE_SPREAD_PTS = 2.0

#: A bin this close to either bound for every arm is saturated.
SATURATION_MARGIN_PTS = 1.0


@dataclass(frozen=True)
class GateAThresholds:
    """Frozen decision thresholds. Set from the noise floor, before outcomes."""

    #: Direct mixed must beat the untrained origin on the hard bin by at least
    #: this to count as "learning" it at all.
    learned_margin_pts: float = 5.0
    #: A curriculum must beat direct mixed by at least this on the hard bin for
    #: curriculum necessity to remain plausible.
    curriculum_margin_pts: float = 5.0
    #: Differences smaller than this are ties on a three-seed screen.
    tie_pts: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BinCandidate:
    """One preset's suitability as the hard informative bin."""

    preset: str
    reference_pts: float
    min_pts: float
    max_pts: float
    spread_pts: float
    banned: bool
    saturated: bool
    rankable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pts(value: float) -> float:
    return 100.0 * float(value)


def bin_candidates(
    per_preset_by_mode: dict[str, dict[str, float]],
    reference_mode: str = "origin",
    banned: Iterable[str] = BANNED_RANKING_PRESETS,
) -> list[BinCandidate]:
    """Describe every preset's fitness to be a ranking bin.

    ``per_preset_by_mode`` maps preset -> mode -> success rate in [0, 1]. Only
    the *reference* mode's value orders the candidates; every other mode is used
    solely to detect saturation, which is a property of the measurement rather
    than of any treatment.
    """
    banned_set = set(banned)
    out: list[BinCandidate] = []
    for preset in sorted(per_preset_by_mode):
        by_mode = per_preset_by_mode[preset]
        if reference_mode not in by_mode:
            continue
        values = [_pts(v) for v in by_mode.values() if v is not None]
        if not values:
            continue
        low, high = min(values), max(values)
        spread = high - low
        is_banned = preset in banned_set
        saturated = (
            high <= SATURATION_MARGIN_PTS
            or low >= 100.0 - SATURATION_MARGIN_PTS
            or spread < MIN_RANKABLE_SPREAD_PTS
        )
        if is_banned:
            reason = "banned by name: a measured floor, retained as a failure bound"
        elif high <= SATURATION_MARGIN_PTS:
            reason = "saturated at the floor for every arm"
        elif low >= 100.0 - SATURATION_MARGIN_PTS:
            reason = "saturated at the ceiling for every arm"
        elif spread < MIN_RANKABLE_SPREAD_PTS:
            reason = f"spread {spread:.2f} pts is below the {MIN_RANKABLE_SPREAD_PTS} pt minimum"
        else:
            reason = "rankable"
        out.append(
            BinCandidate(
                preset=preset,
                reference_pts=_pts(by_mode[reference_mode]),
                min_pts=low,
                max_pts=high,
                spread_pts=spread,
                banned=is_banned,
                saturated=saturated,
                rankable=not is_banned and not saturated,
                reason=reason,
            )
        )
    return out


def select_hard_bin(
    per_preset_by_mode: dict[str, dict[str, float]],
    reference_mode: str = "origin",
    banned: Iterable[str] = BANNED_RANKING_PRESETS,
) -> tuple[BinCandidate | None, list[BinCandidate]]:
    """The hardest rankable bin, by the *reference* arm's success there.

    Returns ``(chosen, all_candidates)``. ``chosen`` is ``None`` when nothing is
    rankable, which is itself a finding: it means the evaluation grid cannot
    currently separate policies anywhere, and the grid has to be fixed before
    any curriculum question can be asked of it.
    """
    candidates = bin_candidates(per_preset_by_mode, reference_mode, banned)
    rankable = [c for c in candidates if c.rankable]
    if not rankable:
        return None, candidates
    # Hardest = lowest reference success. Ties broken by preset name so the
    # choice is a function of the data and nothing else.
    chosen = min(rankable, key=lambda c: (c.reference_pts, c.preset))
    return chosen, candidates


@dataclass(frozen=True)
class GateAResult:
    """The gate's verdict, with everything it was decided from."""

    preset: str
    thresholds: GateAThresholds
    origin_pts: float
    direct_mixed_pts: float
    direct_mixed_minus_origin_pts: float
    best_curriculum_arm: str | None
    best_curriculum_pts: float | None
    curriculum_minus_direct_pts: float | None
    direct_mixed_learns_it: bool
    curriculum_adds_on_it: bool | None
    verdict: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["thresholds"] = self.thresholds.to_dict()
        return out


def score_gate_a(
    per_preset_by_mode: dict[str, dict[str, float]],
    preset: str,
    direct_mixed_mode: str = "fixed",
    reference_mode: str = "origin",
    curriculum_modes: Sequence[str] = (),
    thresholds: GateAThresholds | None = None,
) -> GateAResult:
    """Decide Gate A on one preset.

    Three verdicts:

    ``curriculum_unnecessary``  direct mixed learns the hard bin and no
                                curriculum materially beats it there. Curriculum
                                is solving a problem that does not arise; record
                                that and stop escalating the branch.
    ``curriculum_plausible``    direct mixed does not learn the hard bin, or a
                                curriculum materially beats it there. The
                                necessity hypothesis survives.
    ``not_evaluable``           the arms needed are not present.
    """
    thresholds = thresholds or GateAThresholds()
    by_mode = per_preset_by_mode.get(preset, {})
    origin = by_mode.get(reference_mode)
    direct = by_mode.get(direct_mixed_mode)
    if origin is None or direct is None:
        return GateAResult(
            preset=preset, thresholds=thresholds,
            origin_pts=_pts(origin) if origin is not None else float("nan"),
            direct_mixed_pts=_pts(direct) if direct is not None else float("nan"),
            direct_mixed_minus_origin_pts=float("nan"),
            best_curriculum_arm=None, best_curriculum_pts=None,
            curriculum_minus_direct_pts=None,
            direct_mixed_learns_it=False, curriculum_adds_on_it=None,
            verdict="not_evaluable",
            rationale=(
                f"need both {reference_mode!r} and {direct_mixed_mode!r} on {preset!r}; "
                f"have {sorted(by_mode)}"
            ),
        )

    origin_pts, direct_pts = _pts(origin), _pts(direct)
    delta_origin = direct_pts - origin_pts
    learns = delta_origin >= thresholds.learned_margin_pts

    available = [m for m in curriculum_modes if by_mode.get(m) is not None]
    best_arm = max(available, key=lambda m: by_mode[m]) if available else None
    best_pts = _pts(by_mode[best_arm]) if best_arm else None
    delta_curriculum = None if best_pts is None else best_pts - direct_pts
    adds = (
        None if delta_curriculum is None
        else delta_curriculum >= thresholds.curriculum_margin_pts
    )

    if not learns or adds:
        verdict = "curriculum_plausible"
    else:
        verdict = "curriculum_unnecessary"

    if not learns:
        rationale = (
            f"direct mixed reaches {direct_pts:.2f} pts on {preset!r} against the "
            f"untrained origin's {origin_pts:.2f} ({delta_origin:+.2f}), below the "
            f"{thresholds.learned_margin_pts} pt margin: it does not learn this bin, "
            "so a curriculum still has something to be necessary for."
        )
    elif adds:
        rationale = (
            f"direct mixed learns {preset!r} ({delta_origin:+.2f} pts over the origin), "
            f"but {best_arm!r} beats it there by {delta_curriculum:+.2f} pts, at or "
            f"above the {thresholds.curriculum_margin_pts} pt margin."
        )
    else:
        gap = "no curriculum arm was supplied" if delta_curriculum is None else (
            f"the best curriculum arm {best_arm!r} is {delta_curriculum:+.2f} pts from it"
        )
        rationale = (
            f"direct mixed learns {preset!r} ({delta_origin:+.2f} pts over the origin) "
            f"and {gap}. On this evidence a curriculum is unnecessary here; record the "
            "negative rather than tuning until it wins."
        )

    return GateAResult(
        preset=preset, thresholds=thresholds,
        origin_pts=origin_pts, direct_mixed_pts=direct_pts,
        direct_mixed_minus_origin_pts=delta_origin,
        best_curriculum_arm=best_arm, best_curriculum_pts=best_pts,
        curriculum_minus_direct_pts=delta_curriculum,
        direct_mixed_learns_it=learns, curriculum_adds_on_it=adds,
        verdict=verdict, rationale=rationale,
    )


def per_preset_by_mode(receipts: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Collapse evaluation receipts into ``preset -> mode -> mean success``."""
    out: dict[str, dict[str, float]] = {}
    for receipt in receipts:
        for preset, modes in (receipt.get("mode_summary") or {}).items():
            for mode, block in modes.items():
                value = (
                    block.get("metrics", {}).get("success_rate", {}).get("mean")
                )
                if value is not None:
                    out.setdefault(preset, {})[mode] = float(value)
    return out
