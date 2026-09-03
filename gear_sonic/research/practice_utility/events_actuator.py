"""Actuator-side randomization, as ordinary event terms the curriculum can move.

The four actuator channels are direct writes on the articulation, not event terms
with ranges, and that difference matters more than it looks. The evaluator refuses
a channel it cannot find in the event manager's baseline, deliberately: a cell
claiming to widen a channel the live config never exposed would report physics
that did not run. The curriculum, the strata dispatchers and the box gate all
read the same baseline. A channel outside it is invisible to every one of them.

So rather than add a parallel path for actuator writes, these terms give the
channels the one property the machinery keys on: a ``params`` entry whose name is
in ``dr_scaling.RANGE_NOMINALS``. Everything else then works unchanged. The
curriculum scales the range before the term runs, so a term never sees lambda and
never needs to; it applies the range it was handed, which is exactly how the six
existing channels behave.

The ranges are written so that lambda zero is a genuine no-op. A scale channel's
nominal multiplier is 1 and an additive channel's is 0, so at lambda zero the
scaled range collapses to a point at the nominal and the write puts back the
value that was already there.

Every term is reset-safe by construction: the draw is always re-derived from a
nominal cached the first time the channel is touched, never from the current
value. That is the bug ``events_reset_safe`` exists to prevent, and it is easy to
reintroduce, so it is pinned by a test for each channel.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from gear_sonic.research.practice_utility import actuator_dr as ADR

#: Where a run's actuator telemetry is stashed on the asset, so a receipt can
#: state what was applied instead of what was configured.
TELEMETRY = "_lucid_actuator_reports"


def _asset(env: Any, asset_cfg: Any) -> Any:
    return env.scene[asset_cfg.name]


def _joint_ids(asset_cfg: Any, asset: Any) -> list[int] | None:
    ids = getattr(asset_cfg, "joint_ids", None)
    if ids is None or isinstance(ids, slice):
        return None
    return list(ids)


def _record(asset: Any, channel: str, report: dict[str, Any]) -> None:
    """Accumulate every sub-call, because a term runs once PER COHORT.

    The strata dispatchers split env_ids and call the underlying sampler once per
    cohort, so a single reset produces several reports for one channel. Keeping
    only the last would make every receipt describe whichever cohort happened to
    run last -- and the cohorts sit at deliberately different intensities, so that
    is precisely the number that must not be reported as the arm's exposure.
    """
    store = getattr(asset, TELEMETRY, None)
    if store is None:
        store = {}
        setattr(asset, TELEMETRY, store)
    store.setdefault(channel, []).append(report)


def actuator_telemetry(asset: Any) -> dict[str, Any]:
    """What each actuator channel applied this reset, aggregated over its cohorts."""
    store = getattr(asset, TELEMETRY, {}) or {}
    out: dict[str, Any] = {}
    for channel, reports in store.items():
        if not reports:
            continue
        last = reports[-1]
        out[channel] = {
            **last,
            "cohorts": len(reports),
            "envs": sum(int(r.get("envs", 0)) for r in reports),
            "written_min": min(float(r["written_min"]) for r in reports),
            "written_max": max(float(r["written_max"]) for r in reports),
            "applied_ranges": [r.get("applied_range") for r in reports],
            "physx_readbacks": sorted({str(r.get("physx_readback")) for r in reports}),
        }
    return out


def clear_actuator_telemetry(asset: Any) -> None:
    """Drop the previous reset's reports so a receipt describes one reset."""
    setattr(asset, TELEMETRY, {})


def _term(channel: str, env: Any, env_ids: Any, asset_cfg: Any,
          value_range: Sequence[float]) -> None:
    asset = _asset(env, asset_cfg)
    low, high = float(value_range[0]), float(value_range[1])
    report = ADR.draw_and_write(asset, channel, low, high, env_ids,
                                joint_ids=_joint_ids(asset_cfg, asset))
    _record(asset, channel, report)


def randomize_joint_effort_limit(
    env: Any, env_ids: torch.Tensor | None, asset_cfg: Any,
    effort_limit_scale_range: Sequence[float] = (1.0, 1.0),
) -> None:
    """Scale each joint's peak torque.

    A deployed motor delivers less than its rating: the battery sags under load,
    the winding heats, and units vary. The simulator applies the peak rating
    forever, so this is the one channel that makes the policy live inside a
    torque budget it might actually meet on hardware.
    """
    _term("effort_limit", env, env_ids, asset_cfg, effort_limit_scale_range)


def randomize_joint_friction(
    env: Any, env_ids: torch.Tensor | None, asset_cfg: Any,
    joint_friction_range: Sequence[float] = (0.0, 0.0),
) -> None:
    """Add Coulomb friction to each joint, in newton-metres.

    The G1's URDF declares no joint dynamics at all, so the simulated gearboxes
    are frictionless while a real harmonic or planetary reducer is not, and is
    measurably stiffer cold than warm. Added rather than scaled because the
    nominal is zero and scaling zero is zero.
    """
    _term("joint_friction", env, env_ids, asset_cfg, joint_friction_range)


def randomize_joint_armature(
    env: Any, env_ids: torch.Tensor | None, asset_cfg: Any,
    armature_scale_range: Sequence[float] = (1.0, 1.0),
) -> None:
    """Scale reflected rotor inertia, a routinely mis-identified quantity."""
    _term("armature", env, env_ids, asset_cfg, armature_scale_range)


def randomize_joint_velocity_limit(
    env: Any, env_ids: torch.Tensor | None, asset_cfg: Any,
    velocity_limit_scale_range: Sequence[float] = (1.0, 1.0),
) -> None:
    """Scale each joint's speed ceiling.

    Back-EMF caps joint speed, and a motor at its limit cannot track a fast
    reference. Note the asymmetry with the other three: on a motion-TRACKING
    task a ceiling below what the clip demands makes the reference physically
    untrackable rather than merely hard, which is a different thing from a
    learnability barrier and must not be reported as one.
    """
    _term("velocity_limit", env, env_ids, asset_cfg, velocity_limit_scale_range)
