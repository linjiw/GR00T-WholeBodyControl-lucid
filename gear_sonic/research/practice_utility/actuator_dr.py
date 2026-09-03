"""Randomize the actuator, not the ground: the axes SONIC leaves fixed.

SONIC randomizes five physical channels and a latency: rigid-body mass, torso
centre of mass, ground material, a joint default-position offset, and external
pushes. Every one of them is a property of the robot's *load* or of the *world*.
Not one is a property of the actuator, and the actuator is where a deployed
humanoid actually differs from its simulator.

What Isaac Lab already exposes per environment and per joint at runtime, and what
this robot currently does with it:

======================  =====================================  ==================
property                writer                                 G1 config today
======================  =====================================  ==================
peak torque             ``write_joint_effort_limit_to_sim``     set to the PEAK
                                                               rating and never
                                                               varied
joint friction          ``write_joint_friction_coefficient      **not set at all**
(static/dynamic/        _to_sim`` and the dynamic and
viscous)                viscous variants
armature                ``write_joint_armature_to_sim``         set from motor
                                                               specs, never varied
speed limit             ``write_joint_velocity_limit_to_sim``   set, never varied
======================  =====================================  ==================

Joint friction is the striking one: the simulated G1 has frictionless gearboxes.
A real harmonic or planetary reducer has Coulomb friction that changes with
temperature, load and wear, and a cold robot is measurably stiffer than a warm
one. That is a gap between this simulator and the target robot, sitting behind an
API that is already there.

Why these are worth a curriculum's attention where mass and friction were not:
each one either *removes capability* (a lower torque or speed ceiling) or *adds a
nonsmooth term* (stiction has a sign discontinuity at zero velocity). The
channels measured so far do neither; they move a smooth parameter of a task the
policy already solves, which is why widening them has never needed staging.

**Whether any of them creates a barrier is a measurement, not a claim.** This
module exists so the question can be asked cheaply, on frozen policies, before
anything is trained.

Conventions this follows, all of them the project's existing ones:

* the nominal is cached on first use and **restored before every draw**, so
  repeated resets re-derive from nominal instead of stacking, which is the bug
  ``events_reset_safe`` was written to avoid;
* ``lam = 0`` reproduces the nominal exactly, so a run with a channel enabled at
  zero is a valid baseline rather than a subtly different robot;
* the term reports what it actually wrote, so a run records its severity instead
  of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

#: How a deviation combines with the nominal. ``scale`` multiplies by
#: ``1 + lam * u`` so a negative range derates; ``add`` adds ``lam * u``, which is
#: the only useful form for a property whose nominal is zero.
Combine = Literal["scale", "add"]


@dataclass(frozen=True)
class ActuatorChannel:
    """One randomizable actuator property."""

    name: str
    #: Attributes on ``articulation.data`` that may hold the nominal, in preference
    #: order. Isaac Lab renames these across releases and keeps the old name as a
    #: deprecated property that logs a warning, so reading the first one that exists
    #: is what keeps this working on both sides of a rename.
    nominal_attrs: tuple[str, ...]
    #: Method on the articulation that writes it.
    writer: str
    combine: Combine
    #: Deviation range at ``lam = 1``, in fractions for ``scale`` and in the
    #: property's own units for ``add``.
    deviation: tuple[float, float]
    #: Values below this are physically meaningless.
    floor: float = 0.0
    #: Extra writers that take the same value (PhysX splits friction three ways).
    also_write: tuple[str, ...] = ()
    #: Method on ``articulation.root_physx_view`` that reads the property BACK out of
    #: the physics engine. The articulation writes its own mirror unconditionally
    #: before calling the simulator, so comparing against ``asset.data`` proves only
    #: that Python was updated. This is the read that proves the value landed.
    physx_getter: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a positional mix-up at import time rather than at first use.

        ``name`` and ``nominal_attrs`` are both about naming a property, and the
        two got swapped twice while this file was being written. The failure was
        silent until a draw was attempted, where a string ``nominal_attrs``
        iterated over its own characters. These checks turn that into an
        immediate, readable error.
        """
        if not isinstance(self.nominal_attrs, tuple):
            raise TypeError(
                f"{self.name!r}: nominal_attrs must be a tuple of attribute names, "
                f"got {type(self.nominal_attrs).__name__}. A bare string iterates "
                "over its characters and silently finds nothing.")
        if not all(isinstance(a, str) and a for a in self.nominal_attrs):
            raise TypeError(f"{self.name!r}: nominal_attrs must all be non-empty strings")
        if not self.writer.startswith("write_joint_"):
            raise ValueError(
                f"{self.name!r}: writer {self.writer!r} is not an articulation joint writer; "
                "the arguments are probably in the wrong order")
        if self.combine not in ("scale", "add"):
            raise ValueError(f"{self.name!r}: unknown combine {self.combine!r}")
        lo, hi = self.deviation
        if hi < lo:
            raise ValueError(f"{self.name!r}: deviation {self.deviation} is inverted")


#: The four actuator-side channels, with ranges set from what the hardware
#: plausibly varies by rather than from what produces an outcome.
CHANNELS: dict[str, ActuatorChannel] = {
    # A deployed motor delivers less than its peak rating: the battery sags under
    # load, the winding heats, and units vary. Down to half of peak at lam = 1.
    "effort_limit": ActuatorChannel(
        "effort_limit", ("joint_effort_limits_sim", "joint_effort_limits"),
        "write_joint_effort_limit_to_sim", "scale", (-0.5, 0.0),
        physx_getter="get_dof_max_forces",
        note="fraction of peak torque removed; nominal is the peak rating in g1.py"),
    # Currently zero in simulation. A few N.m of Coulomb friction is ordinary for
    # a reducer of this size, and it rises sharply when the robot is cold.
    "joint_friction": ActuatorChannel(
        "joint_friction", ("default_joint_friction_coeff", "default_joint_friction"),
        "write_joint_friction_coefficient_to_sim", "add", (0.0, 6.0),
        # Static and dynamic Coulomb friction share the same units (N.m) and the
        # same physical magnitude, so one draw is right for both. VISCOUS friction
        # is deliberately NOT written here: it is a velocity-proportional damping
        # coefficient in N.m.s/rad, and feeding it a Coulomb magnitude would add a
        # large damping term while the docstring claimed to be modelling stiction.
        # A viscous channel needs its own units and its own range.
        also_write=("write_joint_dynamic_friction_coefficient_to_sim",),
        note="N.m of Coulomb (static and dynamic) gearbox friction; the asset declares none"),
    # Reflected rotor inertia, set from motor specs and never varied.
    "armature": ActuatorChannel(
        "armature", ("default_joint_armature",), "write_joint_armature_to_sim",
        "scale", (-0.3, 0.6), physx_getter="get_dof_armatures"),
    # Back-EMF caps joint speed; a motor at its limit cannot track a fast reference.
    # joint_velocity_limits is deprecated in favour of joint_vel_limits and logs a
    # warning on every read; prefer the new name and fall back to the old one.
    "velocity_limit": ActuatorChannel(
        "velocity_limit", ("joint_vel_limits", "joint_velocity_limits"),
        "write_joint_velocity_limit_to_sim", "scale", (-0.4, 0.0),
        physx_getter="get_dof_max_velocities"),
}

# The dict key and the channel's own name must agree, or a lookup by key would
# report a different channel than the one it configured.
for _key, _channel in CHANNELS.items():
    if _key != _channel.name:
        raise ValueError(f"channel key {_key!r} does not match its name {_channel.name!r}")

_CACHE = "_lucid_actuator_nominal"


def _nominal(asset: Any, channel: ActuatorChannel) -> torch.Tensor:
    """The nominal value, cached on the asset the first time it is needed.

    Properties with a ``default_*`` field are read from it. The rest are read
    once from the live value, which is the configured rating before anything has
    written to it, and cached so a later draw never re-derives from a randomized
    value.
    """
    cache = getattr(asset, _CACHE, None)
    if cache is None:
        cache = {}
        setattr(asset, _CACHE, cache)
    if channel.name in cache:
        return cache[channel.name]
    data = asset.data
    value = None
    source = None
    for attr in channel.nominal_attrs:
        candidate = getattr(data, attr, None)
        if candidate is not None:
            value, source = candidate, attr
            break
    if value is None:
        if channel.name == "joint_friction":
            # The simulated G1 has no gearbox friction, so there may be no field at
            # all. Zero is the correct nominal and the reason this channel ADDS.
            value, source = torch.zeros_like(data.joint_pos), "assumed zero"
        else:
            raise KeyError(
                f"no nominal source for {channel.name!r}; tried {channel.nominal_attrs}")
    cache[channel.name] = torch.as_tensor(value).clone()
    cache.setdefault("_sources", {})[channel.name] = source
    return cache[channel.name]


def nominal_source(asset: Any, channel_name: str) -> str | None:
    """Which data field the nominal was read from, for the run receipt."""
    cache = getattr(asset, _CACHE, None) or {}
    return (cache.get("_sources") or {}).get(channel_name)


def draw_and_write(
    asset: Any,
    channel_name: str,
    low: float,
    high: float,
    env_ids: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
    joint_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Draw uniformly in an ALREADY-SCALED range and write it.

    This is the entry point an event term uses. The curriculum scales a term's
    range parameter before the term runs, exactly as it does for every other
    channel, so the term itself needs no knowledge of lambda: it applies the
    range it was handed. Keeping it that way is what lets the strata machinery,
    the box gate and the evaluator treat an actuator channel as just another
    channel.
    """
    channel = CHANNELS[channel_name]
    nominal = _nominal(asset, channel)
    if nominal.ndim == 1:
        nominal = nominal.unsqueeze(0).expand(asset.data.joint_pos.shape[0], -1)
    num_envs, num_joints = nominal.shape
    if env_ids is None:
        env_ids = torch.arange(num_envs)
    env_ids = torch.as_tensor(env_ids).reshape(-1)
    columns = list(range(num_joints)) if joint_ids is None else list(joint_ids)

    # Always re-derived from the cached nominal, so calling this every reset does
    # not stack draws on the previous episode's values.
    base = nominal[env_ids][:, columns].clone().to(torch.float32)
    if high < low:
        raise ValueError(f"{channel_name}: range ({low}, {high}) is inverted")
    # The draw must land on the same device as the nominal. A CUDA nominal with a
    # CPU deviation raises on the first reset, which is the good failure, but a
    # generator is device-bound so the draw is made where the generator lives and
    # moved afterwards.
    if low == high:
        step = torch.full(base.shape, float(low), dtype=torch.float32, device=base.device)
    else:
        draw_device = getattr(generator, "device", None) if generator is not None else base.device
        step = torch.rand(base.shape, generator=generator, dtype=torch.float32,
                          device=draw_device) * (high - low) + low
        step = step.to(base.device)
    drawn = base * step if channel.combine == "scale" else base + step
    drawn = drawn.clamp(min=channel.floor)

    writers = (channel.writer, *channel.also_write)
    written = 0
    for name in writers:
        writer = getattr(asset, name, None)
        if writer is None:
            continue  # an older Isaac Sim splits friction fewer ways; report it
        writer(drawn, joint_ids=columns, env_ids=env_ids)
        written += 1

    readback = physx_readback(asset, channel, drawn, env_ids, columns)
    return {
        "channel": channel_name,
        "combine": channel.combine,
        "applied_range": [float(low), float(high)],
        "envs": int(env_ids.numel()),
        "joints": len(columns),
        # How many writer methods this Isaac Sim exposed and accepted the call.
        # It is NOT proof the value reached PhysX: the articulation writes its own
        # mirror unconditionally before the sim call, and older Isaac Sim builds
        # log a warning and return from the dynamic-friction writer. A run that
        # depends on this channel should read the physx view back.
        "writers_called": written,
        "writers_available": len(writers),
        "nominal_mean": float(base.mean()),
        "written_mean": float(drawn.mean()),
        "written_min": float(drawn.min()),
        "written_max": float(drawn.max()),
        "note": channel.note,
        "nominal_source": nominal_source(asset, channel_name),
        **readback,
    }


def physx_readback(asset, channel, drawn, env_ids, columns) -> dict[str, Any]:
    """Read the property back out of the physics engine and compare.

    Returns ``{"physx_readback": "unavailable"}`` where the view or the getter is
    not exposed, which is the honest answer on a fake or an older build. Where it
    IS available, a mismatch means the simulator did not take the value, and that
    is the only evidence that distinguishes a live channel from one that merely
    updated Python.
    """
    view = getattr(asset, "root_physx_view", None)
    getter = getattr(view, channel.physx_getter, None) if (view and channel.physx_getter) else None
    if getter is None:
        return {"physx_readback": "unavailable",
                "physx_reason": ("no getter for this channel" if not channel.physx_getter
                                 else "articulation exposes no root_physx_view")}
    try:
        live = torch.as_tensor(getter()).to(torch.float32)
        seen = live[env_ids.to(live.device)][:, columns].to(drawn.device)
        gap = float((seen - drawn).abs().max())
    except Exception as error:  # a shape or device surprise must not kill the run
        return {"physx_readback": "error", "physx_reason": repr(error)[:200]}
    return {"physx_readback": "matched" if gap <= 1e-3 else "MISMATCH",
            "physx_max_abs_gap": gap}


def apply(
    asset: Any,
    channel_name: str,
    lam: float,
    env_ids: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
    joint_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Draw and write one actuator channel for the given environments.

    Always re-derives from the cached nominal, so calling it every reset does not
    stack draws. At ``lam = 0`` it writes the nominal back unchanged.
    """
    if lam < 0.0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    channel = CHANNELS[channel_name]
    lo, hi = channel.deviation
    # The curriculum's own convention: nominal +/- lam * deviation, so lam = 0
    # reproduces the nominal exactly. For a scale channel the nominal multiplier
    # is 1, for an additive one it is 0.
    centre = 1.0 if channel.combine == "scale" else 0.0
    report = draw_and_write(asset, channel_name, centre + lam * lo, centre + lam * hi,
                            env_ids, generator=generator, joint_ids=joint_ids)
    report["lam"] = lam
    report["deviation_at_lam_1"] = list(channel.deviation)
    return report
