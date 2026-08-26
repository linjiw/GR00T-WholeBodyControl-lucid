"""Enable SONIC's delayed actuators without editing the robot definition.

``DelayedImplicitActuator`` already exists in SONIC and is referenced nowhere;
G1 declares all five joint groups as plain ``ImplicitActuatorCfg``. Turning
latency on therefore means swapping five actuator configs.

Editing ``robots/g1.py`` would do it in five lines, but it would change the
robot for *every* run in the checkout, including the no-latency baseline arms of
a comparison. A curriculum experiment whose control arm quietly acquired a delay
buffer would be measuring the wrong thing. So the swap is done here, in place,
on the module-level config object, and only by a caller that asked for it.

Two properties make this safe rather than clever:

* ``max_delay = 0`` builds ``DelayBuffer(0)``, i.e. a single-slot circular
  buffer whose only entry is the newest command. The delayed actuator is then
  behaviourally identical to the implicit one, so a run with latency "enabled at
  zero" is a valid baseline rather than a subtly different robot.
* The swap is reported and verifiable: :func:`describe_actuators` states what
  each group is now, so a run can record whether it actually has latency instead
  of assuming it.

Timing matters. ``modular_tracking_env_cfg`` resolves the robot through a Python
mapping to ``g1.G1_CYLINDER_MODEL_12_DEX_CFG`` and then calls ``.replace(...)``,
so the patch must be applied *before* the environment config is constructed --
which for a training run means before ``train_agent_trl.main`` builds the env.
"""

from __future__ import annotations

from typing import Any

#: Config objects in ``robots/g1.py`` that a run might use.
G1_CONFIG_NAMES = (
    "G1_CYLINDER_MODEL_12_DEX_CFG",
    "G1_MODEL_12_DEX_CFG",
    "G1_CFG",
)


def enable_delayed_actuators(
    max_delay: int,
    min_delay: int = 0,
    config_names: tuple[str, ...] = G1_CONFIG_NAMES,
) -> dict[str, Any]:
    """Swap implicit actuators for delayed ones, in physics steps.

    Args:
        max_delay: buffer capacity, in physics steps. This is a *construction
            time ceiling* -- ``set_time_lag`` raises above it -- so choose it for
            the widest latency the curriculum may ever request, not for the
            value it starts at. At the configured 200 Hz, 8 steps = 40 ms, which
            is LUCID's training range.
        min_delay: lower bound the actuator resamples from on its own resets. Left
            at 0 so the curriculum's reset-mode event term is the sole source of
            the lag; otherwise every reset would first draw an unscaled uniform
            lag and the curriculum would be fighting it.

    Returns a report naming every group swapped, so a run can record whether it
    really has latency.
    """
    if max_delay < 0:
        raise ValueError(f"max_delay must be >= 0, got {max_delay}")
    if not 0 <= min_delay <= max_delay:
        raise ValueError(f"require 0 <= min_delay <= max_delay, got {min_delay}, {max_delay}")

    from isaaclab.actuators import ImplicitActuatorCfg

    from gear_sonic.envs.manager_env.mdp.actuators import DelayedImplicitActuatorCfg
    from gear_sonic.envs.manager_env.robots import g1

    swapped: dict[str, list[str]] = {}
    for name in config_names:
        robot_cfg = getattr(g1, name, None)
        actuators = getattr(robot_cfg, "actuators", None)
        if not isinstance(actuators, dict):
            continue
        groups = _replace_actuator_mapping(
            actuators,
            ImplicitActuatorCfg,
            DelayedImplicitActuatorCfg,
            min_delay,
            max_delay,
        )
        if groups:
            swapped[name] = groups

    return {
        "max_delay_steps": max_delay,
        "min_delay_steps": min_delay,
        "swapped": swapped,
        "num_groups": sum(len(v) for v in swapped.values()),
    }


def enable_delayed_actuators_on_cfg(
    robot_cfg: Any,
    max_delay: int,
    min_delay: int = 0,
) -> dict[str, Any]:
    """Swap the actuators on the resolved environment robot configuration.

    Patching the module-level G1 template is not sufficient evidence: Hydra's
    environment construction may already hold a copied ``scene.robot``.  This
    function operates on that resolved object at the ``custom_instantiate``
    seam, immediately before Isaac constructs the live articulation.
    """
    if max_delay < 0 or not 0 <= min_delay <= max_delay:
        raise ValueError(f"require 0 <= min_delay <= max_delay, got {min_delay}, {max_delay}")

    from isaaclab.actuators import ImplicitActuatorCfg

    from gear_sonic.envs.manager_env.mdp.actuators import DelayedImplicitActuatorCfg

    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return {"num_groups": 0, "groups": [], "resolved": False}
    groups = _replace_actuator_mapping(
        actuators,
        ImplicitActuatorCfg,
        DelayedImplicitActuatorCfg,
        min_delay,
        max_delay,
    )
    return {
        "num_groups": len(groups),
        "groups": groups,
        "resolved": True,
        "max_delay_steps": max_delay,
        "min_delay_steps": min_delay,
    }


def _replace_actuator_mapping(
    actuators: dict[str, Any],
    implicit_cls: type,
    delayed_cls: type,
    min_delay: int,
    max_delay: int,
) -> list[str]:
    groups = []
    for group, cfg in list(actuators.items()):
        if isinstance(cfg, delayed_cls):
            groups.append(f"{group} (already delayed)")
            continue
        if not isinstance(cfg, implicit_cls):
            continue
        actuators[group] = _to_delayed(cfg, delayed_cls, min_delay, max_delay)
        groups.append(group)
    return groups


def describe_actuators(config_name: str = G1_CONFIG_NAMES[0]) -> dict[str, str]:
    """Report the actuator class of each joint group, for a run receipt."""
    from gear_sonic.envs.manager_env.robots import g1

    robot_cfg = getattr(g1, config_name, None)
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return {}
    return {group: type(cfg).__name__ for group, cfg in actuators.items()}


def _to_delayed(cfg: Any, delayed_cls: type, min_delay: int, max_delay: int) -> Any:
    """Rebuild an implicit actuator config as its delayed subclass.

    Fields are copied by name rather than by ``replace``: the delayed class adds
    two fields the parent does not have, so a plain copy would drop them, and a
    positional rebuild would silently reorder anything upstream adds later.
    """
    import copy
    import dataclasses

    values = {}
    for field in dataclasses.fields(cfg):
        # ``class_type`` is the factory discriminator. Carrying the parent's
        # ImplicitActuator value into the delayed config creates an object that
        # prints as DelayedImplicitActuatorCfg but still instantiates the plain
        # actuator -- exactly the live-only defect the lag audit caught.
        if field.name in ("class_type", "min_delay", "max_delay"):
            continue
        values[field.name] = copy.deepcopy(getattr(cfg, field.name))
    values["min_delay"] = min_delay
    values["max_delay"] = max_delay
    return delayed_cls(**values)
