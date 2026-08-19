"""Reset-safe, per-environment versions of SONIC's startup-only DR terms.

A curriculum can only schedule a randomization channel whose ranges are re-read
after startup. Three of SONIC's five channels are declared ``mode: "startup"``,
and simply switching them to ``"reset"`` produces wrong physics. This module
provides drop-in replacements that are correct at every reset, plus the one
helper that makes friction schedulable at all.

What was actually wrong with each
---------------------------------
``randomize_rigid_body_mass`` (IsaacLab) is the model to copy: it restores
``default_mass`` for the selected environments *before* randomizing, so repeated
calls re-derive from the nominal instead of stacking on the previous draw.

``randomize_rigid_body_com`` (SONIC) does ``coms[:, body_ids, :3] += rand`` on
the *current* CoM. Called every reset, offsets accumulate without bound. It also
computes ``rand_samples`` for ``len(env_ids)`` rows but adds them into all
``num_envs`` rows, which is silently wrong for any partial reset -- it only
happens to work at startup, where ``env_ids`` is every environment.

``randomize_joint_default_pos`` (SONIC) adds to the current default and
overwrites its own ``default_joint_pos_nominal`` from the already-randomized
value on each call, so the "nominal" drifts along with everything else.

``randomize_rigid_body_material`` (IsaacLab) samples its material buckets once
in ``__init__`` from the configured ranges; ``__call__`` only draws bucket
*indices*. Rescaling the range parameters at runtime therefore has **no effect
whatsoever** -- the friction distribution never moves. Only
:func:`resample_material_buckets` actually changes it.

Nominal state is cached on the asset under a private attribute the first time a
term runs, so these functions are safe to call at startup and at every
subsequent reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:  # pragma: no cover - import-time IsaacLab is unavailable
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg

# IsaacLab modules cannot be imported before SimulationApp is instantiated, so
# nothing from Omniverse is imported at module scope. Sampling and the
# add/scale/abs operation are reimplemented below rather than pulled from
# isaaclab.envs.mdp.events, which also removes a dependency on a private helper
# whose signature upstream is free to change.

#: Where per-asset nominal values are cached.
NOMINAL_COM = "_practice_nominal_com"
NOMINAL_JOINT_POS = "_practice_nominal_default_joint_pos"


def randomize_rigid_body_com(
    env: Any,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Per-env CoM randomization that does not accumulate across resets.

    Restores the cached nominal CoM for the selected environments, then applies a
    fresh offset -- the pattern ``randomize_rigid_body_mass`` uses. Indexing is
    done with an explicit ``env_ids`` row selector so a partial reset writes only
    the rows it sampled.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    if env_ids.numel() == 0:
        return

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()
    nominal = getattr(asset, NOMINAL_COM, None)
    if nominal is None:
        # First call defines the nominal, before any randomization is applied.
        nominal = coms.clone()
        setattr(asset, NOMINAL_COM, nominal)

    rows = env_ids[:, None]
    coms[rows, body_ids] = nominal[rows, body_ids].clone()

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    ranges = torch.tensor(range_list, dtype=coms.dtype, device="cpu")
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (env_ids.numel(), 3)).unsqueeze(1)

    coms[rows, body_ids, :3] += samples
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_joint_default_pos(
    env: Any,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Per-env joint-default randomization that re-derives from the nominal.

    Also keeps the action manager's offset in step, exactly as SONIC's original
    does -- without that, the action space and the joint defaults drift apart and
    a commanded zero stops meaning the default pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if pos_distribution_params is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if env_ids.numel() == 0:
        return

    nominal = getattr(asset, NOMINAL_JOINT_POS, None)
    if nominal is None:
        nominal = asset.data.default_joint_pos.clone()
        setattr(asset, NOMINAL_JOINT_POS, nominal)
        # Preserve SONIC's export field, taken from the true nominal.
        asset.data.default_joint_pos_nominal = nominal[0].clone()

    if asset_cfg.joint_ids == slice(None):
        joint_ids = torch.arange(
            asset.data.default_joint_pos.shape[1], dtype=torch.long, device=asset.device
        )
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)

    rows = env_ids[:, None]
    # Restore before randomizing so 'add' cannot compound across resets.
    asset.data.default_joint_pos[rows, joint_ids] = nominal[rows, joint_ids].clone()

    base = asset.data.default_joint_pos[rows, joint_ids]
    low, high = float(pos_distribution_params[0]), float(pos_distribution_params[1])
    samples = sample_uniform(
        torch.tensor(low, device=base.device), torch.tensor(high, device=base.device),
        tuple(base.shape),
    ).to(base.dtype)
    asset.data.default_joint_pos[rows, joint_ids] = apply_operation(base, samples, operation)

    _sync_action_offset(env, asset, env_ids)


def resample_material_buckets(
    term: Any,
    static_friction_range: tuple[float, float] | None = None,
    dynamic_friction_range: tuple[float, float] | None = None,
    restitution_range: tuple[float, float] | None = None,
    make_consistent: bool = False,
) -> bool:
    """Rebuild a material term's buckets so a range change actually takes effect.

    ``randomize_rigid_body_material`` samples ``num_buckets`` materials once, in
    ``__init__``, and thereafter only chooses among them. Scaling its range
    parameters at runtime moves nothing. This resamples the buckets in place,
    which is the only way a friction curriculum can exist.

    Bucket *count* is unchanged on purpose: PhysX caps the scene at 64000 unique
    materials, which is why the buckets are finite in the first place.

    Returns whether the term was resampled.
    """
    buckets = getattr(term, "material_buckets", None)
    if buckets is None:
        return False

    current = buckets.clone()
    ranges = [
        static_friction_range if static_friction_range is not None
        else (float(current[:, 0].min()), float(current[:, 0].max())),
        dynamic_friction_range if dynamic_friction_range is not None
        else (float(current[:, 1].min()), float(current[:, 1].max())),
        restitution_range if restitution_range is not None
        else (float(current[:, 2].min()), float(current[:, 2].max())),
    ]
    bounds = torch.tensor(ranges, dtype=current.dtype, device=current.device)
    resampled = sample_uniform(bounds[:, 0], bounds[:, 1], (current.shape[0], 3))
    if make_consistent:
        # Physics requires dynamic friction <= static friction.
        resampled[:, 1] = torch.min(resampled[:, 0], resampled[:, 1])
    term.material_buckets = resampled.to(current.dtype)
    return True


def sample_uniform(low: torch.Tensor, high: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Uniform samples in ``[low, high]``, broadcasting over the trailing dims."""
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=torch.float32)
    return torch.rand(shape, device=low.device) * (high - low) + low


def apply_operation(
    base: torch.Tensor, samples: torch.Tensor, operation: str
) -> torch.Tensor:
    """Combine a random draw with a nominal value, as IsaacLab's ops do."""
    if operation == "add":
        return base + samples
    if operation == "scale":
        return base * samples
    if operation == "abs":
        return samples
    raise ValueError(f"unsupported operation {operation!r}; expected add/scale/abs")


def _sync_action_offset(env: Any, asset: Articulation, env_ids: torch.Tensor) -> None:
    """Keep the joint-position action offset aligned with the new defaults."""
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return
    try:
        term = manager.get_term("joint_pos")
    except Exception:
        return
    action_names = getattr(term, "_joint_names", None)
    offset = getattr(term, "_offset", None)
    if not action_names or offset is None or not hasattr(offset, "shape"):
        return
    if getattr(offset, "ndim", 0) < 2:
        return

    asset_names = list(asset.joint_names)
    shared = [n for n in action_names if n in set(asset_names)]
    if not shared:
        return
    action_idx = torch.tensor(
        [action_names.index(n) for n in shared], dtype=torch.long, device=offset.device
    )
    asset_idx = torch.tensor(
        [asset_names.index(n) for n in shared], dtype=torch.long, device=asset.device
    )
    rows = env_ids.to(asset.device)[:, None]
    values = asset.data.default_joint_pos[rows, asset_idx]
    offset[env_ids.to(offset.device)[:, None], action_idx] = values.to(offset.device)


# --------------------------------------------------------------------------
# Actuation latency
# --------------------------------------------------------------------------
#
# SONIC already ships a complete, correct DelayedImplicitActuator
# (gear_sonic/envs/manager_env/mdp/actuators.py) that wraps each actuator's
# command in an IsaacLab DelayBuffer and resamples a per-environment lag on
# every reset. A repo-wide search finds no reference to it anywhere: it is dead
# code, and G1 uses plain ImplicitActuatorCfg for all five joint groups.
#
# That makes it the right place to inject latency, for three reasons:
#   * resolution is one physics step (5 ms at the configured 200 Hz), not one
#     control step (20 ms), so 0/5/.../40 ms are all expressible;
#   * per-env lag and per-reset resampling already exist and are exercised by
#     IsaacLab's own DelayedPDActuator;
#   * it sits *downstream* of everything the learner sees. action_manager.action,
#     the `actions` observation history, PPO's stored actions and
#     extras["env_actions"] all keep meaning "commanded", exactly as on hardware.
#     Delaying further upstream would feed the policy its own executed action and
#     quietly break the train/deploy correspondence.
#
# The event term below is what makes latency *curriculum-scalable*: the actuator
# resamples from its own cfg during scene reset, and reset-mode event terms run
# afterwards, so this overwrites that draw with a lambda-scaled one.

#: Attribute names of the delay buffers on a delayed actuator.
DELAY_BUFFERS = ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer")


def randomize_action_delay(
    env: Any,
    env_ids: torch.Tensor | None,
    delay_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: "SceneEntityCfg | None" = None,
) -> int:
    """Sample a per-environment actuation delay, in physics steps.

    ``delay_range`` is expressed in physics steps and is scaled by the LUCID
    curriculum like any other range: at λ=0 it collapses to ``[0, 0]``, which
    makes the delay buffer return its newest sample and the run bit-identical to
    no delay at all. That exact identity is what gives the curriculum a clean A/B
    baseline.

    Returns the number of actuators whose lag was set, so a config that forgot to
    use ``DelayedImplicitActuatorCfg`` reports zero instead of silently training
    without latency.
    """
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    actuators = getattr(asset, "actuators", None)
    if not actuators:
        return 0

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if env_ids.numel() == 0:
        return 0

    low = max(0, int(round(float(delay_range[0]))))
    high = max(low, int(round(float(delay_range[1]))))

    touched = 0
    for actuator in actuators.values():
        buffers = [getattr(actuator, name, None) for name in DELAY_BUFFERS]
        buffers = [b for b in buffers if b is not None]
        if not buffers:
            continue
        capacity = _buffer_capacity(buffers[0])
        if capacity is not None:
            # max_delay sizes the buffer at construction and set_time_lag raises
            # above it, so clamp rather than let a curriculum crash the run.
            high = min(high, capacity)
            low = min(low, high)
        lags = torch.randint(
            low, high + 1, (env_ids.numel(),), dtype=torch.int, device=asset.device
        )
        for buffer in buffers:
            buffer.set_time_lag(lags, env_ids)
            # Resetting matters: without it the buffer still holds targets built
            # from the previous episode's joint-default offset, which
            # randomize_joint_default_pos has just changed.
            buffer.reset(env_ids)
        touched += 1
    return touched


def _buffer_capacity(buffer: Any) -> int | None:
    for attribute in ("history_length", "max_length", "_max_len"):
        value = getattr(buffer, attribute, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


# --------------------------------------------------------------------------
# Piecewise-stationary randomization ("sticky" DR)
# --------------------------------------------------------------------------
#
# Reset-mode terms redraw every episode, so each env's dynamics change
# constantly. That is textbook domain randomization, but it maximises gradient
# variance: every episode is a different MDP, and the policy never gets a
# stationary stretch to actually fit.
#
# Holding a draw for K consecutive episodes decouples two timescales that are
# otherwise fused:
#   * how *wide* the randomization is        -> lambda, the curriculum (slow)
#   * how *often* an env sees a new draw     -> resample_every (fast, but tunable)
#
# With K > 1 each environment trains on one fixed set of physics for K episodes,
# so within a block the problem is stationary; across the population the policy
# still sees the full distribution, because different envs are on different
# phases of their counters.

#: Where the per-env resample counter is cached on the environment.
RESAMPLE_COUNTER = "_practice_resample_counter"


def due_for_resample(env: Any, env_ids: torch.Tensor, every: int, key: str) -> torch.Tensor:
    """Subset of ``env_ids`` whose turn it is to draw new randomization.

    Counters live per (channel) key, so channels can stick for different numbers
    of episodes -- mass might hold for ten while pushes redraw every episode.

    ``every <= 1`` returns ``env_ids`` unchanged, which is the ordinary
    per-episode behaviour and costs nothing.
    """
    if every <= 1:
        return env_ids
    counters = getattr(env, RESAMPLE_COUNTER, None)
    if counters is None:
        counters = {}
        setattr(env, RESAMPLE_COUNTER, counters)
    num_envs = getattr(getattr(env, "scene", None), "num_envs", None) or int(env_ids.max()) + 1
    counter = counters.get(key)
    if counter is None or counter.numel() < num_envs:
        # Seed each env at a random phase. A zero-initialised counter makes the
        # whole population redraw on the same reset, so the entire batch changes
        # physics at once -- a synchronised shock, and the opposite of what
        # holding a draw is meant to buy. Staggered phases mean roughly 1/every
        # of the envs redraw at any reset, so the population distribution stays
        # smooth while each env still gets its stationary block.
        counter = torch.randint(0, max(every, 1), (num_envs,), dtype=torch.long)
        counters[key] = counter

    ids = env_ids.detach().cpu().long()
    due = (counter[ids] % every) == 0
    counter[ids] += 1
    return env_ids[due.to(env_ids.device)]


def sticky(inner: Any, key: str, every: int = 1):
    """Wrap a reset-mode event function so it only fires every ``every`` resets.

    The wrapped function keeps its original signature, so this is a drop-in for
    any ``mode: "reset"`` term.
    """

    def wrapper(env: Any, env_ids: torch.Tensor | None, *args: Any, **kwargs: Any):
        if env_ids is None:
            num_envs = env.scene.num_envs
            env_ids = torch.arange(num_envs)
        selected = due_for_resample(env, env_ids, every, key)
        if selected.numel() == 0:
            return None
        return inner(env, selected, *args, **kwargs)

    wrapper.__name__ = f"sticky_{getattr(inner, '__name__', key)}"
    wrapper.__doc__ = (
        f"{getattr(inner, '__doc__', '') or ''}\n\n"
        f"Wrapped to resample only every {every} episode resets (channel {key!r})."
    )
    return wrapper


# --------------------------------------------------------------------------
# Event config carrying the latency channel
# --------------------------------------------------------------------------


def _build_event_cfg():
    """Subclass SONIC's EventCfg with a latency slot.

    SONIC's ``EventCfg`` declares its terms as fixed class attributes, so a new
    randomization channel needs a new field. Subclassing here rather than editing
    the upstream class keeps every unmodified run genuinely unmodified -- and a
    baseline arm that shares a mutated robot or event config with the treatment
    arm is not a baseline.

    Built lazily because the parent lives behind an Isaac Sim import.
    """
    from isaaclab.utils import configclass

    from gear_sonic.envs.manager_env.mdp.events import EventCfg

    @configclass
    class LucidEventCfg(EventCfg):
        """SONIC's events plus a curriculum-scalable actuation-latency term."""

        randomize_action_delay = None

    return LucidEventCfg


def __getattr__(name: str):
    # Module-level lazy attribute so `LucidEventCfg` resolves only once Isaac is
    # importable, while `_target_` strings referencing it stay valid in configs.
    if name == "LucidEventCfg":
        return _build_event_cfg()
    raise AttributeError(name)
