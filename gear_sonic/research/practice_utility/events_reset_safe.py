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
