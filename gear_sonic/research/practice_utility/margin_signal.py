"""The termination-margin signal: how close each environment is to being killed.

Everything this programme measured about the latent command-execution gap says
it is the wrong instrument. It embeds the pre-delay PD target against the joint
state, so it is torque over stiffness -- an effort meter -- posture-dominated and
blind to foot placement, the one thing that ends episodes here. It reads from a
single environment. And the two arms that sat at lambda = 1 under identical
physics for 2,000 iterations read 0.565 and 0.305: it separates two policies
under one physics as strongly as it separates lambda = 0 from lambda = 1. No
set-point can bind a signal like that.

The termination terms already compute the quantity that matters and throw it
away: each one evaluates a per-body error, compares it to a threshold, and
reduces with ``.any()``. This module keeps the pre-image. For every environment
and control step,

    m_feet   = max over ankle bodies of ||e_b||          / theta_foot
    m_ee     = max over ankle+wrist bodies of |e_b,z|    / theta_ee(env)
    m_pelvis = |e_anchor,z|                              / theta_anchor(env)
    m_ori    = angle(q_ref, q_robot)^2                   / theta_ori
    M        = max(m_feet, m_ee, m_pelvis, m_ori)

with every theta read from the live termination manager, never from yaml, so
``M = 1`` is exactly the kill boundary and the unit is "fraction of the way to
termination". The argmax names the culprit body, which is the decomposition a
curriculum can act on.

Two more choices, each answering a measured defect:

* **Horizon-matched.** The logged kinematic errors turned out to be the error at
  the moment an episode *ended* (IsaacLab logs command metrics over the envs
  being reset), which is why they tracked competence at -0.98. The margin is
  therefore summarised as a prefix mean over the first ``K`` steps of each
  episode, so short and long episodes contribute the same window, and the
  coverage ``P(len < K)`` is reported beside it.
* **Self-referenced.** A yardstick cohort of environments is held at
  lambda = 0 in the same run, and the controller input is the ratio
  ``R = q_focus / q_yardstick`` of per-cohort medians. R is 1 at lambda = 0 by
  construction, measures how much the current dose degrades *this* policy
  relative to its own nominal execution, and needs no constant calibrated in
  another regime -- the failure mode of both ``delta_target`` and the absolute
  return floor.

Nothing here imports IsaacLab; the command term and termination manager are
duck-typed so the arithmetic is testable on a CPU with fakes.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch

ANKLES = ("left_ankle_roll_link", "right_ankle_roll_link")
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
CULPRITS = ("feet", "ee", "pelvis", "ori")


@dataclass(frozen=True)
class Thresholds:
    """The in-force termination thresholds, read from the live manager."""

    foot: float
    ee: float
    ee_down: float
    ee_adaptive: bool
    anchor: float
    anchor_down: float
    anchor_adaptive: bool
    root_height: float
    ori: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "foot_pos_xyz": self.foot,
            "ee_body_pos": {"threshold": self.ee, "down": self.ee_down, "adaptive": self.ee_adaptive},
            "anchor_pos": {"threshold": self.anchor, "down": self.anchor_down, "adaptive": self.anchor_adaptive},
            "root_height_threshold": self.root_height,
            "anchor_ori_full": self.ori,
        }


def read_thresholds(termination_manager: Any) -> Thresholds:
    """Pull the thresholds the simulator is actually using.

    Reading yaml would repeat the mistake that made every early from-scratch
    number wrong: the exp preset overrides the term files, and the launcher
    overrides the preset. The manager holds what is in force.
    """
    names = list(getattr(termination_manager, "active_terms", None) or getattr(termination_manager, "_term_names", []))
    cfgs = list(getattr(termination_manager, "_term_cfgs", []))
    params = {name: dict(getattr(cfg, "params", {}) or {}) for name, cfg in zip(names, cfgs)}

    def need(term: str) -> dict[str, Any]:
        if term not in params:
            raise KeyError(f"termination term {term!r} is not active; the margin needs it")
        return params[term]

    foot = need("foot_pos_xyz")
    ee = need("ee_body_pos")
    anchor = need("anchor_pos")
    ori = need("anchor_ori_full")
    return Thresholds(
        foot=float(foot["threshold"]),
        ee=float(ee["threshold"]),
        ee_down=float(ee.get("down_threshold", ee["threshold"])),
        ee_adaptive=bool(ee.get("threshold_adaptive", False)),
        anchor=float(anchor["threshold"]),
        anchor_down=float(anchor.get("down_threshold", anchor["threshold"])),
        anchor_adaptive=bool(anchor.get("threshold_adaptive", False)),
        root_height=float(
            ee.get("root_height_threshold", anchor.get("root_height_threshold", 0.5))
        ),
        ori=float(ori["threshold"]),
    )


def _indices(command: Any, names: Iterable[str]) -> list[int]:
    # TrackingCommand keeps the tracked list as cmd_body_names (== cfg.body_names),
    # which is what the termination terms index into via _get_body_indexes.
    cfg = getattr(command, "cfg", None)
    body_names = list(
        getattr(command, "cmd_body_names", None)
        or getattr(cfg, "body_names", None)
        or getattr(command, "body_names", [])
    )
    missing = [n for n in names if n not in body_names]
    if missing:
        raise KeyError(f"tracked bodies missing from the command term: {missing}")
    return [body_names.index(n) for n in names]


def quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation angle between two (w, x, y, z) quaternions, in radians.

    Same magnitude as IsaacLab's ``quat_error_magnitude``; written out so this
    module needs nothing from Omniverse to be tested.
    """
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


@dataclass(frozen=True)
class Margins:
    """Per-environment margins for one control step."""

    feet: torch.Tensor
    ee: torch.Tensor
    pelvis: torch.Tensor
    ori: torch.Tensor

    @property
    def overall(self) -> torch.Tensor:
        return torch.stack([self.feet, self.ee, self.pelvis, self.ori], dim=-1).max(dim=-1).values

    @property
    def culprit(self) -> torch.Tensor:
        """Index into :data:`CULPRITS` of the binding margin per env."""
        return torch.stack([self.feet, self.ee, self.pelvis, self.ori], dim=-1).argmax(dim=-1)


def termination_margins(command: Any, thresholds: Thresholds) -> Margins:
    """The termination pre-image, normalised so 1.0 is the kill boundary."""
    ref = command.body_pos_relative_w
    robot = command.robot_body_pos_w
    ankles = _indices(command, ANKLES)
    ee_bodies = _indices(command, ANKLES + WRISTS)

    diff = ref[:, ankles] - robot[:, ankles]
    feet = diff.norm(dim=-1).max(dim=-1).values / thresholds.foot

    ref_root = command.running_ref_root_height
    low = ref_root < thresholds.root_height

    z_err = (ref[:, ee_bodies, 2] - robot[:, ee_bodies, 2]).abs().max(dim=-1).values
    ee_theta = torch.full_like(z_err, thresholds.ee)
    if thresholds.ee_adaptive:
        ee_theta = torch.where(low, torch.full_like(z_err, thresholds.ee_down), ee_theta)
    ee = z_err / ee_theta

    pelvis_err = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    anchor_theta = torch.full_like(pelvis_err, thresholds.anchor)
    if thresholds.anchor_adaptive:
        anchor_theta = torch.where(low, torch.full_like(pelvis_err, thresholds.anchor_down), anchor_theta)
    pelvis = pelvis_err / anchor_theta

    ori = quat_angle(command.anchor_quat_w, command.robot_anchor_quat_w).square() / thresholds.ori
    return Margins(feet=feet, ee=ee, pelvis=pelvis, ori=ori)


@dataclass
class PrefixAccumulator:
    """Per-environment prefix mean of the margin over the first ``K`` steps.

    ``push`` is called once per control step with the pre-step margins and the
    per-env episode age; ``finish`` is called with the envs whose episodes ended
    this step and returns their prefix means. Episodes shorter than ``K``
    contribute what they had; the caller reports coverage.
    """

    num_envs: int
    horizon: int
    device: Any = "cpu"
    sums: torch.Tensor = field(init=False)
    counts: torch.Tensor = field(init=False)
    culprit_hist: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.sums = torch.zeros(self.num_envs, device=self.device)
        self.counts = torch.zeros(self.num_envs, device=self.device)
        self.culprit_hist = torch.zeros(self.num_envs, len(CULPRITS), device=self.device)

    def push(self, margins: Margins, age: torch.Tensor) -> None:
        """Accumulate for envs whose age (steps already taken) is below K.

        Age 0 is skipped on purpose: the command buffers still hold the previous
        episode's last frame at the first step after a reset.
        """
        live = (age > 0) & (age <= self.horizon)
        overall = margins.overall
        self.sums = torch.where(live, self.sums + overall, self.sums)
        self.counts = torch.where(live, self.counts + 1.0, self.counts)
        onehot = torch.nn.functional.one_hot(margins.culprit, len(CULPRITS)).to(self.sums.dtype)
        self.culprit_hist = torch.where(live[:, None], self.culprit_hist + onehot, self.culprit_hist)

    def finish(self, ended: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefix means, step counts and culprit shares of the episodes that ended.

        Returns ``(env_ids, prefix_mean, count, culprit_share)`` and clears
        those envs. Envs that ended with zero accumulated steps are dropped.
        """
        ids = ended.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            empty = torch.zeros(0, device=self.device)
            return ids, empty, empty, torch.zeros(0, len(CULPRITS), device=self.device)
        counts = self.counts[ids]
        keep = counts > 0
        ids = ids[keep]
        counts = counts[keep]
        means = self.sums[ids] / counts
        shares = self.culprit_hist[ids] / counts[:, None]
        self.sums[ids] = 0.0
        self.counts[ids] = 0.0
        self.culprit_hist[ids] = 0.0
        return ids, means, counts, shares


def cohort_median(values: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> float | None:
    """Median of ``values`` over the ended envs that belong to ``mask``."""
    select = mask.to(ids.device)[ids]
    chosen = values[select]
    if chosen.numel() == 0:
        return None
    # torch.median returns the lower middle element for an even count; the
    # interpolated median is what "half the cohort is worse" means.
    return float(torch.quantile(chosen.float(), 0.5).item())


class Ema:
    """Exponential moving average with an explicit time constant in updates."""

    def __init__(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.alpha = 1.0 / float(tau)
        self.value: float | None = None
        self.count = 0

    def update(self, x: float | None) -> float | None:
        if x is None:
            return self.value
        self.count += 1
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value


@dataclass
class MarginRatio:
    """R = q_focus / q_yardstick, the self-referenced controller input."""

    tau: float = 20.0
    focus: Ema = field(init=False)
    yardstick: Ema = field(init=False)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.focus = Ema(self.tau)
        self.yardstick = Ema(self.tau)

    def update(self, q_focus: float | None, q_yardstick: float | None) -> float | None:
        f = self.focus.update(q_focus)
        y = self.yardstick.update(q_yardstick)
        ratio = None if (f is None or y is None or y <= 0.0) else f / y
        self.history.append({"q_focus": q_focus, "q_yardstick": q_yardstick, "ema_focus": f, "ema_yardstick": y, "ratio": ratio})
        return ratio

    @property
    def ready(self) -> bool:
        return self.focus.count >= 1 and self.yardstick.count >= 1


def band_error(ratio: float | None, lo: float, hi: float) -> float:
    """Signed distance outside the band ``[lo, hi]``; zero inside it.

    Positive means the focus cohort is doing *better* than the band wants
    (raise the dose); negative means worse (lower it). Fed to the PI
    controller in place of ``delta_target - gap``, so a ratio inside the band
    holds lambda still instead of drifting on noise.
    """
    if ratio is None:
        return 0.0
    if not 0.0 < lo <= hi:
        raise ValueError(f"need 0 < lo <= hi, got {lo}, {hi}")
    if ratio < lo:
        return lo - ratio
    if ratio > hi:
        return hi - ratio
    return 0.0
