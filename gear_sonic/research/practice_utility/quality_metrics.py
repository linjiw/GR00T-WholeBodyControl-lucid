"""Physical-quality metrics and quality-qualified success.

Success alone is a bad outcome measure for a humanoid curriculum. A policy can
stay upright while chattering its ankles, skating its feet, slamming its
contacts, or sitting on its torque limits -- all of which are fine in
simulation and unacceptable on hardware. A method tuned against success alone
will happily buy success with exactly those artifacts.

So an episode counts as a success here only if it completes *and* passes every
physical gate. That is ``QSuccess``, and the deployment objective ``J_eff`` is
its macro-mean over motion families, not a micro-average over episodes: a large
family must not be able to drown a small one.

Everything is recomputed from simulator state and actions. Reward terms are
deliberately not reused as outcomes -- a reward term is a training signal that
the policy is optimizing against, so scoring the policy with it measures how
well it gamed the objective, not how well it moved.

Thresholds are frozen before the main comparison and are never retuned per
method. :func:`quality_success` returns the reasons an episode failed, so a
gate that is doing too much work is visible rather than buried in an aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

#: Value assigned to the latent gap after a fall, matching LUCID v1's
#: fixed-horizon convention so terminated and completed episodes stay
#: comparable.
TERMINATED_GAP_FILL = 2.0


@dataclass(frozen=True)
class QualityThresholds:
    """Frozen gates for quality-qualified success.

    Sources, in order of preference: simulator/robot safety limits, the nominal
    policy's own rollout distribution, a hardware-safe pilot, and failing those
    a pre-registered quantile of the baseline distribution.
    """

    max_mpjpe: float = 0.30              # rad
    max_foot_slip: float = 0.10          # m per episode
    max_hf_action_ratio: float = 0.25    # share of action energy above cutoff
    max_contact_impulse: float = 400.0   # N*s
    max_torque_saturation: float = 0.05  # fraction of (step, joint) samples
    min_completion: float = 0.95         # share of the reference completed

    def as_dict(self) -> dict[str, float]:
        return {
            "max_mpjpe": self.max_mpjpe,
            "max_foot_slip": self.max_foot_slip,
            "max_hf_action_ratio": self.max_hf_action_ratio,
            "max_contact_impulse": self.max_contact_impulse,
            "max_torque_saturation": self.max_torque_saturation,
            "min_completion": self.min_completion,
        }


@dataclass
class EpisodeQuality:
    """Every quality outcome for one episode, plus its verdict."""

    completed: bool
    completion_fraction: float
    mpjpe: float
    action_rate: float
    action_acceleration: float
    hf_action_ratio: float
    foot_slip: float
    contact_impulse: float
    undesired_contact_rate: float
    torque_saturation: float
    joint_limit_proximity: float
    energy_proxy: float
    episode_length: int
    termination_reason: str = "unknown"
    family: str = "other"
    failed_gates: list[str] = field(default_factory=list)

    @property
    def quality_success(self) -> bool:
        return self.completed and not self.failed_gates

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items()}
        payload["quality_success"] = self.quality_success
        return payload


# --------------------------------------------------------------- action ----


def action_rate(actions: torch.Tensor) -> float:
    """Mean squared first difference of the action sequence."""
    actions = _as_2d(actions, "actions")
    if actions.shape[0] < 2:
        return 0.0
    return float((actions[1:] - actions[:-1]).pow(2).sum(dim=-1).mean())


def action_acceleration(actions: torch.Tensor) -> float:
    """Mean squared second difference -- the jerk-like term hardware feels."""
    actions = _as_2d(actions, "actions")
    if actions.shape[0] < 3:
        return 0.0
    second = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]
    return float(second.pow(2).sum(dim=-1).mean())


def high_frequency_action_ratio(
    actions: torch.Tensor, control_hz: float = 50.0, cutoff_hz: float = 10.0
) -> float:
    """Share of action energy above ``cutoff_hz``.

    A ratio rather than an absolute energy, so it does not simply track how
    vigorous the motion is: a fast run and a jittery stand should be
    distinguishable, and only the second should be penalized.

    The mean is removed first because a constant offset is not oscillation and
    would otherwise dominate the spectrum.
    """
    actions = _as_2d(actions, "actions")
    steps = actions.shape[0]
    if steps < 4:
        return 0.0
    if control_hz <= 0:
        raise ValueError(f"control_hz must be > 0, got {control_hz}")
    if not 0 < cutoff_hz < control_hz / 2:
        raise ValueError(
            f"cutoff_hz must lie in (0, Nyquist={control_hz / 2}), got {cutoff_hz}"
        )

    centred = actions.to(torch.float64) - actions.to(torch.float64).mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(centred, dim=0).abs().pow(2)
    freqs = torch.fft.rfftfreq(steps, d=1.0 / control_hz)

    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    return float(spectrum[freqs >= cutoff_hz].sum() / total)


# ----------------------------------------------------------------- feet ----


def foot_slip(
    foot_velocity_xy: torch.Tensor, contact_mask: torch.Tensor, dt: float
) -> float:
    """Total horizontal distance the feet travelled *while in contact*.

    A foot in contact should be stationary; distance accumulated under contact
    is skating. Measured in metres so it can be compared against a physical
    tolerance rather than an arbitrary score.
    """
    if foot_velocity_xy.ndim != 3 or foot_velocity_xy.shape[-1] != 2:
        raise ValueError(
            f"foot_velocity_xy must be (T, feet, 2), got {tuple(foot_velocity_xy.shape)}"
        )
    if contact_mask.shape != foot_velocity_xy.shape[:2]:
        raise ValueError(
            f"contact_mask must be (T, feet), got {tuple(contact_mask.shape)}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    speed = foot_velocity_xy.to(torch.float64).norm(dim=-1)
    return float((speed * contact_mask.to(torch.float64)).sum() * dt)


def contact_impulse(contact_forces: torch.Tensor, dt: float) -> tuple[float, float]:
    """Return ``(peak_force, impulse_integral)`` over all contact bodies."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    forces = contact_forces.to(torch.float64)
    if forces.numel() == 0:
        return 0.0, 0.0
    magnitude = forces.norm(dim=-1) if forces.ndim == 3 else forces.abs()
    return float(magnitude.max()), float(magnitude.sum() * dt)


def undesired_contact_rate(undesired_contacts: torch.Tensor) -> float:
    """Share of timesteps with contact on a body that should not touch ground."""
    if undesired_contacts.numel() == 0:
        return 0.0
    per_step = undesired_contacts.to(torch.bool).reshape(undesired_contacts.shape[0], -1).any(dim=1)
    return float(per_step.to(torch.float64).mean())


# ------------------------------------------------------------- actuator ----


def torque_saturation(
    torques: torch.Tensor, torque_limits: torch.Tensor, threshold: float = 0.95
) -> float:
    """Fraction of (timestep, joint) samples at or beyond ``threshold`` of limit."""
    torques = _as_2d(torques, "torques")
    limits = torque_limits.to(torch.float64).reshape(-1)
    if limits.numel() != torques.shape[1]:
        raise ValueError(
            f"torque_limits has {limits.numel()} entries for {torques.shape[1]} joints"
        )
    if bool((limits <= 0).any()):
        raise ValueError("torque_limits must all be positive")
    ratio = torques.to(torch.float64).abs() / limits
    return float((ratio >= threshold).to(torch.float64).mean())


def joint_limit_proximity(
    positions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> float:
    """Mean normalized closeness to a joint limit, in ``[0, 1]``.

    0 is mid-range, 1 is on the stop. Reported as a mean because occasional
    limit contact is normal; sustained proximity is what indicates a policy
    bracing against its own kinematics.
    """
    positions = _as_2d(positions, "positions")
    lower = lower.to(torch.float64).reshape(-1)
    upper = upper.to(torch.float64).reshape(-1)
    if lower.numel() != positions.shape[1] or upper.numel() != positions.shape[1]:
        raise ValueError("joint limit vectors must match the number of joints")
    if bool((upper <= lower).any()):
        raise ValueError("upper joint limits must exceed lower limits")
    centre = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower)
    return float((positions.to(torch.float64) - centre).abs().div(half_range).clamp(0, 1).mean())


def energy_proxy(torques: torch.Tensor, joint_velocities: torch.Tensor) -> float:
    """Mean absolute mechanical power, summed over joints."""
    torques = _as_2d(torques, "torques")
    velocities = _as_2d(joint_velocities, "joint_velocities")
    if torques.shape != velocities.shape:
        raise ValueError(
            f"torques {tuple(torques.shape)} and velocities {tuple(velocities.shape)} must align"
        )
    return float((torques.to(torch.float64) * velocities.to(torch.float64)).abs().sum(dim=1).mean())


# ------------------------------------------------------------- verdicts ----


def evaluate_gates(quality: EpisodeQuality, thresholds: QualityThresholds) -> list[str]:
    """Names of the gates this episode fails, in a stable order."""
    checks = (
        ("completion", quality.completion_fraction < thresholds.min_completion),
        ("mpjpe", quality.mpjpe > thresholds.max_mpjpe),
        ("foot_slip", quality.foot_slip > thresholds.max_foot_slip),
        ("hf_action", quality.hf_action_ratio > thresholds.max_hf_action_ratio),
        ("contact_impulse", quality.contact_impulse > thresholds.max_contact_impulse),
        ("torque_saturation", quality.torque_saturation > thresholds.max_torque_saturation),
    )
    return [name for name, failed in checks if failed]


def apply_gates(
    episodes: Iterable[EpisodeQuality], thresholds: QualityThresholds
) -> list[EpisodeQuality]:
    """Populate ``failed_gates`` on each episode and return them."""
    result = []
    for episode in episodes:
        episode.failed_gates = evaluate_gates(episode, thresholds)
        result.append(episode)
    return result


def macro_mean_quality_success(episodes: Sequence[EpisodeQuality]) -> float:
    """``J_eff``: mean over families of the per-family quality-success rate.

    Macro, not micro. A micro-average lets a large family determine the score,
    which is how a method that helps common motions and harms rare ones can look
    like an improvement.
    """
    if not episodes:
        return 0.0
    by_family: dict[str, list[bool]] = {}
    for episode in episodes:
        by_family.setdefault(episode.family, []).append(episode.quality_success)
    rates = [sum(v) / len(v) for v in by_family.values()]
    return sum(rates) / len(rates)


def family_success_rates(episodes: Sequence[EpisodeQuality]) -> dict[str, float]:
    """Per-family quality-success rate, for worst-family reporting."""
    by_family: dict[str, list[bool]] = {}
    for episode in episodes:
        by_family.setdefault(episode.family, []).append(episode.quality_success)
    return {f: sum(v) / len(v) for f, v in sorted(by_family.items())}


def gate_failure_counts(episodes: Sequence[EpisodeQuality]) -> dict[str, int]:
    """How often each gate fired -- exposes a gate doing all the work."""
    counts: dict[str, int] = {}
    for episode in episodes:
        for gate in episode.failed_gates:
            counts[gate] = counts.get(gate, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def summarize(
    episodes: Sequence[EpisodeQuality], thresholds: QualityThresholds
) -> dict[str, Any]:
    """Aggregate report for one evaluation suite."""
    episodes = apply_gates(list(episodes), thresholds)
    families = family_success_rates(episodes)
    raw_success = (
        sum(e.completed for e in episodes) / len(episodes) if episodes else 0.0
    )
    return {
        "num_episodes": len(episodes),
        "raw_completion_rate": raw_success,
        "quality_success_rate_micro": (
            sum(e.quality_success for e in episodes) / len(episodes) if episodes else 0.0
        ),
        "j_eff_macro": macro_mean_quality_success(episodes),
        "worst_family": min(families, key=lambda f: families[f]) if families else None,
        "worst_family_rate": min(families.values()) if families else 0.0,
        "family_success_rates": families,
        "gate_failure_counts": gate_failure_counts(episodes),
        "thresholds": thresholds.as_dict(),
        "mean_quality": {
            name: sum(getattr(e, name) for e in episodes) / len(episodes)
            for name in (
                "mpjpe", "action_rate", "action_acceleration", "hf_action_ratio",
                "foot_slip", "contact_impulse", "torque_saturation", "energy_proxy",
            )
        } if episodes else {},
    }


def _as_2d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be (T, D), got shape {tuple(tensor.shape)}")
    return tensor
