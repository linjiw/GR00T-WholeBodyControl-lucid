"""Localized intervention kernels and identity-preserving distribution mixing.

This module is the mathematical core of the paired branch-and-continue design
and is deliberately free of Isaac Lab and ``MotionLibBase`` imports, so its
guarantees can be proven on CPU before any simulator time is spent.

Two operations live here:

**Intervention** (measurement phase). Given SONIC's native bin distribution
``rho`` and a candidate context ``b``, form

    rho_eps = (1 - eps) * rho + eps * kappa_b

where ``kappa_b`` is a local kernel over ``b`` and its neighbours *within the
same motion clip*. This is a reallocation, not an addition: the total sampling
mass and the PPO budget are unchanged, so the control and intervention branches
remain equal-compute.

**Residual reweighting** (method phase, gated). Given per-bin utility scores,
form ``q = (1 - alpha) * rho + alpha * rho * exp(s / tau) / Z`` subject to a KL
radius, coverage floors, and concentration caps.

Both operations have an exact identity point -- ``eps = 0``, ``alpha = 0``, or
constant scores return ``rho`` unchanged -- which is what makes a null result
interpretable rather than a confound.

Why a kernel rather than a single bin: a bin boundary is arbitrary with respect
to the motion. A difficult transition routinely straddles two bins, and the
first frames of a bin may lack the entry pose that makes the segment
executable. Concentrating all added mass on one 50-frame bin would also produce
an unnatural phase distribution. Radius 1 spreads the dose over roughly three
seconds of local motion neighbourhood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: Probability mass below which a distribution entry is treated as zero.
PROB_TOL = 1e-12


@dataclass(frozen=True)
class KernelSpec:
    """Shape of the local intervention kernel."""

    radius_bins: int = 1
    #: Decay length in frames for the exponential falloff between bin centres.
    sigma_frames: float = 50.0

    def __post_init__(self) -> None:
        if self.radius_bins < 0:
            raise ValueError(f"radius_bins must be >= 0, got {self.radius_bins}")
        if self.sigma_frames <= 0:
            raise ValueError(f"sigma_frames must be > 0, got {self.sigma_frames}")


def build_local_kernel(
    target_position: int,
    bin_positions: torch.Tensor,
    bin_motion_ids: torch.Tensor,
    target_motion_id: int,
    bin_centre_frames: torch.Tensor,
    spec: KernelSpec = KernelSpec(),
) -> torch.Tensor:
    """Build a normalized kernel over bins local to one context.

    Args:
        target_position: index *into the provided arrays* of the target bin.
        bin_positions: per-entry bin index within its own motion clip.
        bin_motion_ids: per-entry originating motion id.
        target_motion_id: motion id of the target bin.
        bin_centre_frames: per-entry centre frame, used for the decay weight.
        spec: kernel shape.

    Returns:
        A non-negative tensor summing to 1, supported only on bins of the target
        motion within ``spec.radius_bins`` of the target.

    The kernel never crosses a motion boundary: neighbouring bins of a
    *different* clip are not a local neighbourhood in any meaningful sense, and
    including them would let the intervention leak into unrelated contexts.
    """
    if bin_positions.shape != bin_motion_ids.shape != bin_centre_frames.shape:
        raise ValueError("bin_positions, bin_motion_ids, bin_centre_frames must align")
    if not 0 <= target_position < bin_positions.numel():
        raise IndexError(f"target_position {target_position} out of range")

    target_bin_pos = bin_positions[target_position]
    same_motion = bin_motion_ids == target_motion_id
    within_radius = (bin_positions - target_bin_pos).abs() <= spec.radius_bins
    support = same_motion & within_radius

    if not bool(support.any()):
        raise ValueError("kernel support is empty; target bin is not in the provided arrays")

    frame_distance = (bin_centre_frames - bin_centre_frames[target_position]).abs().to(torch.float64)
    weights = torch.exp(-frame_distance / spec.sigma_frames)
    weights = torch.where(support, weights, torch.zeros_like(weights))

    total = weights.sum()
    if total <= PROB_TOL:
        raise ValueError("kernel weights vanished; check sigma_frames against bin spacing")
    return weights / total


def mix_intervention(
    base_prob: torch.Tensor,
    kernel: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return ``(1 - epsilon) * base_prob + epsilon * kernel``.

    ``epsilon == 0`` returns the base distribution unchanged, bitwise -- that
    exact identity is what the epsilon=0 branch test relies on to isolate branch
    noise from intervention effect.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    _validate_distribution(base_prob, "base_prob")
    _validate_distribution(kernel, "kernel")
    if base_prob.shape != kernel.shape:
        raise ValueError(f"shape mismatch: base {tuple(base_prob.shape)} vs kernel {tuple(kernel.shape)}")

    if epsilon == 0.0:
        return base_prob.clone()
    return (1.0 - epsilon) * base_prob + epsilon * kernel


def residual_distribution(
    base_prob: torch.Tensor,
    scores: torch.Tensor,
    alpha: float,
    temperature: float,
    max_kl: float | None = None,
    max_prob_ratio: float | None = None,
    coverage_floor: float = 0.0,
) -> torch.Tensor:
    """Identity-preserving residual reweighting of ``base_prob`` by ``scores``.

    The exponential tilt is applied to the base distribution and then blended
    back with weight ``alpha``, so the base curriculum remains the default and
    the learned component can only perturb it within a bounded radius.

    If ``max_kl`` is given, the temperature is raised by bisection until
    ``KL(q || base) <= max_kl``. Raising temperature flattens the tilt, so the
    constraint is always satisfiable -- in the limit the tilt vanishes and
    ``q -> base``.

    Returns ``base_prob`` unchanged when ``alpha == 0`` or when all finite
    scores are equal.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if coverage_floor < 0.0:
        raise ValueError(f"coverage_floor must be >= 0, got {coverage_floor}")
    _validate_distribution(base_prob, "base_prob")
    if base_prob.shape != scores.shape:
        raise ValueError("base_prob and scores must align")

    if alpha == 0.0:
        return base_prob.clone()

    supported = base_prob > PROB_TOL
    if not bool(supported.any()):
        raise ValueError("base_prob has empty support")
    if _scores_are_constant(scores, supported):
        return base_prob.clone()

    def tilt(tau: float) -> torch.Tensor:
        centred = scores.to(torch.float64) - scores[supported].to(torch.float64).max()
        weights = torch.exp(centred / tau)
        weights = torch.where(supported, weights, torch.zeros_like(weights))
        tilted = base_prob.to(torch.float64) * weights
        total = tilted.sum()
        if total <= PROB_TOL:
            return base_prob.to(torch.float64).clone()
        tilted = tilted / total
        if max_prob_ratio is not None:
            tilted = _cap_probability_ratio(tilted, base_prob.to(torch.float64), max_prob_ratio)
        return (1.0 - alpha) * base_prob.to(torch.float64) + alpha * tilted

    candidate = tilt(temperature)
    if max_kl is not None:
        candidate = _shrink_to_kl_radius(base_prob, tilt, temperature, max_kl, candidate)

    if coverage_floor > 0.0:
        candidate = _apply_coverage_floor(candidate, supported, coverage_floor)

    return candidate.to(base_prob.dtype)


def _cap_probability_ratio(
    tilted: torch.Tensor, base: torch.Tensor, max_ratio: float
) -> torch.Tensor:
    """Cap ``tilted_i / base_i`` at ``max_ratio``, redistributing the excess.

    This is the same concentration guard SONIC applies to its own adaptive
    sampler (``max_prob_per_bin``) and that LACE's intervention plan expresses
    as ``max_probability_ratio``: no context may claim more than ``max_ratio``
    times its base share, however extreme its score.

    Excess mass is returned to the uncapped support in proportion to current
    mass, and the process is iterated because redistribution can push a
    previously-uncapped entry over the cap.
    """
    if max_ratio <= 0:
        raise ValueError(f"max_prob_ratio must be > 0, got {max_ratio}")
    cap = base * float(max_ratio)
    result = tilted.clone()
    for _ in range(100):
        over = result > cap + PROB_TOL
        if not bool(over.any()):
            break
        excess = float((result[over] - cap[over]).sum())
        result = torch.where(over, cap, result)
        room = (~over) & (base > PROB_TOL)
        headroom = torch.where(room, (cap - result).clamp(min=0.0), torch.zeros_like(result))
        total_headroom = float(headroom.sum())
        if total_headroom <= PROB_TOL:
            # Every supported entry is at its cap: the cap is globally binding
            # and the best feasible answer is the capped distribution itself.
            break
        result = result + headroom * (excess / total_headroom)
    total = result.sum()
    return result / total if total > PROB_TOL else tilted


def kl_divergence(q: torch.Tensor, p: torch.Tensor) -> float:
    """KL(q || p) in nats, over the support of ``q``."""
    q64, p64 = q.to(torch.float64), p.to(torch.float64)
    mask = q64 > PROB_TOL
    if not bool(mask.any()):
        return 0.0
    if bool(((p64 <= PROB_TOL) & mask).any()):
        return math.inf
    return float((q64[mask] * (q64[mask] / p64[mask]).log()).sum())


def _shrink_to_kl_radius(base_prob, tilt, temperature, max_kl, candidate):
    """Bisect on temperature until the tilt fits inside the KL radius."""
    if kl_divergence(candidate, base_prob) <= max_kl:
        return candidate
    low, high = temperature, temperature
    for _ in range(60):
        high *= 2.0
        if kl_divergence(tilt(high), base_prob) <= max_kl:
            break
    else:
        return base_prob.to(torch.float64).clone()
    for _ in range(100):
        mid = 0.5 * (low + high)
        if kl_divergence(tilt(mid), base_prob) <= max_kl:
            high = mid
        else:
            low = mid
    return tilt(high)


def _apply_coverage_floor(candidate, supported, coverage_floor):
    """Guarantee every supported entry keeps at least ``coverage_floor`` mass."""
    n_supported = int(supported.sum())
    if coverage_floor * n_supported > 1.0:
        raise ValueError(
            f"coverage_floor {coverage_floor} infeasible for {n_supported} supported bins"
        )
    floored = torch.where(
        supported,
        candidate.clamp(min=coverage_floor),
        torch.zeros_like(candidate),
    )
    excess = float(floored.sum()) - 1.0
    if excess <= 0:
        return floored / floored.sum()
    # Reclaim the excess only from entries that are strictly above the floor.
    headroom = (floored - coverage_floor).clamp(min=0.0)
    total_headroom = float(headroom.sum())
    if total_headroom <= PROB_TOL:
        return floored / floored.sum()
    return floored - headroom * (excess / total_headroom)


def _scores_are_constant(scores: torch.Tensor, supported: torch.Tensor) -> bool:
    values = scores[supported].to(torch.float64)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return True
    return bool((finite.max() - finite.min()) <= 1e-12)


def _validate_distribution(prob: torch.Tensor, name: str) -> None:
    if prob.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(prob.shape)}")
    if prob.numel() == 0:
        raise ValueError(f"{name} is empty")
    if bool((prob < -PROB_TOL).any()):
        raise ValueError(f"{name} contains negative entries")
    total = float(prob.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} sums to {total}, expected 1.0")
