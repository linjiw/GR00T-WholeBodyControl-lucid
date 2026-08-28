"""Apply a scalar DR intensity to SONIC's event-term ranges.

LUCID controls one number, ``lambda`` in [0, 1], and every randomization channel
is scaled by it around its nominal value::

    phi ~ U(phi_0 - lambda * dev_lo,  phi_0 + lambda * dev_hi)

At ``lambda = 0`` every range collapses to its nominal, i.e. no randomization;
at ``lambda = 1`` the configured maximum ranges are restored exactly. Keeping the
channels' *relative* magnitudes fixed and moving only their common scale is the
whole point -- it turns a many-dimensional curriculum into a one-dimensional
control problem, which is what makes a PI controller a defensible choice.

Which terms can actually be scaled at runtime
---------------------------------------------
IsaacLab event terms declare a mode:

``startup``   applied once when the scene is built. Rescaling later does nothing
              and are therefore **outside** a runtime curriculum.
``reset``     re-applied on every episode reset. Scalable.
``interval``  re-applied periodically. Scalable.

This is a property of the configuration, not a limitation of the controller, and
it is reported rather than worked around: :func:`scalable_terms` names exactly
which channels a run is really scheduling, so a curriculum cannot silently claim
credit for randomization it never moved. The LUCID event preset supplies
reset-safe versions of the stock startup-only channels and exposes all six.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: Event modes whose ranges take effect again after startup.
RUNTIME_MODES = ("reset", "interval")

#: Range parameters belonging to the material term, whose buckets are sampled
#: once at construction and must be resampled for a range change to have effect.
MATERIAL_RANGE_KEYS = (
    "static_friction_range",
    "dynamic_friction_range",
    "restitution_range",
)

#: Parameter names understood as randomization ranges, with their nominal value.
#: The nominal is the value at which the channel contributes no randomization.
RANGE_NOMINALS: dict[str, float] = {
    "mass_distribution_params": 1.0,  # multiplicative: 1.0 is unscaled
    "pos_distribution_params": 0.0,  # additive offsets
    "velocity_range": 0.0,
    "com_range": 0.0,
    "static_friction_range": None,  # nominal taken from the midpoint
    "dynamic_friction_range": None,
    "restitution_range": 0.0,
    # Actuation latency, in physics steps. Nominal 0 means lambda=0 collapses to
    # [0, 0] -- zero delay, bit-identical to a run with no latency channel.
    "delay_range": 0.0,
}


@dataclass
class ScalingReport:
    """What a scaling pass actually changed."""

    lambda_value: float
    scaled_terms: list[str]
    skipped_startup_terms: list[str]
    skipped_unknown_params: list[str]
    material_terms_resampled: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_value": self.lambda_value,
            "scaled_terms": self.scaled_terms,
            "skipped_startup_terms": self.skipped_startup_terms,
            "skipped_unknown_params": self.skipped_unknown_params,
            "material_terms_resampled": self.material_terms_resampled,
            "num_scaled": len(self.scaled_terms),
        }


#: How far past the configured envelope an *evaluation* may extrapolate. The
#: curriculum itself is still hard-capped at 1: training beyond the envelope
#: the policy is later scored on would make the training distribution
#: unfalsifiable. Evaluation has the opposite need -- a deployment claim is
#: about conditions the policy was never trained on -- so the evaluator, and
#: only the evaluator, may ask for more.
MAX_EXTRAPOLATION = 4.0


def scale_range(
    baseline: Iterable[float],
    lambda_value: float,
    nominal: float | None = None,
    allow_extrapolation: bool = False,
) -> list[float]:
    """Shrink a ``[lo, hi]`` range toward its nominal by ``lambda``.

    ``lambda = 1`` returns the baseline unchanged -- exactly, not approximately,
    so a curriculum that reaches full intensity is indistinguishable from fixed
    randomization at the configured maximum.

    With ``allow_extrapolation`` the same affine map continues past 1, widening
    the range about its nominal. That is how a policy is scored on physics
    *outside* anything it trained under, which is the only honest sim-side
    proxy for a deployment the randomization envelope did not anticipate.
    """
    values = [float(v) for v in baseline]
    if len(values) != 2:
        raise ValueError(f"expected a [lo, hi] range, got {values}")
    low, high = values
    if high < low:
        raise ValueError(f"range is inverted: [{low}, {high}]")
    ceiling = MAX_EXTRAPOLATION if allow_extrapolation else 1.0
    if not 0.0 <= lambda_value <= ceiling:
        raise ValueError(f"lambda must be in [0, {ceiling}], got {lambda_value}")

    centre = 0.5 * (low + high) if nominal is None else float(nominal)
    return [
        centre - lambda_value * (centre - low),
        centre + lambda_value * (high - centre),
    ]


def scale_params(
    baseline: Any,
    lambda_value: float,
    nominal: float | None,
    allow_extrapolation: bool = False,
) -> Any:
    """Scale a range, or a dict of named ranges, leaving anything else alone."""
    if isinstance(baseline, dict):
        return {
            k: scale_params(v, lambda_value, nominal, allow_extrapolation)
            for k, v in baseline.items()
        }
    if (
        isinstance(baseline, (list, tuple))
        and len(baseline) == 2
        and all(isinstance(v, (int, float)) for v in baseline)
    ):
        return scale_range(baseline, lambda_value, nominal, allow_extrapolation)
    return baseline


def scaled_term_params(baseline_term: dict[str, Any], lambda_value: float) -> dict[str, Any]:
    """Every range of one term, scaled to ``lambda``, without touching the term.

    :func:`apply_lambda` writes the scaled ranges back onto the live event
    config, which is what a single-intensity curriculum wants. A stratified
    curriculum needs the same arithmetic for several intensities at once, to
    hand to the per-stratum samplers, so the pure computation lives here.
    """
    return {
        key: scale_params(original, lambda_value, RANGE_NOMINALS[key])
        for key, original in baseline_term.items()
        if key in RANGE_NOMINALS
    }


def scalable_terms(event_manager: Any) -> list[str]:
    """Names of event terms whose ranges a runtime curriculum can actually move."""
    names = []
    for name, cfg in _iter_terms(event_manager):
        if getattr(cfg, "mode", None) in RUNTIME_MODES:
            names.append(name)
    return sorted(names)


def apply_lambda(
    event_manager: Any,
    baseline: dict[str, dict[str, Any]],
    lambda_value: float,
    include_startup: bool = False,
    exclude_terms: Iterable[str] = (),
    allow_extrapolation: bool = False,
) -> ScalingReport:
    """Rescale every runtime-scalable range to ``lambda`` of its baseline.

    Args:
        baseline: the *original* parameters, captured once by
            :func:`capture_baseline`. Scaling is always computed from the
            baseline, never from the current values -- compounding
            ``lambda`` epoch after epoch would drive every range to zero.
    """
    scaled, skipped_startup, skipped_unknown, material_resampled = [], [], [], []
    excluded = set(exclude_terms)
    for name, cfg in _iter_terms(event_manager):
        if name not in baseline or name in excluded:
            continue
        mode = getattr(cfg, "mode", None)
        if mode not in RUNTIME_MODES and not include_startup:
            skipped_startup.append(name)
            continue
        params = getattr(cfg, "params", None)
        if not isinstance(params, dict):
            continue
        touched = False
        for key, original in baseline[name].items():
            if key not in params:
                continue
            if key not in RANGE_NOMINALS:
                skipped_unknown.append(f"{name}.{key}")
                continue
            params[key] = scale_params(
                original, lambda_value, RANGE_NOMINALS[key], allow_extrapolation
            )
            touched = True
        if touched:
            scaled.append(name)
            # A material term only ever chooses among buckets sampled in its
            # __init__, so rewriting its range parameters changes nothing until
            # the buckets themselves are resampled.
            if any(key in params for key in MATERIAL_RANGE_KEYS):
                resampled = _resample_material(cfg, params)
                if resampled:
                    material_resampled.append(name)
    return ScalingReport(
        lambda_value=lambda_value,
        scaled_terms=sorted(scaled),
        skipped_startup_terms=sorted(skipped_startup),
        skipped_unknown_params=sorted(set(skipped_unknown)),
        material_terms_resampled=sorted(material_resampled),
    )


def capture_baseline(event_manager: Any) -> dict[str, dict[str, Any]]:
    """Snapshot every term's range parameters before any scaling is applied."""
    baseline: dict[str, dict[str, Any]] = {}
    for name, cfg in _iter_terms(event_manager):
        params = getattr(cfg, "params", None)
        if not isinstance(params, dict):
            continue
        captured = {
            key: _deep_copy(value) for key, value in params.items() if key in RANGE_NOMINALS
        }
        if captured:
            baseline[name] = captured
    return baseline


def _resample_material(cfg: Any, params: dict[str, Any]) -> bool:
    """Rebuild a material term's buckets from its freshly scaled ranges.

    IsaacLab instantiates class-based event terms and stores the instance back
    on ``cfg.func``, so that is where the live buckets live.
    """
    from gear_sonic.research.practice_utility.events_reset_safe import (
        resample_material_buckets,
    )

    term = getattr(cfg, "func", None)
    if term is None:
        return False
    return resample_material_buckets(
        term,
        static_friction_range=_as_pair(params.get("static_friction_range")),
        dynamic_friction_range=_as_pair(params.get("dynamic_friction_range")),
        restitution_range=_as_pair(params.get("restitution_range")),
    )


def _as_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return None


def _iter_terms(event_manager: Any):
    """Yield ``(name, cfg)`` for every event term, across manager layouts."""
    if event_manager is None:
        return
    names = getattr(event_manager, "active_terms", None)
    configs = getattr(event_manager, "_term_cfgs", None)
    if names is not None and configs is not None:
        flat_names = names if isinstance(names, list) else []
        if flat_names and isinstance(flat_names[0], list):
            flat_names = [n for group in flat_names for n in group]
        for name, cfg in zip(flat_names, configs):
            yield name, cfg
        return
    for attribute in ("cfg", "_cfg"):
        cfg = getattr(event_manager, attribute, None)
        if cfg is None:
            continue
        for name in dir(cfg):
            if name.startswith("_"):
                continue
            term = getattr(cfg, name, None)
            if hasattr(term, "params") and hasattr(term, "mode"):
                yield name, term
        return


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_copy(v) for v in value]
    return value
