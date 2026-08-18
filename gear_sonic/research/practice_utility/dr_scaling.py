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
Only some. IsaacLab event terms declare a mode:

``startup``   applied once when the scene is built. Rescaling later does nothing
              -- friction, base CoM, and joint-default offsets are in this group
              and are therefore **outside** a runtime curriculum.
``reset``     re-applied on every episode reset. Scalable. (body mass)
``interval``  re-applied periodically. Scalable. (pushes)

This is a property of the configuration, not a limitation of the controller, and
it is reported rather than worked around: :func:`scalable_terms` names exactly
which channels a run is really scheduling, so a curriculum cannot silently claim
credit for randomization it never moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

#: Event modes whose ranges take effect again after startup.
RUNTIME_MODES = ("reset", "interval")

#: Parameter names understood as randomization ranges, with their nominal value.
#: The nominal is the value at which the channel contributes no randomization.
RANGE_NOMINALS: dict[str, float] = {
    "mass_distribution_params": 1.0,       # multiplicative: 1.0 is unscaled
    "pos_distribution_params": 0.0,        # additive offsets
    "velocity_range": 0.0,
    "com_range": 0.0,
    "static_friction_range": None,         # nominal taken from the midpoint
    "dynamic_friction_range": None,
    "restitution_range": 0.0,
}


@dataclass
class ScalingReport:
    """What a scaling pass actually changed."""

    lambda_value: float
    scaled_terms: list[str]
    skipped_startup_terms: list[str]
    skipped_unknown_params: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_value": self.lambda_value,
            "scaled_terms": self.scaled_terms,
            "skipped_startup_terms": self.skipped_startup_terms,
            "skipped_unknown_params": self.skipped_unknown_params,
            "num_scaled": len(self.scaled_terms),
        }


def scale_range(
    baseline: Iterable[float], lambda_value: float, nominal: float | None = None
) -> list[float]:
    """Shrink a ``[lo, hi]`` range toward its nominal by ``lambda``.

    ``lambda = 1`` returns the baseline unchanged -- exactly, not approximately,
    so a curriculum that reaches full intensity is indistinguishable from fixed
    randomization at the configured maximum.
    """
    values = [float(v) for v in baseline]
    if len(values) != 2:
        raise ValueError(f"expected a [lo, hi] range, got {values}")
    low, high = values
    if high < low:
        raise ValueError(f"range is inverted: [{low}, {high}]")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lambda_value}")

    centre = 0.5 * (low + high) if nominal is None else float(nominal)
    return [
        centre - lambda_value * (centre - low),
        centre + lambda_value * (high - centre),
    ]


def scale_params(baseline: Any, lambda_value: float, nominal: float | None) -> Any:
    """Scale a range, or a dict of named ranges, leaving anything else alone."""
    if isinstance(baseline, dict):
        return {k: scale_params(v, lambda_value, nominal) for k, v in baseline.items()}
    if isinstance(baseline, (list, tuple)) and len(baseline) == 2 and all(
        isinstance(v, (int, float)) for v in baseline
    ):
        return scale_range(baseline, lambda_value, nominal)
    return baseline


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
) -> ScalingReport:
    """Rescale every runtime-scalable range to ``lambda`` of its baseline.

    Args:
        baseline: the *original* parameters, captured once by
            :func:`capture_baseline`. Scaling is always computed from the
            baseline, never from the current values -- compounding
            ``lambda`` epoch after epoch would drive every range to zero.
    """
    scaled, skipped_startup, skipped_unknown = [], [], []
    for name, cfg in _iter_terms(event_manager):
        if name not in baseline:
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
            params[key] = scale_params(original, lambda_value, RANGE_NOMINALS[key])
            touched = True
        if touched:
            scaled.append(name)
    return ScalingReport(
        lambda_value=lambda_value,
        scaled_terms=sorted(scaled),
        skipped_startup_terms=sorted(skipped_startup),
        skipped_unknown_params=sorted(set(skipped_unknown)),
    )


def capture_baseline(event_manager: Any) -> dict[str, dict[str, Any]]:
    """Snapshot every term's range parameters before any scaling is applied."""
    baseline: dict[str, dict[str, Any]] = {}
    for name, cfg in _iter_terms(event_manager):
        params = getattr(cfg, "params", None)
        if not isinstance(params, dict):
            continue
        captured = {
            key: _deep_copy(value)
            for key, value in params.items()
            if key in RANGE_NOMINALS
        }
        if captured:
            baseline[name] = captured
    return baseline


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
