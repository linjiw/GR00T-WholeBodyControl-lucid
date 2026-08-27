"""Target-Anchored Curriculum Exposure (TACE) for the LUCID DR curriculum.

The measured deficit of the gap-gated curriculum is a *support* deficit: LUCID
ended its budget at lambda = 0.756 and never trained on the full randomization
envelope it is later evaluated on, so it trails fixed full-range DR on that
envelope while beating it on nominal physics. The literature on curricula that
actually beat mixed training agrees on the fix -- keep the target distribution
inside the training mixture at all times -- and TACE is the smallest version of
that fix for SONIC.

Mechanism
---------
A fixed, seeded fraction ``alpha`` of the parallel environments is tagged
``anchor``; the rest are ``focus``. At every reset (and every interval firing)
an anchor environment samples its randomization from the *baseline* ranges
(lambda = 1, the evaluation envelope) while a focus environment samples from the
curriculum's current lambda-scaled ranges. Nothing else changes: PPO still sees
one on-policy rollout buffer, the policy observation carries no cohort tag, and
the curriculum controller still reads its gap from a focus environment only.

How it is installed
-------------------
IsaacLab's event manager calls ``term_cfg.func(env, env_ids, **term_cfg.params)``.
LUCID's curriculum already rewrites ``term_cfg.params`` to the lambda-scaled
ranges. TACE wraps ``term_cfg.func`` with :class:`CohortDispatch`, which splits
``env_ids`` by cohort and calls the *original* sampler twice: once for the focus
subset with the (lambda-scaled) params it was handed, once for the anchor subset
with the captured baseline params. The samplers themselves are untouched, so a
run with ``alpha = 0`` is the ordinary curriculum and ``alpha = 1`` is fixed DR.

The one class-based term, IsaacLab's material randomizer, draws from a bucket
tensor built once at construction. The curriculum resamples those buckets on
every lambda change; for the anchor subset the dispatcher swaps in a copy of the
buckets taken *before* any scaling -- i.e. the configured full-range buckets --
for the duration of the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from gear_sonic.research.practice_utility import dr_scaling as DS

ANCHOR = "anchor"
FOCUS = "focus"


@dataclass(frozen=True)
class CohortAssignment:
    """Fixed anchor/focus partition of the parallel environments."""

    num_envs: int
    anchor_ratio: float
    seed: int
    anchor_ids: tuple[int, ...]
    reserved_focus_ids: tuple[int, ...]

    @property
    def num_anchor(self) -> int:
        return len(self.anchor_ids)

    @property
    def num_focus(self) -> int:
        return self.num_envs - self.num_anchor

    def mask(self, device: Any = None) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.anchor_ids:
            mask[list(self.anchor_ids)] = True
        return mask.to(device) if device is not None else mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "anchor_ratio": self.anchor_ratio,
            "seed": self.seed,
            "num_anchor": self.num_anchor,
            "num_focus": self.num_focus,
            "reserved_focus_ids": list(self.reserved_focus_ids),
            "anchor_ids": list(self.anchor_ids),
        }


def assign_cohorts(
    num_envs: int,
    anchor_ratio: float,
    seed: int,
    reserved_focus_ids: tuple[int, ...] | list[int] = (),
) -> CohortAssignment:
    """Draw a seeded permutation and tag exactly ``round(alpha * N)`` anchors.

    ``reserved_focus_ids`` are never anchors: the observer measures the gap on
    one tracked environment and the controller must only ever see focus-cohort
    evidence, otherwise the anchor's deliberately out-of-frontier samples pull
    lambda down for doing exactly what they were meant to do.
    """
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if not 0.0 <= anchor_ratio <= 1.0:
        raise ValueError(f"anchor_ratio must be in [0, 1], got {anchor_ratio}")
    reserved = tuple(sorted({int(i) for i in reserved_focus_ids if 0 <= int(i) < num_envs}))
    num_anchor = int(round(anchor_ratio * num_envs))
    num_anchor = min(num_anchor, num_envs - len(reserved))
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(num_envs, generator=generator).tolist()
    reserved_set = set(reserved)
    candidates = [i for i in permutation if i not in reserved_set]
    anchors = tuple(sorted(candidates[:num_anchor]))
    return CohortAssignment(
        num_envs=num_envs,
        anchor_ratio=float(anchor_ratio),
        seed=int(seed),
        anchor_ids=anchors,
        reserved_focus_ids=reserved,
    )


class CohortDispatch:
    """Route each event call to the baseline or the curriculum sampler by cohort.

    Wraps one event term's ``func``. Attribute access falls through to the
    wrapped term, so code that reaches for ``cfg.func.material_buckets`` (the
    curriculum's bucket resampler) keeps working unchanged.
    """

    def __init__(
        self,
        inner: Callable[..., Any],
        term_name: str,
        anchor_params: dict[str, Any],
        anchor_mask: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "term_name", term_name)
        object.__setattr__(self, "anchor_params", DS._deep_copy(anchor_params))
        object.__setattr__(self, "_anchor_mask", anchor_mask.detach().cpu().bool())
        object.__setattr__(self, "_anchor_buckets", _clone_buckets(inner))
        object.__setattr__(self, "calls", {ANCHOR: 0, FOCUS: 0})
        object.__setattr__(self, "env_counts", {ANCHOR: 0, FOCUS: 0})
        object.__setattr__(self, "all_envs_mode", False)

    # --------------------------------------------------------------- proxy --

    @property
    def inner(self) -> Callable[..., Any]:
        return object.__getattribute__(self, "_inner")

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the dispatcher itself.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("all_envs_mode",):
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_inner"), name, value)

    # ---------------------------------------------------------------- call --

    def __call__(self, env: Any, env_ids: Any, **params: Any) -> Any:
        ids = _normalize_env_ids(env, env_ids, self._anchor_mask.numel())
        if self.all_envs_mode:
            # Consolidation: every environment trains on the target envelope.
            anchor_ids, focus_ids = ids, ids[:0]
        else:
            mask = self._anchor_mask[ids.cpu()]
            anchor_ids = ids[mask.to(ids.device)]
            focus_ids = ids[(~mask).to(ids.device)]

        results = []
        if focus_ids.numel() > 0:
            self.calls[FOCUS] += 1
            self.env_counts[FOCUS] += int(focus_ids.numel())
            results.append(self.inner(env, focus_ids, **params))
        if anchor_ids.numel() > 0:
            self.calls[ANCHOR] += 1
            self.env_counts[ANCHOR] += int(anchor_ids.numel())
            merged = dict(params)
            merged.update(self.anchor_params)
            results.append(self._call_anchor(env, anchor_ids, merged))
        return _merge_results(results)

    def _call_anchor(self, env: Any, env_ids: torch.Tensor, params: dict[str, Any]) -> Any:
        inner = self.inner
        buckets = self._anchor_buckets
        if buckets is None:
            return inner(env, env_ids, **params)
        live = getattr(inner, "material_buckets", None)
        inner.material_buckets = buckets
        try:
            return inner(env, env_ids, **params)
        finally:
            inner.material_buckets = live

    def telemetry(self) -> dict[str, Any]:
        return {
            "term": self.term_name,
            "calls": dict(self.calls),
            "env_counts": dict(self.env_counts),
            "anchor_params": DS._deep_copy(self.anchor_params),
        }


def install(
    event_manager: Any,
    baseline: dict[str, dict[str, Any]],
    assignment: CohortAssignment,
) -> dict[str, CohortDispatch]:
    """Wrap every runtime-scalable term in ``baseline`` with a dispatcher.

    Idempotent: a term already dispatched is left alone.
    """
    mask = assignment.mask()
    installed: dict[str, CohortDispatch] = {}
    for name, cfg in DS._iter_terms(event_manager):
        if name not in baseline:
            continue
        if getattr(cfg, "mode", None) not in DS.RUNTIME_MODES:
            continue
        func = getattr(cfg, "func", None)
        if func is None:
            continue
        if isinstance(func, CohortDispatch):
            installed[name] = func
            continue
        dispatch = CohortDispatch(func, name, baseline[name], mask)
        cfg.func = dispatch
        installed[name] = dispatch
    return installed


def uninstall(event_manager: Any) -> list[str]:
    restored = []
    for name, cfg in DS._iter_terms(event_manager):
        func = getattr(cfg, "func", None)
        if isinstance(func, CohortDispatch):
            cfg.func = func.inner
            restored.append(name)
    return sorted(restored)


def cohort_delay_stats(asset: Any, anchor_mask: torch.Tensor) -> dict[str, Any]:
    """Mean installed actuator lag per cohort -- realized latency dose."""
    actuators = getattr(asset, "actuators", None) or {}
    lags = []
    for actuator in actuators.values():
        buffer = getattr(actuator, "positions_delay_buffer", None)
        tensor = getattr(buffer, "time_lags", None)
        if isinstance(tensor, torch.Tensor):
            lags.append(tensor.detach().long().flatten().cpu())
    if not lags:
        return {}
    stacked = torch.stack(lags).float()  # groups x envs
    mask = anchor_mask.detach().cpu().bool()
    if mask.numel() != stacked.shape[1]:
        return {"cohort_delay_mask_mismatch": [int(mask.numel()), int(stacked.shape[1])]}
    out: dict[str, Any] = {}
    for label, select in ((ANCHOR, mask), (FOCUS, ~mask)):
        if select.any():
            subset = stacked[:, select]
            out[f"{label}_delay_mean_steps"] = float(subset.mean().item())
            out[f"{label}_delay_max_steps"] = int(subset.max().item())
            out[f"{label}_delay_nonzero_fraction"] = float((subset > 0).float().mean().item())
    return out


# ------------------------------------------------------------------ helpers --


def _normalize_env_ids(env: Any, env_ids: Any, num_envs: int) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        scene = getattr(env, "scene", None)
        count = getattr(scene, "num_envs", None) or num_envs
        return torch.arange(int(count))
    if isinstance(env_ids, torch.Tensor):
        return env_ids.long()
    return torch.as_tensor(list(env_ids), dtype=torch.long)


def _clone_buckets(term: Any) -> torch.Tensor | None:
    buckets = getattr(term, "material_buckets", None)
    return buckets.clone() if isinstance(buckets, torch.Tensor) else None


def _merge_results(results: list[Any]) -> Any:
    if not results:
        return None
    if all(isinstance(r, (int, float)) and not isinstance(r, bool) for r in results):
        return sum(results)
    return results[0]
