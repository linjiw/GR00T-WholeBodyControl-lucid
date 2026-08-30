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
YARDSTICK = "yardstick"


@dataclass(frozen=True)
class CohortAssignment:
    """Fixed anchor/focus partition of the parallel environments.

    ``focus_strata`` sub-divides the focus cohort for the support-expanding
    variant (LUCID-S): stratum ``k`` of ``K`` trains at ``lambda * (k+1)/K``, so
    the training mixture spans ``(0, lambda]`` instead of sitting at the single
    point ``lambda``. With ``K = 1`` the tuple holds one stratum containing
    every focus environment and the behaviour is exactly plain TACE.
    """

    num_envs: int
    anchor_ratio: float
    seed: int
    anchor_ids: tuple[int, ...]
    reserved_focus_ids: tuple[int, ...]
    focus_strata: tuple[tuple[int, ...], ...] = ()
    #: Environments held at lambda = 0 as the self-reference for the margin
    #: signal. Neither anchor nor focus; excluded from every stratum.
    yardstick_ids: tuple[int, ...] = ()

    @property
    def num_anchor(self) -> int:
        return len(self.anchor_ids)

    @property
    def num_yardstick(self) -> int:
        return len(self.yardstick_ids)

    @property
    def num_focus(self) -> int:
        return self.num_envs - self.num_anchor - self.num_yardstick

    def yardstick_mask(self, device: Any = None) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.yardstick_ids:
            mask[list(self.yardstick_ids)] = True
        return mask.to(device) if device is not None else mask

    def focus_mask(self, device: Any = None) -> torch.Tensor:
        mask = ~(self.mask() | self.yardstick_mask())
        return mask.to(device) if device is not None else mask

    @property
    def num_strata(self) -> int:
        return max(1, len(self.focus_strata))

    def stratum_masks(self) -> tuple[torch.Tensor, ...]:
        """One boolean mask per focus stratum, ordered low to high intensity."""
        if not self.focus_strata:
            return ()
        masks = []
        for ids in self.focus_strata:
            mask = torch.zeros(self.num_envs, dtype=torch.bool)
            if ids:
                mask[list(ids)] = True
            masks.append(mask)
        return tuple(masks)

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
            "num_yardstick": self.num_yardstick,
            "yardstick_ids": list(self.yardstick_ids),
            "num_strata": self.num_strata,
            "stratum_sizes": [len(ids) for ids in self.focus_strata],
            "stratum_weights": list(stratum_weights(self.num_strata)),
            "reserved_focus_ids": list(self.reserved_focus_ids),
            "anchor_ids": list(self.anchor_ids),
        }


def stratum_weights(num_strata: int) -> tuple[float, ...]:
    """Fractions of the controller's lambda, one per focus stratum.

    The top stratum is always at ``1.0`` -- it *is* the curriculum frontier the
    controller believes it has reached, so LUCID-S never trains below the
    scalar curriculum, it only adds easier company underneath it.
    """
    if num_strata < 1:
        raise ValueError(f"num_strata must be >= 1, got {num_strata}")
    return tuple((k + 1) / num_strata for k in range(num_strata))


def assign_cohorts(
    num_envs: int,
    anchor_ratio: float,
    seed: int,
    reserved_focus_ids: tuple[int, ...] | list[int] = (),
    num_strata: int = 1,
    num_yardstick: int = 0,
) -> CohortAssignment:
    """Draw a seeded permutation and tag exactly ``round(alpha * N)`` anchors.

    ``reserved_focus_ids`` are never anchors: the observer measures the gap on
    one tracked environment and the controller must only ever see focus-cohort
    evidence, otherwise the anchor's deliberately out-of-frontier samples pull
    lambda down for doing exactly what they were meant to do. For the same
    reason they are placed in the **top** focus stratum: the controller must
    read the frontier it is deciding whether to expand, not the easier company
    training underneath it.
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
    if num_strata < 1:
        raise ValueError(f"num_strata must be >= 1, got {num_strata}")
    if num_yardstick < 0 or num_anchor + num_yardstick + len(reserved) > num_envs:
        raise ValueError("anchor + yardstick + reserved exceed the environment count")
    # Yardstick envs come next in the same seeded permutation, so they are a
    # fixed random subset, never the reserved (observer) envs, never anchors.
    yardstick = tuple(sorted(candidates[num_anchor : num_anchor + num_yardstick]))
    anchor_set = set(anchors) | set(yardstick)
    focus_pool = [i for i in permutation if i not in anchor_set]
    if num_strata == 1:
        strata: tuple[tuple[int, ...], ...] = (tuple(sorted(focus_pool)),)
    else:
        groups: list[list[int]] = [[] for _ in range(num_strata)]
        # Round-robin over the seeded permutation: near-equal sizes, no bias
        # toward any environment index, and deterministic given the seed.
        for position, env_id in enumerate(i for i in focus_pool if i not in reserved_set):
            groups[position % num_strata].append(env_id)
        groups[-1].extend(reserved)
        strata = tuple(tuple(sorted(group)) for group in groups)
    return CohortAssignment(
        num_envs=num_envs,
        anchor_ratio=float(anchor_ratio),
        seed=int(seed),
        anchor_ids=anchors,
        reserved_focus_ids=reserved,
        focus_strata=strata,
        yardstick_ids=yardstick,
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
        stratum_masks: tuple[torch.Tensor, ...] = (),
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "term_name", term_name)
        object.__setattr__(self, "anchor_params", DS._deep_copy(anchor_params))
        object.__setattr__(self, "_anchor_mask", anchor_mask.detach().cpu().bool())
        object.__setattr__(self, "_anchor_buckets", _clone_buckets(inner))
        # Sub-strata of the focus cohort, low intensity first. The *top*
        # stratum is deliberately not given its own parameters: it is served
        # the params the event manager already holds, which the curriculum has
        # scaled to lambda. So K = 1 is byte-for-byte the un-stratified path.
        masks = tuple(m.detach().cpu().bool() for m in stratum_masks)
        object.__setattr__(self, "_stratum_masks", masks if len(masks) > 1 else ())
        object.__setattr__(self, "_stratum_params", [None] * len(masks))
        object.__setattr__(self, "_stratum_buckets", [None] * len(masks))
        object.__setattr__(self, "calls", {ANCHOR: 0, FOCUS: 0})
        object.__setattr__(self, "env_counts", {ANCHOR: 0, FOCUS: 0})
        object.__setattr__(self, "all_envs_mode", False)
        object.__setattr__(self, "_yardstick_mask", None)
        object.__setattr__(self, "_yardstick_params", None)
        object.__setattr__(self, "_yardstick_buckets", None)

    def set_yardstick(
        self, mask: torch.Tensor, params: dict[str, Any], buckets: torch.Tensor | None = None
    ) -> None:
        """Hold these environments at the given (lambda = 0) parameters."""
        object.__setattr__(self, "_yardstick_mask", mask.detach().cpu().bool())
        object.__setattr__(self, "_yardstick_params", DS._deep_copy(params))
        object.__setattr__(
            self, "_yardstick_buckets", buckets.clone() if isinstance(buckets, torch.Tensor) else None
        )

    # ------------------------------------------------------------- strata --

    @property
    def num_strata(self) -> int:
        return len(object.__getattribute__(self, "_stratum_masks")) or 1

    def set_stratum(
        self,
        index: int,
        params: dict[str, Any] | None,
        buckets: torch.Tensor | None = None,
    ) -> None:
        """Install the sampling parameters for one sub-stratum.

        Called by the curriculum after every lambda change. The top stratum
        (``index == num_strata - 1``) must be left at ``None``: it uses the
        event manager's own params, which is what keeps an un-stratified run
        identical to the code before strata existed.
        """
        masks = object.__getattribute__(self, "_stratum_masks")
        if not masks:
            return
        if not 0 <= index < len(masks):
            raise IndexError(f"stratum {index} out of range for {len(masks)} strata")
        object.__getattribute__(self, "_stratum_params")[index] = (
            DS._deep_copy(params) if params is not None else None
        )
        object.__getattribute__(self, "_stratum_buckets")[index] = (
            buckets.clone() if isinstance(buckets, torch.Tensor) else None
        )

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
        ymask = object.__getattribute__(self, "_yardstick_mask")
        if ymask is not None and not self.all_envs_mode:
            in_yard = ymask[focus_ids.cpu()].to(focus_ids.device)
            yard_ids, focus_ids = focus_ids[in_yard], focus_ids[~in_yard]
            if yard_ids.numel() > 0:
                self.calls[YARDSTICK] = self.calls.get(YARDSTICK, 0) + 1
                self.env_counts[YARDSTICK] = self.env_counts.get(YARDSTICK, 0) + int(yard_ids.numel())
                merged = dict(params)
                merged.update(object.__getattribute__(self, "_yardstick_params"))
                results.append(
                    self._call_with_buckets(
                        env, yard_ids, merged, object.__getattribute__(self, "_yardstick_buckets")
                    )
                )
        if focus_ids.numel() > 0:
            self.calls[FOCUS] += 1
            self.env_counts[FOCUS] += int(focus_ids.numel())
            results.extend(self._call_focus(env, focus_ids, params))
        if anchor_ids.numel() > 0:
            self.calls[ANCHOR] += 1
            self.env_counts[ANCHOR] += int(anchor_ids.numel())
            merged = dict(params)
            merged.update(self.anchor_params)
            results.append(self._call_anchor(env, anchor_ids, merged))
        return _merge_results(results)

    def _call_focus(self, env: Any, env_ids: torch.Tensor, params: dict[str, Any]) -> list[Any]:
        """Sample the focus environments, split by intensity stratum."""
        masks = object.__getattribute__(self, "_stratum_masks")
        if not masks:
            return [self.inner(env, env_ids, **params)]
        stratum_params = object.__getattribute__(self, "_stratum_params")
        stratum_buckets = object.__getattribute__(self, "_stratum_buckets")
        results = []
        for index, mask in enumerate(masks):
            selected = env_ids[mask[env_ids.cpu()].to(env_ids.device)]
            if selected.numel() == 0:
                continue
            key = f"{FOCUS}_s{index}"
            self.calls[key] = self.calls.get(key, 0) + 1
            self.env_counts[key] = self.env_counts.get(key, 0) + int(selected.numel())
            override = stratum_params[index]
            if override is None:
                results.append(self.inner(env, selected, **params))
                continue
            merged = dict(params)
            merged.update(override)
            results.append(self._call_with_buckets(env, selected, merged, stratum_buckets[index]))
        return results

    def _call_with_buckets(
        self,
        env: Any,
        env_ids: torch.Tensor,
        params: dict[str, Any],
        buckets: torch.Tensor | None,
    ) -> Any:
        """Run the wrapped sampler with a substituted material bucket tensor."""
        inner = self.inner
        if buckets is None:
            return inner(env, env_ids, **params)
        live = getattr(inner, "material_buckets", None)
        inner.material_buckets = buckets
        try:
            return inner(env, env_ids, **params)
        finally:
            inner.material_buckets = live

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
        out = {
            "term": self.term_name,
            "calls": dict(self.calls),
            "env_counts": dict(self.env_counts),
            "anchor_params": DS._deep_copy(self.anchor_params),
        }
        if object.__getattribute__(self, "_stratum_masks"):
            out["num_strata"] = self.num_strata
            out["stratum_params"] = [
                DS._deep_copy(p) for p in object.__getattribute__(self, "_stratum_params")
            ]
        return out


def install(
    event_manager: Any,
    baseline: dict[str, dict[str, Any]],
    assignment: CohortAssignment,
    anchor_params: dict[str, dict[str, Any]] | None = None,
    anchor_buckets: dict[str, torch.Tensor] | None = None,
) -> dict[str, CohortDispatch]:
    """Wrap every runtime-scalable term in ``baseline`` with a dispatcher.

    The anchor cohort trains on **the arm's own target envelope**, which is not
    always the captured baseline. An arm that pins or caps a channel has a
    narrower target than the config declares, and an anchor drawing from the
    config would hand half the population the very exposure the arm exists to
    withhold -- silently, since nothing else about the run would look wrong.
    ``anchor_params`` and ``anchor_buckets`` name that target per term; a term
    absent from them keeps the baseline, which is the un-restricted case.

    Idempotent: a term already dispatched is left alone.
    """
    mask = assignment.mask()
    stratum_masks = assignment.stratum_masks()
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
        target = (anchor_params or {}).get(name, baseline[name])
        dispatch = CohortDispatch(func, name, target, mask, stratum_masks)
        override_buckets = (anchor_buckets or {}).get(name)
        if override_buckets is not None:
            object.__setattr__(dispatch, "_anchor_buckets", override_buckets.clone())
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


def cohort_delay_stats(
    asset: Any,
    anchor_mask: torch.Tensor,
    stratum_masks: tuple[torch.Tensor, ...] = (),
) -> dict[str, Any]:
    """Mean installed actuator lag per cohort -- realized latency dose.

    This is the measurement that separates "the curriculum wrote different
    ranges" from "the simulator actually installed different physics". For a
    stratified run it is also the only direct evidence that the intensity
    *mixture* exists: the per-stratum means should come out ordered, roughly in
    proportion to the stratum weights.
    """
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
    groups: list[tuple[str, torch.Tensor]] = [(ANCHOR, mask), (FOCUS, ~mask)]
    for index, stratum in enumerate(stratum_masks):
        stratum = stratum.detach().cpu().bool()
        if stratum.numel() == mask.numel():
            groups.append((f"{FOCUS}_s{index}", stratum))
    out: dict[str, Any] = {}
    for label, select in groups:
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
