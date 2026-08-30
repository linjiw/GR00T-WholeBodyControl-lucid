"""Collects the termination margin from every environment, every step.

Wraps ``env.step`` the way :class:`PracticeObserverCallback` does, but reads
the whole population rather than one environment, and reads *before* the step
so the margin belongs to the state the policy acted on. After the step it sees
which episodes ended and closes their prefix means.

Once per PPO iteration it reduces the ended episodes to a median per cohort --
focus and yardstick, as assigned by the curriculum -- and updates the ratio
``R = q_focus / q_yardstick`` that the controller consumes. Everything it saw
goes to a jsonl beside the curriculum's, so the signal can be validated after
the fact against outcomes it never looked at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.research.practice_utility import margin_signal as MS

_ACTIVE: dict[str, "MarginObserverCallback"] = {}


def register_margin_observer(observer: "MarginObserverCallback") -> None:
    _ACTIVE[observer.branch_id] = observer


def get_active_margin_observer(branch_id: str | None = None) -> "MarginObserverCallback | None":
    if branch_id is not None:
        return _ACTIVE.get(branch_id)
    if len(_ACTIVE) == 1:
        return next(iter(_ACTIVE.values()))
    return None


def clear_margin_observers() -> None:
    _ACTIVE.clear()


try:  # pragma: no cover
    from transformers import TrainerCallback
except Exception:  # pragma: no cover

    class TrainerCallback:  # type: ignore[no-redef]
        """Stand-in so this module imports without transformers."""


def _inner_env(env: Any) -> Any:
    """The IsaacLab env that owns the managers, through whichever wrapper."""
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        if candidate is not None and hasattr(candidate, "termination_manager"):
            return candidate
    return None


class MarginObserverCallback(TrainerCallback):
    """Population-wide termination margin, reduced per cohort each iteration."""

    def __init__(
        self,
        enabled: bool = False,
        branch_id: str = "unbound",
        output_dir: str | None = None,
        command_name: str = "motion",
        horizon: int = 12,
        tau: float = 20.0,
        band_lo: float = 1.10,
        band_hi: float = 1.30,
    ) -> None:
        self.enabled = bool(enabled)
        self.branch_id = branch_id
        self.output_dir = output_dir
        self.command_name = command_name
        self.horizon = int(horizon)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        if not 0.0 < self.band_lo <= self.band_hi:
            raise ValueError(f"need 0 < band_lo <= band_hi, got {band_lo}, {band_hi}")
        self.ratio = MS.MarginRatio(tau=float(tau))
        self.thresholds: MS.Thresholds | None = None
        self.accumulator: MS.PrefixAccumulator | None = None
        self.focus_mask: torch.Tensor | None = None
        self.yardstick_mask: torch.Tensor | None = None
        self.history: list[dict[str, Any]] = []
        self._env: Any = None
        self._inner: Any = None
        self._original_step: Any = None
        self._ended: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._steps = 0
        self._errors: set[str] = set()
        self._flushed_step: int | None = None

    # ------------------------------------------------------------ lifecycle --

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.enabled:
            self._install(kwargs.get("env"))
            register_margin_observer(self)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if self._env is None:
            self._install(kwargs.get("env"))
        step = getattr(state, "global_step", 0) if state is not None else 0
        self.ensure_flushed(step)
        return control

    def ensure_flushed(self, step: int) -> None:
        """Reduce this iteration exactly once, whoever asks first.

        Callback order is dict order in the Hydra config. The curriculum may
        run before or after this observer; either way the iteration's episodes
        must be reduced once, before the controller reads them.
        """
        if self._flushed_step == step:
            return
        self._flushed_step = step
        self.history.append(self._flush(step))

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        self.uninstall()
        return control

    def set_cohorts(self, focus_mask: torch.Tensor, yardstick_mask: torch.Tensor) -> None:
        """Told by the curriculum which environments are which."""
        if bool((focus_mask & yardstick_mask).any()):
            raise ValueError("an environment cannot be both focus and yardstick")
        self.focus_mask = focus_mask.detach().cpu().bool()
        self.yardstick_mask = yardstick_mask.detach().cpu().bool()

    # -------------------------------------------------------------- install --

    def _install(self, env: Any) -> None:
        if env is None or not hasattr(env, "step"):
            return
        inner = _inner_env(env)
        if inner is None:
            self._errors.add("no_inner_env")
            return
        self._env, self._inner = env, inner
        self.thresholds = MS.read_thresholds(inner.termination_manager)
        num_envs = int(inner.episode_length_buf.numel())
        device = inner.episode_length_buf.device
        self.accumulator = MS.PrefixAccumulator(num_envs=num_envs, horizon=self.horizon, device=device)
        self._original_step = env.step
        observer = self

        def step_and_observe(actions, *a, **k):
            observer._before(inner)
            result = observer._original_step(actions, *a, **k)
            observer._after(inner, result)
            return result

        env.step = step_and_observe

    def uninstall(self) -> None:
        if self._env is not None and self._original_step is not None:
            self._env.__dict__.pop("step", None)
        self._env = self._inner = self._original_step = None

    # ----------------------------------------------------------- per step --

    def _before(self, inner: Any) -> None:
        try:
            command = inner.command_manager.get_term(self.command_name)
            margins = MS.termination_margins(command, self.thresholds)
            age = inner.episode_length_buf
            self.accumulator.push(margins, age)
            self._steps += 1
        except Exception as error:  # telemetry must never kill a branch
            self._errors.add(f"before:{type(error).__name__}")

    def _after(self, inner: Any, result: Any) -> None:
        try:
            dones = result[2] if isinstance(result, (tuple, list)) and len(result) >= 3 else None
            if dones is None:
                self._errors.add("no_dones")
                return
            ended = torch.as_tensor(dones).bool()
            extras = result[3] if len(result) >= 4 and isinstance(result[3], dict) else {}
            time_outs = extras.get("time_outs")
            timed = torch.as_tensor(time_outs).bool() if time_outs is not None else torch.zeros_like(ended)
            ids, means, counts, shares = self.accumulator.finish(ended)
            if ids.numel():
                self._ended.append((ids, means, counts, shares, timed[ids]))
        except Exception as error:
            self._errors.add(f"after:{type(error).__name__}")

    # -------------------------------------------------------- per iteration --

    def _flush(self, step: int) -> dict[str, Any]:
        if self._ended:
            ids = torch.cat([e[0] for e in self._ended])
            means = torch.cat([e[1] for e in self._ended])
            counts = torch.cat([e[2] for e in self._ended])
            shares = torch.cat([e[3] for e in self._ended])
            timed = torch.cat([e[4] for e in self._ended])
        else:
            ids = torch.zeros(0, dtype=torch.long); means = torch.zeros(0); counts = torch.zeros(0)
            shares = torch.zeros(0, len(MS.CULPRITS)); timed = torch.zeros(0, dtype=torch.bool)
        self._ended = []

        q_focus = q_yard = None
        cov_focus = cov_yard = None
        if self.focus_mask is not None and ids.numel():
            q_focus = MS.cohort_median(means, ids, self.focus_mask)
            sel = self.focus_mask[ids.cpu()]
            cov_focus = float((counts[sel] < self.horizon).float().mean()) if sel.any() else None
        if self.yardstick_mask is not None and ids.numel():
            q_yard = MS.cohort_median(means, ids, self.yardstick_mask)
            sel = self.yardstick_mask[ids.cpu()]
            cov_yard = float((counts[sel] < self.horizon).float().mean()) if sel.any() else None
        ratio = self.ratio.update(q_focus, q_yard)

        record = {
            "global_step": step,
            "episodes_ended": int(ids.numel()),
            "steps_observed": self._steps,
            "q_focus": q_focus,
            "q_yardstick": q_yard,
            "ratio": ratio,
            "band": [self.band_lo, self.band_hi],
            "band_error": MS.band_error(ratio, self.band_lo, self.band_hi),
            "coverage_short_of_horizon": {"focus": cov_focus, "yardstick": cov_yard},
            "timeout_fraction": float(timed.float().mean()) if timed.numel() else None,
            "culprit_share": (
                dict(zip(MS.CULPRITS, [float(v) for v in shares.mean(dim=0)])) if shares.numel() else None
            ),
            "population_median": float(torch.quantile(means, 0.5)) if means.numel() else None,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "errors": sorted(self._errors),
        }
        self._steps = 0
        self._write(record)
        return record

    # -------------------------------------------------------------- consume --

    def current_error(self) -> float:
        """The band error the controller should act on this iteration."""
        if not self.history:
            return 0.0
        return float(self.history[-1]["band_error"])

    def current_ratio(self) -> float | None:
        return self.history[-1]["ratio"] if self.history else None

    def _write(self, record: dict[str, Any]) -> None:
        if not self.output_dir:
            return
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"margin_{self.branch_id}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
