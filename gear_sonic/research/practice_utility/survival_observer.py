"""Per-stratum episode survival, reduced once per PPO iteration.

Wraps ``env.step`` the way :mod:`margin_observer` does, but reads only what an
episode's *ending* says: whether it ran to the end of the clip (a time-out, and
therefore a survival) or was cut short by a termination condition. Each ended
episode is attributed to the intensity stratum its environment belongs to, so
the curriculum can ask a question no population-wide average can answer -- how
is the *probe* cohort doing, the one training above the current frontier.

Why the ending and not the reward
---------------------------------
Reward is unbounded, drifts in scale across a run, and rises when the
environment gets easier; the campaign that motivated this module measured it
rising as robustness fell. An episode ending is bounded, defined by an outcome
rather than by a learned statistic, and observed over the whole population
rather than one tracked environment. It is the one quantity in this trainer
that already satisfies all three admissibility requirements.

Attribution is by environment id. Strata are fixed for a run (TACE assigns
them once), so an episode that ends in environment *i* is unambiguously an
episode of whichever stratum owns *i*, with no need to track which intensity
was in force when the episode started -- every environment in a stratum has
been at that stratum's intensity for the whole episode unless the frontier
moved mid-episode, which the gate's dwell makes rare and which biases the
estimate toward the *lower* (older, easier) intensity, i.e. conservatively
against expanding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

_ACTIVE: dict[str, "SurvivalObserverCallback"] = {}


def register_survival_observer(observer: "SurvivalObserverCallback") -> None:
    _ACTIVE[observer.branch_id] = observer


def get_active_survival_observer(
    branch_id: str | None = None,
) -> "SurvivalObserverCallback | None":
    if branch_id is not None:
        return _ACTIVE.get(branch_id)
    if len(_ACTIVE) == 1:
        return next(iter(_ACTIVE.values()))
    return None


def clear_survival_observers() -> None:
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


class SurvivalObserverCallback(TrainerCallback):
    """Population survival, split by intensity stratum, once per iteration."""

    def __init__(
        self,
        enabled: bool = False,
        branch_id: str = "unbound",
        output_dir: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.branch_id = branch_id
        self.output_dir = output_dir
        self.history: list[dict[str, Any]] = []
        #: Stratum membership masks, low intensity first, set by the curriculum.
        self.stratum_masks: tuple[torch.Tensor, ...] = ()
        self.probe_index: int | None = None
        self._env: Any = None
        self._inner: Any = None
        self._original_step: Any = None
        self._ended: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._steps = 0
        self._errors: set[str] = set()
        self._flushed_step: int | None = None

    # ------------------------------------------------------------ lifecycle --

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.enabled:
            self._install(kwargs.get("env"))
            register_survival_observer(self)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if self._env is None:
            self._install(kwargs.get("env"))
        step = getattr(state, "global_step", 0) if state is not None else 0
        self.ensure_flushed(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        self.uninstall()
        return control

    def ensure_flushed(self, step: int) -> None:
        """Reduce this iteration exactly once, whoever asks first.

        Callback order is dict order in the Hydra config, so the curriculum may
        run before or after this observer. Either way the iteration's episodes
        are reduced once, before the gate reads them.
        """
        if self._flushed_step == step:
            return
        self._flushed_step = step
        self.history.append(self._flush(step))

    def set_strata(
        self,
        masks: Sequence[torch.Tensor],
        probe_index: int | None = None,
    ) -> None:
        """Told by the curriculum which environments are which stratum."""
        resolved = tuple(mask.detach().cpu().bool() for mask in masks)
        if probe_index is not None and not 0 <= probe_index < len(resolved):
            raise ValueError(f"probe_index {probe_index} outside {len(resolved)} strata")
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                if bool((resolved[i] & resolved[j]).any()):
                    raise ValueError(f"strata {i} and {j} share an environment")
        self.stratum_masks = resolved
        self.probe_index = probe_index

    # -------------------------------------------------------------- install --

    def _install(self, env: Any) -> None:
        if env is None or not hasattr(env, "step"):
            return
        inner = _inner_env(env)
        if inner is None:
            self._errors.add("no_inner_env")
            return
        self._env, self._inner = env, inner
        self._original_step = env.step
        observer = self

        def step_and_observe(actions, *a, **k):
            result = observer._original_step(actions, *a, **k)
            observer._after(result)
            return result

        env.step = step_and_observe

    def uninstall(self) -> None:
        if self._env is not None and self._original_step is not None:
            self._env.__dict__.pop("step", None)
        self._env = self._inner = self._original_step = None

    # ------------------------------------------------------------- per step --

    def _after(self, result: Any) -> None:
        try:
            self._steps += 1
            dones = result[2] if isinstance(result, (tuple, list)) and len(result) >= 3 else None
            if dones is None:
                self._errors.add("no_dones")
                return
            ended = torch.as_tensor(dones).reshape(-1).bool().cpu()
            if not bool(ended.any()):
                return
            extras = result[3] if len(result) >= 4 and isinstance(result[3], dict) else {}
            time_outs = extras.get("time_outs")
            if time_outs is None:
                self._errors.add("no_time_outs")
                timed = torch.zeros_like(ended)
            else:
                timed = torch.as_tensor(time_outs).reshape(-1).bool().cpu()
            ids = torch.nonzero(ended, as_tuple=False).reshape(-1)
            self._ended.append((ids, timed[ids]))
        except Exception as error:  # telemetry must never kill a branch
            self._errors.add(f"after:{type(error).__name__}")

    # -------------------------------------------------------- per iteration --

    def _flush(self, step: int) -> dict[str, Any]:
        if self._ended:
            ids = torch.cat([e[0] for e in self._ended])
            timed = torch.cat([e[1] for e in self._ended])
        else:
            ids = torch.zeros(0, dtype=torch.long)
            timed = torch.zeros(0, dtype=torch.bool)
        self._ended = []

        per_stratum: list[dict[str, Any]] = []
        for index, mask in enumerate(self.stratum_masks):
            if ids.numel():
                selected = mask[ids]
                episodes = int(selected.sum())
                survival = float(timed[selected].float().mean()) if episodes else None
            else:
                episodes = 0
                survival = None
            per_stratum.append({"stratum": index, "episodes": episodes, "survival": survival})

        record = {
            "global_step": step,
            "steps_observed": self._steps,
            "episodes_ended": int(ids.numel()),
            "survival": float(timed.float().mean()) if timed.numel() else None,
            "per_stratum": per_stratum,
            "probe_index": self.probe_index,
            "errors": sorted(self._errors),
        }
        self._steps = 0
        self._write(record)
        return record

    # -------------------------------------------------------------- consume --

    def current_probe(self) -> tuple[float | None, int]:
        """This iteration's probe survival rate and the episodes behind it.

        ``(None, 0)`` when no probe episode ended, which the gate must read as
        "no evidence" and hold on -- never as a pass and never as a failure.
        """
        if not self.history or self.probe_index is None:
            return None, 0
        strata = self.history[-1].get("per_stratum") or []
        if self.probe_index >= len(strata):
            return None, 0
        entry = strata[self.probe_index]
        return entry.get("survival"), int(entry.get("episodes", 0))

    def current_population(self) -> float | None:
        return self.history[-1]["survival"] if self.history else None

    def _write(self, record: dict[str, Any]) -> None:
        if not self.output_dir:
            return
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"survival_{self.branch_id}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
