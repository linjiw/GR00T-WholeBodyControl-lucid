"""The LUCID curriculum callback: latent gap in, DR intensity out.

Ties together the three pieces the manuscript describes -- a frozen encoder
measuring command-execution mismatch, a PI controller over a scalar intensity,
and that intensity scaling every randomization channel -- and runs them once per
PPO iteration against a live SONIC environment.

What this callback deliberately does *not* do is decide whether the curriculum
worked. It logs ``lambda``, the gap that drove it, and which terms it actually
moved; the outcome is measured elsewhere, from simulator state, under evaluation
presets the curriculum never sees. Scoring a scheduler with its own control
signal is the mistake this whole programme is organised to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import observer as OBS
from gear_sonic.research.practice_utility.dr_controller import LucidDRController, PIConfig

try:  # pragma: no cover
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        """Stand-in so this module imports without transformers."""


class LucidCurriculumCallback(TrainerCallback):
    """Schedule DR intensity from the live latent command-execution gap."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "lucid",
        observer_branch_id: str | None = None,
        output_dir: str | None = None,
        branch_id: str = "unbound",
        initial_lambda: float = 0.0,
        fixed_lambda: float = 1.0,
        delta_target: float = 0.1,
        kp: float = 1.0,
        ki: float = 0.1,
        alpha: float = 0.05,
        integral_max: float = 5.0,
        quantile: float = 0.9,
        return_floor: float | None = None,
        return_decay: float = 0.5,
        update_every: int = 1,
    ) -> None:
        if mode not in ("lucid", "fixed", "off"):
            raise ValueError(f"unknown curriculum mode {mode!r}; expected lucid/fixed/off")
        self.enabled = bool(enabled)
        self.mode = mode
        self.observer_branch_id = observer_branch_id
        self.output_dir = output_dir
        self.branch_id = branch_id
        self.fixed_lambda = float(fixed_lambda)
        self.update_every = max(1, int(update_every))

        self.controller = LucidDRController(
            PIConfig(
                kp=kp, ki=ki, alpha=alpha, integral_max=integral_max, quantile=quantile,
                delta_target=delta_target, return_floor=return_floor, return_decay=return_decay,
            ),
            initial_lambda=initial_lambda if mode == "lucid" else fixed_lambda,
        )
        self.baseline: dict[str, dict[str, Any]] | None = None
        self.scalable: list[str] = []
        self.history: list[dict[str, Any]] = []
        self._event_manager: Any = None

    # ------------------------------------------------------------ lifecycle --

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        self._bind(kwargs.get("env"))
        # Apply the starting intensity before the first rollout, so iteration 1
        # already trains under the curriculum rather than under whatever the
        # config happened to declare.
        self._apply(self.controller.lambda_value)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if self._event_manager is None:
            self._bind(kwargs.get("env"))
        step = getattr(state, "global_step", 0) if state is not None else 0
        if step % self.update_every:
            return control

        if self.mode == "fixed":
            record = {"global_step": step, "mode": "fixed",
                      "lambda": self.fixed_lambda, "gap_quantile": None}
            self._apply(self.fixed_lambda)
        elif self.mode == "off":
            record = {"global_step": step, "mode": "off", "lambda": 0.0, "gap_quantile": None}
            self._apply(0.0)
        else:
            gaps = self._gaps()
            mean_return = self._mean_return(state)
            outcome = self.controller.update(gaps=gaps, mean_return=mean_return)
            self._apply(outcome.lambda_after)
            record = {"global_step": step, "mode": "lucid", "lambda": outcome.lambda_after,
                      **outcome.to_dict()}

        record["scalable_terms"] = self.scalable
        self.history.append(record)
        self._write(record)
        return control

    # ------------------------------------------------------------- internals --

    def _bind(self, env: Any) -> None:
        manager = _event_manager_of(env)
        if manager is None:
            return
        self._event_manager = manager
        self.baseline = DS.capture_baseline(manager)
        self.scalable = DS.scalable_terms(manager)

    def _apply(self, lambda_value: float) -> None:
        if self._event_manager is None or self.baseline is None:
            return
        DS.apply_lambda(self._event_manager, self.baseline, lambda_value)

    def _gaps(self) -> list[float]:
        observer = OBS.get_active_observer(self.observer_branch_id)
        return observer.drain_gaps() if observer is not None else []

    @staticmethod
    def _mean_return(state: Any) -> float | None:
        """Most recent mean reward from the trainer's log history, if present."""
        history = getattr(state, "log_history", None) or []
        for entry in reversed(history):
            for key in ("mean_reward", "Mean rewards", "rewards/mean", "train/mean_reward"):
                if key in entry:
                    try:
                        return float(entry[key])
                    except (TypeError, ValueError):
                        continue
        return None

    def _write(self, record: dict[str, Any]) -> None:
        if not self.output_dir:
            return
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"curriculum_{self.branch_id}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def _event_manager_of(env: Any) -> Any:
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        if candidate is None:
            continue
        manager = getattr(candidate, "event_manager", None)
        if manager is not None:
            return manager
    return None
