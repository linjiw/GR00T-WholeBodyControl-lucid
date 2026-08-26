"""Per-step observation of a live rollout: LUCID's gap and quality telemetry.

Two measurements need data at *every* control step, not once per PPO iteration:

**The LUCID latent gap** compares short windows of commanded joint targets
against realized joint motion, so it needs H consecutive frames of both.
**Foot slip and contact impulse** are integrals over time and lose their meaning
if sampled once per iteration.

Trainer callbacks only fire per iteration, so this patches
``ManagerEnvWrapper.step`` -- the same narrow monkeypatch already proven for the
sampler -- and restores it exactly on train end.

On what counts as "commanded"
-----------------------------
LUCID's signal is the mismatch between what the policy asked for and what the
body did. The faithful ``q_cmd`` is therefore the PD *target*
(``robot.data.joint_pos_target``), not the raw policy output, because the action
manager rescales and offsets actions before they reach the controller. The
observer prefers the target and falls back to the wrapper's ``env_actions``,
recording which source it used -- a gap computed against a different quantity is
not comparable, so the source travels with the measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from gear_sonic.research.practice_utility import latent_gap_probe as L
from gear_sonic.research.practice_utility import quality_telemetry as QT

try:  # pragma: no cover
    from transformers import TrainerCallback
except Exception:  # pragma: no cover

    class TrainerCallback:  # type: ignore[no-redef]
        """Stand-in so this module imports without transformers."""


#: Active observers, keyed by branch id. The DR curriculum needs the gap the
#: observer is already collecting, and Hydra instantiates the two callbacks
#: independently with no reference between them. A tiny registry beats making
#: the curriculum patch ``step`` a second time and collect the same data twice.
_ACTIVE_OBSERVERS: dict[str, "PracticeObserverCallback"] = {}


def register_observer(observer: "PracticeObserverCallback") -> None:
    _ACTIVE_OBSERVERS[observer.branch_id] = observer


def get_active_observer(branch_id: str | None = None) -> "PracticeObserverCallback | None":
    """The observer for ``branch_id``, or the only active one if unambiguous."""
    if branch_id is not None:
        return _ACTIVE_OBSERVERS.get(branch_id)
    if len(_ACTIVE_OBSERVERS) == 1:
        return next(iter(_ACTIVE_OBSERVERS.values()))
    return None


def clear_observers() -> None:
    _ACTIVE_OBSERVERS.clear()


@dataclass
class CommandExecutionBuffer:
    """Ring buffer of commanded and realized joint positions for one env.

    Only a single environment is tracked. The gap is a per-trajectory quantity
    and averaging windows across environments mid-episode would blend unrelated
    motions; one env sampled consistently is a cleaner estimator than many
    blended badly.
    """

    capacity: int
    commanded: list[torch.Tensor] = field(default_factory=list)
    executed: list[torch.Tensor] = field(default_factory=list)

    def append(self, command: torch.Tensor, execution: torch.Tensor) -> None:
        self.commanded.append(command.detach().float().cpu())
        self.executed.append(execution.detach().float().cpu())
        if len(self.commanded) > self.capacity:
            self.commanded.pop(0)
            self.executed.pop(0)

    def clear(self) -> None:
        self.commanded.clear()
        self.executed.clear()

    @property
    def ready(self) -> bool:
        return len(self.commanded) >= self.capacity

    def stacks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(self.commanded), torch.stack(self.executed)


class PracticeObserverCallback(TrainerCallback):
    """Collects the LUCID gap and quality telemetry from a live rollout."""

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str | None = None,
        branch_id: str = "unbound",
        encoder_path: str | None = None,
        window_length: int = 16,
        window_stride: int = 1,
        sample_every: int = 1,
        tracked_env: int = 0,
        step_dt: float = 0.02,
        collect_quality: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.branch_id = branch_id
        self.sample_every = max(1, int(sample_every))
        self.tracked_env = int(tracked_env)
        self.collect_quality = bool(collect_quality)

        self.spec = L.WindowSpec(length=window_length, stride=window_stride)
        self.buffer = CommandExecutionBuffer(capacity=self.spec.span)
        self.quality = QT.QualityTelemetryCollector(step_dt=step_dt)

        self.encoder: L.TemporalVAE | None = None
        self.encoder_fingerprint: str | None = None
        self._encoder_path = encoder_path

        self.command_source: str | None = None
        self.history: list[dict[str, Any]] = []
        self._latent_gaps: list[float] = []
        self._raw_gaps: list[float] = []
        self._last_flushed_gaps: list[float] = []
        self._sample_counter = 0

        self._env: Any = None
        self._original_step: Any = None

    # ------------------------------------------------------------ lifecycle --

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.enabled:
            self._load_encoder()
            self._install(kwargs.get("env"))
            register_observer(self)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if self._env is None:
            self._install(kwargs.get("env"))
        step = getattr(state, "global_step", 0) if state is not None else 0
        self.history.append(self._flush(step))
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.enabled and self._latent_gaps:
            self.history.append(self._flush(getattr(state, "global_step", 0)))
        self.uninstall()
        return control

    # -------------------------------------------------------------- install --

    def _install(self, env: Any) -> None:
        if env is None or not hasattr(env, "step"):
            return
        self._env = env
        self._original_step = env.step
        observer = self

        def step_and_observe(actions, *args, **kwargs):
            result = observer._original_step(actions, *args, **kwargs)
            try:
                observer.observe(env, result)
            except Exception:
                # Telemetry must never be able to kill a branch. A failure here
                # loses a measurement; raising would lose the whole run.
                observer.quality.accumulator.missing_signals.add("observer_error")
            return result

        env.step = step_and_observe

    def uninstall(self) -> None:
        if self._env is not None and self._original_step is not None:
            self._env.__dict__.pop("step", None)
        self._env = None
        self._original_step = None

    @staticmethod
    def is_patched(env: Any) -> bool:
        return "step" in vars(env)

    # ----------------------------------------------------------- observation --

    def observe(self, env: Any, step_result: Any = None) -> None:
        """Record one control step."""
        self._sample_counter += 1
        if self._sample_counter % self.sample_every:
            return

        robot = QT._scene_entity(env, "robot")
        if robot is None:
            self.quality.accumulator.missing_signals.add("robot")
            return

        command = self._commanded(robot, step_result)
        executed = QT._data(robot, "joint_pos")
        if command is not None and executed is not None:
            index = min(self.tracked_env, command.shape[0] - 1, executed.shape[0] - 1)
            width = min(command.shape[-1], executed.shape[-1])
            self.buffer.append(command[index, :width], executed[index, :width])
            self._accumulate_gap()

        if self.collect_quality:
            actions = command if command is not None else None
            self.quality.observe(env, actions)

    def _commanded(self, robot: Any, step_result: Any) -> torch.Tensor | None:
        target = QT._data(robot, "joint_pos_target")
        if target is not None:
            self.command_source = self.command_source or "joint_pos_target"
            return target
        if isinstance(step_result, (tuple, list)) and len(step_result) >= 4:
            extras = step_result[3]
            actions = extras.get("env_actions") if isinstance(extras, dict) else None
            if isinstance(actions, torch.Tensor):
                self.command_source = self.command_source or "env_actions"
                return actions
        self.command_source = self.command_source or "unavailable"
        return None

    def _accumulate_gap(self) -> None:
        if not self.buffer.ready:
            return
        command, execution = self.buffer.stacks()
        command_windows = L.build_windows(command, self.spec)
        execution_windows = L.build_windows(execution, self.spec)
        if command_windows.shape[0] == 0:
            return
        self._raw_gaps.append(float(L.raw_mismatch(command_windows, execution_windows)[-1]))
        if self.encoder is not None and command_windows.shape[-1] == self.encoder.num_joints:
            gap = L.latent_gap(
                self.encoder.embed(command_windows[-1:]),
                self.encoder.embed(execution_windows[-1:]),
            )
            self._latent_gaps.append(float(gap[0]))

    # ------------------------------------------------------------- reporting --

    def drain_gaps(self) -> list[float]:
        """Hand the epoch's latent gaps to a consumer.

        Returns the current epoch's gaps, or the previous epoch's if this one has
        already been flushed. Callback order is decided by dict order in the
        Hydra config, and a consumer that happens to run *after* the observer's
        ``on_step_end`` would otherwise read an empty list and silently believe
        the epoch produced no evidence -- which is exactly what happened on the
        first live curriculum run: sixteen iterations of ``num_gap_samples = 0``
        and a controller that correctly, uselessly, held lambda at zero.

        Keeping the last flushed epoch makes the consumer order-independent at
        the cost of at most one epoch of staleness, which a curriculum updating
        once per iteration can absorb.
        """
        if self._latent_gaps:
            return list(self._latent_gaps)
        return list(self._last_flushed_gaps)

    def _flush(self, global_step: int) -> dict[str, Any]:
        latent = L.summarize_gap(torch.tensor(self._latent_gaps)) if self._latent_gaps else None
        raw = L.summarize_gap(torch.tensor(self._raw_gaps)) if self._raw_gaps else None
        delay_stats: dict[str, Any] = {}
        robot = QT._scene_entity(self._env, "robot") if self._env is not None else None
        if robot is not None:
            from gear_sonic.research.practice_utility.events_reset_safe import (
                action_delay_stats,
            )

            delay_stats = action_delay_stats(robot)
        record: dict[str, Any] = {
            "global_step": global_step,
            "branch_id": self.branch_id,
            "command_source": self.command_source,
            "encoder_fingerprint": self.encoder_fingerprint,
            "num_gap_samples": len(self._latent_gaps),
            **(
                {f"latent_{k.replace('gap_', '')}": v for k, v in latent.to_dict().items()}
                if latent
                else {}
            ),
            **(
                {f"raw_{k.replace('gap_', '')}": v for k, v in raw.to_dict().items()} if raw else {}
            ),
            **self.quality.snapshot(),
            **delay_stats,
        }
        self._last_flushed_gaps = list(self._latent_gaps)
        self._latent_gaps.clear()
        self._raw_gaps.clear()
        self.quality.reset()
        if self.output_dir:
            directory = Path(self.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"observer_{self.branch_id}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    def _load_encoder(self) -> None:
        if not self._encoder_path or self.encoder is not None:
            return
        blob = torch.load(Path(self._encoder_path), weights_only=False)
        model = L.TemporalVAE(blob["num_joints"], blob["window_length"], blob["latent_dim"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.encoder = model
        self.encoder_fingerprint = blob.get("encoder_fingerprint")
        # A gap is only comparable against gaps from the same instrument, so the
        # window geometry comes from the artifact rather than the caller.
        self.spec = L.WindowSpec(blob["window_length"], blob.get("window_stride", 1))
        self.buffer = CommandExecutionBuffer(capacity=self.spec.span)
