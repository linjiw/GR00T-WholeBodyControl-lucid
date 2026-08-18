"""Collect physical-quality telemetry from a live SONIC rollout.

:mod:`quality_metrics` defines *what* to measure; this collects the state those
functions need, from the simulator, during training. It exists so a branch
reports its harm vector without a separate evaluation pass -- an evaluation pass
per branch per horizon is the single most expensive item in the campaign, and
most of the harm signal does not need one.

Signals and where they come from (verified against the G1 configuration):

``env.scene["contact_forces"]``
    A ``ContactSensor`` over every robot body, ``force_threshold=10.0``,
    ``track_air_time=True``. Gives real contact forces, so foot slip is measured
    under genuine contact rather than a height heuristic.
``left_ankle_roll_link`` / ``right_ankle_roll_link``
    The feet. These are exactly the bodies SONIC's own ``undesired_contacts``
    term *excludes* (with wrists and elbows), which is the configuration's own
    statement of which contacts are legitimate.
``robot.data``
    ``applied_torque``, ``joint_vel``, ``joint_pos``, ``body_lin_vel_w``,
    ``soft_joint_pos_limits``.

Access is defensive: the wrapper, the underlying env, and the scene are all
reached through several candidate paths, because a hard-coded chain is exactly
what broke the first live run.

This is telemetry, not the deployment objective. ``J_eff`` remains a macro-mean
of quality-qualified success on a frozen dev suite; these per-iteration
aggregates are the harm vector and a mechanism diagnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from gear_sonic.research.practice_utility import quality_metrics as QM

try:  # pragma: no cover - present in the training env
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        """Stand-in so this module imports without transformers."""

#: The bodies that may legitimately touch the ground.
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

#: Contact-force magnitude (N) above which a body counts as in contact.
CONTACT_THRESHOLD = 10.0


@dataclass
class QualityAccumulator:
    """Running per-iteration aggregates over rollout steps."""

    steps: int = 0
    action_rate_sum: float = 0.0
    action_accel_sum: float = 0.0
    foot_slip_total: float = 0.0
    contact_impulse_total: float = 0.0
    contact_force_peak: float = 0.0
    undesired_contact_steps: float = 0.0
    torque_saturation_sum: float = 0.0
    joint_limit_proximity_sum: float = 0.0
    energy_sum: float = 0.0
    missing_signals: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        n = max(self.steps, 1)
        return {
            "steps": self.steps,
            "action_rate": self.action_rate_sum / n,
            "action_acceleration": self.action_accel_sum / n,
            "foot_slip_total_m": self.foot_slip_total,
            "foot_slip_per_step_m": self.foot_slip_total / n,
            "contact_impulse_total": self.contact_impulse_total,
            "contact_force_peak": self.contact_force_peak,
            "undesired_contact_rate": self.undesired_contact_steps / n,
            "torque_saturation": self.torque_saturation_sum / n,
            "joint_limit_proximity": self.joint_limit_proximity_sum / n,
            "energy_proxy": self.energy_sum / n,
            "missing_signals": sorted(self.missing_signals),
        }

    def reset(self) -> None:
        missing = self.missing_signals
        self.__init__()  # type: ignore[misc]
        self.missing_signals = missing


class QualityTelemetryCollector:
    """Samples simulator state and accumulates quality aggregates.

    Separate from the callback so it can be exercised on CPU against a fake
    environment; the callback is a thin lifecycle shell around it.
    """

    def __init__(
        self,
        foot_bodies: tuple[str, ...] = FOOT_BODIES,
        contact_threshold: float = CONTACT_THRESHOLD,
        step_dt: float = 0.02,
    ) -> None:
        self.foot_bodies = foot_bodies
        self.contact_threshold = contact_threshold
        self.step_dt = step_dt
        self.accumulator = QualityAccumulator()
        self._previous_action: torch.Tensor | None = None
        self._previous_delta: torch.Tensor | None = None
        self._foot_indices: list[int] | None = None
        self._undesired_indices: list[int] | None = None

    # ----------------------------------------------------------- collection --

    def observe(self, env: Any, actions: torch.Tensor | None = None) -> None:
        """Sample one rollout step. Never raises on a missing signal.

        A quality channel that is unavailable is recorded in
        ``missing_signals`` and omitted, rather than silently reported as zero:
        a zero would read as "no slip" when it means "not measured".
        """
        robot = _scene_entity(env, "robot")
        if robot is None:
            self.accumulator.missing_signals.add("robot")
            return
        self.accumulator.steps += 1

        if actions is not None:
            self._accumulate_action(actions.detach())
        self._accumulate_feet(env, robot)
        self._accumulate_contacts(env, robot)
        self._accumulate_actuator(robot)

    def _accumulate_action(self, action: torch.Tensor) -> None:
        action = action.reshape(action.shape[0], -1).to(torch.float64)
        if self._previous_action is not None:
            delta = action - self._previous_action
            self.accumulator.action_rate_sum += float(delta.pow(2).sum(dim=-1).mean())
            if self._previous_delta is not None:
                second = delta - self._previous_delta
                self.accumulator.action_accel_sum += float(second.pow(2).sum(dim=-1).mean())
            self._previous_delta = delta
        self._previous_action = action

    def _accumulate_feet(self, env: Any, robot: Any) -> None:
        forces = self._contact_forces(env)
        velocity = _data(robot, "body_lin_vel_w")
        if forces is None or velocity is None:
            self.accumulator.missing_signals.add("foot_slip")
            return
        indices = self._resolve_feet(robot)
        if not indices:
            self.accumulator.missing_signals.add("foot_bodies")
            return
        magnitude = forces[:, indices].norm(dim=-1)
        in_contact = (magnitude > self.contact_threshold).to(torch.float64)
        horizontal = velocity[:, indices, :2].to(torch.float64).norm(dim=-1)
        # Mean over envs so the figure is per-robot, not per-batch.
        self.accumulator.foot_slip_total += float(
            (horizontal * in_contact).sum(dim=-1).mean() * self.step_dt
        )

    def _accumulate_contacts(self, env: Any, robot: Any) -> None:
        forces = self._contact_forces(env)
        if forces is None:
            self.accumulator.missing_signals.add("contact")
            return
        magnitude = forces.norm(dim=-1).to(torch.float64)
        self.accumulator.contact_impulse_total += float(magnitude.sum(dim=-1).mean() * self.step_dt)
        self.accumulator.contact_force_peak = max(
            self.accumulator.contact_force_peak, float(magnitude.max())
        )
        undesired = self._resolve_undesired(robot)
        if undesired:
            touching = (magnitude[:, undesired] > self.contact_threshold).any(dim=-1)
            self.accumulator.undesired_contact_steps += float(touching.to(torch.float64).mean())

    def _accumulate_actuator(self, robot: Any) -> None:
        torque = _data(robot, "applied_torque")
        joint_vel = _data(robot, "joint_vel")
        joint_pos = _data(robot, "joint_pos")
        limits = _data(robot, "soft_joint_pos_limits")

        if torque is not None and joint_vel is not None:
            self.accumulator.energy_sum += float(
                (torque.to(torch.float64) * joint_vel.to(torch.float64)).abs().sum(dim=-1).mean()
            )
        else:
            self.accumulator.missing_signals.add("energy")

        effort = _data(robot, "joint_effort_limits")
        if torque is not None and effort is not None:
            ratio = torque.to(torch.float64).abs() / effort.to(torch.float64).clamp(min=1e-6)
            self.accumulator.torque_saturation_sum += float((ratio >= 0.95).to(torch.float64).mean())
        else:
            self.accumulator.missing_signals.add("torque_saturation")

        if joint_pos is not None and limits is not None and limits.shape[-1] == 2:
            lower, upper = limits[..., 0].to(torch.float64), limits[..., 1].to(torch.float64)
            centre = 0.5 * (upper + lower)
            half = (0.5 * (upper - lower)).clamp(min=1e-6)
            proximity = ((joint_pos.to(torch.float64) - centre).abs() / half).clamp(0.0, 1.0)
            self.accumulator.joint_limit_proximity_sum += float(proximity.mean())
        else:
            self.accumulator.missing_signals.add("joint_limit_proximity")

    # -------------------------------------------------------------- helpers --

    def _contact_forces(self, env: Any) -> torch.Tensor | None:
        sensor = _scene_entity(env, "contact_forces")
        if sensor is None:
            return None
        forces = _data(sensor, "net_forces_w")
        if forces is None or forces.ndim != 3:
            return None
        return forces

    def _resolve_feet(self, robot: Any) -> list[int]:
        if self._foot_indices is None:
            self._foot_indices = _find_bodies(robot, list(self.foot_bodies))
        return self._foot_indices

    def _resolve_undesired(self, robot: Any) -> list[int]:
        """Every body that is not a foot, wrist, or elbow.

        Mirrors SONIC's own ``undesired_contacts`` exclusion list, so the metric
        agrees with the configuration's notion of a legitimate contact.
        """
        if self._undesired_indices is None:
            names = getattr(robot, "body_names", None)
            if not names:
                self._undesired_indices = []
            else:
                allowed = set(self.foot_bodies) | {
                    "left_wrist_yaw_link", "right_wrist_yaw_link",
                    "left_elbow_link", "right_elbow_link",
                }
                self._undesired_indices = [
                    i for i, name in enumerate(names) if name not in allowed
                ]
        return self._undesired_indices

    def snapshot(self) -> dict[str, Any]:
        return self.accumulator.as_dict()

    def reset(self) -> None:
        self.accumulator.reset()
        self._previous_action = None
        self._previous_delta = None


class PracticeQualityCallback(TrainerCallback):
    """Writes per-iteration quality telemetry beside the dose reports."""

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str | None = None,
        branch_id: str = "unbound",
        sample_every: int = 1,
        step_dt: float = 0.02,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.branch_id = branch_id
        self.sample_every = max(1, int(sample_every))
        self.collector = QualityTelemetryCollector(step_dt=step_dt)
        self.history: list[dict[str, Any]] = []
        self._sample_counter = 0

    def observe(self, env: Any, actions: torch.Tensor | None = None) -> None:
        """Call from the rollout loop; cheap and never raises."""
        if not self.enabled:
            return
        self._sample_counter += 1
        if self._sample_counter % self.sample_every == 0:
            self.collector.observe(env, actions)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        step = getattr(state, "global_step", 0) if state is not None else 0
        record = {"global_step": step, "branch_id": self.branch_id, **self.collector.snapshot()}
        self.history.append(record)
        self.collector.reset()
        if self.output_dir:
            directory = Path(self.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"quality_{self.branch_id}.jsonl").open("a").write(
                json.dumps(record) + "\n"
            )
        return control

    def thresholds_report(self, thresholds: QM.QualityThresholds) -> dict[str, Any]:
        """Which gates the collected telemetry would trip, for a quick read."""
        if not self.history:
            return {}
        latest = self.history[-1]
        return {
            "foot_slip_exceeds": latest["foot_slip_total_m"] > thresholds.max_foot_slip,
            "torque_saturation_exceeds": (
                latest["torque_saturation"] > thresholds.max_torque_saturation
            ),
            "contact_impulse_exceeds": (
                latest["contact_impulse_total"] > thresholds.max_contact_impulse
            ),
        }


def _scene_entity(env: Any, name: str) -> Any:
    """Reach a scene entity through whichever wrapper layer is present."""
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        scene = getattr(candidate, "scene", None) if candidate is not None else None
        if scene is None:
            continue
        try:
            return scene[name]
        except (KeyError, TypeError):
            continue
    return None


def _data(entity: Any, attribute: str) -> torch.Tensor | None:
    data = getattr(entity, "data", None)
    value = getattr(data, attribute, None) if data is not None else None
    return value if isinstance(value, torch.Tensor) else None


def _find_bodies(robot: Any, names: list[str]) -> list[int]:
    finder = getattr(robot, "find_bodies", None)
    if callable(finder):
        try:
            indices = finder(names)[0]
            return [int(i) for i in indices]
        except Exception:
            pass
    body_names = getattr(robot, "body_names", None)
    if not body_names:
        return []
    return [i for i, name in enumerate(body_names) if name in set(names)]
