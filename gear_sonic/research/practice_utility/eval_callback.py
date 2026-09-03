"""Deployment-oriented evaluation callback for frozen SONIC policies.

This extends SONIC's imitation callback without changing its success, progress,
or MPJPE calculations.  It only adds physical-quality telemetry and a live audit
of the delayed-actuator buffers to the resulting ``metrics_eval.json``.
"""

from __future__ import annotations

from typing import Any

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import events_actuator as EAC
from gear_sonic.research.practice_utility import events_reset_safe as ERS
from gear_sonic.research.practice_utility.quality_telemetry import (
    QualityTelemetryCollector,
    _scene_entity,
)
from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback


class PracticeRobustnessEvalCallback(ImEvalCallback):
    """Run SONIC's frozen-policy evaluation with deployment telemetry."""

    def __init__(
        self,
        eval_frequency: int = 1,
        empty_cache_freq: int = 20,
        eval_only: bool = True,
        output_dir: str | None = None,
        log_keys: list[str] | None = None,
        preset_id: str = "unbound",
        branch_id: str = "unbound",
        step_dt: float = 0.02,
        non_latency_dr_scale: float | None = None,
        fixed_latency_steps: int | None = None,
        channel_dr_scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            eval_frequency=eval_frequency,
            empty_cache_freq=empty_cache_freq,
            eval_only=eval_only,
            output_dir=output_dir,
            log_keys=log_keys,
        )
        self.preset_id = preset_id
        self.branch_id = branch_id
        self.quality = QualityTelemetryCollector(step_dt=step_dt)
        self.non_latency_dr_scale = (
            None if non_latency_dr_scale is None else float(non_latency_dr_scale)
        )
        # Above 1 the evaluation extrapolates past the training envelope: the
        # robustness profile's whole point is that the last cell is physics the
        # policy was never trained on.
        if self.non_latency_dr_scale is not None and not (
            0.0 <= self.non_latency_dr_scale <= DS.MAX_EXTRAPOLATION
        ):
            raise ValueError(f"non_latency_dr_scale must be in [0, {DS.MAX_EXTRAPOLATION}]")
        # Pin actuation latency to exactly this many physics steps, on top of
        # whatever the preset says. The stock ``latency_60ms`` preset stacks a
        # fixed 60 ms on top of the *full* six-channel envelope, and every arm
        # ever measured on it -- including the untrained origin -- scores 0.00%.
        # A saturated cell cannot discriminate between policies, so the
        # deployment-latency endpoint needs a ladder against clean physics,
        # which is also the question a deployment actually asks: the real robot
        # has some actuation delay; does the policy survive it?
        self.fixed_latency_steps = (
            None if fixed_latency_steps is None else int(fixed_latency_steps)
        )
        if self.fixed_latency_steps is not None and self.fixed_latency_steps < 0:
            raise ValueError("fixed_latency_steps must be >= 0")
        self._latency_report: dict[str, Any] | None = None
        self._dr_scale_report: dict[str, Any] | None = None
        # Per-channel intensities, applied AFTER the scalar scale and only to
        # the named event terms. The scalar ladder moves every channel together,
        # so a drop at phys_150 says nothing about which physics broke the
        # policy; a cell that widens one term while holding the others at their
        # training envelope is the only way to attribute a failure to a channel.
        # Each value is the same affine intensity ``dr_scaling.scale_range``
        # uses, so ``{"physics_material": 1.5}`` is the friction/restitution
        # marginal of phys_150 with mass, CoM, joint offsets and pushes at 1.0.
        self.channel_dr_scales: dict[str, float] | None = None
        if channel_dr_scales:
            resolved: dict[str, float] = {}
            for name, value in dict(channel_dr_scales).items():
                scale = float(value)
                if not 0.0 <= scale <= DS.MAX_EXTRAPOLATION:
                    raise ValueError(
                        f"channel_dr_scales[{name!r}] must be in [0, {DS.MAX_EXTRAPOLATION}], got {scale}"
                    )
                resolved[str(name)] = scale
            self.channel_dr_scales = resolved

    def _pre_evaluate_policy(self, reset_env: bool = True) -> None:
        self.quality.reset()
        event_manager = _event_manager(self.env)
        if self.non_latency_dr_scale is not None:
            baseline = DS.capture_baseline(event_manager)
            report = DS.apply_lambda(
                event_manager,
                baseline,
                self.non_latency_dr_scale,
                exclude_terms=("randomize_action_delay", "randomize_action_delay_interval"),
                allow_extrapolation=True,
            )
            self._dr_scale_report = report.to_dict()
            if self.non_latency_dr_scale > 1.0:
                self._dr_scale_report["physical_clamp"] = DS.clamp_physical(event_manager)
        if self.channel_dr_scales:
            baseline = DS.capture_baseline(event_manager) if self.non_latency_dr_scale is None else baseline
            missing = sorted(name for name in self.channel_dr_scales if name not in baseline)
            if missing:
                # Fail closed: a cell that claims to widen a channel the live
                # event manager does not expose would be reported as a
                # measurement of physics that never ran.
                raise ValueError(
                    f"channel_dr_scales names terms with no scalable range: {missing}; "
                    f"available: {sorted(baseline)}"
                )
            channels: dict[str, Any] = {}
            for name, scale in self.channel_dr_scales.items():
                report = DS.apply_lambda(
                    event_manager,
                    {name: baseline[name]},
                    scale,
                    allow_extrapolation=True,
                )
                channels[name] = report.to_dict()
                if name not in report.scaled_terms:
                    raise ValueError(
                        f"channel_dr_scales[{name!r}] was requested but the term was not scaled: "
                        f"{report.to_dict()}"
                    )
            if self._dr_scale_report is None:
                self._dr_scale_report = {}
            self._dr_scale_report["channels"] = channels
            if any(scale > 1.0 for scale in self.channel_dr_scales.values()):
                self._dr_scale_report["physical_clamp_channels"] = DS.clamp_physical(event_manager)
        if self.fixed_latency_steps is not None:
            self._latency_report = _pin_action_delay(
                event_manager, float(self.fixed_latency_steps)
            )
        robot = _scene_entity(self.env, "robot")
        if robot is not None:
            ERS.reset_action_delay_process(robot)
            # Clear the previous cell's actuator reports so the receipt describes
            # this one. What the terms then record includes a PhysX read-back, which
            # is the only evidence that distinguishes a channel that did nothing
            # physically from one whose write never reached the engine: the
            # articulation updates its own mirror unconditionally, so every
            # event-manager read passes either way.
            EAC.clear_actuator_telemetry(robot)
        super()._pre_evaluate_policy(reset_env=reset_env)

    def env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        actor_state = super().env_step(actor_state)
        self.quality.observe(self.env, actor_state.get("actions"))
        return actor_state

    def _post_evaluate_policy(self, eval_res: dict[str, Any]) -> dict[str, Any]:
        metrics = super()._post_evaluate_policy(eval_res)
        metrics["eval/protocol/preset_id"] = self.preset_id
        metrics["eval/protocol/branch_id"] = self.branch_id
        metrics["eval/protocol/non_latency_dr_scale"] = self.non_latency_dr_scale
        metrics["eval/protocol/dr_scale_report"] = self._dr_scale_report
        metrics["eval/protocol/channel_dr_scales"] = self.channel_dr_scales
        metrics["eval/protocol/fixed_latency_steps"] = self.fixed_latency_steps
        metrics["eval/protocol/fixed_latency_report"] = self._latency_report
        robot = _scene_entity(self.env, "robot")
        metrics["eval/protocol/actuator_telemetry"] = (
            EAC.actuator_telemetry(robot) if robot is not None else None)
        event_manager = _event_manager(self.env)
        metrics["eval/protocol/active_dr_terms"] = DS.scalable_terms(event_manager)
        metrics["eval/protocol/dr_ranges"] = DS.capture_baseline(event_manager)
        for key, value in self.quality.snapshot().items():
            metrics[f"eval/quality/{key}"] = value

        robot = _scene_entity(self.env, "robot")
        delay = (
            ERS.action_delay_stats(robot)
            if robot is not None
            else {
                "action_delay_actuator_groups": 0,
                "action_delay_num_lags": 0,
            }
        )
        for key, value in delay.items():
            metrics[f"eval/delay/{key}"] = value
        return metrics


def _pin_action_delay(event_manager: Any, steps: float) -> dict[str, Any]:
    """Force every actuation-delay term to the closed range ``[steps, steps]``.

    Reported rather than assumed: the returned record names the terms that were
    actually pinned, so a receipt cannot claim a latency level that no live term
    ever received.
    """
    pinned: list[str] = []
    for name, cfg in DS._iter_terms(event_manager):
        params = getattr(cfg, "params", None)
        if isinstance(params, dict) and "delay_range" in params:
            params["delay_range"] = [steps, steps]
            pinned.append(name)
    return {"requested_steps": steps, "pinned_terms": sorted(pinned)}


def _event_manager(env: Any) -> Any:
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        manager = getattr(candidate, "event_manager", None) if candidate is not None else None
        if manager is not None:
            return manager
    return None
