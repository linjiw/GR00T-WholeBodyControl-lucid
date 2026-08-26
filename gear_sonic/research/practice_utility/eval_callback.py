"""Deployment-oriented evaluation callback for frozen SONIC policies.

This extends SONIC's imitation callback without changing its success, progress,
or MPJPE calculations.  It only adds physical-quality telemetry and a live audit
of the delayed-actuator buffers to the resulting ``metrics_eval.json``.
"""

from __future__ import annotations

from typing import Any

from gear_sonic.research.practice_utility import dr_scaling as DS
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
        if self.non_latency_dr_scale is not None and not 0.0 <= self.non_latency_dr_scale <= 1.0:
            raise ValueError("non_latency_dr_scale must be in [0, 1]")
        self._dr_scale_report: dict[str, Any] | None = None

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
            )
            self._dr_scale_report = report.to_dict()
        robot = _scene_entity(self.env, "robot")
        if robot is not None:
            ERS.reset_action_delay_process(robot)
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


def _event_manager(env: Any) -> Any:
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        manager = getattr(candidate, "event_manager", None) if candidate is not None else None
        if manager is not None:
            return manager
    return None
