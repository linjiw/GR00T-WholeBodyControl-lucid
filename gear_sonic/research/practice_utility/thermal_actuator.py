"""Apply the thermal budget to a running simulation, without subclassing an actuator.

The obvious way to add a torque budget is a new actuator class, mirroring SONIC's
``DelayedImplicitActuator``. It is the wrong way here. That class inherits from
IsaacLab's ``ImplicitActuator``, which cannot be imported without the USD runtime,
so every line of it would be untested until a GPU run, and an actuator that
silently mis-clamps is exactly the kind of defect this project has been bitten by
before.

IsaacLab already exposes what is needed at runtime:
``Articulation.write_joint_effort_limit_to_sim(limits, joint_ids, env_ids)`` sets
the solver's per-environment, per-joint maximum force, and
``Articulation.data.applied_torque`` reports what was actually delivered. For an
implicit actuator the PD law is evaluated inside PhysX, so clamping the maximum
force is precisely a torque budget rather than an approximation of one.

This module is therefore a thin adapter: it reads the delivered torque, advances
:class:`~gear_sonic.research.practice_utility.thermal.ThermalState`, and writes the
new limit back. Every rule lives in the model, which is tested on a CPU; the
adapter is duck-typed against the two methods above so it can be tested against a
fake articulation too. Nothing here is trusted because it looks right.

Order matters and is stated rather than assumed: the limit written at step *k* is
the budget implied by the heat accumulated up to and including step *k*, so a
joint is never punished for torque it has not yet drawn, and never escapes torque
it has.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from gear_sonic.research.practice_utility.thermal import ThermalConfig, ThermalState


class ThermalBudget:
    """Holds the thermal state for one articulation and keeps the sim in step with it."""

    def __init__(
        self,
        articulation: Any,
        *,
        lam: float = 1.0,
        config: ThermalConfig | None = None,
        joint_ids: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.articulation = articulation
        data = articulation.data
        peak = torch.as_tensor(data.joint_effort_limits, dtype=torch.float64)
        if peak.ndim == 2:
            # Per-environment limits; they are identical across environments at reset,
            # and the first row is the hardware rating this budget is a fraction of.
            peak = peak[0]
        self.joint_ids = list(range(peak.numel())) if joint_ids is None else list(joint_ids)
        peak = peak[self.joint_ids]
        num_envs = int(getattr(articulation, "num_instances", None) or data.joint_pos.shape[0])
        self.state = ThermalState(num_envs, peak, lam=lam, config=config,
                                  device=peak.device, generator=generator)
        self.enabled = lam > 0.0
        self._applied_limit: torch.Tensor | None = None

    # ------------------------------------------------------------------ hooks

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """New episode: redraw the starting temperature and write the fresh budget."""
        if not self.enabled:
            return
        self.state.reset(env_ids)
        self._write()

    def step(self, dt_s: float) -> None:
        """Advance the budget by one control step using the torque actually delivered."""
        if not self.enabled:
            return
        delivered = torch.as_tensor(self.articulation.data.applied_torque, dtype=torch.float64)
        self.state.step(delivered[:, self.joint_ids], dt_s)
        self._write()

    def _write(self) -> None:
        limit = self.state.available_torque()
        self._applied_limit = limit
        self.articulation.write_joint_effort_limit_to_sim(
            limit.to(torch.float32), joint_ids=self.joint_ids
        )

    # --------------------------------------------------------------- telemetry

    def report(self) -> dict[str, Any]:
        report = self.state.report()
        report["enabled"] = self.enabled
        report["joint_ids"] = list(self.joint_ids)
        report["wrote_limit"] = self._applied_limit is not None
        return report
