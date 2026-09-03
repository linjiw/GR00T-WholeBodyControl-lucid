"""Motor thermal derating: a randomization channel the simulator does not have.

Every randomization channel in this project so far perturbs a task the policy
already solves. Mass, centre of mass, joint offset, friction, latency and pushes
are bounded, memoryless draws, and a policy trained at the full envelope keeps
82% success when all five are doubled. There is no barrier anywhere in that
space, which is why a curriculum has had nothing to add: staging the approach to
a difficulty that direct training already reaches cannot help.

Thermal derating is different in kind, and it is not invented for this purpose.
The G1's controller limits current when a joint exceeds its safe temperature, and
Unitree documents continuous operation ending in either a flat battery or that
derating. Our simulator has no such limit: ``effort_limit_sim`` is the PEAK
rating, applied forever. A policy trained here is free to draw peak torque
indefinitely, and the robot it is trained for is not. This module closes that
specific gap.

What makes it a different kind of channel:

* **The difficulty is endogenous.** Heating is driven by the policy's own torque,
  so an inefficient policy creates its own harder environment. No other channel
  here does that.
* **It is history-dependent within an episode.** The constraint arrives late,
  after heat accumulates, so early-episode reward says little about whether the
  episode will survive.
* **Escaping it needs a global change, not a reflex.** A policy cannot react its
  way out of a torque budget; it has to move more economically.

Those three properties are the reason to expect a learnability barrier: a
from-scratch policy is at its least efficient exactly when it can least afford
to be. **That expectation is a hypothesis, and this module exists so it can be
measured rather than assumed.** Whether a barrier exists, and where, is what the
severity sweep in the preregistration answers; whether a curriculum crosses one
is a separate question asked only where a barrier is found.

The model
---------

Per joint, a normalized temperature ``T`` in [0, inf) with a first-order balance::

    dT/dt = (tau / tau_peak)^2 / heat_tau  -  T / cool_tau

Heating goes as torque squared because ohmic loss goes as current squared and
current is proportional to torque in a brushless motor; cooling relaxes toward
ambient. The available torque is the peak rating reduced once ``T`` passes an
onset::

    available = tau_peak * (1 - depth * clip((T - onset) / (1 - onset), 0, 1))

``depth`` is set from the motor's continuous-to-peak ratio, not chosen to
produce an outcome: ``depth = 1 - continuous/peak``.

Intensity
---------

One knob, ``lam``, scaling the channel exactly as every other channel here is
scaled, with ``lam = 0`` collapsing to nominal. At ``lam = 0`` the heating rate
is zero, ``T`` stays at its initial zero, and the available torque is the peak
rating at every step — the model is then a bit-identical no-op, so a run with the
channel "enabled at zero" is a valid baseline rather than a subtly different
robot. That property is pinned by a test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

#: Continuous-to-peak torque ratio used to set the derating depth. Brushless
#: servo actuators of this class sustain roughly half their peak rating
#: indefinitely; Unitree publishes the peak figures and documents that continuous
#: operation ends in thermal derating without giving a curve. This is therefore a
#: STATED ASSUMPTION, recorded here so a severity is never mistaken for a
#: measurement, and it is the first thing to replace with a bench measurement.
DEFAULT_CONTINUOUS_OVER_PEAK = 0.5

#: Seconds of sustained peak torque that take a cold joint to the onset of
#: derating at lam = 1. Chosen to be commensurate with an episode, so the effect
#: is visible inside one; a real robot's constant is minutes, which no episode
#: would ever reach. This is a MODELLING CHOICE about timescale, not a claim
#: about the hardware, and it is what the severity ladder scales.
DEFAULT_HEAT_SECONDS = 4.0

#: Cooling time constant, as a multiple of the heating one. Motors shed heat far
#: more slowly than they make it, which is what makes the constraint accumulate.
DEFAULT_COOL_MULTIPLE = 6.0

#: Fraction of the way to saturation at which derating begins.
DEFAULT_ONSET = 0.35


@dataclass(frozen=True)
class ThermalConfig:
    """Everything that fixes the channel's behaviour, before any intensity."""

    continuous_over_peak: float = DEFAULT_CONTINUOUS_OVER_PEAK
    heat_seconds: float = DEFAULT_HEAT_SECONDS
    cool_multiple: float = DEFAULT_COOL_MULTIPLE
    onset: float = DEFAULT_ONSET
    #: Highest initial temperature an episode may start at, at lam = 1. A robot
    #: that has been walking for an hour does not start cold, and this is the
    #: per-episode draw the curriculum widens.
    initial_temperature_max: float = 0.5

    @property
    def depth(self) -> float:
        """How much of peak torque is lost when fully saturated."""
        return 1.0 - float(self.continuous_over_peak)

    def as_dict(self) -> dict[str, Any]:
        return {
            "continuous_over_peak": self.continuous_over_peak,
            "heat_seconds": self.heat_seconds,
            "cool_multiple": self.cool_multiple,
            "onset": self.onset,
            "initial_temperature_max": self.initial_temperature_max,
            "depth": self.depth,
        }


class ThermalState:
    """Per-environment, per-joint temperature and the torque budget it implies.

    Holds no opinion about the simulator: it takes torques and a timestep, and
    returns the limit that should be applied. That keeps every rule in this file
    testable on a CPU without Isaac.
    """

    def __init__(
        self,
        num_envs: int,
        peak_torque: torch.Tensor,
        *,
        lam: float = 1.0,
        config: ThermalConfig | None = None,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if lam < 0.0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        self.config = config or ThermalConfig()
        if not 0.0 <= self.config.onset < 1.0:
            raise ValueError(f"onset must be in [0, 1), got {self.config.onset}")
        if not 0.0 < self.config.continuous_over_peak <= 1.0:
            raise ValueError(
                f"continuous_over_peak must be in (0, 1], got {self.config.continuous_over_peak}")
        if self.config.heat_seconds <= 0.0:
            raise ValueError("heat_seconds must be positive")
        self.lam = float(lam)
        self.device = torch.device(device)
        self.peak = torch.as_tensor(peak_torque, dtype=torch.float64, device=self.device).reshape(-1)
        if bool((self.peak <= 0).any()):
            raise ValueError("peak_torque must all be positive")
        self.num_envs = int(num_envs)
        self.num_joints = int(self.peak.numel())
        self.generator = generator
        self.temperature = torch.zeros((self.num_envs, self.num_joints),
                                       dtype=torch.float64, device=self.device)
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Draw a starting temperature for the given environments.

        At ``lam = 0`` the draw is exactly zero, so the channel is off rather
        than merely mild.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if env_ids.numel() == 0:
            return
        high = self.lam * self.config.initial_temperature_max
        if high <= 0.0:
            self.temperature[env_ids] = 0.0
            return
        shape = (env_ids.numel(), 1)  # one draw per environment, shared across joints
        draw = torch.rand(shape, dtype=torch.float64, device=self.device,
                          generator=self.generator) * high
        self.temperature[env_ids] = draw.expand(-1, self.num_joints)

    # ----------------------------------------------------------------- limits

    def available_torque(self) -> torch.Tensor:
        """The per-environment, per-joint torque budget at the current state."""
        onset = self.config.onset
        over = (self.temperature - onset) / max(1e-12, 1.0 - onset)
        derate = self.config.depth * over.clamp(0.0, 1.0)
        return self.peak.unsqueeze(0) * (1.0 - derate)

    def step(self, applied_torque: torch.Tensor, dt_s: float) -> torch.Tensor:
        """Advance the temperature by one control step and return the new budget.

        ``applied_torque`` is what the actuator actually delivered, which is what
        heats the motor. Passing the *commanded* torque instead would let a
        saturated joint heat as though it had produced torque it could not.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        tau = torch.as_tensor(applied_torque, dtype=torch.float64,
                              device=self.device).reshape(self.num_envs, self.num_joints)
        if self.lam > 0.0:
            duty = (tau.abs() / self.peak.unsqueeze(0)).clamp(min=0.0) ** 2
            # lam scales the thermal budget: at twice the intensity a joint reaches
            # the same temperature in half the sustained time.
            heating = duty * (self.lam / self.config.heat_seconds)
            cooling = self.temperature / (self.config.heat_seconds * self.config.cool_multiple)
            self.temperature = (self.temperature + dt_s * (heating - cooling)).clamp(min=0.0)
        return self.available_torque()

    def clamp(self, commanded_torque: torch.Tensor) -> torch.Tensor:
        """Limit a commanded torque to the current budget, preserving its sign."""
        tau = torch.as_tensor(commanded_torque, dtype=torch.float64,
                              device=self.device).reshape(self.num_envs, self.num_joints)
        limit = self.available_torque()
        return tau.sign() * torch.minimum(tau.abs(), limit)

    # --------------------------------------------------------------- telemetry

    def report(self) -> dict[str, Any]:
        """What a run should record so a severity is never assumed after the fact."""
        limit = self.available_torque()
        headroom = (limit / self.peak.unsqueeze(0))
        return {
            "lam": self.lam,
            "mean_temperature": float(self.temperature.mean()),
            "max_temperature": float(self.temperature.max()),
            "fraction_derated": float((self.temperature > self.config.onset).to(torch.float64).mean()),
            "mean_available_fraction_of_peak": float(headroom.mean()),
            "min_available_fraction_of_peak": float(headroom.min()),
            "config": self.config.as_dict(),
        }


def sustained_duty_to_onset(config: ThermalConfig, lam: float, duty: float) -> float:
    """Seconds at a constant duty before a COLD joint begins to derate.

    ``inf`` when it never does. An episode that begins warm derates sooner than this;
    the cold-start time is the conservative figure, and it is the one the severity
    ladder is set from.

    The steady state of the balance is ``T_inf = duty^2 * lam * cool_multiple``, so a
    duty whose steady state sits below the onset never derates however long it runs.
    This is the function that says where a severity ladder actually bites, and it is
    how the ladder was chosen rather than guessed.
    """
    if lam <= 0.0 or duty <= 0.0:
        return float("inf")
    cool_tau = config.heat_seconds * config.cool_multiple
    steady = (duty ** 2) * lam * config.cool_multiple
    if steady <= config.onset:
        return float("inf")
    # T(t) = steady * (1 - exp(-t / cool_tau)); solve T(t) = onset.
    import math
    return float(-cool_tau * math.log(1.0 - config.onset / steady))
