"""A per-channel difficulty gate: one probe, many frontiers.

Why this exists
---------------
The scalar curriculum this programme started from moves six randomization
channels with one number. That was a deliberate simplification -- it turns a
many-dimensional schedule into a one-dimensional control problem -- and it has
a cost the single-channel attribution sweep measures: a humanoid does not fail
along a ray. If friction is what breaks the policy at lambda 1.5 while mass,
CoM, joint offsets and pushes are still free, a scalar frontier that stalls on
friction also withholds the four channels the policy could have widened, and a
scalar frontier that advances on the other four drags friction with it.

This module keeps everything the scalar gate earned -- monotone by
construction, probe measured *above* the frontier, survival as the signal,
freeze as the only guard action -- and changes exactly one thing: the frontier
is a vector, one entry per channel, and each entry rises on its own evidence.

Design
------
*One probe stratum, visited channel by channel.* The population can carry one
probe stratum of useful size (128 of 1,024 environments gives ~6 episodes per
iteration; splitting it six ways would not). So the box gate probes one
channel at a time: the probe stratum runs at the frontier with the *active*
channel raised one step, and the active channel rotates round-robin. A channel
whose probe passes steps its frontier up and hands the probe on; a channel
whose full window falls below threshold is marked blocked for this round and
hands the probe on; a channel that has held the probe for ``channel_budget``
iterations without a decision is timed out and hands the probe on. Blocked
marks clear when a round completes, so a channel that failed at iteration
1,000 is retried once the policy has trained further.

*Per-channel controllers, one shared clock.* Each channel owns a
:class:`~survival_gate.SurvivalGateController`, so the window, dwell,
threshold and ceiling semantics are exactly the scalar gate's and are tested
there. Every controller is advanced every iteration -- with probe evidence
only for the active channel -- so dwell counts wall iterations and the return
guard sees the same return series on every channel. A channel's window is
cleared whenever the probe leaves it: each visit must earn a fresh window.

*Same maximum support as the scalar arms.* With every channel ceiling equal to
the scalar ceiling, the box's terminal support is at most the scalar arm's on
every channel. The box can therefore only differ from the scalar gate in
*which* channels it widens and *when* -- never by training on more. That is
what makes it a fair arm in the support-expansion screen.

Telemetry
---------
Every iteration records the frontier vector before and after, the probe
vector, the active channel, the active channel's gate step (survival, window,
withheld reason), any rotation and why, and the round index. ``lambda`` for
downstream readers is the mean of the frontier vector; the vector itself is
``frontier_vector``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gear_sonic.research.practice_utility.survival_gate import (
    GateStep,
    SurvivalGateConfig,
    SurvivalGateController,
)

#: Why the probe moved to another channel.
ROTATIONS = ("fired", "blocked", "timeout", "at_ceiling", "start")


@dataclass(frozen=True)
class BoxGateConfig:
    """Per-channel gate law: the scalar law, applied per channel, plus rotation."""

    #: Ordered channel names (event term names). Order is the probe rotation.
    channels: tuple[str, ...]
    #: Frontier ceiling per channel. A scalar applies to every channel.
    lambda_max: float | dict[str, float] = 1.5
    #: Probe ceiling per channel; defaults to ``lambda_max``.
    probe_max: float | dict[str, float] | None = None
    threshold: float = 0.80
    window: int = 200
    step_size: float = 0.125
    probe_offset: float = 0.125
    dwell: int = 200
    min_episodes: int = 200
    #: Iterations one channel may hold the probe without a fire or a full
    #: below-threshold window before the probe moves on. Zero disables.
    channel_budget: int = 0
    return_relative_drop: float = 0.25
    return_window: int = 8
    guard_action: str = "freeze"
    guard_freeze_iterations: int = 500

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("box gate needs at least one channel")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"duplicate channels: {self.channels}")
        if self.channel_budget < 0:
            raise ValueError("channel_budget must be >= 0")
        for name in self.channels:
            self.channel_config(name)  # validates per-channel numbers

    def ceiling(self, name: str) -> float:
        value = self.lambda_max
        return float(value[name] if isinstance(value, dict) else value)

    def probe_ceiling(self, name: str) -> float:
        if self.probe_max is None:
            return self.ceiling(name)
        value = self.probe_max
        return float(value[name] if isinstance(value, dict) else value)

    def channel_config(self, name: str) -> SurvivalGateConfig:
        if name not in self.channels:
            raise KeyError(name)
        return SurvivalGateConfig(
            threshold=self.threshold,
            window=self.window,
            step_size=self.step_size,
            probe_offset=self.probe_offset,
            dwell=self.dwell,
            min_episodes=self.min_episodes,
            lambda_max=self.ceiling(name),
            probe_max=self.probe_ceiling(name),
            return_relative_drop=self.return_relative_drop,
            return_window=self.return_window,
            guard_action=self.guard_action,
            guard_freeze_iterations=self.guard_freeze_iterations,
        )


@dataclass
class BoxGateStep:
    """One iteration of the box gate, recorded for audit."""

    iteration: int
    active_channel: str | None
    frontier_before: dict[str, float]
    frontier_after: dict[str, float]
    probe: dict[str, float]
    #: The active channel's own gate step, when there is one.
    channel_step: dict[str, Any] | None = None
    fired: bool = False
    withheld: str | None = None
    #: Set when the probe moved on this iteration, with the reason.
    rotation: str | None = None
    rotated_to: str | None = None
    round_index: int = 0
    blocked: tuple[str, ...] = ()
    channels_at_ceiling: int = 0
    all_at_ceiling: bool = False
    guard_tripped: bool = False
    applied_decrease: bool = False
    #: Iterations the active channel has held the probe, after this update.
    channel_hold: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["blocked"] = list(self.blocked)
        out["lambda_mean"] = mean_frontier(self.frontier_after)
        out["lambda_max_channel"] = max(self.frontier_after.values()) if self.frontier_after else None
        return out


def mean_frontier(frontier: dict[str, float]) -> float | None:
    if not frontier:
        return None
    return float(sum(frontier.values()) / len(frontier))


class BoxGateController:
    """Monotone per-channel frontier expansion, one probe visited in rotation."""

    def __init__(
        self,
        config: BoxGateConfig,
        initial_lambda: float | dict[str, float] = 1.0,
    ) -> None:
        self.config = config
        self.gates: dict[str, SurvivalGateController] = {}
        for name in config.channels:
            start = initial_lambda[name] if isinstance(initial_lambda, dict) else initial_lambda
            self.gates[name] = SurvivalGateController(
                config.channel_config(name), initial_lambda=float(min(start, config.ceiling(name)))
            )
        self.iteration = 0
        self.round_index = 0
        self.expansions: dict[str, int] = {name: 0 for name in config.channels}
        self.visits: dict[str, int] = {name: 0 for name in config.channels}
        self.blocked: set[str] = set()
        self.visited_this_round: set[str] = set()
        self.history: list[BoxGateStep] = []
        self._active: str | None = None
        self._hold = 0
        self._pending_rotation: str = "start"
        self._select_next()

    # ------------------------------------------------------------- geometry --

    @property
    def frontier(self) -> dict[str, float]:
        return {name: gate.frontier for name, gate in self.gates.items()}

    @property
    def active_channel(self) -> str | None:
        return self._active

    @property
    def probe(self) -> dict[str, float]:
        """The probe stratum's intensity vector: frontier, active channel raised."""
        vector = self.frontier
        if self._active is not None:
            vector[self._active] = self.gates[self._active].probe_lambda
        return vector

    @property
    def all_at_ceiling(self) -> bool:
        return all(gate.at_ceiling for gate in self.gates.values())

    @property
    def incidents(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, gate in self.gates.items():
            out.extend({"channel": name, **incident} for incident in gate.incidents)
        return out

    # ------------------------------------------------------------- rotation --

    def _eligible(self) -> list[str]:
        return [
            name
            for name in self.config.channels
            if not self.gates[name].at_ceiling and name not in self.blocked
        ]

    def _select_next(self) -> str | None:
        """Move the probe to the next eligible channel after the current one.

        A round is one visit to every channel that was eligible; when the
        round closes the blocked marks are lifted so a channel that failed
        earlier is retried against a better-trained policy.
        """
        previous = self._active
        if previous is not None:
            self.gates[previous].clear_window()
        eligible = self._eligible()
        if not eligible:
            if self.blocked and not self.all_at_ceiling:
                # Everyone still below ceiling is blocked: close the round.
                self.round_index += 1
                self.blocked.clear()
                self.visited_this_round.clear()
                eligible = self._eligible()
        if not eligible:
            self._active = None
            self._hold = 0
            return None
        order = list(self.config.channels)
        start = 0 if previous is None else (order.index(previous) + 1) % len(order)
        rotated = order[start:] + order[:start]
        chosen = next(name for name in rotated if name in eligible)
        if chosen in self.visited_this_round and set(eligible) <= self.visited_this_round:
            self.round_index += 1
            self.visited_this_round.clear()
        self.visited_this_round.add(chosen)
        self.visits[chosen] += 1
        self._active = chosen
        self._hold = 0
        return chosen

    # ----------------------------------------------------------------- step --

    def update(
        self,
        probe_survival: float | None = None,
        probe_episodes: int = 0,
        mean_return: float | None = None,
    ) -> BoxGateStep:
        """Advance one iteration.

        Probe evidence belongs to the active channel only. Every other
        channel's controller is advanced with no evidence so that dwell and the
        return guard run on the shared clock.
        """
        self.iteration += 1
        before = self.frontier
        active = self._active
        rotation = self._pending_rotation
        self._pending_rotation = None  # type: ignore[assignment]
        rotated_to = active if rotation is not None else None

        steps: dict[str, GateStep] = {}
        for name, gate in self.gates.items():
            if name == active:
                steps[name] = gate.update(
                    probe_survival=probe_survival,
                    probe_episodes=probe_episodes,
                    mean_return=mean_return,
                )
            else:
                steps[name] = gate.update(probe_survival=None, probe_episodes=0, mean_return=mean_return)

        fired = False
        withheld: str | None = None
        channel_step: dict[str, Any] | None = None
        next_rotation: str | None = None
        if active is not None:
            self._hold += 1
            step = steps[active]
            channel_step = step.to_dict()
            fired = step.fired
            withheld = step.withheld
            if fired:
                self.expansions[active] += 1
                next_rotation = "fired"
            elif step.at_ceiling:
                next_rotation = "at_ceiling"
            elif withheld == "below_threshold":
                self.blocked.add(active)
                next_rotation = "blocked"
            elif self.config.channel_budget and self._hold >= self.config.channel_budget:
                self.blocked.add(active)
                withheld = "channel_timeout"
                next_rotation = "timeout"
        elif not self.all_at_ceiling:
            # No active channel but some channel is still below ceiling: a
            # round just closed with everything blocked; reopen.
            next_rotation = "start"

        guard_tripped = any(step.guard_tripped for step in steps.values())
        applied_decrease = any(step.applied_decrease for step in steps.values())
        after = self.frontier
        hold = self._hold

        record = BoxGateStep(
            iteration=self.iteration,
            active_channel=active,
            frontier_before=before,
            frontier_after=after,
            probe=self.probe,
            channel_step=channel_step,
            fired=fired,
            withheld=withheld if active is not None else ("all_at_ceiling" if self.all_at_ceiling else "no_active_channel"),
            rotation=rotation,
            rotated_to=rotated_to,
            round_index=self.round_index,
            blocked=tuple(sorted(self.blocked)),
            channels_at_ceiling=sum(1 for gate in self.gates.values() if gate.at_ceiling),
            all_at_ceiling=self.all_at_ceiling,
            guard_tripped=guard_tripped,
            applied_decrease=applied_decrease,
            channel_hold=hold,
        )
        for name in self.config.channels:
            if after[name] < before[name] - 1e-12 and not applied_decrease:
                raise AssertionError(f"box gate lowered channel {name!r} without recording an incident")
        self.history.append(record)

        if next_rotation is not None:
            self._select_next()
            self._pending_rotation = next_rotation
        return record

    # ---------------------------------------------------------- persistence --

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "channels": list(self.config.channels),
            "gates": {name: gate.state_dict() for name, gate in self.gates.items()},
            "iteration": self.iteration,
            "round_index": self.round_index,
            "expansions": dict(self.expansions),
            "visits": dict(self.visits),
            "blocked": sorted(self.blocked),
            "visited_this_round": sorted(self.visited_this_round),
            "active": self._active,
            "hold": self._hold,
            "pending_rotation": self._pending_rotation,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore mid-run. Frontiers are never rolled back (see survival_gate)."""
        for name, gate in self.gates.items():
            if name in state.get("gates", {}):
                gate.load_state_dict(state["gates"][name])
        self.iteration = int(state.get("iteration", self.iteration))
        self.round_index = int(state.get("round_index", self.round_index))
        self.expansions.update({k: int(v) for k, v in state.get("expansions", {}).items() if k in self.expansions})
        self.visits.update({k: int(v) for k, v in state.get("visits", {}).items() if k in self.visits})
        self.blocked = {name for name in state.get("blocked", []) if name in self.gates}
        self.visited_this_round = {name for name in state.get("visited_this_round", []) if name in self.gates}
        active = state.get("active")
        self._active = active if active in self.gates else None
        self._hold = int(state.get("hold", 0))
        self._pending_rotation = state.get("pending_rotation")
        if self._active is None and not self.all_at_ceiling:
            self._select_next()
            self._pending_rotation = "start"
