"""A difficulty gate that cannot invert.

Why this exists
---------------
The latent-gap PI controller this programme started from failed in a specific,
reproducible way: two of six 8,000-iteration runs *evacuated* difficulty,
ending at lambda 0.062 and 0.012 after holding lambda 1.0 for thousands of
iterations, with zero guard trips. The mechanism is not a tuning error. Any
scheduler whose signal is measured on the distribution it controls has the same
fixed point: lowering difficulty makes environments easier, which improves the
signal, which justifies lowering difficulty further.

Two properties are therefore built in rather than tuned:

*Monotone by construction.* The frontier only ever rises. There is no gain, no
sign, and no integral that could produce a downward move, so the evacuation
path does not exist to be avoided. The return guard is the sole exception and
its default action is to *freeze* expansion rather than contract support --
contracting would discard training support the policy has already paid for,
which is the cost the campaign measured (a collapsed run scored 7.97 points of
frontier success AUC below its own mid-training capsule).

*Measured off-distribution.* The gate reads survival in a small **probe**
stratum held one step *above* the current frontier -- the difficulty we are
considering moving to, not the one we are already training on. A signal
measured at the candidate level cannot be improved by making the current level
easier, which is exactly the loop the gap and the return both closed.

Signal admissibility
--------------------
The gate consumes the probe stratum's episode time-out rate: the fraction of
episodes that ended by reaching the end of the clip rather than by a
termination condition. Three properties earn it the job, and the alternatives
fail at least one:

============= ========  ===============  =========================
signal        anchored  population-wide  bounded / outcome-defined
============= ========  ===============  =========================
latent gap    NO        no (one env)     no (learned, drifting)
mean return   yes       yes              NO (unbounded, scale-drifts)
time-out rate yes       yes              yes ([0, 1], an outcome)
============= ========  ===============  =========================

Anchoring was measured, not assumed, before this module was written. Across
five runs whose applied lambda is pinned at 1.0 -- three fixed-DR arms and two
monotone-ratchet arms -- the rank correlation of each signal against the
iteration index was:

    latent gap p90   -0.30 to +0.11   (mean -0.04, 54% monotone, 19 reversals)
    time-out rate    +0.985 to +0.992 (mean +0.987, 92% monotone, 5 reversals)
    mean return      +0.967 to +0.980 (mean +0.973, 95% monotone, 3 reversals)

Difficulty is constant in those runs, so competence is the only thing left
moving. The latent gap does not track it. That is a disqualification measured
on five independent runs, and it holds regardless of how the controller
consuming it is tuned.

The gap is worse than merely uninformative: its direction is set by the arm
rather than by the policy. Against the same iteration index it correlates
-0.66 in the no-randomization arm, about zero in the fixed arms, and +0.39 and
+0.50 in the two arms that evacuated difficulty. In those two the gap *rose*
while lambda was being cut, which is precisely the sign that drives a PI
controller to cut further. The collapse is that loop, and it is visible in
the instrument itself.

Mean return, by contrast, *is* anchored -- the table's earlier draft claimed
otherwise and the audit above corrected it. Return is disqualified for a
different reason: it is unbounded and its scale drifts across a run (roughly
1.4 to 11-12 on these arms, and to 15.9 on a collapsed one), so no fixed
threshold has a stable meaning. A gate needs an absolute level to compare
against, and only a bounded outcome rate supplies one.

The caveat that remains, stated because it is the whole point of the module:
the time-out rate is *difficulty-relative*, exactly as return is. The
collapsed run reached 0.986, above every healthy fixed-DR arm, because it had
made its own exam easier. Survival is admissible as a gate signal only under a
monotone actuator and only when read at the probe level -- never at the level
currently being trained. Both conditions are structural here rather than
configured.

The law, once per iteration::

    hold the trailing window of W probe survival rates
    fire when   len(window) == W
          and   episodes_in_window >= min_episodes        (coverage floor)
          and   mean(window) >= threshold                 (competence)
          and   iterations_since_last_step >= dwell       (settling)
    then  frontier <- min(frontier + step_size, lambda_max)
          and clear the window                            (hysteresis)

Clearing the window after a step is what stops a single stretch of good
evidence from ratcheting the frontier several steps in a row: each new level
must earn its own full window of fresh evidence at that level.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

#: Actions the return guard may take. ``freeze`` suspends expansion for
#: ``guard_freeze_iterations`` and leaves applied support untouched; ``decay``
#: additionally contracts the frontier, which is a support loss and is always
#: reported as an incident.
GUARD_ACTIONS = ("freeze", "decay")


@dataclass(frozen=True)
class SurvivalGateConfig:
    """Gate law and its guards.

    Defaults are the preregistered Phase-2 values. ``threshold`` sits below the
    ~0.95 survival that fixed DR reaches at lambda = 1 because the probe runs
    *above* the frontier and is expected to be harder than the frontier itself.
    """

    #: Survival (episode time-out) rate the probe must average to expand.
    threshold: float = 0.80
    #: Trailing window of iterations the mean is taken over.
    window: int = 200
    #: Frontier increment per expansion.
    step_size: float = 0.125
    #: Probe stratum offset above the frontier.
    probe_offset: float = 0.125
    #: Iterations that must pass after an expansion before another may fire.
    dwell: int = 200
    #: Episodes that must have ended inside the window for it to count. No
    #: evidence is a hold, never an expansion.
    min_episodes: int = 200
    #: Frontier ceiling. The probe may sit ``probe_offset`` above this.
    lambda_max: float = 1.5
    #: Hard ceiling on the probe itself, independent of the frontier ceiling.
    probe_max: float = 2.0
    #: Relative return guard: trips when the trailing mean return falls below
    #: ``(1 - return_relative_drop)`` times the best trailing mean seen.
    return_relative_drop: float = 0.25
    return_window: int = 8
    guard_action: str = "freeze"
    guard_freeze_iterations: int = 500
    #: Multiplier applied to the frontier when ``guard_action == "decay"``.
    guard_decay: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if self.step_size <= 0.0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")
        if self.probe_offset <= 0.0:
            raise ValueError(f"probe_offset must be > 0, got {self.probe_offset}")
        if self.dwell < 0:
            raise ValueError(f"dwell must be >= 0, got {self.dwell}")
        if self.min_episodes < 0:
            raise ValueError(f"min_episodes must be >= 0, got {self.min_episodes}")
        if self.lambda_max <= 0.0:
            raise ValueError(f"lambda_max must be > 0, got {self.lambda_max}")
        if self.probe_max < self.lambda_max:
            raise ValueError(f"probe_max {self.probe_max} must be >= lambda_max {self.lambda_max}")
        if not 0.0 < self.return_relative_drop < 1.0:
            raise ValueError(
                f"return_relative_drop must be in (0, 1), got {self.return_relative_drop}"
            )
        if self.return_window < 2:
            raise ValueError(f"return_window must be >= 2, got {self.return_window}")
        if self.guard_action not in GUARD_ACTIONS:
            raise ValueError(
                f"guard_action must be one of {GUARD_ACTIONS}, got {self.guard_action!r}"
            )
        if self.guard_freeze_iterations < 0:
            raise ValueError("guard_freeze_iterations must be >= 0")
        if not 0.0 < self.guard_decay <= 1.0:
            raise ValueError(f"guard_decay must be in (0, 1], got {self.guard_decay}")


@dataclass
class GateStep:
    """One iteration of the gate, recorded for audit.

    ``applied_decrease`` is the field that matters for the safety claim: it is
    the count this arm's report must show as zero, and any nonzero value is an
    incident rather than an adaptation.
    """

    iteration: int
    frontier_before: float
    frontier_after: float
    probe_lambda: float
    probe_survival: float | None = None
    probe_episodes: int = 0
    window_mean: float | None = None
    window_episodes: int = 0
    window_full: bool = False
    fired: bool = False
    #: Why the gate did not fire this iteration, when it did not.
    withheld: str | None = None
    mean_return: float | None = None
    return_reference: float | None = None
    guard_tripped: bool = False
    guard_action: str | None = None
    frozen_until: int = 0
    applied_decrease: bool = False
    at_ceiling: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Window:
    """Trailing survival evidence at the current frontier level."""

    rates: deque[float] = field(default_factory=deque)
    episodes: deque[int] = field(default_factory=deque)

    def clear(self) -> None:
        self.rates.clear()
        self.episodes.clear()


class SurvivalGateController:
    """Monotone frontier expansion gated on probe-stratum survival."""

    def __init__(
        self,
        config: SurvivalGateConfig | None = None,
        initial_lambda: float = 1.0,
    ) -> None:
        self.config = config or SurvivalGateConfig()
        if not 0.0 <= initial_lambda <= self.config.lambda_max:
            raise ValueError(
                f"initial_lambda {initial_lambda} outside [0, {self.config.lambda_max}]"
            )
        self.frontier = float(initial_lambda)
        self.iteration = 0
        self.expansions = 0
        #: Every downward movement of applied support, whatever its cause. The
        #: safety claim is that this list stays empty.
        self.incidents: list[dict[str, Any]] = []
        self.history: list[GateStep] = []
        self._window = _Window()
        self._last_step_iteration = -(10**9)
        self._frozen_until = 0
        self._returns: deque[float] = deque(maxlen=max(2, self.config.return_window))
        self._best_return_mean: float | None = None

    # ------------------------------------------------------------- geometry --

    @property
    def probe_lambda(self) -> float:
        """Where the probe stratum runs: one step above the frontier."""
        return min(self.frontier + self.config.probe_offset, self.config.probe_max)

    @property
    def at_ceiling(self) -> bool:
        return self.frontier >= self.config.lambda_max - 1e-12

    # ---------------------------------------------------------------- guard --

    def _observe_return(self, mean_return: float | None) -> float | None:
        """Track the trailing mean return and its running best."""
        if mean_return is None:
            return self._best_return_mean
        self._returns.append(float(mean_return))
        if len(self._returns) < self._returns.maxlen:
            return self._best_return_mean
        window_mean = sum(self._returns) / len(self._returns)
        if self._best_return_mean is None or window_mean > self._best_return_mean:
            self._best_return_mean = window_mean
        return self._best_return_mean

    def _guard_trips(self) -> bool:
        """Has return collapsed relative to this run's own recent best?

        Relative, never absolute: the reward scale is not stationary across a
        run, and an absolute floor cannot tell "training on a harder
        distribution" apart from "the policy is failing".
        """
        if self._best_return_mean is None or self._best_return_mean <= 0.0:
            return False
        if len(self._returns) < self._returns.maxlen:
            return False
        window_mean = sum(self._returns) / len(self._returns)
        return window_mean < (1.0 - self.config.return_relative_drop) * self._best_return_mean

    # ----------------------------------------------------------------- step --

    def update(
        self,
        probe_survival: float | None = None,
        probe_episodes: int = 0,
        mean_return: float | None = None,
    ) -> GateStep:
        """Advance one iteration.

        Args:
            probe_survival: fraction of probe-stratum episodes that ended by
                time-out this iteration, or None when none ended.
            probe_episodes: how many probe episodes that fraction came from.
            mean_return: mean episodic return, for the guard only.

        An iteration with no probe evidence holds the frontier. It does not
        contribute to the window either: an absent measurement must not be
        read as a passing one, and must not be read as a failing one.
        """
        self.iteration += 1
        config = self.config
        before = self.frontier
        reference = self._observe_return(mean_return)

        guard_tripped = self._guard_trips()
        guard_action: str | None = None
        applied_decrease = False
        if guard_tripped:
            guard_action = config.guard_action
            self._frozen_until = self.iteration + config.guard_freeze_iterations
            self._window.clear()
            if config.guard_action == "decay":
                decayed = max(0.0, self.frontier * config.guard_decay)
                if decayed < self.frontier - 1e-12:
                    applied_decrease = True
                    self.incidents.append(
                        {
                            "iteration": self.iteration,
                            "cause": "return_guard_decay",
                            "frontier_before": self.frontier,
                            "frontier_after": decayed,
                            "mean_return": mean_return,
                            "return_reference": reference,
                        }
                    )
                    self.frontier = decayed

        if probe_survival is not None and probe_episodes > 0:
            self._window.rates.append(float(probe_survival))
            self._window.episodes.append(int(probe_episodes))
            while len(self._window.rates) > config.window:
                self._window.rates.popleft()
                self._window.episodes.popleft()

        window_full = len(self._window.rates) >= config.window
        window_episodes = int(sum(self._window.episodes))
        window_mean = (
            sum(self._window.rates) / len(self._window.rates) if self._window.rates else None
        )

        fired = False
        withheld: str | None = None
        if self.at_ceiling:
            withheld = "at_ceiling"
        elif self.iteration < self._frozen_until:
            withheld = "guard_freeze"
        elif not window_full:
            withheld = "window_not_full"
        elif window_episodes < config.min_episodes:
            withheld = "insufficient_episodes"
        elif self.iteration - self._last_step_iteration < config.dwell:
            withheld = "dwell"
        elif window_mean is None or window_mean < config.threshold:
            withheld = "below_threshold"
        else:
            fired = True
            self.frontier = min(self.frontier + config.step_size, config.lambda_max)
            self.expansions += 1
            self._last_step_iteration = self.iteration
            # Fresh evidence is required at the new level: without clearing,
            # one good stretch would ratchet several steps in succession.
            self._window.clear()

        step = GateStep(
            iteration=self.iteration,
            frontier_before=before,
            frontier_after=self.frontier,
            probe_lambda=self.probe_lambda,
            probe_survival=probe_survival,
            probe_episodes=int(probe_episodes),
            window_mean=window_mean,
            window_episodes=window_episodes,
            window_full=window_full,
            fired=fired,
            withheld=withheld,
            mean_return=mean_return,
            return_reference=reference,
            guard_tripped=guard_tripped,
            guard_action=guard_action,
            frozen_until=self._frozen_until,
            applied_decrease=applied_decrease,
            at_ceiling=self.at_ceiling,
        )
        if step.frontier_after < step.frontier_before - 1e-12 and not applied_decrease:
            # Unreachable by construction; kept as an assertion in production
            # because the entire safety claim of this module is this invariant.
            raise AssertionError("survival gate lowered the frontier without recording an incident")
        self.history.append(step)
        return step

    # ---------------------------------------------------------- persistence --

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "frontier": self.frontier,
            "iteration": self.iteration,
            "expansions": self.expansions,
            "window_rates": list(self._window.rates),
            "window_episodes": list(self._window.episodes),
            "last_step_iteration": self._last_step_iteration,
            "frozen_until": self._frozen_until,
            "returns": list(self._returns),
            "best_return_mean": self._best_return_mean,
            "incidents": list(self.incidents),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a gate mid-run.

        The frontier is restored as-is and never rolled back: a resume that
        lowered applied support would be the exact failure this module deletes.
        """
        self.frontier = float(state.get("frontier", self.frontier))
        self.iteration = int(state.get("iteration", self.iteration))
        self.expansions = int(state.get("expansions", self.expansions))
        self._window.clear()
        for rate, episodes in zip(state.get("window_rates", []), state.get("window_episodes", [])):
            self._window.rates.append(float(rate))
            self._window.episodes.append(int(episodes))
        self._last_step_iteration = int(state.get("last_step_iteration", self._last_step_iteration))
        self._frozen_until = int(state.get("frozen_until", self._frozen_until))
        self._returns.clear()
        for value in state.get("returns", []):
            self._returns.append(float(value))
        best = state.get("best_return_mean")
        self._best_return_mean = None if best is None else float(best)
        self.incidents = list(state.get("incidents", []))


def linear_ramp_lambda(
    iteration: int,
    *,
    start_lambda: float,
    end_lambda: float,
    begin_iteration: int,
    end_iteration: int,
) -> float:
    """The open-loop schedule the gate is credited against.

    LUCID's monotone ratchet turned out to be distributionally identical to
    fixed randomization -- one distinct applied lambda over 98.75% of training
    -- which made its noninferiority test near-tautological. A feedback arm is
    therefore compared against a schedule of the same shape and the same
    terminal support, so that "feedback helped" cannot be satisfied by
    "difficulty rose". This function is that schedule: monotone, open loop, and
    reading nothing.
    """
    if end_iteration <= begin_iteration:
        raise ValueError("end_iteration must be > begin_iteration")
    if iteration <= begin_iteration:
        return float(start_lambda)
    if iteration >= end_iteration:
        return float(end_lambda)
    span = float(end_iteration - begin_iteration)
    fraction = (float(iteration) - float(begin_iteration)) / span
    return float(start_lambda) + fraction * (float(end_lambda) - float(start_lambda))
