"""LUCID's PI controller over domain-randomization intensity.

A faithful implementation of the scheduler from the LUCID manuscript, with the
one change the rest of this programme insists on: the gap that drives it is
recorded as a *signal*, and the outcomes used to judge the resulting policy are
measured separately (see ``quality_metrics``). Driving a curriculum with a
quantity and then scoring the curriculum with that same quantity would make any
improvement partly definitional.

The loop, once per curriculum epoch:

    delta_k  = Quantile({latent gap}, p)        high quantile, not the mean
    e_k      = delta_target - delta_k
    I_k      = clip(I_{k-1} + e_k, -I_max, I_max)
    u_k      = clip(Kp e_k + Ki I_k, -1, 1)
    lambda   = clip(lambda + alpha u_k, 0, 1)

A *positive* error means mismatch is below target -- the robot is tracking its
own commands well -- so randomization may increase. Negative means the
environment is already harder than the policy can absorb, and it backs off.

Three guards, each earning its place:

``I_max``   anti-windup. Without it a long stretch at saturation accumulates
            integral the controller then has to unwind, overshooting badly.
``alpha``   bounds how far lambda can move in one epoch, so a single noisy gap
            estimate cannot slam the environment to full difficulty.
``return guard``
            if mean episodic return stays below a floor for two consecutive
            epochs, the integral is reset and lambda is decayed. The gap can
            look healthy while the policy is quietly failing -- a fallen robot
            tracks its commands beautifully at the point of no return -- and
            this is the term that catches that.

Two guards, and why there are two
---------------------------------
``return_guard = "absolute"`` is the manuscript's floor. It is only meaningful
while the reward scale is stationary, and the 128-iteration horizon study showed
it is not: training reward roughly halved between the 32- and 128-iteration
regimes for *every* arm, so a floor calibrated at 32 iterations fired
continuously later on and decayed lambda from 1.0 to 0.4-0.6 -- not because the
policy was failing but because the yardstick had moved. Worse, a floor conflates
two different things: a harder environment legitimately returns less, and a
failing policy returns less, and an absolute threshold cannot tell them apart.

``return_guard = "relative"`` compares the current return against the best of a
trailing window of its own recent history. It asks "is this policy worse than it
recently was", which is scale-free, survives reward drift, and does not punish an
arm for training on a harder distribution. It is the guard every arm after the
horizon study should use; ``absolute`` remains the default so that every existing
receipt's controller reproduces exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class PIConfig:
    """Gains and guards. Defaults follow the manuscript."""

    kp: float = 1.0
    ki: float = 0.1
    alpha: float = 0.05
    integral_max: float = 5.0
    quantile: float = 0.9
    delta_target: float = 0.1
    return_floor: float | None = None
    return_decay: float = 0.5
    low_return_patience: int = 2
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    #: "absolute" trips below ``return_floor``; "relative" trips below
    #: ``(1 - return_relative_drop)`` times the best return in the trailing
    #: ``return_window`` epochs.
    return_guard: str = "absolute"
    return_relative_drop: float = 0.25
    return_window: int = 8
    #: Upward-only lambda moves: the PI law may raise difficulty but never
    #: lower it; the return guard stays the sole downward path. Deletes the
    #: observed anti-gate collapse mode by construction. Default off so every
    #: existing receipt's controller reproduces exactly.
    monotonic: bool = False
    #: Signal-agnostic anti-gate protection: while the trailing window of mean
    #: returns sits at >= ``latch_threshold`` x the run's best trailing-window
    #: mean and is non-decreasing, lambda decreases from the PI law are
    #: refused (guard
    #: trips still lower lambda). The two observed collapses -- lambda cut at
    #: peak competence with zero guard trips -- are exactly the state this
    #: latch binds in. Default off; H_M2 must stay falsifiable for arms whose
    #: preregistration reports anti-gating as an outcome.
    competence_latch: bool = False
    latch_threshold: float = 0.95
    latch_window: int = 500

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.integral_max <= 0:
            raise ValueError(f"integral_max must be > 0, got {self.integral_max}")
        if not 0.0 <= self.lambda_min < self.lambda_max <= 1.0:
            raise ValueError("require 0 <= lambda_min < lambda_max <= 1")
        if not 0.0 < self.return_decay <= 1.0:
            raise ValueError(f"return_decay must be in (0, 1], got {self.return_decay}")
        if self.low_return_patience < 1:
            raise ValueError("low_return_patience must be >= 1")
        if self.return_guard not in ("absolute", "relative"):
            raise ValueError(
                f"return_guard must be 'absolute' or 'relative', got {self.return_guard!r}"
            )
        if not 0.0 < self.return_relative_drop < 1.0:
            raise ValueError(
                f"return_relative_drop must be in (0, 1), got {self.return_relative_drop}"
            )
        if self.return_window < 2:
            raise ValueError(f"return_window must be >= 2, got {self.return_window}")
        if not 0.0 < self.latch_threshold <= 1.0:
            raise ValueError(
                f"latch_threshold must be in (0, 1], got {self.latch_threshold}"
            )
        if self.latch_window < 10:
            raise ValueError(f"latch_window must be >= 10, got {self.latch_window}")


@dataclass
class ControllerStep:
    """One epoch of the loop, recorded for audit."""

    epoch: int
    gap_quantile: float
    error: float
    integral: float
    control: float
    lambda_before: float
    lambda_after: float
    mean_return: float | None = None
    guard_tripped: bool = False
    low_return_streak: int = 0
    num_gap_samples: int = 0
    return_reference: float | None = None
    #: True on epochs where the competence latch (or monotonic mode) refused a
    #: lambda decrease the PI law asked for -- the audit trail that keeps a
    #: bound latch visible as evidence of signal invalidity.
    latch_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LucidDRController:
    """Scalar DR intensity driven by the latent command-execution gap."""

    def __init__(self, config: PIConfig | None = None, initial_lambda: float = 0.0) -> None:
        self.config = config or PIConfig()
        if not self.config.lambda_min <= initial_lambda <= self.config.lambda_max:
            raise ValueError(
                f"initial_lambda {initial_lambda} outside "
                f"[{self.config.lambda_min}, {self.config.lambda_max}]"
            )
        self.lambda_value = float(initial_lambda)
        self.integral = 0.0
        self.epoch = 0
        self.low_return_streak = 0
        self.return_window: deque[float] = deque(maxlen=max(2, self.config.return_window))
        self.history: list[ControllerStep] = []
        # Competence-latch state: the best *window mean* seen so far and the
        # current trailing window. Comparing a window mean with the best
        # single noisy iteration makes the 0.95 threshold effectively
        # unreachable on the observed runs, so both sides of the comparison
        # deliberately have the same aggregation.
        self.latch_best_mean: float | None = None
        self.latch_returns: deque[float] = deque(maxlen=max(10, self.config.latch_window))

    def _observe_return(self, mean_return: float | None) -> None:
        if mean_return is None:
            return
        self.latch_returns.append(float(mean_return))
        if len(self.latch_returns) < self.config.latch_window:
            return
        window_mean = sum(self.latch_returns) / len(self.latch_returns)
        if self.latch_best_mean is None or window_mean > self.latch_best_mean:
            self.latch_best_mean = window_mean

    def _latch_binds(self) -> bool:
        """Is the policy demonstrably thriving right now?

        A *full* trailing-window mean at >= ``latch_threshold`` x the best
        trailing-window mean so far, and the window's second half no worse
        than its first: the exact regime both observed anti-gate collapses ran
        their lambda cuts in.
        """
        window = list(self.latch_returns)
        if len(window) < self.config.latch_window:
            return False
        if self.latch_best_mean is None or self.latch_best_mean <= 0.0:
            return False
        mean = sum(window) / len(window)
        if mean < self.config.latch_threshold * self.latch_best_mean:
            return False
        half = len(window) // 2
        first = sum(window[:half]) / half
        second = sum(window[half:]) / (len(window) - half)
        return second >= first

    def _apply_floor(self, lambda_before: float, lambda_after: float) -> tuple[float, bool]:
        """Refuse PI-law decreases in monotonic mode or while the latch binds."""
        if lambda_after >= lambda_before:
            return lambda_after, False
        if self.config.monotonic:
            return lambda_before, True
        if self.config.competence_latch and self._latch_binds():
            return lambda_before, True
        return lambda_after, False

    def update(
        self,
        gaps: Sequence[float] | None = None,
        mean_return: float | None = None,
        gap_quantile: float | None = None,
    ) -> ControllerStep:
        """Advance one curriculum epoch and return what happened.

        Args:
            gaps: per-step latent gaps collected during the epoch.
            mean_return: mean episodic return, for the guard.
            gap_quantile: a pre-computed quantile, if the caller already has one.

        An epoch with no gap samples holds ``lambda`` still rather than guessing.
        Moving difficulty on no evidence is the failure mode the whole scheduler
        exists to avoid.
        """
        self.epoch += 1
        config = self.config
        before = self.lambda_value
        self._observe_return(mean_return)

        if gap_quantile is None:
            gap_quantile = _quantile(gaps or [], config.quantile)
        num_samples = len(gaps) if gaps is not None else 0

        reference = self.return_reference
        guard = self._guard(mean_return)
        if guard:
            # Reset the integral as well as decaying lambda: leaving accumulated
            # integral in place would immediately push difficulty back up.
            self.integral = 0.0
            self.lambda_value = max(config.lambda_min, self.lambda_value * config.return_decay)
            step = ControllerStep(
                epoch=self.epoch, gap_quantile=gap_quantile, error=0.0,
                integral=self.integral, control=0.0, lambda_before=before,
                lambda_after=self.lambda_value, mean_return=mean_return,
                guard_tripped=True, low_return_streak=self.low_return_streak,
                num_gap_samples=num_samples, return_reference=reference,
            )
            self.history.append(step)
            return step

        if num_samples == 0 and gap_quantile == 0.0 and gaps is not None:
            step = ControllerStep(
                epoch=self.epoch, gap_quantile=0.0, error=0.0, integral=self.integral,
                control=0.0, lambda_before=before, lambda_after=before,
                mean_return=mean_return, low_return_streak=self.low_return_streak,
                num_gap_samples=0, return_reference=reference,
            )
            self.history.append(step)
            return step

        error = config.delta_target - gap_quantile
        self.integral = _clip(self.integral + error, -config.integral_max, config.integral_max)
        control = _clip(config.kp * error + config.ki * self.integral, -1.0, 1.0)
        self.lambda_value = _clip(
            self.lambda_value + config.alpha * control, config.lambda_min, config.lambda_max
        )
        self.lambda_value, latched = self._apply_floor(before, self.lambda_value)

        step = ControllerStep(
            epoch=self.epoch, gap_quantile=gap_quantile, error=error, integral=self.integral,
            control=control, lambda_before=before, lambda_after=self.lambda_value,
            mean_return=mean_return, low_return_streak=self.low_return_streak,
            num_gap_samples=num_samples, return_reference=reference, latch_active=latched,
        )
        self.history.append(step)
        return step

    def update_with_error(self, error: float, mean_return: float | None = None) -> ControllerStep:
        """Advance one epoch on an error the caller already computed.

        The same PI law and the same guards as :meth:`update`; only the error
        term is supplied instead of derived from ``delta_target - gap``. Used by
        the margin signal, whose dead band makes "no error" a real state.
        """
        self.epoch += 1
        config = self.config
        before = self.lambda_value
        self._observe_return(mean_return)
        reference = self.return_reference
        if self._guard(mean_return):
            self.integral = 0.0
            self.lambda_value = max(config.lambda_min, self.lambda_value * config.return_decay)
            step = ControllerStep(
                epoch=self.epoch, gap_quantile=float("nan"), error=0.0, integral=self.integral,
                control=0.0, lambda_before=before, lambda_after=self.lambda_value,
                mean_return=mean_return, guard_tripped=True, low_return_streak=self.low_return_streak,
                num_gap_samples=0, return_reference=reference,
            )
            self.history.append(step)
            return step
        error = float(error)
        self.integral = _clip(self.integral + error, -config.integral_max, config.integral_max)
        control = _clip(config.kp * error + config.ki * self.integral, -1.0, 1.0)
        self.lambda_value = _clip(
            self.lambda_value + config.alpha * control, config.lambda_min, config.lambda_max
        )
        self.lambda_value, latched = self._apply_floor(before, self.lambda_value)
        step = ControllerStep(
            epoch=self.epoch, gap_quantile=float("nan"), error=error, integral=self.integral,
            control=control, lambda_before=before, lambda_after=self.lambda_value,
            mean_return=mean_return, low_return_streak=self.low_return_streak,
            num_gap_samples=0, return_reference=reference, latch_active=latched,
        )
        self.history.append(step)
        return step

    @property
    def return_reference(self) -> float | None:
        """Best return seen in the trailing window, or ``None`` before any."""
        return max(self.return_window) if self.return_window else None

    def _guard(self, mean_return: float | None) -> bool:
        if self.config.return_guard == "relative":
            return self._relative_guard(mean_return)
        floor = self.config.return_floor
        if floor is None or mean_return is None:
            return False
        if mean_return < floor:
            self.low_return_streak += 1
        else:
            self.low_return_streak = 0
        if self.low_return_streak >= self.config.low_return_patience:
            self.low_return_streak = 0
            return True
        return False

    def _relative_guard(self, mean_return: float | None) -> bool:
        """Trip on a fall relative to this run's own recent best.

        The reference is read *before* the new sample joins the window, so a
        single collapse cannot lower the bar it is being judged against. A
        window that is not yet full holds lambda's guard open: with fewer than
        two prior epochs there is no history to be worse than.
        """
        if mean_return is None:
            return False
        reference = self.return_reference
        self.return_window.append(float(mean_return))
        if reference is None or len(self.return_window) < 3 or reference <= 0.0:
            self.low_return_streak = 0
            return False
        if mean_return < (1.0 - self.config.return_relative_drop) * reference:
            self.low_return_streak += 1
        else:
            self.low_return_streak = 0
        if self.low_return_streak >= self.config.low_return_patience:
            self.low_return_streak = 0
            # Drop the window as well: after a deliberate back-off the old peak
            # is no longer the standard this policy should be held to, and
            # keeping it would re-trip the guard on the very next epoch.
            self.return_window.clear()
            if mean_return is not None:
                self.return_window.append(float(mean_return))
            return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "lambda_value": self.lambda_value,
            "integral": self.integral,
            "epoch": self.epoch,
            "low_return_streak": self.low_return_streak,
            "return_window": list(self.return_window),
            "latch_best_mean": self.latch_best_mean,
            "latch_returns": list(self.latch_returns),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lambda_value = float(state["lambda_value"])
        self.integral = float(state["integral"])
        self.epoch = int(state["epoch"])
        self.low_return_streak = int(state.get("low_return_streak", 0))
        self.return_window.clear()
        self.return_window.extend(float(v) for v in state.get("return_window", ()))
        self.latch_returns.clear()
        self.latch_returns.extend(float(v) for v in state.get("latch_returns", ()))
        best_mean = state.get("latch_best_mean")
        self.latch_best_mean = float(best_mean) if best_mean is not None else None
        # Compatibility with any state written by the short-lived development
        # implementation: recompute a like-for-like window statistic instead
        # of restoring its incomparable best single-iteration return.
        if self.latch_best_mean is None and len(self.latch_returns) >= self.config.latch_window:
            self.latch_best_mean = sum(self.latch_returns) / len(self.latch_returns)


def calibrate_target(
    nominal_gaps: Sequence[float], num_sigma: float = 3.0, quantile: float = 0.9
) -> float:
    """Set ``delta_target`` from nominal rollouts, as the manuscript specifies.

    The target is ``mu + num_sigma * sigma`` of the gap measured at
    ``lambda = 0``: a "stable tracking" reference for *this* policy on *this*
    encoder. A hand-picked constant would not transfer between encoders, whose
    latent scales differ, so the target must be calibrated rather than assumed.
    """
    values = [float(v) for v in nominal_gaps]
    if len(values) < 2:
        raise ValueError("need at least two nominal gap samples to calibrate a target")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean + num_sigma * (variance**0.5)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
