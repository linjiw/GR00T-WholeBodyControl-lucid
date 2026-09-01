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

from gear_sonic.research.practice_utility import (
    dr_scaling as DS,
    observer as OBS,
    quality_telemetry as QT,
    tace as TACE,
)
from gear_sonic.research.practice_utility.dr_controller import (
    LucidDRController,
    PIConfig,
)

#: Active curriculum callbacks, so a checkpoint writer can capture their state
#: without Hydra having to wire the two callbacks together.
_ACTIVE_CURRICULA: dict[str, "LucidCurriculumCallback"] = {}


def register_curriculum(callback: "LucidCurriculumCallback") -> None:
    _ACTIVE_CURRICULA[callback.branch_id] = callback


def get_active_curriculum(branch_id: str | None = None) -> "LucidCurriculumCallback | None":
    if branch_id is not None:
        return _ACTIVE_CURRICULA.get(branch_id)
    if len(_ACTIVE_CURRICULA) == 1:
        return next(iter(_ACTIVE_CURRICULA.values()))
    return None


def clear_curricula() -> None:
    _ACTIVE_CURRICULA.clear()


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
        return_guard: str = "absolute",
        return_relative_drop: float = 0.25,
        return_window: int = 8,
        update_every: int = 1,
        warmup_iterations: int = 0,
        resume_state_path: str | None = None,
        max_lambda_step_on_resume: float | None = None,
        anchor_ratio: float = 0.0,
        anchor_seed: int | None = None,
        anchor_reserved_focus_envs: tuple[int, ...] | list[int] = (0,),
        consolidation_fraction: float = 0.0,
        yoked_schedule_path: str | None = None,
        term_lambda_overrides: dict[str, float] | None = None,
        term_lambda_caps: dict[str, float] | None = None,
        spread_strata: int = 1,
        stratum_sizes: tuple[int, ...] | list[int] | None = None,
        monotonic: bool = False,
        competence_latch: bool = False,
        latch_threshold: float = 0.95,
        latch_window: int = 500,
        signal: str = "gap",
        yardstick_envs: int = 0,
        margin_branch_id: str | None = None,
        allow_extrapolation: bool = False,
        survival_branch_id: str | None = None,
        gate_threshold: float = 0.80,
        gate_window: int = 200,
        gate_step: float = 0.125,
        gate_probe_offset: float = 0.125,
        gate_dwell: int = 200,
        gate_min_episodes: int = 200,
        gate_lambda_max: float = 1.5,
        gate_probe_max: float = 2.0,
        gate_guard_action: str = "freeze",
        gate_tail_fraction: float = 0.25,
        #: Where an expansion arm starts. Support expansion begins where fixed
        #: randomization already sits, so the arm's first difference from fixed
        #: DR is an expansion rather than a warm-up. Read instead of
        #: ``initial_lambda``, which every launcher passes as 0.0.
        gate_initial_frontier: float = 1.0,
        ramp_start_lambda: float = 1.0,
        ramp_end_lambda: float = 1.5,
        ramp_begin_iteration: int = 1000,
        ramp_end_iteration: int = 5000,
    ) -> None:
        if mode not in ("lucid", "fixed", "off", "yoked", "gate", "ramp"):
            raise ValueError(
                f"unknown curriculum mode {mode!r}; expected " "lucid/fixed/off/yoked/gate/ramp"
            )
        if not 0.0 <= float(anchor_ratio) <= 1.0:
            raise ValueError(f"anchor_ratio must be in [0, 1], got {anchor_ratio}")
        if not 0.0 <= float(consolidation_fraction) < 1.0:
            raise ValueError(
                f"consolidation_fraction must be in [0, 1), got {consolidation_fraction}"
            )
        if mode == "yoked" and not yoked_schedule_path:
            raise ValueError("mode='yoked' requires yoked_schedule_path")
        if int(spread_strata) < 1:
            raise ValueError(f"spread_strata must be >= 1, got {spread_strata}")
        if stratum_sizes is not None and len(tuple(stratum_sizes)) != int(spread_strata):
            raise ValueError(
                f"stratum_sizes has {len(tuple(stratum_sizes))} entries "
                f"for spread_strata={spread_strata}"
            )
        if signal not in ("gap", "margin"):
            raise ValueError(f"signal must be 'gap' or 'margin', got {signal!r}")
        if signal == "margin" and int(yardstick_envs) < 1:
            raise ValueError("signal='margin' needs yardstick_envs >= 1: the ratio is against them")
        # --- the controller input ------------------------------------------
        # "gap": the frozen-encoder latent gap from one tracked env (the
        # manuscript's signal). "margin": the termination margin from every
        # env, as a ratio against a yardstick cohort held at lambda = 0 in the
        # same run, with a dead band -- see margin_signal.py for why.
        self.signal = signal
        self.yardstick_envs = int(yardstick_envs)
        self.margin_branch_id = margin_branch_id
        # --- TACE: target-anchored exposure -------------------------------
        # A fixed cohort of environments always samples the full (lambda = 1)
        # envelope; the curriculum only moves the rest. alpha = 0 is the plain
        # curriculum, alpha = 1 is fixed DR under another name.
        self.anchor_ratio = float(anchor_ratio)
        # --- LUCID-S: expand the support, do not move a point ---------------
        # With K > 1 the focus cohort is split into K intensity strata, stratum
        # k training at lambda * (k+1)/K. The controller still moves one number;
        # what that number now sets is the *upper edge* of the training
        # mixture rather than its single value. The measured motivation is the
        # 32 -> 128 iteration collapse: every arm that trained at one intensity
        # lost 23-27 points of clean success, and the arm with the widest
        # realized intensity mixture (50% anchor) lost the least.
        self.spread_strata = int(spread_strata)
        # Explicit final stratum sizes (low first, top last, observer envs
        # counted in the top). None keeps the round-robin equal split that
        # every pre-existing arm trained with.
        self.stratum_sizes = (
            tuple(int(s) for s in stratum_sizes) if stratum_sizes is not None else None
        )
        self.anchor_seed = anchor_seed
        self.anchor_reserved_focus_envs = tuple(int(i) for i in anchor_reserved_focus_envs)
        self.consolidation_fraction = float(consolidation_fraction)
        self.assignment: TACE.CohortAssignment | None = None
        self.dispatchers: dict[str, TACE.CohortDispatch] = {}
        self._consolidating = False
        self.yoked_schedule_path = yoked_schedule_path
        # Per-term fixed intensities applied after the global lambda: a channel
        # attribution tool (e.g. everything at 1 except latency at 0).
        self.term_lambda_overrides = {
            str(k): float(v)
            for k, v in (dict(term_lambda_overrides) if term_lambda_overrides else {}).items()
        }
        for name, value in self.term_lambda_overrides.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"term_lambda_overrides[{name!r}] must be in [0, 1], got {value}")
        # A *cap* is the per-channel ceiling the scalar curriculum cannot
        # express: the channel follows lambda up to its own limit and no
        # further. An override pins a channel at a constant regardless of
        # lambda; a cap lets it still be scheduled. "Everything except
        # actuation latency" is a cap, not an override, and it is the shape
        # LUCID-MC needs if one channel turns out to carry the harm.
        self.term_lambda_caps = {
            str(k): float(v)
            for k, v in (dict(term_lambda_caps) if term_lambda_caps else {}).items()
        }
        for name, value in self.term_lambda_caps.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"term_lambda_caps[{name!r}] must be in [0, 1], got {value}")
            if name in self.term_lambda_overrides:
                raise ValueError(
                    f"term {name!r} has both an override and a cap; pick one -- an override "
                    "pins the channel and a cap schedules it, and applying both hides which won"
                )
        self._yoked_schedule: list[float] = []
        if mode == "yoked":
            self._yoked_schedule = _load_yoked_schedule(yoked_schedule_path)
        self.enabled = bool(enabled)
        self.mode = mode
        self.observer_branch_id = observer_branch_id
        self.output_dir = output_dir
        self.branch_id = branch_id
        self.fixed_lambda = float(fixed_lambda)
        self.allow_extrapolation = bool(allow_extrapolation)
        self._clamp_report: dict[str, Any] | None = None
        # The eval/training asymmetry: evaluation may score a policy outside the
        # envelope, but training crosses lambda = 1 only through this explicit
        # flag, and only for modes whose applied support cannot contract -- a
        # *bidirectional* controller that can also extrapolate is how a support
        # experiment becomes an uncontrolled one. The gap-driven "lucid" mode
        # stays hard-capped at 1 for exactly that reason; "gate" and "ramp" are
        # admitted because both are monotone by construction (see
        # survival_gate.py) and neither can lower applied support by its own
        # decision rule.
        if self.allow_extrapolation and mode not in ("fixed", "gate", "ramp"):
            raise ValueError(
                "allow_extrapolation is a support extension for the monotone "
                f"modes fixed/gate/ramp; mode {mode!r} stays hard-capped at lambda = 1"
            )
        if self.fixed_lambda > 1.0 and not self.allow_extrapolation:
            raise ValueError(
                f"fixed_lambda={self.fixed_lambda} is outside the training envelope; "
                "pass allow_extrapolation=true to extend support past lambda = 1 deliberately"
            )
        if not 0.0 <= self.fixed_lambda <= DS.MAX_EXTRAPOLATION:
            raise ValueError(
                f"fixed_lambda must be in [0, {DS.MAX_EXTRAPOLATION}], got {self.fixed_lambda}"
            )
        self.update_every = max(1, int(update_every))
        # After a (re)start the gap and return estimates are dominated by the
        # restart transient -- measured at up to 28% relative spread in the first
        # iterations. Reacting to that moves difficulty on noise, so hold lambda
        # until the run has settled.
        self.warmup_iterations = max(0, int(warmup_iterations))
        self.resume_state_path = resume_state_path
        self.max_lambda_step_on_resume = max_lambda_step_on_resume
        self._start_step: int | None = None
        self._resumed_from: dict[str, Any] | None = None

        # --- monotone support expansion (gate / ramp) -----------------------
        # Both modes drive a *frontier* rather than a point: the focus cohort is
        # split into a retained tail, a frontier stratum, and -- for the gate --
        # a probe stratum one step above the frontier. They share this entire
        # layout so that the only difference between the feedback arm and its
        # control is how the frontier moves, which is the comparison the ratchet
        # result showed the programme was missing.
        self.gate_probe_offset = float(gate_probe_offset)
        # Ceiling on the probe itself. Setting it equal to the frontier ceiling
        # makes the arm's MAXIMUM applied intensity equal to its frontier
        # ceiling, so an expansion arm and an open-loop arm at the same nominal
        # lambda train on the same maximum support. Without that, the probe
        # would carry a gate arm one step past every arm it is compared with,
        # and a frontier difference would be confounded with a support
        # difference. It also means the probe only ever explores levels the
        # frontier could actually reach.
        self.gate_probe_max = float(gate_probe_max)
        self.gate_tail_fraction = float(gate_tail_fraction)
        self.ramp_start_lambda = float(ramp_start_lambda)
        self.ramp_end_lambda = float(ramp_end_lambda)
        self.ramp_begin_iteration = int(ramp_begin_iteration)
        self.ramp_end_iteration = int(ramp_end_iteration)
        self.survival_branch_id = survival_branch_id
        self.gate: Any = None
        self._frontier_lambda: float | None = None
        #: Absolute per-stratum intensities for the current iteration, low
        #: first. When set, ``_apply_strata`` uses these instead of scaling the
        #: frontier by the TACE weights, which is what lets one stratum sit
        #: *above* the frontier.
        self._stratum_lambdas_absolute: tuple[float, ...] | None = None
        if mode in ("gate", "ramp"):
            if not 0.0 <= self.gate_tail_fraction < 1.0:
                raise ValueError(f"gate_tail_fraction must be in [0, 1), got {gate_tail_fraction}")
            ceiling = (
                gate_lambda_max
                if mode == "gate"
                else max(self.ramp_start_lambda, self.ramp_end_lambda)
            )
            if ceiling > 1.0 and not self.allow_extrapolation:
                raise ValueError(
                    f"mode {mode!r} targets lambda {ceiling} outside the training "
                    "envelope; pass allow_extrapolation=true to extend support "
                    "past lambda = 1 deliberately"
                )
            if self.spread_strata < 2:
                raise ValueError(
                    f"mode {mode!r} needs spread_strata >= 2: a frontier stratum "
                    "and, for the gate, a probe stratum above it"
                )
            if self.consolidation_fraction > 0.0:
                # Consolidation pins every cohort at lambda = 1.0 for the last
                # stretch of training. For an arm whose frontier is above 1.0
                # that is a support CONTRACTION -- applied silently, and
                # invisible to the gate's own incident log, because the gate
                # never asked for it. Refused before the run starts rather than
                # discovered 5 GPU-hours in.
                raise ValueError(
                    f"consolidation pins lambda at 1.0 and would contract the "
                    f"support of monotone mode {mode!r}; the two are incompatible"
                )
            if mode == "ramp" and self.ramp_end_lambda < self.ramp_start_lambda:
                raise ValueError(
                    "ramp_end_lambda must be >= ramp_start_lambda: the open-loop "
                    "control is monotone for the same reason the gate is"
                )
        if mode == "gate":
            from gear_sonic.research.practice_utility import survival_gate as SG

            self.gate = SG.SurvivalGateController(
                SG.SurvivalGateConfig(
                    threshold=float(gate_threshold),
                    window=int(gate_window),
                    step_size=float(gate_step),
                    probe_offset=self.gate_probe_offset,
                    dwell=int(gate_dwell),
                    min_episodes=int(gate_min_episodes),
                    lambda_max=float(gate_lambda_max),
                    probe_max=float(gate_probe_max),
                    return_relative_drop=float(return_relative_drop),
                    return_window=int(return_window),
                    guard_action=str(gate_guard_action),
                ),
                initial_lambda=float(min(gate_initial_frontier, gate_lambda_max)),
            )
            self._frontier_lambda = self.gate.frontier
        elif mode == "ramp":
            self._frontier_lambda = self.ramp_start_lambda

        if mode == "lucid":
            starting_lambda = initial_lambda
        elif mode in ("gate", "ramp"):
            # The PI controller is inert in these modes; keep its config
            # envelope-bound so no controller path can follow the frontier
            # above 1, exactly as fixed mode does.
            starting_lambda = min(float(self._frontier_lambda or 0.0), 1.0)
        elif mode == "fixed":
            # The controller is inert in fixed mode (_apply reads fixed_lambda
            # directly); its config stays envelope-bound even when a support
            # extension trains past 1, so no controller path can ever follow.
            starting_lambda = min(fixed_lambda, 1.0)
        elif mode == "yoked":
            starting_lambda = self._yoked_schedule[0] if self._yoked_schedule else initial_lambda
        else:
            starting_lambda = 0.0

        self.controller = LucidDRController(
            PIConfig(
                kp=kp,
                ki=ki,
                alpha=alpha,
                integral_max=integral_max,
                quantile=quantile,
                delta_target=delta_target,
                return_floor=return_floor,
                return_decay=return_decay,
                return_guard=return_guard,
                return_relative_drop=return_relative_drop,
                return_window=return_window,
                monotonic=bool(monotonic),
                competence_latch=bool(competence_latch),
                latch_threshold=float(latch_threshold),
                latch_window=int(latch_window),
            ),
            initial_lambda=starting_lambda,
        )
        self.baseline: dict[str, dict[str, Any]] | None = None
        self.scalable: list[str] = []
        self.history: list[dict[str, Any]] = []
        self._event_manager: Any = None
        self._env: Any = None
        # The controller deliberately remains capped at one even for a
        # fixed-mode support extension. Keep the dose actually sent to the
        # event manager separately so startup, warmup, and telemetry cannot
        # accidentally report/apply the controller's capped placeholder.
        if self.mode == "fixed":
            self._last_applied_lambda = float(self.fixed_lambda)
        elif self._frontier_lambda is not None:
            self._last_applied_lambda = float(self._frontier_lambda)
        else:
            self._last_applied_lambda = float(self.controller.lambda_value)

    # ------------------------------------------------------------ lifecycle --

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        self._bind(kwargs.get("env"))
        register_curriculum(self)
        # Restore before the first rollout. Without this a resumed run silently
        # restarts the curriculum at initial_lambda: the environment jumps back
        # to easy, the policy is briefly trained on a distribution it has already
        # outgrown, and the curriculum re-climbs from zero. Nothing errors, and
        # the learning curve simply looks worse than it should.
        if self.resume_state_path:
            self.load_state_file(self.resume_state_path)
        # Apply the starting intensity before the first rollout, so iteration 1
        # already trains under the curriculum rather than under whatever the
        # config happened to declare.
        self._apply(self._mode_lambda())
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if self._event_manager is None:
            self._bind(kwargs.get("env"))
        step = getattr(state, "global_step", 0) if state is not None else 0
        if self._start_step is None:
            self._start_step = step
        if step - self._start_step < self.warmup_iterations:
            # Hold, but keep applying, so the restored intensity is in force.
            warmup_lambda = self._mode_lambda()
            self._apply(warmup_lambda)
            record = {
                "global_step": step,
                "mode": self.mode,
                "lambda": warmup_lambda,
                "gap_quantile": None,
                "warmup_hold": True,
                "scalable_terms": self.scalable,
            }
            self.history.append(record)
            self._write(record)
            self.save_state_file()
            return control
        if step % self.update_every:
            return control

        if self._enter_consolidation_if_due(args, state, step):
            record = {
                "global_step": step,
                "mode": self.mode,
                "lambda": 1.0,
                "gap_quantile": None,
                "consolidation": True,
            }
            self._apply(1.0)
        elif self.mode == "fixed":
            self._apply(self.fixed_lambda)
            record = {
                "global_step": step,
                "mode": "fixed",
                "lambda": self.fixed_lambda,
                "gap_quantile": None,
            }
            if self.allow_extrapolation:
                record["allow_extrapolation"] = True
                record["physical_clamp"] = sorted((self._clamp_report or {}).get("clamped", {}))
        elif self.mode == "off":
            record = {"global_step": step, "mode": "off", "lambda": 0.0, "gap_quantile": None}
            self._apply(0.0)
        elif self.mode == "gate":
            observer = self._survival_observer()
            mean_return = self._mean_return(state)
            probe_survival, probe_episodes = (None, 0)
            if observer is not None:
                observer.ensure_flushed(step)
                probe_survival, probe_episodes = observer.current_probe()
            outcome = self.gate.update(
                probe_survival=probe_survival,
                probe_episodes=probe_episodes,
                mean_return=mean_return,
            )
            self._frontier_lambda = outcome.frontier_after
            self._apply(outcome.frontier_after)
            record = {
                "global_step": step,
                "mode": "gate",
                "signal": "survival",
                # ``lambda`` stays the frontier for every downstream reader that
                # already parses this field as "the intensity the arm trains at".
                "lambda": outcome.frontier_after,
                "gap_quantile": None,
                "frontier_lambda": outcome.frontier_after,
                "population_survival": (
                    observer.current_population() if observer is not None else None
                ),
                "survival_observer_present": observer is not None,
                "allow_extrapolation": self.allow_extrapolation,
                "physical_clamp": sorted((self._clamp_report or {}).get("clamped", {})),
                **outcome.to_dict(),
            }
        elif self.mode == "ramp":
            from gear_sonic.research.practice_utility import survival_gate as SG

            observer = self._survival_observer()
            if observer is not None:
                observer.ensure_flushed(step)
            index = step - (self._start_step or 0)
            frontier = SG.linear_ramp_lambda(
                index,
                start_lambda=self.ramp_start_lambda,
                end_lambda=self.ramp_end_lambda,
                begin_iteration=self.ramp_begin_iteration,
                end_iteration=self.ramp_end_iteration,
            )
            self._frontier_lambda = frontier
            self._apply(frontier)
            probe_survival, probe_episodes = (
                observer.current_probe() if observer is not None else (None, 0)
            )
            record = {
                "global_step": step,
                "mode": "ramp",
                # Open loop: the schedule reads nothing. Probe survival is
                # recorded at the same place the gate would read it, so the two
                # arms produce comparable telemetry and the control can be
                # checked to have gated on nothing.
                "signal": "none",
                "lambda": frontier,
                "gap_quantile": None,
                "frontier_lambda": frontier,
                "probe_lambda": frontier + self.gate_probe_offset,
                "probe_survival": probe_survival,
                "probe_episodes": probe_episodes,
                "population_survival": (
                    observer.current_population() if observer is not None else None
                ),
                "ramp_index": index,
                "mean_return": self._mean_return(state),
                "allow_extrapolation": self.allow_extrapolation,
                "physical_clamp": sorted((self._clamp_report or {}).get("clamped", {})),
            }
        elif self.mode == "yoked":
            # Replay a recorded lambda trajectory, indexed by iteration since
            # (re)start, with no feedback. The attribution control for TA-LUCID:
            # same schedule shape and dose, minus the online gap.
            index = step - (self._start_step or 0)
            lam = self._yoked_lambda(index)
            self.controller.lambda_value = lam
            self._apply(lam)
            record = {
                "global_step": step,
                "mode": "yoked",
                "lambda": lam,
                "gap_quantile": None,
                "yoked_index": index,
                "yoked_schedule_length": len(self._yoked_schedule),
            }
        elif self.signal == "margin":
            observer = self._margin_observer()
            mean_return = self._mean_return(state)
            if observer is None:
                # No signal is not "raise the dose": hold.
                outcome = self.controller.update_with_error(0.0, mean_return=mean_return)
                ratio = None
            else:
                observer.ensure_flushed(step)
                outcome = self.controller.update_with_error(
                    observer.current_error(), mean_return=mean_return
                )
                ratio = observer.current_ratio()
            self._apply(outcome.lambda_after)
            record = {
                "global_step": step,
                "mode": "lucid",
                "signal": "margin",
                "lambda": outcome.lambda_after,
                "margin_ratio": ratio,
                "margin_observer_present": observer is not None,
                **outcome.to_dict(),
            }
        else:
            gaps = self._gaps()
            mean_return = self._mean_return(state)
            outcome = self.controller.update(gaps=gaps, mean_return=mean_return)
            self._apply(outcome.lambda_after)
            record = {
                "global_step": step,
                "mode": "lucid",
                "signal": "gap",
                "lambda": outcome.lambda_after,
                **outcome.to_dict(),
            }

        record["scalable_terms"] = self.scalable
        if self.term_lambda_overrides:
            record["term_lambda_overrides"] = self.term_lambda_overrides
        if self.term_lambda_caps:
            record["term_lambda_caps"] = self.term_lambda_caps
            record["realized_channel_lambdas"] = {
                name: self._channel_lambda(name, self._last_applied_lambda)
                for name in sorted(self.term_lambda_caps)
            }
        if self._resumed_from is not None:
            record["resumed_from"] = self._resumed_from
        if self.assignment is not None:
            record["tace"] = self._tace_telemetry()
        self.history.append(record)
        self._write(record)
        # Persist every update, so a run killed between checkpoints still
        # resumes with the curriculum it had rather than with lambda = 0.
        self.save_state_file()
        return control

    # ------------------------------------------------------------- internals --

    def _mode_lambda(self) -> float:
        """Intensity in force before a feedback update for the current mode."""
        if self.mode == "fixed":
            return self.fixed_lambda
        if self.mode == "off":
            return 0.0
        if self._frontier_lambda is not None:
            return float(self._frontier_lambda)
        return self.controller.lambda_value

    # ------------------------------------------------- support expansion --

    def _survival_observer(self):
        if self.mode not in ("gate", "ramp"):
            return None
        from gear_sonic.research.practice_utility import survival_observer as SO

        return SO.get_active_survival_observer(self.survival_branch_id or self.branch_id)

    def _probe_stratum_index(self) -> int | None:
        """The stratum held above the frontier: always the top one.

        Present in both gate and ramp so the two arms allocate environments
        identically. In ramp mode it is measured and logged but gates nothing,
        which is what makes the pair a clean attribution control.
        """
        if self.mode not in ("gate", "ramp") or self.spread_strata < 2:
            return None
        return self.spread_strata - 1

    def _expansion_stratum_lambdas(self, frontier: float) -> tuple[float, ...]:
        """Absolute intensity per stratum: retained tail, frontier, probe.

        Support is *expanded*, never moved. The tail strata keep sampling the
        intensities the policy already trained on -- the campaign measured a
        collapsed arm scoring below its own mid-training capsule, so retention
        is not free -- while the frontier stratum carries the mass and the
        probe sits one step beyond it.
        """
        count = self.spread_strata
        probe_index = self._probe_stratum_index()
        if probe_index is None:
            return tuple(frontier * w for w in TACE.stratum_weights(count))
        # The probe is derived from the frontier actually being applied, never
        # from the gate's own copy of it: the two can differ during warm-up or a
        # resume, and taking the gate's copy there would place the probe *below*
        # the frontier and quietly delete the stratum the whole design depends
        # on. Capped at gate_probe_max, which both modes share so that a gate
        # arm and its open-loop control have identical maximum support.
        probe = min(frontier + self.gate_probe_offset, self.gate_probe_max)
        probe = max(float(probe), float(frontier))
        # Strata 0 .. count-3 span the retained tail, stratum count-2 is the
        # frontier itself, stratum count-1 is the probe.
        tail = count - 2
        lambdas: list[float] = []
        for index in range(tail):
            # Evenly spaced across (0, frontier), never including 0: a stratum
            # at exactly zero is the "off" arm, which is a different question.
            lambdas.append(frontier * float(index + 1) / float(tail + 1))
        lambdas.append(float(frontier))
        lambdas.append(float(probe))
        return tuple(lambdas)

    def _bind(self, env: Any) -> None:
        manager = _event_manager_of(env)
        if manager is None:
            return
        self._event_manager = manager
        self.baseline = DS.capture_baseline(manager)
        self.scalable = DS.scalable_terms(manager)
        self._env = env
        if (
            self.anchor_ratio > 0.0 or self.spread_strata > 1 or self.yardstick_envs > 0
        ) and self.assignment is None:
            num_envs = _num_envs_of(env, manager)
            if num_envs is None:
                raise RuntimeError("TACE needs the environment count to assign cohorts")
            reserved = self.anchor_reserved_focus_envs
            observer = OBS.get_active_observer(self.observer_branch_id)
            if observer is not None:
                reserved = tuple(sorted({*reserved, int(observer.tracked_env)}))
            seed = self.anchor_seed if self.anchor_seed is not None else 0
            self.assignment = TACE.assign_cohorts(
                num_envs,
                self.anchor_ratio,
                seed,
                reserved,
                num_strata=self.spread_strata,
                num_yardstick=self.yardstick_envs,
                stratum_sizes=self.stratum_sizes,
            )
            anchor_params, anchor_buckets = self._anchor_target(manager)
            self.dispatchers = TACE.install(
                manager, self.baseline, self.assignment, anchor_params, anchor_buckets
            )
            if self.yardstick_envs > 0:
                self._install_yardstick()
            margin = self._margin_observer()
            if margin is not None:
                margin.set_cohorts(self.assignment.focus_mask(), self.assignment.yardstick_mask())
            survival = self._survival_observer()
            if survival is not None:
                # Attribution is by environment id, fixed for the run, so the
                # gate can read the probe cohort apart from the frontier.
                survival.set_strata(self.assignment.stratum_masks(), self._probe_stratum_index())

    def _install_yardstick(self) -> None:
        """Hold the yardstick cohort at lambda = 0 on every dispatched term."""
        from gear_sonic.research.practice_utility import events_reset_safe as ERS

        assert self.assignment is not None and self.baseline is not None
        mask = self.assignment.yardstick_mask()
        for name, dispatch in self.dispatchers.items():
            base = self.baseline.get(name)
            if base is None:
                continue
            params = DS.scaled_term_params(base, 0.0)
            buckets = None
            if any(key in params for key in DS.MATERIAL_RANGE_KEYS):
                buckets = ERS.draw_material_buckets(
                    dispatch.inner,
                    static_friction_range=DS._as_pair(params.get("static_friction_range")),
                    dynamic_friction_range=DS._as_pair(params.get("dynamic_friction_range")),
                    restitution_range=DS._as_pair(params.get("restitution_range")),
                )
            dispatch.set_yardstick(mask, params, buckets)

    def _margin_observer(self):
        if self.signal != "margin":
            return None
        from gear_sonic.research.practice_utility import margin_observer as MO

        return MO.get_active_margin_observer(self.margin_branch_id or self.branch_id)

    def _anchor_target(self, manager: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """The envelope the anchor cohort should sample: *this arm's* target.

        For an unrestricted channel that is the captured baseline. For a channel
        this arm pins or caps, it is the baseline scaled to that channel's own
        ceiling -- otherwise the anchor would train half the population on
        exposure the arm was defined to withhold, and the pin or cap would be a
        claim about only half the run.
        """
        from gear_sonic.research.practice_utility import events_reset_safe as ERS

        params: dict[str, dict[str, Any]] = {}
        buckets: dict[str, Any] = {}
        if self.baseline is None:
            return params, buckets
        ceilings = {**self.term_lambda_overrides, **self.term_lambda_caps}
        if not ceilings:
            return params, buckets
        terms = dict(DS._iter_terms(manager))
        for name, ceiling in ceilings.items():
            base = self.baseline.get(name)
            if base is None:
                continue
            scaled = DS.scaled_term_params(base, ceiling, self.allow_extrapolation)
            if self.allow_extrapolation:
                scaled, _ = DS.clamp_params_physical(scaled)
            params[name] = scaled
            cfg = terms.get(name)
            func = getattr(cfg, "func", None) if cfg is not None else None
            if func is None or not any(key in scaled for key in DS.MATERIAL_RANGE_KEYS):
                continue
            drawn = ERS.draw_material_buckets(
                func,
                static_friction_range=DS._as_pair(scaled.get("static_friction_range")),
                dynamic_friction_range=DS._as_pair(scaled.get("dynamic_friction_range")),
                restitution_range=DS._as_pair(scaled.get("restitution_range")),
            )
            if drawn is not None:
                buckets[name] = drawn
        return params, buckets

    def _channel_lambda(self, name: str, lambda_value: float) -> float:
        """The intensity one channel actually runs at, after overrides and caps."""
        if name in self.term_lambda_overrides:
            return self.term_lambda_overrides[name]
        cap = self.term_lambda_caps.get(name)
        return lambda_value if cap is None else min(lambda_value, cap)

    def _apply(self, lambda_value: float) -> None:
        self._last_applied_lambda = float(lambda_value)
        if self._event_manager is None or self.baseline is None:
            return
        per_term = tuple(self.term_lambda_overrides) + tuple(self.term_lambda_caps)
        DS.apply_lambda(
            self._event_manager,
            self.baseline,
            lambda_value,
            exclude_terms=per_term,
            allow_extrapolation=self.allow_extrapolation,
        )
        for name in per_term:
            if name in self.baseline:
                DS.apply_lambda(
                    self._event_manager,
                    {name: self.baseline[name]},
                    self._channel_lambda(name, lambda_value),
                    allow_extrapolation=self.allow_extrapolation,
                )
        if self.allow_extrapolation:
            # Affine extension past lambda = 1 can leave the physically valid
            # region (friction went negative at 1.5x before the eval-side clamp
            # existed); clamp the live config and keep the report for the record.
            self._clamp_report = DS.clamp_physical(self._event_manager)
        if self.mode in ("gate", "ramp") and self.spread_strata > 1:
            self._stratum_lambdas_absolute = self._expansion_stratum_lambdas(float(lambda_value))
        self._apply_strata(lambda_value)

    def _apply_strata(self, lambda_value: float) -> None:
        """Give each focus stratum below the frontier its own share of lambda.

        The top stratum is deliberately left at ``None``: it samples from the
        event manager's own params, which ``apply_lambda`` has just written.
        That is what makes ``spread_strata = 1`` identical to the curriculum
        before strata existed -- the difference is additive, never a rewrite of
        the frontier the controller believes it set.
        """
        if self.spread_strata <= 1 or not self.dispatchers or self.baseline is None:
            return
        from gear_sonic.research.practice_utility import events_reset_safe as ERS

        absolute = self._stratum_lambdas_absolute
        weights = TACE.stratum_weights(self.spread_strata)
        for name, dispatch in self.dispatchers.items():
            base = self.baseline.get(name)
            if base is None:
                continue
            channel_lambda = self._channel_lambda(name, lambda_value)
            for index, weight in enumerate(weights):
                if absolute is None and index == len(weights) - 1:
                    # The top stratum samples the event manager's own params,
                    # which apply_lambda has just written. That is what makes
                    # spread_strata = 1 identical to the pre-strata curriculum,
                    # so it is preserved for every arm that does not place a
                    # stratum above the frontier.
                    dispatch.set_stratum(index, None)
                    continue
                if absolute is not None:
                    # A stratum may sit above the frontier, so the intensity is
                    # absolute rather than a fraction of it. Channel caps and
                    # overrides still apply, per channel, at that intensity.
                    stratum_lambda = self._channel_lambda(name, float(absolute[index]))
                else:
                    stratum_lambda = channel_lambda * weight
                params = DS.scaled_term_params(base, stratum_lambda, self.allow_extrapolation)
                if self.allow_extrapolation:
                    params, _ = DS.clamp_params_physical(params)
                buckets = None
                if any(key in params for key in DS.MATERIAL_RANGE_KEYS):
                    buckets = ERS.draw_material_buckets(
                        dispatch.inner,
                        static_friction_range=DS._as_pair(params.get("static_friction_range")),
                        dynamic_friction_range=DS._as_pair(params.get("dynamic_friction_range")),
                        restitution_range=DS._as_pair(params.get("restitution_range")),
                    )
                dispatch.set_stratum(index, params, buckets)

    def _gaps(self) -> list[float]:
        observer = OBS.get_active_observer(self.observer_branch_id)
        return observer.drain_gaps() if observer is not None else []

    @staticmethod
    def _mean_return(state: Any) -> float | None:
        """Most recent mean reward from the trainer's log history, if present."""
        history = getattr(state, "log_history", None) or []
        for entry in reversed(history):
            for key in (
                "objective/rewards",
                "mean_reward",
                "Mean rewards",
                "rewards/mean",
                "train/mean_reward",
            ):
                if key in entry:
                    try:
                        return float(entry[key])
                    except (TypeError, ValueError):
                        continue
        return None

    # ------------------------------------------------------------------ TACE --

    def _enter_consolidation_if_due(self, args: Any, state: Any, step: int) -> bool:
        """Final target-only phase: every cohort on the full envelope."""
        if self.consolidation_fraction <= 0.0:
            return False
        if self.mode in ("gate", "ramp"):
            # Backstop only: the constructor already refuses this combination,
            # so reaching here means consolidation_fraction was mutated after
            # construction. Raising is still correct -- see the constructor.
            raise RuntimeError(
                "consolidation pins lambda at 1.0 and would contract the support "
                f"of monotone mode {self.mode!r}; the two are incompatible"
            )
        if self._consolidating:
            return True
        total = None
        for source in (state, args):
            value = getattr(source, "max_steps", None) if source is not None else None
            if isinstance(value, (int, float)) and value > 0:
                total = float(value)
                break
        if total is None:
            return False
        if step >= total * (1.0 - self.consolidation_fraction):
            self._consolidating = True
            for dispatch in self.dispatchers.values():
                dispatch.all_envs_mode = True
            return True
        return False

    def _yoked_lambda(self, index: int) -> float:
        if not self._yoked_schedule:
            return self.controller.lambda_value
        index = max(0, min(int(index), len(self._yoked_schedule) - 1))
        return float(self._yoked_schedule[index])

    def _tace_telemetry(self) -> dict[str, Any]:
        assert self.assignment is not None
        out: dict[str, Any] = {
            "num_anchor": self.assignment.num_anchor,
            "num_focus": self.assignment.num_focus,
            "anchor_ratio": self.assignment.anchor_ratio,
            "num_strata": self.assignment.num_strata,
            "stratum_sizes": [len(ids) for ids in self.assignment.focus_strata],
            "stratum_lambdas": (
                list(self._stratum_lambdas_absolute)
                if self._stratum_lambdas_absolute is not None
                else [
                    self._last_applied_lambda * w
                    for w in TACE.stratum_weights(self.assignment.num_strata)
                ]
            ),
            "probe_stratum": self._probe_stratum_index(),
            "consolidating": self._consolidating,
            "dispatch": {name: d.telemetry() for name, d in self.dispatchers.items()},
        }
        robot = QT._scene_entity(self._env, "robot") if self._env is not None else None
        if robot is not None:
            out.update(
                TACE.cohort_delay_stats(
                    robot,
                    self.assignment.mask(),
                    self.assignment.stratum_masks() if self.spread_strata > 1 else (),
                )
            )
        return out

    # ------------------------------------------------------------ persistence --

    def state_dict(self) -> dict[str, Any]:
        """Curriculum state that must ride along with a checkpoint."""
        return {
            "schema_version": 1,
            "mode": self.mode,
            "branch_id": self.branch_id,
            "spread_strata": self.spread_strata,
            "signal": self.signal,
            "yardstick_envs": self.yardstick_envs,
            "controller": self.controller.state_dict(),
            "gate": self.gate.state_dict() if self.gate is not None else None,
            "frontier_lambda": self._frontier_lambda,
            "scalable_terms": self.scalable,
            "term_lambda_overrides": self.term_lambda_overrides,
            "term_lambda_caps": self.term_lambda_caps,
            "tace": self.assignment.to_dict() if self.assignment is not None else None,
            "consolidating": self._consolidating,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore lambda and the integral, optionally rate-limiting the jump.

        ``max_lambda_step_on_resume`` exists for the case where the restored
        intensity does not match the environment the run is about to train in --
        a changed config, a different pool, a hand-edited lambda. Jumping
        straight there reintroduces exactly the shock the curriculum is meant to
        avoid, so the move can be capped and closed over subsequent epochs.
        """
        gate_state = state.get("gate")
        if gate_state and self.gate is not None:
            # Restored as-is and never rolled back: a resume that lowered
            # applied support would be the failure this arm exists to delete.
            self.gate.load_state_dict(gate_state)
            self._frontier_lambda = self.gate.frontier
            self._resumed_from = {
                "frontier": self.gate.frontier,
                "expansions": self.gate.expansions,
                "iteration": self.gate.iteration,
            }
        elif state.get("frontier_lambda") is not None and self.mode == "ramp":
            self._frontier_lambda = float(state["frontier_lambda"])
        controller_state = state.get("controller")
        if not controller_state:
            return
        target = float(controller_state.get("lambda_value", self.controller.lambda_value))
        self.controller.load_state_dict(controller_state)
        if self.max_lambda_step_on_resume is not None:
            current = float(self.controller.config.lambda_min)
            capped = min(target, current + abs(self.max_lambda_step_on_resume))
            self.controller.lambda_value = capped
        self._resumed_from = {
            "lambda": target,
            "applied_lambda": self.controller.lambda_value,
            "epoch": self.controller.epoch,
        }

    def save_state_file(self, path: str | None = None) -> str | None:
        target = (
            Path(path)
            if path
            else (
                Path(self.output_dir) / f"curriculum_state_{self.branch_id}.json"
                if self.output_dir
                else None
            )
        )
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_dict(), indent=2, default=str))
        return str(target)

    def load_state_file(self, path: str) -> bool:
        source = Path(path)
        if not source.exists():
            return False
        self.load_state_dict(json.loads(source.read_text()))
        return True

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


def _num_envs_of(env: Any, manager: Any) -> int | None:
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None), manager):
        if candidate is None:
            continue
        scene = getattr(candidate, "scene", None)
        for holder in (scene, candidate):
            value = getattr(holder, "num_envs", None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _load_yoked_schedule(path: str | None) -> list[float]:
    """Lambda per iteration from a curriculum jsonl written by a lucid run."""
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"yoked schedule not found: {path}")
    values: list[float] = []
    for line in source.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "lambda" in record:
            values.append(float(record["lambda"]))
    if not values:
        raise ValueError(f"yoked schedule has no lambda records: {path}")
    return values
