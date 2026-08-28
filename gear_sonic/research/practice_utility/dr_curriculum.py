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

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import observer as OBS
from gear_sonic.research.practice_utility import quality_telemetry as QT
from gear_sonic.research.practice_utility import tace as TACE
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
    ) -> None:
        if mode not in ("lucid", "fixed", "off", "yoked"):
            raise ValueError(
                f"unknown curriculum mode {mode!r}; expected lucid/fixed/off/yoked"
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
            str(k): float(v) for k, v in (dict(term_lambda_overrides) if term_lambda_overrides else {}).items()
        }
        for name, value in self.term_lambda_overrides.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'term_lambda_overrides[{name!r}] must be in [0, 1], got {value}')
        # A *cap* is the per-channel ceiling the scalar curriculum cannot
        # express: the channel follows lambda up to its own limit and no
        # further. An override pins a channel at a constant regardless of
        # lambda; a cap lets it still be scheduled. "Everything except
        # actuation latency" is a cap, not an override, and it is the shape
        # LUCID-MC needs if one channel turns out to carry the harm.
        self.term_lambda_caps = {
            str(k): float(v) for k, v in (dict(term_lambda_caps) if term_lambda_caps else {}).items()
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

        if mode == "lucid":
            starting_lambda = initial_lambda
        elif mode == "fixed":
            starting_lambda = fixed_lambda
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
            ),
            initial_lambda=starting_lambda,
        )
        self.baseline: dict[str, dict[str, Any]] | None = None
        self.scalable: list[str] = []
        self.history: list[dict[str, Any]] = []
        self._event_manager: Any = None
        self._env: Any = None

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
        self._apply(self.controller.lambda_value)
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
            self._apply(self.controller.lambda_value)
            record = {
                "global_step": step,
                "mode": self.mode,
                "lambda": self.controller.lambda_value,
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
            record = {
                "global_step": step,
                "mode": "fixed",
                "lambda": self.fixed_lambda,
                "gap_quantile": None,
            }
            self._apply(self.fixed_lambda)
        elif self.mode == "off":
            record = {"global_step": step, "mode": "off", "lambda": 0.0, "gap_quantile": None}
            self._apply(0.0)
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
        else:
            gaps = self._gaps()
            mean_return = self._mean_return(state)
            outcome = self.controller.update(gaps=gaps, mean_return=mean_return)
            self._apply(outcome.lambda_after)
            record = {
                "global_step": step,
                "mode": "lucid",
                "lambda": outcome.lambda_after,
                **outcome.to_dict(),
            }

        record["scalable_terms"] = self.scalable
        if self.term_lambda_overrides:
            record["term_lambda_overrides"] = self.term_lambda_overrides
        if self.term_lambda_caps:
            record["term_lambda_caps"] = self.term_lambda_caps
            record["realized_channel_lambdas"] = {
                name: self._channel_lambda(name, self.controller.lambda_value)
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

    def _bind(self, env: Any) -> None:
        manager = _event_manager_of(env)
        if manager is None:
            return
        self._event_manager = manager
        self.baseline = DS.capture_baseline(manager)
        self.scalable = DS.scalable_terms(manager)
        self._env = env
        if (self.anchor_ratio > 0.0 or self.spread_strata > 1) and self.assignment is None:
            num_envs = _num_envs_of(env, manager)
            if num_envs is None:
                raise RuntimeError("TACE needs the environment count to assign cohorts")
            reserved = self.anchor_reserved_focus_envs
            observer = OBS.get_active_observer(self.observer_branch_id)
            if observer is not None:
                reserved = tuple(sorted({*reserved, int(observer.tracked_env)}))
            seed = self.anchor_seed if self.anchor_seed is not None else 0
            self.assignment = TACE.assign_cohorts(
                num_envs, self.anchor_ratio, seed, reserved, num_strata=self.spread_strata
            )
            anchor_params, anchor_buckets = self._anchor_target(manager)
            self.dispatchers = TACE.install(
                manager, self.baseline, self.assignment, anchor_params, anchor_buckets
            )

    def _anchor_target(
        self, manager: Any
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
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
            scaled = DS.scaled_term_params(base, ceiling)
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
        if self._event_manager is None or self.baseline is None:
            return
        per_term = tuple(self.term_lambda_overrides) + tuple(self.term_lambda_caps)
        DS.apply_lambda(
            self._event_manager,
            self.baseline,
            lambda_value,
            exclude_terms=per_term,
        )
        for name in per_term:
            if name in self.baseline:
                DS.apply_lambda(
                    self._event_manager,
                    {name: self.baseline[name]},
                    self._channel_lambda(name, lambda_value),
                )
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

        weights = TACE.stratum_weights(self.spread_strata)
        for name, dispatch in self.dispatchers.items():
            base = self.baseline.get(name)
            if base is None:
                continue
            channel_lambda = self._channel_lambda(name, lambda_value)
            for index, weight in enumerate(weights):
                if index == len(weights) - 1:
                    dispatch.set_stratum(index, None)
                    continue
                params = DS.scaled_term_params(base, channel_lambda * weight)
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
            "stratum_lambdas": [
                self.controller.lambda_value * w
                for w in TACE.stratum_weights(self.assignment.num_strata)
            ],
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
            "controller": self.controller.state_dict(),
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
