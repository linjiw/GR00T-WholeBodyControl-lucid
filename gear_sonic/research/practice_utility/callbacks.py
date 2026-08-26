"""Trainer callbacks that wire practice-utility measurement into a SONIC run.

Two callbacks, both strict no-ops when disabled:

:class:`PracticeContextCallback`
    Installs a :class:`PracticeSamplerAdapter` onto the live motion library,
    optionally arms an intervention, and writes the realized-dose receipt. It
    wraps three methods across two live objects:
    ``update_adaptive_sampling_probabilities`` for the optional intervention
    and ``sample_motion_ids_and_time_steps`` for draw accounting on the motion
    library, plus ``ManagerEnvWrapper.step`` for passive completed-step
    accounting. Every wrapper calls the native method unchanged and preserves
    its return; only the probability wrapper post-processes native output when
    an intervention is armed. The step wrapper freezes the transition's motion
    context before IsaacLab can reset and resample terminated environments.

:class:`PracticeCapsuleCallback`
    Saves branch capsules at the horizon checkpoints, carrying the RNG state and
    sampler-to-pool binding that ``ModelSaveCallback`` does not.

``PracticeContextCallback`` re-arms after every motion resample. SONIC
periodically reloads the resident motion batch, which invalidates a kernel built
over the previous batch; the adapter detects a stale kernel and raises rather
than silently misattributing dose, so the callback must re-arm at that point.

Configured through Hydra like every other SONIC callback::

    practice_context:
      _target_: gear_sonic.research.practice_utility.callbacks.PracticeContextCallback
      enabled: true
      role: intervention
      pair_id: pair_017
      context_id: 3ba90f8b99da380f
      epsilon: 0.10
"""

# Ruff's import sorter conflicts with the repository's authoritative isort profile.
# ruff: noqa: I001

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from gear_sonic.research.practice_utility import dose_plan as DP
from gear_sonic.research.practice_utility.branch_capsule import (
    Provenance,
    load_capsule,
    save_capsule,
)
from gear_sonic.research.practice_utility.rng_capsule import RngState
from gear_sonic.research.practice_utility.sampler_adapter import PracticeSamplerAdapter
from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    MotionPoolManifest,
    sha256_of,
)

try:  # pragma: no cover - transformers is present in the training env
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - keeps CPU tests import-light

    class TrainerCallback:  # type: ignore[no-redef]
        """Minimal stand-in so this module imports without transformers."""


class PracticeContextCallback(TrainerCallback):
    """Install the sampler adapter and, optionally, an intervention."""

    def __init__(
        self,
        enabled: bool = False,
        role: str = "control",
        pair_id: str = "unbound",
        branch_id: str | None = None,
        context: dict[str, Any] | None = None,
        epsilon: float = 0.0,
        kernel_radius_bins: int = 1,
        dose_report_dir: str | None = None,
        dose_report_frequency: int = 0,
        manifest_path: str | None = None,
        snapshot_path: str | None = None,
        snapshot_at_step: int = 0,
        snapshot_timeline_fps: float = 50.0,
        claim_mode: bool = False,
        dose_plan_path: str | None = None,
        dose_plan_sha256: str | None = None,
        dose_plan_stage: str | None = None,
        dose_report_horizons: dict[str, int] | None = None,
        dose_origin_global_step: int | None = None,
        dose_num_steps_per_iteration: int | None = None,
        dose_num_envs: int | None = None,
        dose_lineage: dict[str, str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.role = role
        self.pair_id = pair_id
        self.branch_id = branch_id or f"{pair_id}_{role}"
        self.context = ContextKey.from_dict(context) if context else None
        self.epsilon = float(epsilon)
        self.kernel_radius_bins = int(kernel_radius_bins)
        self.dose_report_dir = dose_report_dir
        self.dose_report_frequency = int(dose_report_frequency)
        self.manifest_path = manifest_path
        self.snapshot_path = snapshot_path
        # Snapshotting at install captures the sampler's *prior*, not its
        # statistics: adp_samp_num_episodes and num_failures both start at
        # init_num_failures, so every failure rate reads exactly 1.0 and carries
        # no information. A campaign stratified on that would have no
        # difficulty axis at all. Snapshot after warm-up instead.
        self.snapshot_at_step = int(snapshot_at_step)
        self.snapshot_timeline_fps = float(snapshot_timeline_fps)
        if self.snapshot_timeline_fps <= 0:
            raise ValueError("snapshot_timeline_fps must be positive")
        self._verified_snapshot_timeline_fps: float | None = None
        self._snapshot_written = False
        self.claim_mode = bool(claim_mode)
        self.dose_plan_path = dose_plan_path
        self.dose_plan_sha256 = dose_plan_sha256
        self.dose_plan_stage = dose_plan_stage
        self.dose_report_horizons = {
            str(label): int(step) for label, step in (dose_report_horizons or {}).items()
        }
        if any(step <= 0 for step in self.dose_report_horizons.values()):
            raise ValueError("dose_report_horizons must contain positive absolute steps")
        self.dose_origin_global_step = (
            int(dose_origin_global_step) if dose_origin_global_step is not None else None
        )
        self.dose_num_steps_per_iteration = (
            int(dose_num_steps_per_iteration) if dose_num_steps_per_iteration is not None else None
        )
        self.dose_num_envs = int(dose_num_envs) if dose_num_envs is not None else None
        self.dose_lineage = dict(dose_lineage or {})
        self._dose_horizons_written: set[str] = set()
        self._dose_errors: list[str] = []
        self._passive_dose_plan: DP.PassiveDosePlan | None = None
        # A context is only armable while its motion is resident. SONIC keeps a
        # subset of the pool loaded (195 of 512 motions in a measured run), so a
        # branch whose context is absent at install must wait for a resample
        # rather than die -- otherwise most of a campaign never starts.
        self._arm_attempts = 0
        self._armed_steps = 0
        self._first_armed_step: int | None = None

        self.adapter: PracticeSamplerAdapter | None = None
        self._env: Any = None
        self._motion_lib: Any = None
        self._original_update: Any = None
        self._original_sample: Any = None
        self._original_env_step: Any = None
        self._original_env_step_instance: tuple[bool, Any] = (False, None)
        self._patched_attributes: list[str] = []
        self._original_instance_attributes: dict[str, tuple[bool, Any]] = {}
        self._armed = False

        if self.enabled and self.role == "intervention" and self.context is None:
            raise ValueError(
                "an intervention branch requires a context; without one the branch "
                "is a control and must be labelled as such"
            )
        if self.claim_mode and not self.dose_plan_path:
            raise ValueError("claim-mode passive dose requires dose_plan_path")
        if self.claim_mode and not self.dose_plan_sha256:
            raise ValueError("claim-mode passive dose requires dose_plan_sha256")
        if self.claim_mode and not self.dose_report_horizons:
            raise ValueError("claim-mode passive dose requires exact dose_report_horizons")
        if self.claim_mode and self.dose_plan_stage is None:
            raise ValueError("claim-mode passive dose requires dose_plan_stage")
        if self.claim_mode and self.dose_origin_global_step is None:
            raise ValueError("claim-mode passive dose requires dose_origin_global_step")
        if self.claim_mode and (
            self.dose_num_steps_per_iteration is None or self.dose_num_steps_per_iteration <= 0
        ):
            raise ValueError(
                "claim-mode passive dose requires positive dose_num_steps_per_iteration"
            )
        if self.claim_mode and (self.dose_num_envs is None or self.dose_num_envs <= 0):
            raise ValueError("claim-mode passive dose requires positive dose_num_envs")
        if self.claim_mode:
            required_lineage = ("campaign_id", "manifest_sha256", "manifest_file_sha256")
            missing = [key for key in required_lineage if not self.dose_lineage.get(key)]
            if missing:
                raise ValueError(
                    "claim-mode passive dose requires lineage fields: " + ", ".join(missing)
                )

    # ------------------------------------------------------------ lifecycle --

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.enabled:
            self._install(kwargs.get("env"))
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if self.adapter is None:
            self._install(kwargs.get("env"))
        step = getattr(state, "global_step", 0) if state is not None else 0
        # A motion resample swaps the resident batch out from under the kernel,
        # and may also bring an absent context into residence.
        self._rearm_if_stale()
        if self.context is not None and not self._armed:
            self._arm(step)
        if self._armed:
            self._armed_steps += 1
        if self.snapshot_path and not self._snapshot_written and step >= self.snapshot_at_step > 0:
            self.write_snapshot(step)
            self._snapshot_written = True
        exact_report_written = False
        for label, horizon in sorted(
            self.dose_report_horizons.items(), key=lambda item: (item[1], item[0])
        ):
            if step == horizon and label not in self._dose_horizons_written:
                self.write_dose_report(step, horizon_label=label)
                self._dose_horizons_written.add(label)
                exact_report_written = True
            elif self.claim_mode and step > horizon and label not in self._dose_horizons_written:
                raise RuntimeError(
                    f"claim-mode passive dose missed exact horizon {label} at step {horizon}"
                )
        if (
            self.dose_report_frequency
            and self.dose_report_dir
            and state is not None
            and getattr(state, "global_step", 0) % self.dose_report_frequency == 0
            and not exact_report_written
        ):
            self.write_dose_report(getattr(state, "global_step", 0))
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        try:
            if self.enabled and self.dose_report_dir:
                step = getattr(state, "global_step", 0)
                for label, horizon in sorted(self.dose_report_horizons.items()):
                    if horizon == step and label not in self._dose_horizons_written:
                        self.write_dose_report(step, horizon_label=label)
                        self._dose_horizons_written.add(label)
                if not self.claim_mode and not any(
                    horizon == step for horizon in self.dose_report_horizons.values()
                ):
                    self.write_dose_report(step)
                missing = sorted(set(self.dose_report_horizons) - self._dose_horizons_written)
                if self.claim_mode and missing:
                    raise RuntimeError(
                        "claim-mode passive dose ended without exact horizon receipts: "
                        + ", ".join(missing)
                    )
        finally:
            self.uninstall()
        return control

    # -------------------------------------------------------------- install --

    def _install(self, env: Any) -> None:
        motion_lib = _motion_lib_of(env)
        if motion_lib is None:
            raise RuntimeError(
                "no motion library on the environment; practice-utility measurement "
                "requires SONIC's adaptive motion sampler"
            )
        if not getattr(motion_lib, "use_adaptive_sampling", False):
            raise RuntimeError(
                "adaptive sampling is disabled; there is no bin distribution to "
                "intervene on and dose cannot be attributed to a context"
            )
        if self.snapshot_path:
            self._verified_snapshot_timeline_fps = self._motion_lib_timeline_fps(motion_lib)

        manifest = self._load_manifest()
        if env is None or not callable(getattr(env, "step", None)):
            raise RuntimeError(
                "practice-utility completed-dose measurement requires a live "
                "ManagerEnvWrapper.step method"
            )
        if getattr(env, "motion_command", None) is None:
            raise RuntimeError(
                "practice-utility completed-dose measurement requires the wrapper's "
                "live motion_command"
            )
        self._env = env
        self._motion_lib = motion_lib
        self.adapter = PracticeSamplerAdapter(
            motion_lib, branch_id=self.branch_id, role=self.role, manifest=manifest
        )
        if self.dose_plan_path:
            self._passive_dose_plan = DP.load_passive_dose_plan(
                self.dose_plan_path,
                expected_file_sha256=self.dose_plan_sha256,
                expected_campaign_id=self.dose_lineage.get("campaign_id"),
                expected_manifest_sha256=self.dose_lineage.get("manifest_sha256"),
                expected_manifest_file_sha256=self.dose_lineage.get("manifest_file_sha256"),
            )
            if not self.dose_plan_stage:
                raise ValueError("dose_plan_stage is required when dose_plan_path is set")
            self._passive_dose_plan.contexts_for(self.dose_plan_stage)
            if self.kernel_radius_bins != self._passive_dose_plan.kernel_radius_bins:
                raise ValueError("callback kernel_radius_bins differs from the passive dose plan")
            live_bin_size = getattr(motion_lib, "adp_samp_bin_size", None)
            if self.claim_mode and (
                isinstance(live_bin_size, bool)
                or not isinstance(live_bin_size, (int, float))
                or float(live_bin_size) != float(self._passive_dose_plan.reference_bin_size_frames)
                or float(live_bin_size) != self._passive_dose_plan.sigma_frames
            ):
                raise ValueError(
                    "live adp_samp_bin_size must equal the passive plan's "
                    "reference_bin_size_frames and sigma_frames"
                )
        live_num_envs = _num_envs_of(env)
        if self.claim_mode and live_num_envs != self.dose_num_envs:
            raise ValueError(
                "live environment count differs from frozen passive-dose count: "
                f"{live_num_envs} != {self.dose_num_envs}"
            )

        self._original_update = motion_lib.update_adaptive_sampling_probabilities
        self._original_sample = motion_lib.sample_motion_ids_and_time_steps
        self._original_env_step = env.step
        self._original_env_step_instance = ("step" in vars(env), vars(env).get("step"))
        adapter = self.adapter

        def update_with_override(*args, **kwargs):
            result = self._original_update(*args, **kwargs)
            motion_lib.adp_sampling_active_prob = adapter.apply(motion_lib.adp_sampling_active_prob)
            return result

        def sample_and_record(n, *args, **kwargs):
            motion_ids, time_steps = self._original_sample(n, *args, **kwargs)
            try:
                adapter.record_draw(_global_bins_for(motion_lib, motion_ids, time_steps))
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
                # Stale kernel: the resident batch changed. on_step_end re-arms;
                # dropping this batch's dose is safer than attributing it wrongly.
                message = f"sampled-context draw was not attributable: {error}"
                self._dose_errors.append(message)
                if self.claim_mode:
                    raise RuntimeError(message) from error
            return motion_ids, time_steps

        def step_and_record(*args, **kwargs):
            # IsaacLab resets terminated environments during the native step,
            # before TrackingCommand updates the sampler. Freeze the context
            # that generated this transition immediately before that step.
            captured_bins = None
            capture_error: Exception | None = None
            try:
                captured_bins = _pre_transition_global_bins(env, motion_lib)
                if self.dose_num_envs is not None and captured_bins.numel() != self.dose_num_envs:
                    raise ValueError(
                        "completion batch size differs from frozen environment count: "
                        f"{captured_bins.numel()} != {self.dose_num_envs}"
                    )
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
                capture_error = error

            # Do not catch or translate native failures, and do not count a
            # transition unless the native step returns successfully.
            result = self._original_env_step(*args, **kwargs)
            try:
                if capture_error is not None:
                    raise capture_error
                assert captured_bins is not None
                dones, time_outs = _termination_flags_from_step_result(result)
                if dones.numel() != captured_bins.numel():
                    raise ValueError(
                        "returned done batch size differs from captured transition contexts: "
                        f"{dones.numel()} != {captured_bins.numel()}"
                    )
                if time_outs.numel() != captured_bins.numel():
                    raise ValueError(
                        "returned time_outs batch size differs from captured transition contexts: "
                        f"{time_outs.numel()} != {captured_bins.numel()}"
                    )
                adapter.record_completion(
                    captured_bins,
                    torch.ones(captured_bins.numel(), device=captured_bins.device),
                    early_terminated=dones & ~time_outs,
                )
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
                adapter.record_dropped_completion_batch()
                message = f"passive completed-step batch was not attributable: {error}"
                self._dose_errors.append(message)
                if self.claim_mode:
                    raise RuntimeError(message) from error
            return result

        if self.snapshot_path and self.snapshot_at_step <= 0:
            self.write_snapshot()
            self._snapshot_written = True

        for name in (
            "update_adaptive_sampling_probabilities",
            "sample_motion_ids_and_time_steps",
        ):
            self._original_instance_attributes[name] = (
                name in vars(motion_lib),
                vars(motion_lib).get(name),
            )
        motion_lib.update_adaptive_sampling_probabilities = update_with_override
        motion_lib.sample_motion_ids_and_time_steps = sample_and_record
        env.step = step_and_record
        self._patched_attributes = [
            "update_adaptive_sampling_probabilities",
            "sample_motion_ids_and_time_steps",
        ]
        self._arm()

    def uninstall(self) -> None:
        """Restore the native sampler exactly as it was.

        The patch is *removed* from the instance dictionary rather than
        overwritten with the original bound method. Overwriting would leave a
        shadowing instance attribute behind -- behaviourally correct, but it
        would also mean a later reinstall wrapped an already-wrapped method, and
        it makes "is this run patched?" unanswerable by inspection.
        """
        if self._motion_lib is not None:
            for name in self._patched_attributes:
                existed, value = self._original_instance_attributes.get(name, (False, None))
                if existed:
                    setattr(self._motion_lib, name, value)
                else:
                    self._motion_lib.__dict__.pop(name, None)
        if self._env is not None:
            existed, value = self._original_env_step_instance
            if existed:
                self._env.step = value
            else:
                self._env.__dict__.pop("step", None)
        self._env = None
        self._motion_lib = None
        self._original_update = None
        self._original_sample = None
        self._original_env_step = None
        self._original_env_step_instance = (False, None)
        self._patched_attributes = []
        self._original_instance_attributes = {}
        self.adapter = None
        self._armed = False
        self._verified_snapshot_timeline_fps = None
        self._passive_dose_plan = None

    def _motion_lib_timeline_fps(self, motion_lib: Any) -> float:
        """Return the live target FPS only when it matches the frozen callback value."""
        raw = getattr(motion_lib, "target_fps", None)
        if isinstance(raw, bool):
            raise RuntimeError("motion library target_fps is not a positive finite number")
        try:
            actual = float(raw)
        except (TypeError, ValueError) as error:
            raise RuntimeError("motion library has no numeric target_fps") from error
        if not math.isfinite(actual) or actual <= 0:
            raise RuntimeError("motion library target_fps is not a positive finite number")
        if not math.isclose(actual, self.snapshot_timeline_fps, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                "live motion library target_fps differs from snapshot_timeline_fps: "
                f"actual={actual}, frozen={self.snapshot_timeline_fps}"
            )
        return actual

    @staticmethod
    def is_patched(motion_lib: Any) -> bool:
        """Whether a motion library currently carries our sampler patch."""
        return "update_adaptive_sampling_probabilities" in vars(motion_lib)

    # ------------------------------------------------------------- arming --

    def _arm(self, global_step: int = 0) -> bool:
        """Arm the intervention if the context is resident. Never fatal.

        Returns whether the branch is armed. A context absent from the current
        motion batch is a normal, transient condition -- SONIC holds only part of
        the pool in memory and rotates it -- so this records the attempt and
        returns False instead of raising. A branch that never manages to arm
        delivers no dose, and ``build_utility_record`` already refuses to emit a
        label in that case, which is where the failure belongs: loud at label
        time, quiet at install.
        """
        if self.adapter is None or self.context is None:
            self._armed = False
            return False
        self._arm_attempts += 1
        try:
            # epsilon == 0 is armed deliberately: the noise-floor branch must
            # travel the same code path as a real intervention.
            self.adapter.set_intervention(
                self.context, epsilon=self.epsilon, kernel_radius=self.kernel_radius_bins
            )
        except ValueError:
            self.adapter.clear_override()
            self._armed = False
            return False
        self._armed = True
        if self._first_armed_step is None:
            self._first_armed_step = global_step
        return True

    def _rearm_if_stale(self) -> None:
        if not self._armed or self.adapter is None or self.context is None:
            return
        resident = getattr(self._motion_lib, "adp_samp_active_motion_bins", None)
        kernel = self.adapter._kernel_weights
        if resident is None or kernel is None or kernel.numel() == resident.numel():
            return
        if not self._arm():
            # The context is no longer resident after the resample; fall back to
            # the native distribution rather than intervene through a stale kernel.
            self.adapter.clear_override()

    def _load_manifest(self) -> MotionPoolManifest | None:
        if not self.manifest_path:
            return None
        payload = json.loads(Path(self.manifest_path).read_text())
        motions = payload.get("motions", [])
        return MotionPoolManifest(
            manifest_id=payload.get("pool_id", "pool"),
            motion_keys=[m["motion_key"] for m in motions],
            motion_hashes={m["motion_key"]: m["content_sha256"] for m in motions},
            source_root=payload.get("source_root", ""),
        )

    # -------------------------------------------------------------- output --

    def write_snapshot(self, global_step: int = 0) -> str | None:
        """Dump the native distribution and every resident context.

        This is how a campaign discovers which contexts actually exist: bin
        boundaries are decided by SONIC at load time from each clip's resampled
        frame count, so they cannot be guessed from the stored motion files.
        A control run writes this, and the probe manifest is built from it.
        """
        if self.adapter is None or not self.snapshot_path:
            return None
        timeline_fps = self._motion_lib_timeline_fps(self._motion_lib)
        self._verified_snapshot_timeline_fps = timeline_fps
        snapshot = self.adapter.snapshot_native_distribution(global_step)
        contexts = []
        for position, bin_id in enumerate(snapshot.active_bin_ids):
            context = self.adapter.context_for_bin(bin_id)
            contexts.append(
                {
                    **context.to_dict(),
                    "context_id": context.context_id,
                    "global_bin_id": bin_id,
                    "sampling_probability": snapshot.active_prob[position],
                    "failure_rate": snapshot.failure_rate_raw[position],
                    "num_episodes": snapshot.num_episodes[position],
                    "num_failures": snapshot.num_failures[position],
                }
            )
        path = Path(self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": "practice_utility_sampler_snapshot",
                    "schema_version": 1,
                    "global_step": global_step,
                    "branch_id": self.branch_id,
                    "snapshot_timeline_fps": timeline_fps,
                    "num_bins": snapshot.num_bins,
                    "num_active_bins": len(snapshot.active_bin_ids),
                    "distribution_sha256": snapshot.distribution_sha256,
                    "effective_num_bins": snapshot.effective_num_bins,
                    "uniform_sampling_rate": snapshot.uniform_sampling_rate,
                    "failure_rate_max_over_mean": snapshot.failure_rate_max_over_mean,
                    "contexts": contexts,
                },
                indent=2,
            )
        )
        return str(path)

    def write_dose_report(
        self, global_step: int, *, horizon_label: str | None = None
    ) -> str | None:
        """Atomically persist a v2 passive completed-dose receipt."""
        if self.adapter is None or not self.dose_report_dir:
            return None
        directory = Path(self.dose_report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        report = self.adapter.get_exact_dose_report()
        contexts: tuple[ContextKey, ...]
        radius, sigma = self.kernel_radius_bins, float(self._motion_lib.adp_samp_bin_size)
        if self._passive_dose_plan is not None:
            assert self.dose_plan_stage is not None
            contexts = self._passive_dose_plan.contexts_for(self.dose_plan_stage)
            radius = self._passive_dose_plan.kernel_radius_bins
            sigma = self._passive_dose_plan.sigma_frames
        elif self.context is not None:
            contexts = (self.context,)
        else:
            contexts = ()

        blockers = list(self._dose_errors)
        verified_registry_sha256: str | None = None
        try:
            verified_registry_sha256 = self.adapter.verify_dose_registry()
        except RuntimeError as error:
            blockers.append(str(error))

        projections = []
        if verified_registry_sha256 is not None:
            for context in contexts:
                try:
                    projection = self.adapter.project_completed_kernel_steps(
                        context,
                        verified_registry_sha256=verified_registry_sha256,
                        kernel_radius=radius,
                        sigma_frames=sigma,
                    )
                except (RuntimeError, ValueError) as error:
                    blockers.append(
                        f"context {context.context_id} passive-dose projection failed: {error}"
                    )
                    continue
                projections.append({"context": context.to_dict(), **projection.to_dict()})
        if report.dropped_completion_batches:
            blockers.append(
                f"{report.dropped_completion_batches} completed-step batches were dropped"
            )
        if report.completion_hook_calls == 0:
            blockers.append("passive completion hook was never observed")
        if self.role == "intervention" and self._first_armed_step is None:
            blockers.append("intervention was never armed")

        expected_env_steps = None
        expected_hook_calls = None
        expected_observations = None
        if (
            self.dose_origin_global_step is not None
            and self.dose_num_steps_per_iteration is not None
            and self.dose_num_envs is not None
        ):
            expected_env_steps = (
                (global_step - self.dose_origin_global_step)
                * self.dose_num_steps_per_iteration
                * self.dose_num_envs
            )
            expected_hook_calls = (
                global_step - self.dose_origin_global_step
            ) * self.dose_num_steps_per_iteration
            expected_observations = expected_hook_calls * self.dose_num_envs
            if expected_env_steps < 0:
                blockers.append("dose horizon precedes the frozen origin")
            elif abs(report.completed_env_steps - expected_env_steps) > 1e-6:
                blockers.append(
                    "completed_env_steps does not match the frozen horizon: "
                    f"{report.completed_env_steps} != {expected_env_steps}"
                )
            if report.completion_hook_calls != expected_hook_calls:
                blockers.append(
                    "completion_hook_calls does not match the frozen horizon: "
                    f"{report.completion_hook_calls} != {expected_hook_calls}"
                )
            if report.completion_observations != expected_observations:
                blockers.append(
                    "completion_observations does not match the frozen horizon: "
                    f"{report.completion_observations} != {expected_observations}"
                )
            if report.termination_observations != expected_observations:
                blockers.append(
                    "termination_observations does not match the frozen horizon: "
                    f"{report.termination_observations} != {expected_observations}"
                )
        if horizon_label is not None:
            expected_horizon = self.dose_report_horizons.get(horizon_label)
            if expected_horizon != global_step:
                blockers.append(
                    f"horizon {horizon_label!r} expected step {expected_horizon}, got {global_step}"
                )
        elif self.claim_mode:
            blockers.append("claim-mode dose receipt is not tied to an exact horizon")
        if self._passive_dose_plan is not None and len(projections) != len(contexts):
            blockers.append("passive dose plan context coverage is incomplete")

        callback_path = Path(__file__).resolve()
        required_lineage = ("campaign_id", "manifest_sha256", "manifest_file_sha256")
        exact_horizon = (
            horizon_label is not None
            and self.dose_report_horizons.get(horizon_label) == global_step
        )
        expected_count_metadata = (
            expected_env_steps is not None
            and expected_hook_calls is not None
            and expected_observations is not None
        )
        complete_lineage = all(self.dose_lineage.get(key) for key in required_lineage)
        valid_for_claim = (
            self.claim_mode
            and not blockers
            and self._passive_dose_plan is not None
            and exact_horizon
            and expected_count_metadata
            and complete_lineage
        )
        payload = {
            "kind": DP.PASSIVE_DOSE_RECEIPT_KIND,
            "schema_version": DP.PASSIVE_DOSE_RECEIPT_SCHEMA_VERSION,
            "status": "blocked" if blockers else "complete",
            "valid_for_claim": valid_for_claim,
            "branch_id": report.branch_id,
            "pair_id": self.pair_id,
            "role": report.role,
            "global_step": global_step,
            "horizon_label": horizon_label,
            "context_id": report.context_id,
            "epsilon": self.epsilon,
            "kernel_radius_bins": self.kernel_radius_bins,
            "measurement_hook": DP.PASSIVE_DOSE_HOOK,
            "armed": self._armed,
            "armed_steps": self._armed_steps,
            "arm_attempts": self._arm_attempts,
            "first_armed_step": self._first_armed_step,
            "never_armed": self._first_armed_step is None,
            "drawn_episodes": report.drawn_episodes,
            "drawn_kernel_mass": report.drawn_kernel_mass,
            "total_episodes": report.total_episodes,
            "total_episode_count_semantics": (
                "sampler_draws_after_instrumentation; excludes episodes resident at install"
            ),
            "completed_env_steps": report.completed_env_steps,
            "expected_env_steps": expected_env_steps,
            "expected_completion_hook_calls": expected_hook_calls,
            "expected_completion_observations": expected_observations,
            "completed_kernel_steps": report.completed_kernel_steps,
            "completion_hook_calls": report.completion_hook_calls,
            "completion_observations": report.completion_observations,
            "termination_observations": report.termination_observations,
            "termination_observation_semantics": (
                "per-environment termination flags observed; not actual terminations"
            ),
            "dropped_completion_batches": report.dropped_completion_batches,
            "early_terminations": report.early_terminations,
            "dose_registry_sha256_at_install": report.dose_registry_sha256,
            "dose_registry_sha256_at_report": verified_registry_sha256,
            "registry_stable": verified_registry_sha256 == report.dose_registry_sha256,
            "per_bin_drawn": {str(k): v for k, v in report.per_bin_drawn.items()},
            "per_bin_completed": {str(k): v for k, v in sorted(report.per_bin_completed.items())},
            "per_motion_drawn": report.per_motion_drawn,
            "context_doses": projections,
            "passive_dose_plan": (
                {
                    "path": str(self._passive_dose_plan.path),
                    "file_sha256": self._passive_dose_plan.file_sha256,
                    "logical_sha256": self._passive_dose_plan.logical_sha256,
                    "stage": self.dose_plan_stage,
                }
                if self._passive_dose_plan is not None
                else None
            ),
            "lineage": self.dose_lineage,
            "implementation": {
                "callback_path": str(callback_path),
                "callback_sha256": hashlib.sha256(callback_path.read_bytes()).hexdigest(),
            },
            "blockers": blockers,
        }
        payload["receipt_payload_sha256"] = sha256_of(payload)
        suffix = f"_{horizon_label}" if horizon_label else ""
        path = directory / f"dose_{self.branch_id}{suffix}_step{global_step:06d}.json"
        content = (json.dumps(payload, indent=2) + "\n").encode()
        if self.claim_mode:
            _publish_exclusive_atomic(path, content)
        else:
            staging = path.with_suffix(path.suffix + ".partial")
            staging.write_bytes(content)
            staging.replace(path)
        if self.claim_mode and blockers:
            raise RuntimeError(
                "claim-mode passive dose receipt failed closed: " + "; ".join(blockers)
            )
        return str(path)


class PracticeCapsuleCallback(TrainerCallback):
    """Save branch capsules -- including RNG -- at the horizon checkpoints."""

    def __init__(
        self,
        enabled: bool = False,
        capsule_dir: str = "capsules",
        horizons: dict[str, int] | None = None,
        pair_id: str = "unbound",
        role: str = "control",
        branch_id: str | None = None,
        provenance: dict[str, str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.capsule_dir = capsule_dir
        self.horizons = dict(horizons or {})
        self.pair_id = pair_id
        self.role = role
        self.branch_id = branch_id or f"{pair_id}_{role}"
        self.provenance = Provenance(
            **{
                "resolved_config_sha256": "",
                "motion_pool_manifest_sha256": "",
                "dev_suite_sha256": "",
                "source_commit": "",
                "checkpoint_sha256": "",
                **(provenance or {}),
            }
        )
        self.saved: dict[str, str] = {}

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled or state is None:
            return control
        step = getattr(state, "global_step", 0)
        for label, horizon in self.horizons.items():
            if step == horizon and label not in self.saved:
                self.saved[label] = self.save(label, step, {**kwargs, "state": state})
        return control

    def save(self, horizon_label: str, global_step: int, kwargs: dict[str, Any]) -> str:
        """Write one capsule. RNG is captured *before* anything else is read.

        The model is stored the way SONIC's own loader expects -- ``policy`` and
        ``value`` state dicts kept apart -- not as one combined ``state_dict()``.
        A combined blob cannot be split back reliably, which would make a capsule
        unusable as a branch origin and confine the whole programme to probing
        one policy stage.
        """
        rng_state = RngState.capture(self.pair_id)
        env = kwargs.get("env")
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")

        model_state = {
            "combined_state_dict": _state_dict_of(model),
            "policy_state_dict": _state_dict_of(getattr(model, "policy", None)),
            "value_state_dict": _state_dict_of(getattr(model, "value_model", None)),
            "lr_scheduler_state_dict": _state_dict_of(lr_scheduler),
        }

        path = Path(self.capsule_dir) / f"{self.branch_id}_{horizon_label}.capsule.pt"
        save_capsule(
            path,
            branch_id=self.branch_id,
            pair_id=self.pair_id,
            role=self.role,
            global_step=global_step,
            model_state=model_state,
            optimizer_state=_state_dict_of(optimizer),
            trainer_state={
                "global_step": global_step,
                "horizon_label": horizon_label,
                "trainer_state_obj": kwargs.get("state"),
            },
            env_state=env.get_env_state_dict() if hasattr(env, "get_env_state_dict") else {},
            native_sampler_state=_sampler_state_of(env),
            rng_state=rng_state,
            provenance=self.provenance,
        )
        return str(path)


class PracticeCapsuleResumeCallback(TrainerCallback):
    """Restore a capsule's RNG stream just before SONIC resets a resumed env.

    SONIC loads model, optimizer, trainer, and environment state while building
    the trainer, but its exported checkpoint format has no RNG field. Loading
    here is deliberately late: ``on_train_begin`` runs after checkpoint loading
    and immediately before ``reset_all()``, so construction-time draws cannot
    displace the resumed rollout stream.
    """

    def __init__(self, enabled: bool = False, capsule_path: str | None = None) -> None:
        self.enabled = bool(enabled)
        self.capsule_path = capsule_path
        self.restored: dict[str, Any] | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        if not self.enabled:
            return control
        if not self.capsule_path:
            raise ValueError("enabled capsule resume requires capsule_path")
        payload = load_capsule(self.capsule_path, restore_rng=True)
        expected_step = int(payload["global_step"])
        actual_step = int(getattr(state, "global_step", 0))
        if actual_step != expected_step:
            raise RuntimeError(
                f"resume step {actual_step} does not match capsule step {expected_step}"
            )
        self.restored = {
            "capsule_path": self.capsule_path,
            "capsule_sha256": payload.get("capsule_sha256"),
            "global_step": expected_step,
            "pair_id": payload.get("pair_id"),
        }
        print(
            f"[practice-resume] restored capsule RNG at step {expected_step}",
            flush=True,
        )
        return control


# ------------------------------------------------------------------ helpers --


def _motion_lib_of(env: Any) -> Any:
    """Reach SONIC's motion library through the wrapper."""
    if env is None:
        return None
    for attribute in ("_motion_lib", "motion_lib"):
        library = getattr(env, attribute, None)
        if library is not None:
            return library
    command = getattr(env, "motion_command", None)
    return getattr(command, "motion_lib", None) if command is not None else None


def _num_envs_of(env: Any) -> int | None:
    """Read the live local environment count without inferring a default."""

    if env is None:
        return None
    value = getattr(env, "num_envs", None)
    if value is None:
        config = getattr(env, "config", None)
        value = getattr(config, "num_envs", None)
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _global_bins_for(motion_lib: Any, motion_ids: Any, time_steps: Any) -> Any:
    """Map sampled ``(motion_id, time_step)`` pairs back to global bin ids.

    Mirrors ``update_adaptive_sampling_stats``: batch-local motion ids are
    resolved to dataset ids, offset by the motion's frame start, then looked up
    in the frame-to-bin table.
    """
    dataset_ids = motion_lib.get_motion_ids_in_dataset(motion_ids)
    frames = motion_lib.adp_samp_length_starts[dataset_ids] + time_steps
    return motion_lib.adp_samp_frame_to_bin[frames.long()]


def _pre_transition_global_bins(env: Any, motion_lib: Any) -> torch.Tensor:
    """Freeze the reference contexts that generate the imminent transition."""

    command = getattr(env, "motion_command", None)
    if command is None:
        raise AttributeError("ManagerEnvWrapper has no live motion_command")
    motion_ids = command.motion_ids.detach().clone().reshape(-1)
    start_steps = command.motion_start_time_steps.detach().clone().reshape(-1)
    elapsed_steps = command.time_steps.detach().clone().reshape(-1)
    if motion_ids.numel() != start_steps.numel() or motion_ids.numel() != elapsed_steps.numel():
        raise ValueError("motion ids, start steps, and elapsed steps must align")
    return _global_bins_for(motion_lib, motion_ids, start_steps + elapsed_steps).detach().clone()


def _termination_flags_from_step_result(result: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract reset and timeout flags from the unchanged wrapper-step result."""

    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError("ManagerEnvWrapper.step must return (obs, reward, dones, extras)")
    dones, extras = result[2], result[3]
    if not isinstance(dones, torch.Tensor):
        raise TypeError("ManagerEnvWrapper.step dones must be a tensor")
    if not isinstance(extras, dict) or not isinstance(extras.get("time_outs"), torch.Tensor):
        raise TypeError("ManagerEnvWrapper.step extras must contain tensor time_outs")
    return (
        dones.detach().clone().reshape(-1).to(torch.bool),
        extras["time_outs"].detach().clone().reshape(-1).to(torch.bool),
    )


def _sampler_state_of(env: Any) -> dict[str, Any]:
    library = _motion_lib_of(env)
    if library is None:
        return {}
    state = {}
    for key in ("adp_samp_num_episodes", "adp_samp_num_failures"):
        value = getattr(library, key, None)
        if value is not None:
            state[key] = value.detach().cpu()
    return state


def _state_dict_of(module: Any) -> dict[str, Any]:
    return module.state_dict() if hasattr(module, "state_dict") else {}


def _publish_exclusive_atomic(path: Path, content: bytes) -> None:
    """Publish immutable claim evidence without replacing an existing receipt."""

    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
