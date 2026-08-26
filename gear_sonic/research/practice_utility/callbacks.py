"""Trainer callbacks that wire practice-utility measurement into a SONIC run.

Two callbacks, both strict no-ops when disabled:

:class:`PracticeContextCallback`
    Installs a :class:`PracticeSamplerAdapter` onto the live motion library,
    optionally arms an intervention, and writes the realized-dose receipt. It
    patches exactly one method -- ``update_adaptive_sampling_probabilities`` --
    and only post-processes its result, so the native failure-rate computation,
    uniform floor, bin weights, and concentration caps all still run.

:class:`PracticeCapsuleCallback`
    Saves branch capsules at the horizon checkpoints, carrying the RNG state and
    sampler-to-pool binding that ``ModelSaveCallback`` does not.

Both re-arm after every motion resample. SONIC periodically reloads the resident
motion batch, which invalidates a kernel built over the previous batch; the
adapter detects a stale kernel and raises rather than silently misattributing
dose, so the callback must re-arm at that point.

Configured through Hydra like every other SONIC callback::

    practice_context:
      _target_: gear_sonic.research.practice_utility.callbacks.PracticeContextCallback
      enabled: true
      role: intervention
      pair_id: pair_017
      context_id: 3ba90f8b99da380f
      epsilon: 0.10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gear_sonic.research.practice_utility.branch_capsule import (
    Provenance,
    load_capsule,
    save_capsule,
)
from gear_sonic.research.practice_utility.rng_capsule import RngState
from gear_sonic.research.practice_utility.sampler_adapter import PracticeSamplerAdapter
from gear_sonic.research.practice_utility.schema import ContextKey, MotionPoolManifest

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
        self._snapshot_written = False
        # A context is only armable while its motion is resident. SONIC keeps a
        # subset of the pool loaded (195 of 512 motions in a measured run), so a
        # branch whose context is absent at install must wait for a resample
        # rather than die -- otherwise most of a campaign never starts.
        self._arm_attempts = 0
        self._armed_steps = 0
        self._first_armed_step: int | None = None

        self.adapter: PracticeSamplerAdapter | None = None
        self._motion_lib: Any = None
        self._original_update: Any = None
        self._original_sample: Any = None
        self._patched_attributes: list[str] = []
        self._armed = False

        if self.enabled and self.role == "intervention" and self.context is None:
            raise ValueError(
                "an intervention branch requires a context; without one the branch "
                "is a control and must be labelled as such"
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
        if (
            self.dose_report_frequency
            and self.dose_report_dir
            and state is not None
            and getattr(state, "global_step", 0) % self.dose_report_frequency == 0
        ):
            self.write_dose_report(getattr(state, "global_step", 0))
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.enabled and self.dose_report_dir:
            self.write_dose_report(getattr(state, "global_step", 0))
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

        manifest = self._load_manifest()
        self._motion_lib = motion_lib
        self.adapter = PracticeSamplerAdapter(
            motion_lib, branch_id=self.branch_id, role=self.role, manifest=manifest
        )

        self._original_update = motion_lib.update_adaptive_sampling_probabilities
        self._original_sample = motion_lib.sample_motion_ids_and_time_steps
        adapter = self.adapter

        def update_with_override(*args, **kwargs):
            result = self._original_update(*args, **kwargs)
            motion_lib.adp_sampling_active_prob = adapter.apply(motion_lib.adp_sampling_active_prob)
            return result

        def sample_and_record(n, *args, **kwargs):
            motion_ids, time_steps = self._original_sample(n, *args, **kwargs)
            try:
                adapter.record_draw(_global_bins_for(motion_lib, motion_ids, time_steps))
            except RuntimeError:
                # Stale kernel: the resident batch changed. on_step_end re-arms;
                # dropping this batch's dose is safer than attributing it wrongly.
                pass
            return motion_ids, time_steps

        if self.snapshot_path and self.snapshot_at_step <= 0:
            self.write_snapshot()
            self._snapshot_written = True

        motion_lib.update_adaptive_sampling_probabilities = update_with_override
        motion_lib.sample_motion_ids_and_time_steps = sample_and_record
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
                self._motion_lib.__dict__.pop(name, None)
        self._motion_lib = None
        self._original_update = None
        self._original_sample = None
        self._patched_attributes = []
        self.adapter = None
        self._armed = False

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

    def write_dose_report(self, global_step: int) -> str | None:
        """Persist the realized-dose receipt for this branch."""
        if self.adapter is None or not self.dose_report_dir:
            return None
        directory = Path(self.dose_report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        report = self.adapter.get_exact_dose_report()
        path = directory / f"dose_{self.branch_id}_step{global_step:06d}.json"
        path.write_text(
            json.dumps(
                {
                    "branch_id": report.branch_id,
                    "pair_id": self.pair_id,
                    "role": report.role,
                    "global_step": global_step,
                    "context_id": report.context_id,
                    "epsilon": self.epsilon,
                    "kernel_radius_bins": self.kernel_radius_bins,
                    "armed": self._armed,
                    "armed_steps": self._armed_steps,
                    "arm_attempts": self._arm_attempts,
                    "first_armed_step": self._first_armed_step,
                    "never_armed": self._first_armed_step is None,
                    "drawn_episodes": report.drawn_episodes,
                    "drawn_kernel_mass": report.drawn_kernel_mass,
                    "completed_env_steps": report.completed_env_steps,
                    "completed_kernel_steps": report.completed_kernel_steps,
                    "early_terminations": report.early_terminations,
                    "per_bin_drawn": {str(k): v for k, v in report.per_bin_drawn.items()},
                    "per_motion_drawn": report.per_motion_drawn,
                },
                indent=2,
            )
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


def _global_bins_for(motion_lib: Any, motion_ids: Any, time_steps: Any) -> Any:
    """Map sampled ``(motion_id, time_step)`` pairs back to global bin ids.

    Mirrors ``update_adaptive_sampling_stats``: batch-local motion ids are
    resolved to dataset ids, offset by the motion's frame start, then looked up
    in the frame-to-bin table.
    """
    dataset_ids = motion_lib.get_motion_ids_in_dataset(motion_ids)
    frames = motion_lib.adp_samp_length_starts[dataset_ids] + time_steps
    return motion_lib.adp_samp_frame_to_bin[frames.long()]


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
