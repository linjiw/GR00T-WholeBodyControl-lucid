"""Adapter that installs interventions into SONIC's live motion sampler.

The adapter never re-implements SONIC's sampler. It reads the genuine
distribution the native code produced and, only when an override is armed,
replaces it for the current epoch. With no override armed every call is a
pass-through, which is what ``test_sampler_identity`` and the no-op parity test
verify.

Where this binds, in ``gear_sonic/utils/motion_lib/motion_lib_base.py``:

``update_adaptive_sampling_probabilities()``
    Produces ``adp_sampling_active_prob`` -- failure rate, clipped at
    ``failure_rate_max_over_mean``, blended with a uniform floor, scaled by bin
    weights, then capped per bin and per motion. The adapter post-processes that
    result; it does not replace the computation.

``sample_motion_ids_and_time_steps(n)``
    The single point where a training context is chosen: one multinomial draw
    over ``adp_sampling_active_prob``, mapped through
    ``adp_samp_active_motion_bins`` to global bin ids, then to
    ``adp_samp_bins[bin] = (orig_motion_id, bin_start, bin_end)``. Dose is
    counted here, at draw time.

Counting dose at draw time matters for episode counts, but completed exposure
is captured separately around ``ManagerEnvWrapper.step``. The callback freezes
the current reference context before IsaacLab can reset and resample a failed
environment, then records the successful transition after the native step
returns. The realized-dose denominator in :meth:`UtilityRecord.utility_at`
therefore reflects steps actually executed inside the kernel rather than the
next episode's newly sampled motion.

The adapter is duck-typed over the motion library so its guarantees can be
tested on CPU against a fake that mirrors the real attribute contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from gear_sonic.research.practice_utility import intervention as I
from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    DoseProjection,
    DoseReport,
    MotionPoolManifest,
    SamplingSnapshot,
    motion_hash,
    sha256_of,
)


class MotionLibLike(Protocol):
    """The subset of ``MotionLibBase`` this adapter depends on."""

    adp_samp_bins: torch.Tensor
    adp_samp_active_motion_bins: torch.Tensor
    adp_sampling_active_prob: torch.Tensor
    adp_samp_bin_size: int
    adp_samp_num_bins: int
    adp_samp_failure_rate_raw: torch.Tensor
    adp_samp_num_episodes: torch.Tensor
    adp_samp_num_failures: torch.Tensor
    uniform_sampling_rate: float
    adp_samp_failure_rate_max_over_mean: float
    _motion_data_keys: Any
    _device: Any


@dataclass
class InterventionSpec:
    """An armed intervention: what context, how much dose, what kernel shape."""

    context: ContextKey
    epsilon: float
    kernel: I.KernelSpec

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")


class StaleDoseRegistryError(RuntimeError):
    """Raised when passive dose can no longer be projected on its frozen bins."""


class PracticeSamplerAdapter:
    """Reads, and optionally overrides, SONIC's bin sampling distribution."""

    def __init__(
        self,
        motion_lib: MotionLibLike,
        branch_id: str = "unbound",
        role: str = "control",
        manifest: MotionPoolManifest | None = None,
    ) -> None:
        self.motion_lib = motion_lib
        self.branch_id = branch_id
        self.role = role
        self.manifest = manifest

        self._intervention: InterventionSpec | None = None
        self._residual_prob: torch.Tensor | None = None
        self._residual_id: str | None = None

        self._drawn_per_bin: dict[int, float] = defaultdict(float)
        self._completed_per_bin: dict[int, float] = defaultdict(float)
        self._kernel_weights: torch.Tensor | None = None
        self._drawn_episodes = 0.0
        self._drawn_kernel_mass = 0.0
        self._completed_env_steps = 0.0
        self._completed_kernel_steps = 0.0
        self._early_terminations = 0
        self._completion_hook_calls = 0
        self._completion_observations = 0
        self._termination_observations = 0
        self._dropped_completion_batches = 0
        self._dose_registry_sha256 = self.dose_registry_sha256()

    # ---------------------------------------------------------------- state --

    @property
    def override_active(self) -> bool:
        """True when the adapter will modify the native distribution."""
        return self._intervention is not None or self._residual_prob is not None

    def clear_override(self) -> None:
        """Return to a pure pass-through. Dose counters are preserved."""
        self._intervention = None
        self._residual_prob = None
        self._residual_id = None
        self._kernel_weights = None

    def reset_dose(self) -> None:
        """Zero the dose counters, e.g. at the start of a continuation."""
        self._drawn_per_bin = defaultdict(float)
        self._completed_per_bin = defaultdict(float)
        self._drawn_episodes = 0.0
        self._drawn_kernel_mass = 0.0
        self._completed_env_steps = 0.0
        self._completed_kernel_steps = 0.0
        self._early_terminations = 0
        self._completion_hook_calls = 0
        self._completion_observations = 0
        self._termination_observations = 0
        self._dropped_completion_batches = 0

    # ------------------------------------------------------------ snapshots --

    def snapshot_native_distribution(self, global_step: int = 0) -> SamplingSnapshot:
        """Capture the live native distribution without perturbing it."""
        lib = self.motion_lib
        active = lib.adp_samp_active_motion_bins.detach().cpu()
        prob = lib.adp_sampling_active_prob.detach().cpu().to(torch.float64)
        total = float(prob.sum())
        if total <= 0:
            raise RuntimeError("native sampling distribution has no mass")
        prob = prob / total  # guard against float32 drift upstream
        return SamplingSnapshot(
            global_step=global_step,
            num_bins=int(lib.adp_samp_num_bins),
            active_bin_ids=[int(b) for b in active.tolist()],
            active_prob=[float(p) for p in prob.tolist()],
            failure_rate_raw=self._gather(lib.adp_samp_failure_rate_raw, active),
            num_episodes=self._gather(lib.adp_samp_num_episodes, active),
            num_failures=self._gather(lib.adp_samp_num_failures, active),
            uniform_sampling_rate=float(lib.uniform_sampling_rate),
            failure_rate_max_over_mean=float(lib.adp_samp_failure_rate_max_over_mean),
            manifest_id=self.manifest.manifest_id if self.manifest else "unbound",
        )

    def context_for_bin(self, global_bin_id: int, encoder_mode: str = "g1") -> ContextKey:
        """Build the stable :class:`ContextKey` for a global bin id."""
        lib = self.motion_lib
        row = lib.adp_samp_bins[int(global_bin_id)]
        orig_motion_id, bin_start, bin_end = (int(row[0]), int(row[1]), int(row[2]))
        motion_key = str(lib._motion_data_keys[orig_motion_id])
        num_frames = self._motion_frame_count(orig_motion_id)
        return ContextKey(
            motion_key=motion_key,
            motion_hash=self._motion_hash_for(motion_key, num_frames),
            bin_index=bin_start // int(lib.adp_samp_bin_size),
            bin_start_frame=bin_start,
            bin_end_frame=bin_end,
            encoder_mode=encoder_mode,
        )

    # --------------------------------------------------------- interventions --

    def set_intervention(
        self,
        context: ContextKey,
        epsilon: float,
        kernel_radius: int = 1,
        sigma_frames: float | None = None,
    ) -> None:
        """Arm a localized intervention on ``context``.

        ``epsilon = 0`` is armed rather than rejected: the epsilon=0 branch is a
        real experimental condition used to measure the branch noise floor, and
        it must traverse exactly the same code path as a real intervention.
        """
        if self._residual_prob is not None:
            raise RuntimeError("cannot arm an intervention while a residual distribution is set")
        lib = self.motion_lib
        spec = I.KernelSpec(
            radius_bins=kernel_radius,
            sigma_frames=float(sigma_frames if sigma_frames is not None else lib.adp_samp_bin_size),
        )
        self._intervention = InterventionSpec(context=context, epsilon=epsilon, kernel=spec)
        self._kernel_weights = self._build_kernel(context, spec)

    def set_residual_distribution(self, probability: torch.Tensor, manifest_id: str) -> None:
        """Install a precomputed residual distribution over the active bins."""
        if self._intervention is not None:
            raise RuntimeError("cannot set a residual distribution while an intervention is armed")
        active_n = int(self.motion_lib.adp_samp_active_motion_bins.numel())
        if probability.numel() != active_n:
            raise ValueError(
                f"residual distribution has {probability.numel()} entries "
                f"for {active_n} active bins"
            )
        prob = probability.detach().to(torch.float64).cpu()
        total = float(prob.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"residual distribution sums to {total}, expected 1.0")
        self._residual_prob = prob
        self._residual_id = manifest_id

    def apply(self, native_prob: torch.Tensor) -> torch.Tensor:
        """Transform the native distribution for this epoch.

        Returns ``native_prob`` unchanged, and with the same dtype and device,
        whenever no override is armed.
        """
        if not self.override_active:
            return native_prob

        base = native_prob.detach().to(torch.float64).cpu()
        base = base / base.sum()

        if self._residual_prob is not None:
            result = self._residual_prob
        else:
            assert self._intervention is not None and self._kernel_weights is not None
            result = I.mix_intervention(base, self._kernel_weights, self._intervention.epsilon)

        return result.to(dtype=native_prob.dtype, device=native_prob.device)

    # ------------------------------------------------------------------ dose --

    def record_draw(self, global_bin_ids: torch.Tensor) -> None:
        """Record contexts drawn by ``sample_motion_ids_and_time_steps``."""
        ids = global_bin_ids.detach().cpu().reshape(-1)
        self._drawn_episodes += float(ids.numel())
        for bin_id, count in zip(*[t.tolist() for t in torch.unique(ids, return_counts=True)]):
            self._drawn_per_bin[int(bin_id)] += float(count)
        if self._kernel_weights is not None:
            active = self.motion_lib.adp_samp_active_motion_bins.detach().cpu()
            membership = self._kernel_membership(active)
            position = self._global_to_active_position(ids, active)
            valid = position >= 0
            if bool(valid.any()):
                self._drawn_kernel_mass += float(membership[position[valid]].sum())

    def record_completion(
        self,
        global_bin_ids: torch.Tensor,
        steps_completed: torch.Tensor,
        early_terminated: torch.Tensor | None = None,
    ) -> None:
        """Record one batch of executed reference-timeline steps.

        The callback wraps ``ManagerEnvWrapper.step`` and passes the contexts
        frozen immediately before each successful native step with unit
        ``steps_completed``. The more general weighted form remains useful for
        CPU contract tests and offline aggregation.
        """
        ids = global_bin_ids.detach().cpu().reshape(-1)
        steps = steps_completed.detach().cpu().reshape(-1).to(torch.float64)
        if ids.numel() != steps.numel():
            raise ValueError("global_bin_ids and steps_completed must align")
        if early_terminated is not None and early_terminated.numel() != ids.numel():
            raise ValueError("early_terminated must align with global_bin_ids")
        if not bool(torch.isfinite(steps).all()) or bool((steps < 0).any()):
            raise ValueError("steps_completed must be finite and non-negative")
        num_bins = int(self.motion_lib.adp_samp_num_bins)
        if bool(((ids < 0) | (ids >= num_bins)).any()):
            raise ValueError("global_bin_ids contains an unmappable bin")

        self._completion_hook_calls += 1
        self._completion_observations += int(ids.numel())
        self._termination_observations += int(ids.numel())
        self._completed_env_steps += float(steps.sum())
        if ids.numel():
            unique_ids, inverse = torch.unique(ids.to(torch.long), return_inverse=True)
            totals = torch.zeros(unique_ids.numel(), dtype=torch.float64)
            totals.scatter_add_(0, inverse, steps)
            for bin_id, value in zip(unique_ids.tolist(), totals.tolist()):
                self._completed_per_bin[int(bin_id)] += float(value)
        if early_terminated is not None:
            self._early_terminations += int(early_terminated.detach().cpu().sum())

        if self._kernel_weights is not None and ids.numel():
            active = self.motion_lib.adp_samp_active_motion_bins.detach().cpu()
            membership = self._kernel_membership(active)
            position = self._global_to_active_position(ids, active)
            valid = position >= 0
            if bool(valid.any()):
                self._completed_kernel_steps += float(
                    (membership[position[valid]] * steps[valid]).sum()
                )

    def record_dropped_completion_batch(self) -> None:
        """Record a batch that could not be attributed exactly.

        Claim-mode callbacks raise immediately after incrementing this counter;
        exploratory callers may continue, but the resulting receipt is marked
        unusable.
        """

        self._dropped_completion_batches += 1

    def dose_registry_sha256(self) -> str:
        """Hash the immutable dataset-global mapping used for passive dose.

        ``adp_samp_active_motion_bins`` is intentionally absent. SONIC may
        rotate the resident batch during a continuation, but
        ``adp_samp_frame_to_bin`` and ``adp_samp_bins`` retain the same
        dataset-global identities. A change to that global mapping is stale;
        an ordinary resident-batch rotation is not.
        """

        lib = self.motion_lib
        rows = lib.adp_samp_bins.detach().cpu().to(torch.long)
        frame_to_bin = lib.adp_samp_frame_to_bin.detach().cpu().to(torch.long)
        keys = lib._motion_data_keys
        listed = keys.tolist() if hasattr(keys, "tolist") else list(keys)
        return sha256_of(
            {
                "global_bin_rows": [[int(value) for value in row] for row in rows.tolist()],
                "frame_to_bin": [int(value) for value in frame_to_bin.tolist()],
                "motion_keys": [str(value) for value in listed],
                "bin_size": int(lib.adp_samp_bin_size),
            }
        )

    def verify_dose_registry(self) -> str:
        """Hash the global registry once and verify it against installation.

        This belongs at receipt time, not in the per-simulation-step completion
        hook. The returned hash is then an explicit verification token for all
        context projections emitted by that receipt.
        """

        current = self.dose_registry_sha256()
        if current != self._dose_registry_sha256:
            raise StaleDoseRegistryError(
                "passive dose registry changed during the continuation: "
                f"{self._dose_registry_sha256} -> {current}"
            )
        return current

    def project_completed_kernel_steps(
        self,
        context: ContextKey,
        *,
        verified_registry_sha256: str,
        kernel_radius: int = 1,
        sigma_frames: float | None = None,
    ) -> DoseProjection:
        """Purely project the histogram using one receipt-time registry check."""

        if verified_registry_sha256 != self._dose_registry_sha256:
            raise StaleDoseRegistryError(
                "projection did not receive the verified installation registry hash"
            )
        spec = I.KernelSpec(
            radius_bins=kernel_radius,
            sigma_frames=float(
                sigma_frames if sigma_frames is not None else self.motion_lib.adp_samp_bin_size
            ),
        )
        # Build over the immutable global registry, not the rotating resident
        # list. This keeps projection pure and lets a shared control cover all
        # planned contexts even when a context is not resident at report time.
        # Duplicate active ids therefore cannot double count dose.
        lib = self.motion_lib
        rows = lib.adp_samp_bins.detach().cpu().to(torch.long)
        motion_ids, starts, ends = rows[:, 0], rows[:, 1], rows[:, 2]
        target_motion_id = self._motion_id_for_key(context.motion_key)
        self._validate_context_bin_identity(context)
        matches = torch.nonzero(
            (motion_ids == target_motion_id)
            & (starts == context.bin_start_frame)
            & (ends == context.bin_end_frame),
            as_tuple=False,
        ).flatten()
        if matches.numel() != 1:
            raise ValueError(f"context {context.context_id} maps to {matches.numel()} global bins")
        weights = I.build_local_kernel(
            target_position=int(matches[0]),
            bin_positions=starts // int(lib.adp_samp_bin_size),
            bin_motion_ids=motion_ids,
            target_motion_id=target_motion_id,
            bin_centre_frames=(starts + ends) // 2,
            spec=spec,
        )
        peak = float(weights.max())
        membership = weights / peak if peak > 0 else weights
        by_bin = {index: float(value) for index, value in enumerate(membership.tolist())}
        by_bin = {key: value for key, value in by_bin.items() if value > 0.0}
        completed = sum(
            self._completed_per_bin.get(key, 0.0) * value for key, value in by_bin.items()
        )
        kernel_hash = sha256_of(
            {
                "context_id": context.context_id,
                "kernel_radius_bins": spec.radius_bins,
                "sigma_frames": spec.sigma_frames,
                "membership_by_global_bin": {
                    str(key): value for key, value in sorted(by_bin.items())
                },
                "dose_registry_sha256": self._dose_registry_sha256,
            }
        )
        return DoseProjection(
            context_id=context.context_id,
            kernel_radius_bins=spec.radius_bins,
            sigma_frames=spec.sigma_frames,
            completed_kernel_steps=float(completed),
            membership_by_global_bin=by_bin,
            kernel_membership_sha256=kernel_hash,
        )

    def get_exact_dose_report(self, context_id: str | None = None) -> DoseReport:
        """Emit the realized-dose receipt for this branch."""
        if context_id is None:
            context_id = self._intervention.context.context_id if self._intervention else "native"
        per_motion: dict[str, float] = defaultdict(float)
        for bin_id, count in self._drawn_per_bin.items():
            per_motion[self.context_for_bin(bin_id).motion_key] += count
        return DoseReport(
            branch_id=self.branch_id,
            context_id=context_id,
            role=self.role,  # type: ignore[arg-type]
            drawn_episodes=self._drawn_episodes,
            drawn_kernel_mass=self._drawn_kernel_mass,
            completed_env_steps=self._completed_env_steps,
            completed_kernel_steps=self._completed_kernel_steps,
            total_env_steps=self._completed_env_steps,
            # This is the observable episode-start count: sampler draws after
            # instrumentation began. It intentionally excludes episodes that
            # were already resident at callback installation.
            total_episodes=self._drawn_episodes,
            per_bin_drawn=dict(self._drawn_per_bin),
            per_bin_completed=dict(self._completed_per_bin),
            per_motion_drawn=dict(per_motion),
            early_terminations=self._early_terminations,
            completion_hook_calls=self._completion_hook_calls,
            completion_observations=self._completion_observations,
            termination_observations=self._termination_observations,
            dropped_completion_batches=self._dropped_completion_batches,
            dose_registry_sha256=self._dose_registry_sha256,
        )

    # --------------------------------------------------------------- helpers --

    def _build_kernel(self, context: ContextKey, spec: I.KernelSpec) -> torch.Tensor:
        """Kernel over active bins, supported only on the context's own clip."""
        lib = self.motion_lib
        active = lib.adp_samp_active_motion_bins.detach().cpu()
        rows = lib.adp_samp_bins.detach().cpu()[active]
        motion_ids, starts, ends = rows[:, 0], rows[:, 1], rows[:, 2]

        target_motion_id = self._motion_id_for_key(context.motion_key)
        self._validate_context_bin_identity(context)
        match = (
            (motion_ids == target_motion_id)
            & (starts == context.bin_start_frame)
            & (ends == context.bin_end_frame)
        )
        positions = torch.nonzero(match, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(
                f"context {context.motion_key}[{context.bin_start_frame}:"
                f"{context.bin_end_frame}] is not resident in the current motion batch"
            )

        return I.build_local_kernel(
            target_position=int(positions[0]),
            bin_positions=starts // int(lib.adp_samp_bin_size),
            bin_motion_ids=motion_ids,
            target_motion_id=target_motion_id,
            bin_centre_frames=(starts + ends) // 2,
            spec=spec,
        )

    def _kernel_membership(self, active: torch.Tensor) -> torch.Tensor:
        """Per-active-bin kernel weight normalized so the peak is 1.0.

        Dose is an exposure count, not a probability, so the membership used to
        weight steps is scaled by its maximum rather than its sum: a step in the
        target bin counts as one unit of practice regardless of how many
        neighbours share the kernel.
        """
        assert self._kernel_weights is not None
        weights = self._kernel_weights
        if weights.numel() != active.numel():
            raise RuntimeError(
                "kernel is stale: the resident motion batch changed after the "
                "intervention was armed; re-arm after every motion resample"
            )
        peak = float(weights.max())
        return weights / peak if peak > 0 else weights

    @staticmethod
    def _global_to_active_position(
        global_bin_ids: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        """Map global bin ids to positions in the active array (-1 if absent)."""
        lookup = {int(b): i for i, b in enumerate(active.tolist())}
        return torch.tensor(
            [lookup.get(int(b), -1) for b in global_bin_ids.tolist()], dtype=torch.long
        )

    def _motion_id_for_key(self, motion_key: str) -> int:
        keys = self.motion_lib._motion_data_keys
        listed = keys.tolist() if hasattr(keys, "tolist") else list(keys)
        for index, key in enumerate(listed):
            if str(key) == motion_key:
                return index
        raise ValueError(f"motion key {motion_key!r} not present in the motion library")

    def _validate_context_bin_identity(self, context: ContextKey) -> None:
        bin_size = int(self.motion_lib.adp_samp_bin_size)
        if (
            context.bin_start_frame % bin_size != 0
            or context.bin_index != context.bin_start_frame // bin_size
        ):
            raise ValueError(
                f"context {context.context_id} bin_index does not match its reference-bin start"
            )

    def _motion_frame_count(self, orig_motion_id: int) -> int:
        frames = getattr(self.motion_lib, "adp_samp_num_frames", None)
        return int(frames[orig_motion_id]) if frames is not None else 0

    def _motion_hash_for(self, motion_key: str, num_frames: int) -> str:
        if self.manifest is not None and motion_key in self.manifest.motion_hashes:
            return self.manifest.motion_hashes[motion_key]
        return motion_hash(motion_key, num_frames, self._timeline_fps())

    def _timeline_fps(self) -> float:
        """Rate of the timeline the bins are defined on.

        ``_sim_fps`` (a scalar, ``1 / step_dt``) is the right choice, not
        ``_motion_fps``: the latter is a per-motion tensor of *source* clip
        rates indexed by batch-local motion id, whereas bin boundaries and
        ``adp_samp_num_frames`` live on the resampled simulation timeline and
        are keyed by dataset-wide motion id. Mixing the two indexing schemes
        would silently attach the wrong rate to a clip.
        """
        for attribute, default in (("_sim_fps", None), ("_motion_fps", None)):
            value = getattr(self.motion_lib, attribute, default)
            if value is None:
                continue
            if hasattr(value, "numel"):
                # A per-motion tensor cannot be reduced to one clip's rate here;
                # only accept it when it is genuinely scalar.
                if value.numel() != 1:
                    continue
                return float(value.reshape(-1)[0])
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 50.0

    @staticmethod
    def _gather(source: torch.Tensor, index: torch.Tensor) -> list[float]:
        return [float(v) for v in source.detach().cpu()[index].tolist()]
