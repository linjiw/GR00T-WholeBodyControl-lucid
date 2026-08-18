"""Data contracts for counterfactual practice-utility measurement.

Every claim-bearing artifact this program produces must be traceable from a
:class:`UtilityRecord` back to the source checkpoint, config, random streams,
motion pool, exact realized dose, and evaluation suite. These dataclasses are
that contract.

Design constraints (see ``lucid-design-implementation-plan.md``):

* Contexts are identified by a **stable motion hash**, never by a loader-order
  integer -- SONIC's ``_curr_motion_ids`` are batch-local and change between
  runs (``MotionLibBase.load_motions`` resamples the resident batch).
* Utility is normalized by **realized** dose, not nominal ``epsilon * H``:
  early termination, motion length, and parallel resampling all mean the
  intervention branch does not receive the dose it was asked for.
* A scalar efficacy delta is never sufficient. Every record carries a harm
  vector and a three-way safety label so that an action-quality regression
  cannot be hidden behind an improvement in success rate.

This module is deliberately free of torch/Isaac imports so contracts can be
validated on CPU, in isolation from the simulator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1

#: Three-way safety label for a measured context.
SafetyLabel = Literal["safe_positive", "neutral", "harmful"]

#: Which branch of a paired experiment produced a measurement.
BranchRole = Literal["control", "intervention"]


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` deterministically, for hashing and receipts."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(payload: Any) -> str:
    """Return the SHA-256 of the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def motion_hash(motion_key: str, num_frames: int, fps: float) -> str:
    """Stable identity for a motion clip.

    Uses the clip's key together with its frame count and sampling rate so a
    renamed-but-identical clip and a same-named-but-resampled clip are
    distinguishable. Claim-bearing artifacts must key on this, not on the
    batch-local motion id.
    """
    return sha256_of({"motion_key": motion_key, "num_frames": int(num_frames), "fps": float(fps)})


@dataclass(frozen=True)
class ContextKey:
    """Identity of a training context ``x``.

    Stage 1 of the program uses motion-only contexts (``perturbation_group``
    stays ``"native"``); the physics extension adds ``(group, severity)`` without
    changing the key's shape, so labels from both stages share one table.
    """

    motion_key: str
    motion_hash: str
    bin_index: int
    bin_start_frame: int
    bin_end_frame: int
    perturbation_group: str = "native"
    severity_level: int = 0
    encoder_mode: str = "g1"

    def __post_init__(self) -> None:
        if self.bin_end_frame <= self.bin_start_frame:
            raise ValueError(
                f"empty bin for {self.motion_key}: "
                f"[{self.bin_start_frame}, {self.bin_end_frame})"
            )
        if self.bin_index < 0:
            raise ValueError(f"negative bin_index for {self.motion_key}: {self.bin_index}")
        if self.severity_level < 0:
            raise ValueError(f"negative severity_level: {self.severity_level}")

    @property
    def num_frames(self) -> int:
        return self.bin_end_frame - self.bin_start_frame

    @property
    def context_id(self) -> str:
        """Stable, filesystem-safe identifier used in manifests and paths."""
        return sha256_of(dataclasses.asdict(self))[:16]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContextKey:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in fields})


@dataclass
class MotionPoolManifest:
    """The frozen set of motions available to every branch in a campaign.

    Control and intervention branches must have *identical* available support;
    otherwise SONIC's periodic motion-library reload becomes a second, unmatched
    treatment variable. ``disable_periodic_reload`` records that the reload was
    actually switched off for the oracle phase.
    """

    manifest_id: str
    motion_keys: list[str]
    motion_hashes: dict[str, str]
    source_root: str
    disable_periodic_reload: bool = True
    lace_split_partition: str | None = None
    lace_split_sha256: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        missing = [k for k in self.motion_keys if k not in self.motion_hashes]
        if missing:
            raise ValueError(f"{len(missing)} motion_keys lack hashes, e.g. {missing[:3]}")
        if len(set(self.motion_keys)) != len(self.motion_keys):
            raise ValueError("motion_keys contains duplicates")

    @property
    def num_motions(self) -> int:
        return len(self.motion_keys)

    @property
    def manifest_sha256(self) -> str:
        return sha256_of(
            {
                "motion_keys": sorted(self.motion_keys),
                "motion_hashes": self.motion_hashes,
                "source_root": self.source_root,
            }
        )


@dataclass
class SamplingSnapshot:
    """A frozen view of SONIC's native sampling distribution at one instant.

    Taken from the live ``MotionLibBase`` state rather than reconstructed, so the
    control branch trains under the genuine native sampler and any difference
    between branches is attributable to the intervention alone.
    """

    global_step: int
    num_bins: int
    active_bin_ids: list[int]
    active_prob: list[float]
    failure_rate_raw: list[float]
    num_episodes: list[float]
    num_failures: list[float]
    uniform_sampling_rate: float
    failure_rate_max_over_mean: float
    manifest_id: str

    def __post_init__(self) -> None:
        n = len(self.active_bin_ids)
        if len(self.active_prob) != n:
            raise ValueError(
                f"active_prob has {len(self.active_prob)} entries for {n} active bins"
            )
        if n and abs(sum(self.active_prob) - 1.0) > 1e-6:
            raise ValueError(f"active_prob sums to {sum(self.active_prob)!r}, expected 1.0")
        if any(p < 0.0 for p in self.active_prob):
            raise ValueError("active_prob contains negative entries")

    @property
    def distribution_sha256(self) -> str:
        return sha256_of({"bins": self.active_bin_ids, "prob": self.active_prob})

    @property
    def effective_num_bins(self) -> float:
        """Inverse-Simpson diversity of the distribution."""
        denom = sum(p * p for p in self.active_prob)
        return (1.0 / denom) if denom > 0 else 0.0


@dataclass
class DoseReport:
    """Realized exposure of one branch to a context's kernel.

    ``drawn_*`` counts sampling decisions; ``completed_*`` counts environment
    steps actually executed inside the kernel. They differ whenever episodes
    terminate early -- which is exactly when a difficult context is being
    probed, so the distinction is not academic.
    """

    branch_id: str
    context_id: str
    role: BranchRole
    drawn_episodes: float = 0.0
    drawn_kernel_mass: float = 0.0
    completed_env_steps: float = 0.0
    completed_kernel_steps: float = 0.0
    total_env_steps: float = 0.0
    total_episodes: float = 0.0
    per_bin_drawn: dict[int, float] = field(default_factory=dict)
    per_motion_drawn: dict[str, float] = field(default_factory=dict)
    early_terminations: int = 0
    distribution_kl: float | None = None
    sampling_entropy: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "drawn_episodes",
            "drawn_kernel_mass",
            "completed_env_steps",
            "completed_kernel_steps",
            "total_env_steps",
            "total_episodes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.completed_kernel_steps > self.completed_env_steps + 1e-6:
            raise ValueError(
                "completed_kernel_steps exceeds completed_env_steps "
                f"({self.completed_kernel_steps} > {self.completed_env_steps})"
            )

    @property
    def kernel_step_fraction(self) -> float:
        if self.completed_env_steps <= 0:
            return 0.0
        return self.completed_kernel_steps / self.completed_env_steps


@dataclass
class RngReceipt:
    """Evidence that two branches started from identical random state.

    GPU physics is not bitwise reproducible, so this is a *receipt*, never a
    proof of identical trajectories. The empirical noise floor comes from the
    epsilon=0 paired branches, not from these hashes.
    """

    python_state_sha256: str
    numpy_state_sha256: str
    torch_cpu_state_sha256: str
    torch_cuda_state_sha256: list[str]
    context_stream_key: str
    counter_rng_enabled: bool
    deterministic_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchCapsule:
    """Everything needed to restart training bit-for-bit-as-close-as-possible.

    SONIC's ``ModelSaveCallback`` persists model, optimizer, and env state but
    **not** RNG state, and not the adaptive-sampler counters in a form tied to a
    motion-pool manifest. Both gaps are fatal for paired continuation, so this
    capsule is the first piece of infrastructure the program builds.
    """

    branch_id: str
    pair_id: str
    role: BranchRole
    global_step: int

    policy_state_path: str
    optimizer_state_path: str
    trainer_state: dict[str, Any]
    env_state: dict[str, Any]
    native_sampler_state: dict[str, Any]

    rng: RngReceipt

    resolved_config_sha256: str
    motion_pool_manifest_sha256: str
    dev_suite_sha256: str
    source_commit: str
    checkpoint_sha256: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.role not in ("control", "intervention"):
            raise ValueError(f"unknown branch role: {self.role!r}")
        if self.global_step < 0:
            raise ValueError(f"negative global_step: {self.global_step}")

    @property
    def capsule_sha256(self) -> str:
        return sha256_of(
            {
                "branch_id": self.branch_id,
                "pair_id": self.pair_id,
                "role": self.role,
                "global_step": self.global_step,
                "config": self.resolved_config_sha256,
                "pool": self.motion_pool_manifest_sha256,
                "checkpoint": self.checkpoint_sha256,
                "commit": self.source_commit,
            }
        )


@dataclass
class HarmVector:
    """Paired deltas on physical-quality outcomes (intervention minus control).

    Sign convention: **positive means worse**. ``clean_delta`` is the exception
    and follows efficacy convention (positive means better on clean control),
    because it gates non-inferiority rather than harm.
    """

    clean_delta: float
    action_rate_delta: float
    slip_delta: float
    contact_impulse_delta: float
    torque_saturation_delta: float

    def exceeds(self, gates: dict[str, float]) -> list[str]:
        """Return the names of harm channels breaching their gate."""
        checks = {
            "action_rate": self.action_rate_delta,
            "slip": self.slip_delta,
            "contact_impulse": self.contact_impulse_delta,
            "torque_saturation": self.torque_saturation_delta,
        }
        breached = [name for name, value in checks.items() if value > gates.get(name, float("inf"))]
        if self.clean_delta < -abs(gates.get("clean_noninferiority", float("inf"))):
            breached.append("clean_noninferiority")
        return breached


@dataclass
class UtilityRecord:
    """One measured context at one checkpoint: the program's unit of evidence.

    ``efficacy_delta`` / ``utility`` are keyed by horizon label (``"H_s"``,
    ``"H_m"``, ``"H_l"``) so a single intervention branch evaluated at nested
    checkpoints yields the full vector without three independent trainings.
    """

    branch_pair_id: str
    context: ContextKey
    policy_stage: str
    seed: int
    horizons: dict[str, int]

    base_distribution_sha256: str
    intervention_distribution_sha256: str
    epsilon: float
    kernel_radius_bins: int

    control_dose: DoseReport
    intervention_dose: DoseReport

    efficacy_delta: dict[str, float] = field(default_factory=dict)
    utility: dict[str, float] = field(default_factory=dict)
    harm: dict[str, HarmVector] = field(default_factory=dict)
    safety_label: dict[str, SafetyLabel] = field(default_factory=dict)

    proxy_features: dict[str, float] = field(default_factory=dict)
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    #: Guard against dividing an efficacy delta by a vanishing realized dose.
    DOSE_EPSILON: float = 1e-6

    def __post_init__(self) -> None:
        if self.control_dose.role != "control":
            raise ValueError("control_dose must have role='control'")
        if self.intervention_dose.role != "intervention":
            raise ValueError("intervention_dose must have role='intervention'")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon out of range: {self.epsilon}")

    @property
    def realized_extra_dose(self) -> float:
        """Extra kernel exposure the intervention branch actually received."""
        return (
            self.intervention_dose.completed_kernel_steps
            - self.control_dose.completed_kernel_steps
        )

    def utility_at(self, horizon_label: str) -> float:
        """Dose-normalized practice utility at one horizon.

        Raises if the realized extra dose is non-positive: a branch whose
        intervention was swallowed by early termination carries no information
        about the context, and silently returning a huge or negative ratio would
        contaminate the label set.
        """
        if horizon_label not in self.efficacy_delta:
            raise KeyError(f"no efficacy delta recorded for horizon {horizon_label!r}")
        dose = self.realized_extra_dose
        if dose <= 0.0:
            raise ValueError(
                f"non-positive realized extra dose ({dose}) for context "
                f"{self.context.context_id}; this pair is not usable as a label"
            )
        return self.efficacy_delta[horizon_label] / (dose + self.DOSE_EPSILON)

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["context_id"] = self.context.context_id
        return payload
