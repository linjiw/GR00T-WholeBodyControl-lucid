"""Counterfactual practice-utility measurement for SONIC motion curricula.

Layering (plan section 3.2):
  T0 infrastructure -> T1 measurement -> T2 method -> T3 physics/transfer.

Nothing in this package may alter SONIC behaviour unless a research feature is
explicitly enabled; ``tests/practice_utility/test_sampler_identity.py`` and the
no-op parity test enforce that.
"""

from gear_sonic.research.practice_utility.branch_capsule import (  # noqa: F401
    CapsuleIntegrityError,
    Provenance,
    assert_fork_identical,
    fork_pair,
    load_capsule,
    save_capsule,
)
from gear_sonic.research.practice_utility.intervention import (  # noqa: F401
    KernelSpec,
    build_local_kernel,
    kl_divergence,
    mix_intervention,
    residual_distribution,
)
from gear_sonic.research.practice_utility.rng_capsule import (  # noqa: F401
    MATCHED_CHANNELS,
    TREATMENT_CHANNEL,
    RngState,
    channel_generator,
    derive_seed,
    enable_determinism,
)
from gear_sonic.research.practice_utility.sampler_adapter import (  # noqa: F401
    InterventionSpec,
    PracticeSamplerAdapter,
)
from gear_sonic.research.practice_utility.schema import (  # noqa: F401
    SCHEMA_VERSION,
    BranchCapsule,
    ContextKey,
    DoseReport,
    HarmVector,
    MotionPoolManifest,
    RngReceipt,
    SamplingSnapshot,
    UtilityRecord,
    motion_hash,
    sha256_of,
)

__all__ = [
    "SCHEMA_VERSION",
    "MATCHED_CHANNELS",
    "TREATMENT_CHANNEL",
    # schema
    "BranchCapsule",
    "ContextKey",
    "DoseReport",
    "HarmVector",
    "MotionPoolManifest",
    "RngReceipt",
    "SamplingSnapshot",
    "UtilityRecord",
    "motion_hash",
    "sha256_of",
    # intervention
    "KernelSpec",
    "build_local_kernel",
    "kl_divergence",
    "mix_intervention",
    "residual_distribution",
    # sampler
    "InterventionSpec",
    "PracticeSamplerAdapter",
    # rng
    "RngState",
    "channel_generator",
    "derive_seed",
    "enable_determinism",
    # capsules
    "CapsuleIntegrityError",
    "Provenance",
    "assert_fork_identical",
    "fork_pair",
    "load_capsule",
    "save_capsule",
]
