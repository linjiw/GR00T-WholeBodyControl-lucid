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
from gear_sonic.research.practice_utility.observer import (  # noqa: F401
    CommandExecutionBuffer,
    PracticeObserverCallback,
)
from gear_sonic.research.practice_utility.quality_telemetry import (  # noqa: F401
    PracticeQualityCallback,
    QualityTelemetryCollector,
)
from gear_sonic.research.practice_utility.motion_pool import (  # noqa: F401
    MotionRecord,
    PoolScan,
    drop_exact_duplicates,
    motion_family,
    parse_motion_key,
    pool_sha256,
    scan_pool,
)
from gear_sonic.research.practice_utility.probe_manifest import (  # noqa: F401
    ContextCandidate,
    ManifestError,
    ProbeManifest,
    build_probe_manifest,
    stratified_select,
    validate_manifest,
)
from gear_sonic.research.practice_utility.proxy_audit import (  # noqa: F401
    GateBReport,
    ProxyResult,
    assess_sufficiency,
    audit_all_proxies,
    audit_proxy,
    count_reversals,
    spearman,
)
from gear_sonic.research.practice_utility.quality_metrics import (  # noqa: F401
    EpisodeQuality,
    QualityThresholds,
    apply_gates,
    evaluate_gates,
    macro_mean_quality_success,
    summarize,
)
from gear_sonic.research.practice_utility.sampler_adapter import (  # noqa: F401
    InterventionSpec,
    PracticeSamplerAdapter,
)
from gear_sonic.research.practice_utility.split import (  # noqa: F401
    DEFAULT_RATIOS,
    MotionSplit,
    SplitError,
    build_groups,
    build_split,
    verify_split,
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
    # motion pool and splits
    "MotionRecord",
    "PoolScan",
    "scan_pool",
    "drop_exact_duplicates",
    "motion_family",
    "parse_motion_key",
    "pool_sha256",
    "MotionSplit",
    "SplitError",
    "DEFAULT_RATIOS",
    "build_groups",
    "build_split",
    "verify_split",
    # campaign design
    "ContextCandidate",
    "ProbeManifest",
    "ManifestError",
    "build_probe_manifest",
    "stratified_select",
    "validate_manifest",
    # outcomes
    "EpisodeQuality",
    "QualityThresholds",
    "apply_gates",
    "evaluate_gates",
    "macro_mean_quality_success",
    "summarize",
    # audit
    "ProxyResult",
    "GateBReport",
    "audit_proxy",
    "audit_all_proxies",
    "assess_sufficiency",
    "count_reversals",
    "spearman",
    # live observation
    "PracticeObserverCallback",
    "CommandExecutionBuffer",
    "PracticeQualityCallback",
    "QualityTelemetryCollector",
    # capsules
    "CapsuleIntegrityError",
    "Provenance",
    "assert_fork_identical",
    "fork_pair",
    "load_capsule",
    "save_capsule",
]
