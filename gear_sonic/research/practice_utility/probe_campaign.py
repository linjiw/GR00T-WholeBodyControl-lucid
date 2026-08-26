"""Fail-closed preflight for a claim-bearing practice-utility screen.

The v1 probe manifest freezes *which* contexts to measure, but it is not by
itself a runnable causal experiment.  In particular, it does not bind every
``(stage, seed)`` to a settled restart origin, prove that the selected contexts
are resident at that origin, freeze the latent proxy features, or specify a
deployment-side efficacy estimand.  Launching directly from that manifest can
therefore produce internally tidy receipts for the wrong comparison.

This module audits a separate, immutable preflight bundle.  It performs no
training, imports no simulator code, and deliberately returns blockers instead
of filling absent fields with defaults.  The bundle contract is intentionally
explicit:

``origin_maps``
    Hashed, per-stage receipts written by
    ``scripts/practice_utility/create_probe_origins.py``.  Each receipt contains
    every manifest seed and transitively binds a full branch capsule, its
    SONIC-compatible exported checkpoint, explicit trailing-window settled
    evidence, and a same-step sampler snapshot by content hash.  The manifest
    must have been selected from the intersection of those snapshots.

``encoder`` and ``proxy_features``
    One frozen encoder and one hashed feature table with exactly one row per
    ``(stage, seed, context)``.  Every row is linked to its origin snapshot and
    contains the latent-gap feature used by the proxy audit.

``efficacy_plan`` and ``noise_floor``
    Hashed JSON artifacts.  The plan defines deployment ``J_eff`` as a
    family-macro-mean of quality-qualified success, uses frozen policies and a
    frozen development suite, and prohibits the training-reward fallback.  The
    epsilon-zero noise floor must carry the *same estimand hash* and use settled
    symmetric fresh restarts.

``gate_preregistration``
    Preregistered Gate-A thresholds and two separately named Gate-B decisions:
    ``latent_proxy_predictiveness`` asks whether the frozen latent proxy predicts
    utility; ``inverse_estimator_authorization`` is the opposite-direction rule
    that authorizes an estimator only when no simple proxy is sufficient.  A
    generic ``gate_b_pass`` is rejected because it conflates those outcomes.
    Naming ``nested_cv_univariate_calibration`` in that file is only a plan: the
    leakage-free calibration is not implemented, and both Gate-B decisions stay
    blocked until an immutable algorithm implementation and its tests exist.

The screen manifest intentionally retains one shared control per
``(stage, seed)``. :func:`audit_probe_campaign` preserves that frozen branch
count and emits deterministic screening branch specifications. Passive
per-global-bin instrumentation and a frozen projection plan now have a CPU
contract, but the audit remains blocked until a hash-bound live-GPU smoke
receipt verifies the production hook end to end. It also warns that
confirmation evidence requires independent paired controls.
"""

# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for mixed import/from-import blocks.
# ruff: noqa: I001

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gear_sonic.research.practice_utility import branch_capsule as BC
from gear_sonic.research.practice_utility import directional_calibration as DC
from gear_sonic.research.practice_utility import dose_plan as DP
from gear_sonic.research.practice_utility.schema import ContextKey, sha256_of

PREFLIGHT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_EFFICACY_WINDOW = 4
MIN_NOISE_REPLICATES = 3

SHARED_CONTROL_STRATEGY = "shared_per_stage_seed"
INDEPENDENT_CONTROL_STRATEGY = "independent_per_context"
STOCHASTIC_RANDOMNESS_CONTRACT = "stochastic_potential_outcomes_no_channelwise_crn"

REQUIRED_QUALITY_THRESHOLDS = (
    "mpjpe",
    "foot_slip",
    "high_frequency_action",
    "undesired_contact",
    "torque_saturation",
)
REQUIRED_QUALITY_CONDITIONS = ("completion", *REQUIRED_QUALITY_THRESHOLDS)
REQUIRED_LATENT_FEATURES = ("latent_gap_p90", "latent_gap_median")
PROXY_FEATURE_KIND = "practice_utility_proxy_features"
NOISE_FLOOR_KIND = "practice_utility_same_estimand_noise_floor"
GATE_PREREGISTRATION_KIND = "practice_utility_gate_preregistration"
DIRECTIONAL_PREREGISTRATION_KIND = "practice_utility_latent_directional_calibration_preregistration"
LIVE_PASSIVE_DOSE_SMOKE_KIND = "practice_utility_live_passive_dose_smoke"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PASSIVE_DOSE_PLAN_LAUNCHER = (
    _REPO_ROOT / "scripts/practice_utility/create_passive_dose_plan.py"
).resolve()
_DIRECTIONAL_CALIBRATION_LAUNCHER = (
    _REPO_ROOT / "scripts/practice_utility/freeze_directional_calibration.py"
).resolve()
_PASSIVE_DOSE_SMOKE_LAUNCHER = (
    _REPO_ROOT / "scripts/practice_utility/run_passive_dose_smoke.py"
).resolve()
_PASSIVE_DOSE_CALLBACK = (
    _REPO_ROOT / "gear_sonic/research/practice_utility/callbacks.py"
).resolve()
_PASSIVE_DOSE_SOURCE_PATHS = {
    "sampler_adapter": _REPO_ROOT / "gear_sonic/research/practice_utility/sampler_adapter.py",
    "dose_plan": _REPO_ROOT / "gear_sonic/research/practice_utility/dose_plan.py",
    "schema": _REPO_ROOT / "gear_sonic/research/practice_utility/schema.py",
    "intervention": _REPO_ROOT / "gear_sonic/research/practice_utility/intervention.py",
    "branch_capsule": _REPO_ROOT / "gear_sonic/research/practice_utility/branch_capsule.py",
    "train_agent": _REPO_ROOT / "gear_sonic/train_agent_trl.py",
    "ppo_trainer": _REPO_ROOT / "gear_sonic/trl/trainer/ppo_trainer.py",
    "manager_env_wrapper": _REPO_ROOT / "gear_sonic/envs/wrapper/manager_env_wrapper.py",
}


@dataclass(frozen=True)
class Finding:
    """One auditable preflight conclusion."""

    code: str
    message: str
    scope: str = "campaign"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "scope": self.scope, "message": self.message}


@dataclass(frozen=True)
class BranchSpec:
    """A deterministic, non-executing branch description.

    Horizons in the frozen manifest are relative continuation lengths.  SONIC's
    resumed trainer expects an absolute target, so both representations are
    included and ``target_global_step`` is always the largest absolute horizon.
    """

    branch_id: str
    role: str
    stage: str
    seed: int
    origin_key: str
    origin_global_step: int
    capsule_path: str
    checkpoint_path: str
    relative_horizons: dict[str, int]
    absolute_horizons: dict[str, int]
    target_global_step: int
    manifest_sha256: str
    context_index: int | None = None
    context_id: str | None = None
    control_branch_id: str | None = None
    screening_control_strategy: str = SHARED_CONTROL_STRATEGY

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "role": self.role,
            "stage": self.stage,
            "seed": self.seed,
            "origin_key": self.origin_key,
            "origin_global_step": self.origin_global_step,
            "capsule_path": self.capsule_path,
            "checkpoint_path": self.checkpoint_path,
            "relative_horizons": self.relative_horizons,
            "absolute_horizons": self.absolute_horizons,
            "target_global_step": self.target_global_step,
            "manifest_sha256": self.manifest_sha256,
            "context_index": self.context_index,
            "context_id": self.context_id,
            "control_branch_id": self.control_branch_id,
            "screening_control_strategy": self.screening_control_strategy,
        }


@dataclass
class PreflightReport:
    """Machine-readable result of :func:`audit_probe_campaign`."""

    manifest_path: str
    preflight_path: str
    manifest_sha256: str | None = None
    manifest_file_sha256: str | None = None
    preflight_file_sha256: str | None = None
    verified: list[Finding] = field(default_factory=list)
    blockers: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    branch_specs: list[BranchSpec] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "practice_utility_probe_preflight_report",
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "ready" if self.ready else "blocked",
            "ready": self.ready,
            "manifest_path": self.manifest_path,
            "preflight_path": self.preflight_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "preflight_file_sha256": self.preflight_file_sha256,
            "verified": [item.to_dict() for item in self.verified],
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "branch_specs": [spec.to_dict() for spec in self.branch_specs],
            "branch_counts": {
                "control": sum(spec.role == "control" for spec in self.branch_specs),
                "intervention": sum(spec.role == "intervention" for spec in self.branch_specs),
                "total": len(self.branch_specs),
            },
            "confirmation_requirement": INDEPENDENT_CONTROL_STRATEGY,
            "randomness_contract": STOCHASTIC_RANDOMNESS_CONTRACT,
            "gate_b_vocabulary": {
                "proxy_question": "latent_proxy_predictiveness",
                "authorization_question": "inverse_estimator_authorization",
            },
        }


@dataclass(frozen=True)
class _ManifestView:
    payload: dict[str, Any]
    campaign_id: str
    stages: tuple[str, ...]
    seeds: tuple[int, ...]
    horizons: dict[str, int]
    contexts: dict[str, tuple[tuple[int, str, dict[str, Any]], ...]]
    manifest_sha256: str

    @property
    def expected_origin_keys(self) -> tuple[str, ...]:
        return tuple(origin_key(stage, seed) for stage in self.stages for seed in self.seeds)

    @property
    def expected_feature_keys(self) -> set[tuple[str, int, str]]:
        return {
            (stage, seed, context_id)
            for stage in self.stages
            for seed in self.seeds
            for _, context_id, _ in self.contexts[stage]
        }


@dataclass(frozen=True)
class _Asset:
    path: Path
    sha256: str


@dataclass
class _Origin:
    key: str
    stage: str
    seed: int
    global_step: int
    capsule: _Asset
    checkpoint: _Asset
    capsule_payload: dict[str, Any]
    checkpoint_payload: dict[str, Any]
    snapshot: _Asset | None = None
    snapshot_payload: dict[str, Any] | None = None

    @property
    def capsule_sha256(self) -> str:
        return str(self.capsule_payload.get("capsule_sha256", ""))


class _Audit:
    def __init__(self, report: PreflightReport) -> None:
        self.report = report

    def verified(self, code: str, message: str, scope: str = "campaign") -> None:
        self.report.verified.append(Finding(code, message, scope))

    def blocker(self, code: str, message: str, scope: str = "campaign") -> None:
        self.report.blockers.append(Finding(code, message, scope))

    def warning(self, code: str, message: str, scope: str = "campaign") -> None:
        self.report.warnings.append(Finding(code, message, scope))


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a checkpoint into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def origin_key(stage: str, seed: int) -> str:
    """Canonical key used by the preflight origin map."""

    return f"{stage}:s{int(seed)}"


def manifest_claim_sha256(payload: Mapping[str, Any]) -> str:
    """Recompute :class:`ProbeManifest`'s claim-bearing logical hash."""

    contexts = payload.get("contexts_per_stage") or {}
    frozen_contexts: dict[str, list[str]] = {}
    for stage, entries in sorted(contexts.items()):
        frozen_contexts[str(stage)] = sorted(
            str(entry.get("context_id") or _context_id(entry.get("context"))) for entry in entries
        )
    return sha256_of(
        {
            "campaign_id": payload.get("campaign_id"),
            "contexts": frozen_contexts,
            "seeds": sorted(payload.get("seeds") or []),
            "epsilon": payload.get("epsilon"),
            "kernel_radius_bins": payload.get("kernel_radius_bins"),
            "horizons": payload.get("horizons") or {},
            "pool_sha256": payload.get("pool_sha256"),
            "split_sha256": payload.get("split_sha256"),
        }
    )


def default_preflight_path(manifest_path: str | Path) -> Path:
    """Return ``<manifest-stem>.preflight.json`` next to a manifest."""

    manifest = Path(manifest_path)
    return manifest.with_name(f"{manifest.stem}.preflight.json")


def audit_probe_campaign(
    manifest_path: str | Path,
    preflight_path: str | Path | None = None,
) -> PreflightReport:
    """Audit a frozen probe screen without launching any branch.

    Missing or malformed evidence is accumulated in ``blockers`` so one run
    reports the whole repair list.  No inferred origin, filename convention, or
    training-side efficacy fallback can make ``ready`` true.
    """

    manifest_path = Path(manifest_path).resolve()
    bundle_path = Path(preflight_path or default_preflight_path(manifest_path)).resolve()
    report = PreflightReport(str(manifest_path), str(bundle_path))
    audit = _Audit(report)

    manifest_payload = _read_json(manifest_path, audit, "manifest", "manifest")
    if manifest_payload is None:
        _missing_bundle_contract(audit, include_bundle=False)
        return report

    report.manifest_file_sha256 = sha256_file(manifest_path)
    manifest = _validate_manifest(manifest_payload, audit)
    if manifest is not None:
        report.manifest_sha256 = manifest.manifest_sha256
        _shared_control_notice(manifest, audit)

    if not bundle_path.is_file():
        audit.blocker(
            "preflight_bundle_missing",
            f"no preregistration bundle exists at {bundle_path}",
            "preflight",
        )
        _missing_bundle_contract(audit, include_bundle=False)
        return report

    report.preflight_file_sha256 = sha256_file(bundle_path)
    bundle = _read_json(bundle_path, audit, "preflight", "preflight")
    if bundle is None or manifest is None:
        return report

    _validate_bundle_header(bundle, manifest, report, audit)
    base_dir = bundle_path.parent

    passive_plan = _validate_passive_dose_plan(
        bundle.get("passive_dose_plan"), manifest, report, base_dir, audit
    )
    live_passive_smoke = _validate_live_passive_dose_smoke(
        bundle.get("live_passive_dose_smoke"), manifest, report, passive_plan, base_dir, audit
    )
    if not live_passive_smoke:
        _block_live_passive_dose(audit)

    directional_asset, directional_preregistration = (
        _validate_directional_calibration_preregistration(
            bundle.get("directional_calibration_preregistration"),
            manifest,
            report,
            base_dir,
            audit,
        )
    )

    efficacy_plan, efficacy_asset, estimand_sha256, dev_suite_sha256 = _validate_efficacy_plan(
        bundle.get("efficacy_plan"), manifest, base_dir, audit
    )
    origins = _validate_origin_maps(
        bundle.get("origin_maps"), manifest, base_dir, dev_suite_sha256, audit
    )
    encoder = _validate_encoder(bundle.get("encoder"), base_dir, audit)
    proxy_asset = _validate_proxy_features(
        bundle.get("proxy_features"), manifest, origins, encoder, base_dir, audit
    )
    noise_asset = _validate_noise_floor(
        bundle.get("noise_floor"),
        manifest,
        efficacy_asset,
        estimand_sha256,
        base_dir,
        audit,
    )
    _validate_gates(
        bundle.get("gate_preregistration"),
        manifest,
        efficacy_plan,
        proxy_asset,
        noise_asset,
        estimand_sha256,
        directional_asset,
        directional_preregistration,
        base_dir,
        audit,
    )

    strategy = bundle.get("screening_control_strategy")
    if strategy != SHARED_CONTROL_STRATEGY:
        audit.blocker(
            "screening_control_strategy_changed",
            f"frozen screen requires {SHARED_CONTROL_STRATEGY!r}, got {strategy!r}; "
            "do not silently change its branch count",
            "preflight",
        )

    if len(origins) == len(manifest.expected_origin_keys):
        report.branch_specs = _build_branch_specs(manifest, origins)
        expected_interventions = sum(len(manifest.contexts[s]) for s in manifest.stages) * len(
            manifest.seeds
        )
        expected_controls = len(manifest.stages) * len(manifest.seeds)
        controls = sum(spec.role == "control" for spec in report.branch_specs)
        interventions = sum(spec.role == "intervention" for spec in report.branch_specs)
        if (controls, interventions) == (expected_controls, expected_interventions):
            audit.verified(
                "branch_specs_deterministic",
                f"derived {controls} shared controls and {interventions} interventions "
                "with origin-relative and absolute horizons",
                "branches",
            )
        else:  # pragma: no cover - defensive invariant
            audit.blocker(
                "branch_count_mismatch",
                f"derived {controls}/{interventions} control/intervention branches, expected "
                f"{expected_controls}/{expected_interventions}",
                "branches",
            )
    else:
        audit.blocker(
            "branch_specs_unavailable",
            "deterministic branch specifications require one verified origin per (stage, seed)",
            "branches",
        )

    if efficacy_plan is not None:
        audit.verified(
            "deployment_efficacy_plan_loaded",
            "deployment J_eff plan is frozen and training-metric fallback is disabled",
            "efficacy_plan",
        )
    return report


def _read_json(path: Path, audit: _Audit, code_prefix: str, scope: str) -> dict[str, Any] | None:
    if not path.is_file():
        audit.blocker(f"{code_prefix}_missing", f"required JSON artifact is missing: {path}", scope)
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        audit.blocker(f"{code_prefix}_invalid_json", f"cannot read {path} as JSON: {error}", scope)
        return None
    if not isinstance(payload, dict):
        audit.blocker(f"{code_prefix}_not_object", f"{path} must contain a JSON object", scope)
        return None
    return payload


def _validate_manifest(payload: dict[str, Any], audit: _Audit) -> _ManifestView | None:
    if payload.get("kind") != "practice_utility_probe_manifest":
        audit.blocker("manifest_kind_invalid", "not a practice-utility probe manifest", "manifest")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        audit.blocker(
            "manifest_schema_invalid",
            f"manifest schema must be {MANIFEST_SCHEMA_VERSION}",
            "manifest",
        )

    campaign = payload.get("campaign_id")
    stages_raw = payload.get("stages")
    seeds_raw = payload.get("seeds")
    horizons_raw = payload.get("horizons")
    contexts_raw = payload.get("contexts_per_stage")
    if not isinstance(campaign, str) or not campaign:
        audit.blocker("manifest_campaign_missing", "manifest has no campaign_id", "manifest")
        return None
    if (
        not isinstance(stages_raw, list)
        or not stages_raw
        or not all(isinstance(stage, str) and stage for stage in stages_raw)
    ):
        audit.blocker(
            "manifest_stages_invalid", "manifest stages must be non-empty strings", "manifest"
        )
        return None
    if len(set(stages_raw)) != len(stages_raw):
        audit.blocker("manifest_stages_duplicate", "manifest stages contain duplicates", "manifest")
    if (
        not isinstance(seeds_raw, list)
        or not seeds_raw
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds_raw)
    ):
        audit.blocker(
            "manifest_seeds_invalid", "manifest seeds must be non-empty integers", "manifest"
        )
        return None
    if len(set(seeds_raw)) != len(seeds_raw):
        audit.blocker("manifest_seeds_duplicate", "manifest seeds contain duplicates", "manifest")
    if not isinstance(horizons_raw, dict) or not horizons_raw:
        audit.blocker("manifest_horizons_invalid", "manifest has no relative horizons", "manifest")
        return None
    horizons: dict[str, int] = {}
    for label, value in horizons_raw.items():
        if (
            not isinstance(label, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            audit.blocker(
                "manifest_horizon_invalid",
                f"horizon {label!r} must be a positive integer, got {value!r}",
                "manifest",
            )
            continue
        horizons[label] = value
    if len(set(horizons.values())) != len(horizons):
        audit.blocker(
            "manifest_horizons_duplicate", "relative horizon values must be unique", "manifest"
        )
    if not horizons:
        return None
    if not isinstance(contexts_raw, dict) or set(contexts_raw) != set(stages_raw):
        audit.blocker(
            "manifest_context_stages_mismatch",
            "contexts_per_stage must cover exactly the declared stages",
            "manifest",
        )
        return None

    contexts: dict[str, tuple[tuple[int, str, dict[str, Any]], ...]] = {}
    for stage in stages_raw:
        entries = contexts_raw.get(stage)
        if not isinstance(entries, list) or not entries:
            audit.blocker(
                "manifest_contexts_missing", f"stage {stage!r} has no contexts", f"stage:{stage}"
            )
            continue
        parsed: list[tuple[int, str, dict[str, Any]]] = []
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            scope = f"stage:{stage}/context:{index}"
            if not isinstance(entry, dict) or not isinstance(entry.get("context"), dict):
                audit.blocker("manifest_context_invalid", "context entry is malformed", scope)
                continue
            context_payload = _canonical_context(entry["context"])
            if context_payload is None:
                audit.blocker("manifest_context_invalid", "ContextKey validation failed", scope)
                continue
            computed = _context_id(context_payload)
            declared = entry.get("context_id")
            if declared != computed:
                audit.blocker(
                    "manifest_context_id_mismatch",
                    f"declared context_id {declared!r} != computed {computed}",
                    scope,
                )
                continue
            if computed in seen:
                audit.blocker("manifest_context_duplicate", f"duplicate context {computed}", scope)
                continue
            seen.add(computed)
            parsed.append((index, computed, context_payload))
        contexts[stage] = tuple(parsed)

    if set(contexts) != set(stages_raw) or any(not contexts.get(stage) for stage in stages_raw):
        return None

    recorded = payload.get("manifest_sha256")
    computed_hash = manifest_claim_sha256(payload)
    if not _is_sha(recorded):
        audit.blocker("manifest_hash_invalid", "manifest_sha256 is not a SHA-256", "manifest")
    elif recorded != computed_hash:
        audit.blocker(
            "manifest_hash_mismatch",
            f"recorded manifest hash {recorded} != recomputed {computed_hash}",
            "manifest",
        )
    else:
        audit.verified("manifest_hash_verified", f"frozen manifest hash {recorded}", "manifest")

    expected_interventions = sum(len(contexts[stage]) for stage in stages_raw) * len(seeds_raw)
    expected_controls = len(stages_raw) * len(seeds_raw)
    if payload.get("num_intervention_branches") != expected_interventions:
        audit.blocker(
            "manifest_intervention_count_mismatch",
            f"manifest declares {payload.get('num_intervention_branches')} interventions, "
            f"but its frozen contexts require {expected_interventions}",
            "manifest",
        )
    if payload.get("num_control_branches") != expected_controls:
        audit.blocker(
            "manifest_control_count_mismatch",
            f"manifest declares {payload.get('num_control_branches')} controls, "
            f"but shared screening requires {expected_controls}",
            "manifest",
        )

    return _ManifestView(
        payload=payload,
        campaign_id=campaign,
        stages=tuple(sorted(stages_raw)),
        seeds=tuple(sorted(seeds_raw)),
        horizons=dict(sorted(horizons.items(), key=lambda item: (item[1], item[0]))),
        contexts=contexts,
        manifest_sha256=str(recorded),
    )


def _shared_control_notice(manifest: _ManifestView, audit: _Audit) -> None:
    controls = len(manifest.stages) * len(manifest.seeds)
    audit.verified(
        "screening_branch_count_preserved",
        f"frozen screen retains {controls} shared controls (one per stage and seed)",
        "branches",
    )
    audit.warning(
        "shared_controls_screening_only",
        "shared controls are preserved for this screen; confirmation must use independent "
        "paired controls for every confirmed context",
        "branches",
    )
    audit.warning(
        "channel_keyed_crn_unimplemented",
        "counter_rng_enabled in historical capsules is self-reported: derive_seed has no "
        "production call sites. This screen uses a stochastic potential-outcome estimand "
        "with symmetric fresh restarts and a measured same-estimand floor; it makes no "
        "channel-wise common-random-number claim.",
        "randomness",
    )


def _validate_bundle_header(
    bundle: dict[str, Any], manifest: _ManifestView, report: PreflightReport, audit: _Audit
) -> None:
    if bundle.get("kind") != "practice_utility_probe_preflight":
        audit.blocker("preflight_kind_invalid", "preflight kind is invalid", "preflight")
    if bundle.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        audit.blocker(
            "preflight_schema_invalid",
            f"preflight schema must be {PREFLIGHT_SCHEMA_VERSION}",
            "preflight",
        )
    if bundle.get("preregistered") is not True:
        audit.blocker(
            "preflight_not_preregistered",
            "preflight must explicitly set preregistered=true before labels are inspected",
            "preflight",
        )
    if bundle.get("manifest_sha256") != manifest.manifest_sha256:
        audit.blocker(
            "preflight_manifest_hash_mismatch",
            "preflight does not bind the frozen logical manifest hash",
            "preflight",
        )
    if bundle.get("manifest_file_sha256") != report.manifest_file_sha256:
        audit.blocker(
            "preflight_manifest_file_hash_mismatch",
            "preflight does not bind the exact manifest file bytes",
            "preflight",
        )
    if bundle.get("randomness_contract") != STOCHASTIC_RANDOMNESS_CONTRACT:
        audit.blocker(
            "randomness_contract_invalid",
            f"preflight must declare {STOCHASTIC_RANDOMNESS_CONTRACT!r}; channel-keyed CRN "
            "is not implemented in production",
            "preflight",
        )
    if (
        bundle.get("manifest_sha256") == manifest.manifest_sha256
        and bundle.get("manifest_file_sha256") == report.manifest_file_sha256
    ):
        audit.verified(
            "preflight_manifest_binding_verified",
            "preflight binds both logical and file hashes of the frozen manifest",
            "preflight",
        )


def _validate_passive_dose_plan(
    reference: Any,
    manifest: _ManifestView,
    report: PreflightReport,
    base_dir: Path,
    audit: _Audit,
) -> DP.PassiveDosePlan | None:
    """Validate the frozen, outcome-blind projection plan for shared controls."""

    scope = "dose_plan"
    asset = _verify_asset(reference, base_dir, audit, "passive_dose_plan", scope)
    if asset is None:
        return None
    try:
        plan = DP.load_passive_dose_plan(
            asset.path,
            expected_file_sha256=asset.sha256,
            expected_campaign_id=manifest.campaign_id,
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_manifest_file_sha256=report.manifest_file_sha256,
        )
    except ValueError as error:
        audit.blocker("passive_dose_plan_invalid", str(error), scope)
        return None

    expected = {
        stage: {context_id: context for _, context_id, context in manifest.contexts[stage]}
        for stage in manifest.stages
    }
    actual = {
        stage: {context.context_id: context.to_dict() for context in plan.contexts_for(stage)}
        for stage in sorted(plan.contexts_per_stage)
    }
    if actual != expected:
        audit.blocker(
            "passive_dose_plan_contexts_mismatch",
            "passive dose plan must contain each manifest context exactly once by stage",
            scope,
        )
    if plan.kernel_radius_bins != manifest.payload.get("kernel_radius_bins"):
        audit.blocker(
            "passive_dose_plan_kernel_mismatch",
            "passive dose projection radius differs from the frozen intervention kernel",
            scope,
        )
    manifest_contexts = [
        ContextKey.from_dict(context)
        for stage in manifest.stages
        for _, _, context in manifest.contexts[stage]
    ]
    try:
        reference_bin_size = DP.derive_reference_bin_size_frames(manifest_contexts)
    except ValueError as error:
        audit.blocker("passive_dose_plan_reference_bin_invalid", str(error), scope)
    else:
        if plan.reference_bin_size_frames != reference_bin_size or plan.sigma_frames != float(
            reference_bin_size
        ):
            audit.blocker(
                "passive_dose_plan_reference_bin_invalid",
                "passive dose plan sigma/reference size differs from the manifest bins",
                scope,
            )
    expected_launcher_sha256 = (
        sha256_file(_PASSIVE_DOSE_PLAN_LAUNCHER) if _PASSIVE_DOSE_PLAN_LAUNCHER.is_file() else None
    )
    provenance_valid = (
        Path(plan.source_manifest_path).resolve() == Path(report.manifest_path).resolve()
        and _FULL_COMMIT.fullmatch(plan.source_commit) is not None
        and Path(plan.launcher_path).resolve() == _PASSIVE_DOSE_PLAN_LAUNCHER
        and plan.launcher_sha256 == expected_launcher_sha256
    )
    if not provenance_valid:
        audit.blocker(
            "passive_dose_plan_provenance_invalid",
            "passive dose plan must bind the exact manifest, a clean full Git commit, "
            "and the current plan-freezing launcher bytes",
            scope,
        )
    if not any(item.scope == scope for item in audit.report.blockers):
        audit.verified(
            "passive_dose_plan_verified",
            f"v2 plan {plan.logical_sha256} covers every frozen context",
            scope,
        )
        return plan
    return None


def _validate_live_passive_dose_smoke(
    reference: Any,
    manifest: _ManifestView,
    report: PreflightReport,
    plan: DP.PassiveDosePlan | None,
    base_dir: Path,
    audit: _Audit,
) -> bool:
    """Require immutable, recomputable live CUDA evidence for the step hook."""

    scope = "live_passive_dose_smoke"
    asset = _verify_asset(reference, base_dir, audit, "live_passive_dose_smoke", scope)
    if asset is None or plan is None:
        return False
    smoke = _read_json(asset.path, audit, "live_passive_dose_smoke", scope)
    if smoke is None:
        return False
    initial_blockers = len(audit.report.blockers)
    if smoke.get("kind") != LIVE_PASSIVE_DOSE_SMOKE_KIND or smoke.get("schema_version") != 1:
        audit.blocker(
            "live_passive_dose_smoke_schema_invalid",
            "live passive-dose smoke has the wrong kind or schema",
            scope,
        )
    recorded_smoke_hash = smoke.get("smoke_sha256")
    smoke_body = dict(smoke)
    smoke_body.pop("smoke_sha256", None)
    if not _is_sha(recorded_smoke_hash) or recorded_smoke_hash != sha256_of(smoke_body):
        audit.blocker(
            "live_passive_dose_smoke_logical_hash_invalid",
            "live passive-dose smoke logical hash is invalid",
            scope,
        )
    expected_links = {
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": report.manifest_file_sha256,
        "measurement_hook": DP.PASSIVE_DOSE_HOOK,
        "execution_mode": "live_gpu_simulator",
        "device_type": "cuda",
        "status": "complete",
    }
    mismatched = [key for key, value in expected_links.items() if smoke.get(key) != value]
    if mismatched:
        audit.blocker(
            "live_passive_dose_smoke_lineage_invalid",
            f"live passive-dose smoke differs on {mismatched}",
            scope,
        )
    if (
        not _FULL_COMMIT.fullmatch(str(smoke.get("source_commit", "")))
        or smoke.get("source_tree_status") != []
    ):
        audit.blocker(
            "live_passive_dose_smoke_commit_invalid",
            "live smoke requires a full 40-character commit and clean source tree",
            scope,
        )
    plan_link = smoke.get("passive_dose_plan")
    if not isinstance(plan_link, dict) or plan_link != {
        "file_sha256": plan.file_sha256,
        "logical_sha256": plan.logical_sha256,
    }:
        audit.blocker(
            "live_passive_dose_smoke_plan_mismatch",
            "live passive-dose smoke does not bind the verified v2 dose plan",
            scope,
        )
    stage = smoke.get("stage")
    if not isinstance(stage, str) or stage not in manifest.stages:
        audit.blocker(
            "live_passive_dose_smoke_stage_invalid",
            "live passive-dose smoke must name a manifest stage",
            scope,
        )

    implementation = smoke.get("implementation")
    implementation_valid = isinstance(implementation, dict)
    if implementation_valid:
        implementation_valid = _binding_matches_current(
            implementation.get("launcher"), _PASSIVE_DOSE_SMOKE_LAUNCHER
        ) and _binding_matches_current(implementation.get("callback"), _PASSIVE_DOSE_CALLBACK)
        implementation_valid = implementation_valid and _binding_is_hash_valid(
            implementation.get("source_config")
        )
        sources = implementation.get("sources")
        implementation_valid = implementation_valid and isinstance(sources, dict)
        if isinstance(sources, dict):
            implementation_valid = (
                implementation_valid
                and set(sources) == set(_PASSIVE_DOSE_SOURCE_PATHS)
                and all(
                    _binding_matches_current(sources.get(name), path)
                    for name, path in _PASSIVE_DOSE_SOURCE_PATHS.items()
                )
            )
    if not implementation_valid:
        audit.blocker(
            "live_passive_dose_smoke_implementation_invalid",
            "live smoke must bind the current launcher, callback, and complete source set",
            scope,
        )

    prereg_asset = _verify_asset(
        smoke.get("preregistration"),
        asset.path.parent,
        audit,
        "live_passive_dose_preregistration",
        scope,
        record_success=False,
    )
    prereg = (
        _read_json(prereg_asset.path, audit, "live_passive_dose_preregistration", scope)
        if prereg_asset is not None
        else None
    )
    prereg_valid = prereg is not None and _validate_passive_smoke_preregistration(
        prereg,
        manifest=manifest,
        report=report,
        plan=plan,
        smoke=smoke,
        implementation=implementation if isinstance(implementation, dict) else {},
    )
    if not prereg_valid:
        audit.blocker(
            "live_passive_dose_preregistration_invalid",
            "live smoke preregistration is not immutable or does not bind its exact inputs/code",
            scope,
        )

    dose_asset = _verify_asset(
        smoke.get("dose_receipt"),
        asset.path.parent,
        audit,
        "live_passive_dose_receipt",
        scope,
        record_success=False,
    )
    dose = (
        _read_json(dose_asset.path, audit, "live_passive_dose_receipt", scope)
        if dose_asset is not None
        else None
    )
    if (
        dose is None
        or prereg is None
        or not _validate_live_dose_invariants(
            dose,
            dose_asset=dose_asset,
            prereg=prereg,
            manifest=manifest,
            report=report,
            plan=plan,
            stage=stage,
            smoke=smoke,
        )
    ):
        audit.blocker(
            "live_passive_dose_receipt_invalid",
            "live dose receipt fails recomputed horizon, accounting, lineage, or projection checks",
            scope,
        )

    runtime = smoke.get("runtime")
    log_asset = None
    if isinstance(runtime, dict):
        log_asset = _verify_asset(
            runtime.get("log"),
            asset.path.parent,
            audit,
            "live_passive_dose_log",
            scope,
            record_success=False,
        )
    runtime_valid = (
        isinstance(runtime, dict)
        and runtime.get("exit_code") == 0
        and _is_positive_number(runtime.get("wall_seconds"))
        and isinstance(runtime.get("cuda_memory_growth_mib"), (int, float))
        and not isinstance(runtime.get("cuda_memory_growth_mib"), bool)
        and float(runtime["cuda_memory_growth_mib"]) >= 512.0
        and isinstance(runtime.get("gpu"), Mapping)
        and bool(runtime["gpu"])
        and log_asset is not None
        and log_asset.path.stat().st_size > 0
    )
    if not runtime_valid:
        audit.blocker(
            "live_passive_dose_runtime_invalid",
            "live smoke must bind a successful CUDA runtime and non-empty hashed log",
            scope,
        )

    checks = smoke.get("checks")
    required_checks = (
        "passive_completion_exact",
        "epsilon_zero_control",
        "dose_registry_stable",
        "exact_context_projection",
        "cuda_execution_verified",
        "atomic_receipt_no_partial",
        "immutable_preregistration_bound",
        "successful_runtime_and_log",
    )
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in required_checks):
        audit.blocker(
            "live_passive_dose_smoke_checks_incomplete",
            f"live smoke must pass {list(required_checks)}",
            scope,
        )
    not_yet_verified = smoke.get("not_yet_verified")
    required_not_yet = {
        "native step return preservation in the live process",
        "native distribution bitwise identity against a paired no-callback reference",
        "callback wrapper removal observed after on_train_end",
    }
    if not isinstance(not_yet_verified, list) or not required_not_yet <= set(
        str(item) for item in not_yet_verified
    ):
        audit.blocker(
            "live_passive_dose_smoke_scope_invalid",
            "live smoke must leave native identity and wrapper-removal claims explicitly "
            "not-yet-verified",
            scope,
        )
    if len(audit.report.blockers) == initial_blockers:
        audit.verified(
            "live_passive_dose_smoke_verified",
            "live CUDA smoke verifies exact pre-transition dose and shared-context projection",
            scope,
        )
        return True
    return False


def _binding_matches_current(binding: Any, expected_path: Path) -> bool:
    """Verify a path/hash binding against the current claim implementation bytes."""

    if not isinstance(binding, dict) or not expected_path.is_file():
        return False
    raw_path = binding.get("path")
    return (
        isinstance(raw_path, str)
        and Path(raw_path).resolve() == expected_path.resolve()
        and binding.get("sha256") == sha256_file(expected_path)
    )


def _binding_is_hash_valid(binding: Any) -> bool:
    """Rehash a non-code artifact whose exact path is frozen by preregistration."""

    if not isinstance(binding, Mapping):
        return False
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not _is_sha(binding.get("sha256")):
        return False
    path = Path(raw_path).resolve()
    return path.is_file() and sha256_file(path) == binding.get("sha256")


def _validate_passive_smoke_preregistration(
    prereg: Mapping[str, Any],
    *,
    manifest: _ManifestView,
    report: PreflightReport,
    plan: DP.PassiveDosePlan,
    smoke: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> bool:
    recorded_hash = prereg.get("preregistration_sha256")
    body = dict(prereg)
    body.pop("preregistration_sha256", None)
    inputs = prereg.get("inputs")
    design = prereg.get("design")
    launcher = prereg.get("launcher")
    command = prereg.get("command")
    raw_outputs = prereg.get("outputs")
    raw_runtime = smoke.get("runtime")
    outputs = raw_outputs if isinstance(raw_outputs, Mapping) else {}
    runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    if not isinstance(inputs, Mapping) or not isinstance(design, Mapping):
        return False
    manifest_ref = inputs.get("manifest")
    plan_ref = inputs.get("passive_dose_plan")
    source_config = design.get("num_steps_source_config")
    origin_map_ref = inputs.get("origin_map")
    capsule_ref = inputs.get("capsule")
    checkpoint_ref = inputs.get("checkpoint")
    snapshot_ref = inputs.get("snapshot")
    localized_ref = inputs.get("localized_checkpoint")
    smoke_origin = smoke.get("origin")
    selected_origin: Mapping[str, Any] = {}
    if _binding_is_hash_valid(origin_map_ref):
        try:
            origin_map_payload = json.loads(Path(origin_map_ref["path"]).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            origin_map_payload = None
        origins = (
            origin_map_payload.get("origins") if isinstance(origin_map_payload, Mapping) else {}
        )
        candidate = origins.get(str(smoke.get("seed"))) if isinstance(origins, Mapping) else None
        if isinstance(candidate, Mapping):
            selected_origin = candidate
    origin_bindings_valid = all(
        _binding_is_hash_valid(item)
        for item in (origin_map_ref, capsule_ref, checkpoint_ref, snapshot_ref, localized_ref)
    )
    if isinstance(localized_ref, Mapping) and isinstance(checkpoint_ref, Mapping):
        origin_bindings_valid = origin_bindings_valid and all(
            (
                localized_ref.get("publication") == "exclusive_byte_copy",
                localized_ref.get("sha256") == checkpoint_ref.get("sha256"),
                Path(str(localized_ref.get("path", ""))).resolve()
                != Path(str(checkpoint_ref.get("path", ""))).resolve(),
                Path(str(localized_ref.get("source_path", ""))).resolve()
                == Path(str(checkpoint_ref.get("path", ""))).resolve(),
            )
        )
    else:
        origin_bindings_valid = False
    origin_record_valid = isinstance(smoke_origin, Mapping) and all(
        (
            smoke_origin.get("origin_map_file_sha256")
            == (origin_map_ref.get("sha256") if isinstance(origin_map_ref, Mapping) else None),
            smoke_origin.get("capsule_file_sha256")
            == (capsule_ref.get("sha256") if isinstance(capsule_ref, Mapping) else None),
            smoke_origin.get("checkpoint_file_sha256")
            == (checkpoint_ref.get("sha256") if isinstance(checkpoint_ref, Mapping) else None),
            smoke_origin.get("snapshot_file_sha256")
            == (snapshot_ref.get("sha256") if isinstance(snapshot_ref, Mapping) else None),
            smoke_origin.get("global_step") == design.get("origin_global_step"),
            selected_origin.get("origin_step") == design.get("origin_global_step"),
            selected_origin.get("seed") == smoke.get("seed"),
            selected_origin.get("capsule_sha256")
            == (capsule_ref.get("sha256") if isinstance(capsule_ref, Mapping) else None),
            selected_origin.get("checkpoint_sha256")
            == (checkpoint_ref.get("sha256") if isinstance(checkpoint_ref, Mapping) else None),
            selected_origin.get("snapshot_sha256")
            == (snapshot_ref.get("sha256") if isinstance(snapshot_ref, Mapping) else None),
        )
    )
    return all(
        (
            prereg.get("kind") == "practice_utility_passive_dose_smoke_preregistration",
            prereg.get("schema_version") == 1,
            prereg.get("immutable") is True,
            prereg.get("outcome_blind") is True,
            _is_sha(recorded_hash) and recorded_hash == sha256_of(body),
            prereg.get("source_commit") == smoke.get("source_commit"),
            prereg.get("source_tree_clean") is True,
            prereg.get("source_tree_status") == [],
            smoke.get("blockers") == [],
            prereg.get("implementation") == implementation,
            launcher == implementation.get("launcher"),
            inputs.get("callback") == implementation.get("callback"),
            source_config == implementation.get("source_config"),
            isinstance(manifest_ref, Mapping)
            and Path(str(manifest_ref.get("path", ""))).resolve()
            == Path(report.manifest_path).resolve()
            and manifest_ref.get("sha256") == report.manifest_file_sha256,
            isinstance(plan_ref, Mapping)
            and Path(str(plan_ref.get("path", ""))).resolve() == plan.path
            and plan_ref.get("sha256") == plan.file_sha256
            and plan_ref.get("logical_sha256") == plan.logical_sha256,
            design.get("campaign_id") == manifest.campaign_id,
            design.get("manifest_sha256") == manifest.manifest_sha256,
            design.get("stage") == smoke.get("stage"),
            design.get("seed") == smoke.get("seed"),
            smoke.get("seed") in manifest.seeds,
            design.get("role") == "control",
            design.get("epsilon") == 0.0,
            origin_bindings_valid,
            origin_record_valid,
            isinstance(command, list),
            prereg.get("command_sha256") == sha256_of({"argv": command}),
            smoke.get("command_sha256") == prereg.get("command_sha256"),
            isinstance(raw_outputs, Mapping),
            isinstance(raw_runtime, Mapping),
            isinstance(runtime.get("log"), Mapping)
            and outputs.get("log") == runtime["log"].get("path"),
            isinstance(smoke.get("dose_receipt"), Mapping)
            and outputs.get("dose_receipt") == smoke["dose_receipt"].get("path"),
        )
    )


def _validate_live_dose_invariants(
    dose: Mapping[str, Any],
    *,
    dose_asset: _Asset | None,
    prereg: Mapping[str, Any],
    manifest: _ManifestView,
    report: PreflightReport,
    plan: DP.PassiveDosePlan,
    stage: Any,
    smoke: Mapping[str, Any],
) -> bool:
    """Recompute the smoke's numerical claims instead of trusting check booleans."""

    if dose_asset is None or not isinstance(stage, str) or stage not in manifest.stages:
        return False
    design = prereg.get("design")
    if not isinstance(design, Mapping):
        return False
    relative = design.get("relative_horizon")
    absolute = design.get("absolute_horizon")
    short = relative.get("H_s") if isinstance(relative, Mapping) else None
    target = absolute.get("H_s") if isinstance(absolute, Mapping) else None
    num_envs = design.get("num_envs")
    num_steps = design.get("num_steps_per_env")
    origin = design.get("origin_global_step")
    integer_values = (short, target, num_envs, num_steps, origin)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        return False
    assert all(isinstance(value, int) for value in integer_values)
    if short <= 0 or num_envs <= 0 or num_steps <= 0 or target != origin + short:
        return False
    if manifest.horizons.get("H_s") != short:
        return False
    expected_total = short * num_steps * num_envs
    expected_hooks = short * num_steps
    expected_observations = expected_hooks * num_envs
    if design.get("expected_completed_env_steps") != expected_total:
        return False

    receipt_hash = dose.get("receipt_payload_sha256")
    receipt_body = dict(dose)
    receipt_body.pop("receipt_payload_sha256", None)
    receipt_plan = dose.get("passive_dose_plan")
    registry = dose.get("dose_registry_sha256_at_install")
    callback = dose.get("implementation")
    base_checks = (
        dose.get("kind") == DP.PASSIVE_DOSE_RECEIPT_KIND,
        dose.get("schema_version") == DP.PASSIVE_DOSE_RECEIPT_SCHEMA_VERSION,
        dose.get("status") == "complete",
        dose.get("valid_for_claim") is True,
        dose.get("blockers") == [],
        dose.get("role") == "control",
        dose.get("epsilon") == 0.0,
        dose.get("context_id") == "native",
        dose.get("armed") is False,
        dose.get("never_armed") is True,
        dose.get("measurement_hook") == DP.PASSIVE_DOSE_HOOK,
        dose.get("global_step") == target,
        dose.get("horizon_label") == "H_s",
        dose.get("completed_env_steps") == expected_total,
        dose.get("expected_env_steps") == expected_total,
        dose.get("completion_hook_calls") == expected_hooks,
        dose.get("expected_completion_hook_calls") == expected_hooks,
        dose.get("completion_observations") == expected_observations,
        dose.get("expected_completion_observations") == expected_observations,
        dose.get("termination_observations") == expected_observations,
        dose.get("dropped_completion_batches") == 0,
        _is_sha(registry),
        dose.get("dose_registry_sha256_at_report") == registry,
        dose.get("registry_stable") is True,
        _is_sha(receipt_hash) and receipt_hash == sha256_of(receipt_body),
        isinstance(receipt_plan, Mapping)
        and Path(str(receipt_plan.get("path", ""))).resolve() == plan.path
        and receipt_plan.get("file_sha256") == plan.file_sha256
        and receipt_plan.get("logical_sha256") == plan.logical_sha256
        and receipt_plan.get("stage") == stage,
        dose.get("lineage")
        == {
            "campaign_id": manifest.campaign_id,
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_file_sha256": report.manifest_file_sha256,
        },
        isinstance(callback, Mapping)
        and _binding_matches_current(
            {
                "path": callback.get("callback_path"),
                "sha256": callback.get("callback_sha256"),
            },
            _PASSIVE_DOSE_CALLBACK,
        ),
        not dose_asset.path.with_suffix(dose_asset.path.suffix + ".partial").exists(),
    )
    if not all(base_checks):
        return False

    per_bin = dose.get("per_bin_completed")
    if not isinstance(per_bin, Mapping):
        return False
    try:
        completed_by_bin = {int(key): float(value) for key, value in per_bin.items()}
    except (TypeError, ValueError):
        return False
    if len(completed_by_bin) != len(per_bin) or any(str(int(key)) != str(key) for key in per_bin):
        return False
    if any(not math.isfinite(value) or value < 0 for value in completed_by_bin.values()):
        return False
    if not math.isclose(
        math.fsum(completed_by_bin.values()), expected_total, rel_tol=0.0, abs_tol=1e-6
    ):
        return False

    expected_contexts = {context_id: context for _, context_id, context in manifest.contexts[stage]}
    rows = dose.get("context_doses")
    if not isinstance(rows, list) or len(rows) != len(expected_contexts):
        return False
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        context_id = row.get("context_id")
        if (
            not isinstance(context_id, str)
            or context_id in seen
            or row.get("context") != expected_contexts.get(context_id)
        ):
            return False
        seen.add(context_id)
        membership = row.get("membership_by_global_bin")
        if not isinstance(membership, Mapping):
            return False
        try:
            weights = {int(key): float(value) for key, value in membership.items()}
        except (TypeError, ValueError):
            return False
        if (
            not weights
            or len(weights) != len(membership)
            or any(str(int(key)) != str(key) for key in membership)
            or any(
                not math.isfinite(value) or not 0.0 < value <= 1.0 + 1e-12
                for value in weights.values()
            )
            or not math.isclose(max(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            return False
        projected = math.fsum(
            completed_by_bin.get(bin_id, 0.0) * weight for bin_id, weight in weights.items()
        )
        try:
            recorded_projected = float(row.get("completed_kernel_steps"))
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(recorded_projected)
            or recorded_projected < 0
            or not math.isclose(projected, recorded_projected, rel_tol=0.0, abs_tol=1e-6)
        ):
            return False
        kernel_payload = {
            "context_id": context_id,
            "kernel_radius_bins": row.get("kernel_radius_bins"),
            "sigma_frames": row.get("sigma_frames"),
            "membership_by_global_bin": {str(key): value for key, value in sorted(weights.items())},
            "dose_registry_sha256": registry,
        }
        if (
            row.get("kernel_radius_bins") != plan.kernel_radius_bins
            or row.get("sigma_frames") != plan.sigma_frames
            or row.get("kernel_membership_sha256") != sha256_of(kernel_payload)
        ):
            return False
    coverage_hash = sha256_of({"stage": stage, "context_ids": sorted(seen)})
    return seen == set(expected_contexts) and coverage_hash == smoke.get("context_coverage_sha256")


def _block_live_passive_dose(audit: _Audit) -> None:
    """Keep the historical blocker until one valid live receipt transitions it."""

    audit.blocker(
        "live_passive_dose_smoke_missing",
        "no valid hash-bound live-GPU smoke proves the passive completion hook and exact "
        "shared-control projections end to end",
        "dose",
    )
    audit.blocker(
        "shared_control_realized_dose_unimplemented",
        "CPU contracts exist, but shared-control realized dose remains claim-blocked until "
        "live_passive_dose_smoke_verified is present",
        "dose",
    )


def _validate_directional_calibration_preregistration(
    reference: Any,
    manifest: _ManifestView,
    report: PreflightReport,
    base_dir: Path,
    audit: _Audit,
) -> tuple[_Asset | None, dict[str, Any] | None]:
    """Validate the external, outcome-free directional design artifact."""

    scope = "latent_direction"
    asset = _verify_asset(
        reference, base_dir, audit, "directional_calibration_preregistration", scope
    )
    if asset is None:
        return None, None
    payload = _read_json(asset.path, audit, "directional_calibration_preregistration", scope)
    if payload is None:
        return asset, None
    initial_blockers = len(audit.report.blockers)
    if (
        payload.get("kind") != DIRECTIONAL_PREREGISTRATION_KIND
        or payload.get("schema_version") != 1
    ):
        audit.blocker(
            "directional_calibration_preregistration_schema_invalid",
            "directional calibration preregistration has the wrong kind or schema",
            scope,
        )
    expected = {
        "frozen_before_outcomes": True,
        "contains_outcomes": False,
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": report.manifest_file_sha256,
    }
    bad = [key for key, value in expected.items() if payload.get(key) != value]
    if bad:
        audit.blocker(
            "directional_calibration_preregistration_binding_invalid",
            f"directional preregistration differs on {bad}",
            scope,
        )
    source_manifest = payload.get("source_manifest")
    git_identity = payload.get("git")
    launcher = payload.get("launcher")
    current_launcher_sha256 = (
        sha256_file(_DIRECTIONAL_CALIBRATION_LAUNCHER)
        if _DIRECTIONAL_CALIBRATION_LAUNCHER.is_file()
        else None
    )
    provenance_valid = (
        isinstance(source_manifest, dict)
        and Path(str(source_manifest.get("path", ""))).resolve()
        == Path(report.manifest_path).resolve()
        and source_manifest.get("logical_sha256") == manifest.manifest_sha256
        and source_manifest.get("file_sha256") == report.manifest_file_sha256
        and isinstance(git_identity, dict)
        and _FULL_COMMIT.fullmatch(str(git_identity.get("sha", ""))) is not None
        and git_identity.get("status_short") == []
        and isinstance(launcher, dict)
        and Path(str(launcher.get("path", ""))).resolve() == _DIRECTIONAL_CALIBRATION_LAUNCHER
        and launcher.get("sha256") == current_launcher_sha256
    )
    if not provenance_valid:
        audit.blocker(
            "directional_calibration_preregistration_provenance_invalid",
            "directional preregistration must bind the exact manifest, a clean full Git "
            "identity, and the current directional-freezing launcher bytes",
            scope,
        )
    body = dict(payload)
    recorded = body.pop("artifact_sha256", None)
    if not _is_sha(recorded) or recorded != sha256_of(body):
        audit.blocker(
            "directional_calibration_preregistration_hash_invalid",
            "directional preregistration logical hash is invalid",
            scope,
        )
    try:
        algorithm = DC.validate_algorithm_artifact(payload.get("algorithm"))
    except ValueError as error:
        audit.blocker("directional_calibration_algorithm_invalid", str(error), scope)
        algorithm = None
    if algorithm is not None:
        rows = []
        raw_contexts = manifest.payload.get("contexts_per_stage") or {}
        for stage in manifest.stages:
            for seed in manifest.seeds:
                for entry in raw_contexts.get(stage, []):
                    rows.append(
                        DC.CalibrationDesignRow(
                            sample_id=f"{stage}|{seed}|{entry.get('context_id')}",
                            context_id=str(entry.get("context_id")),
                            motion_family=str(entry.get("family", "")),
                        )
                    )
        try:
            expected_support = DC.validate_design_support(rows, algorithm)
        except ValueError as error:
            audit.blocker("directional_calibration_design_invalid", str(error), scope)
        else:
            if payload.get("design_support") != expected_support:
                audit.blocker(
                    "directional_calibration_design_mismatch",
                    "recorded directional design support/folds do not match the manifest",
                    scope,
                )
            if expected_support.get("status") != "ready":
                audit.blocker(
                    "directional_calibration_design_not_ready",
                    "frozen manifest cannot support the preregistered nested grouped folds",
                    scope,
                )
    if len(audit.report.blockers) == initial_blockers:
        audit.verified(
            "directional_calibration_preregistration_verified",
            f"outcome-free directional design is READY [{recorded}]",
            scope,
        )
        return asset, payload
    return asset, None


def _validate_efficacy_plan(
    reference: Any,
    manifest: _ManifestView,
    base_dir: Path,
    audit: _Audit,
) -> tuple[dict[str, Any] | None, _Asset | None, str | None, str | None]:
    asset = _verify_asset(reference, base_dir, audit, "efficacy_plan", "efficacy_plan")
    if asset is None:
        return None, None, None, None
    plan = _read_json(asset.path, audit, "efficacy_plan", "efficacy_plan")
    if plan is None:
        return None, asset, None, None

    if plan.get("kind") != "practice_utility_deployment_efficacy_plan":
        audit.blocker(
            "efficacy_plan_kind_invalid", "efficacy plan kind is invalid", "efficacy_plan"
        )
    if plan.get("schema_version") != 1:
        audit.blocker(
            "efficacy_plan_schema_invalid", "efficacy plan schema must be 1", "efficacy_plan"
        )
    if plan.get("frozen") is not True or plan.get("frozen_before_campaign") is not True:
        audit.blocker(
            "efficacy_plan_not_frozen",
            "deployment efficacy plan must be frozen before the campaign",
            "efficacy_plan",
        )
    if plan.get("manifest_sha256") != manifest.manifest_sha256:
        audit.blocker(
            "efficacy_plan_manifest_mismatch",
            "efficacy plan is not linked to the frozen manifest",
            "efficacy_plan",
        )

    efficacy = plan.get("efficacy")
    required_efficacy = {
        "name": "j_eff_macro_quality_success",
        "units": "success_fraction",
        "utility_units": "success_fraction_per_completed_kernel_step",
        "window": DEFAULT_EFFICACY_WINDOW,
        "quality_qualified": True,
        "macro_average_group": "motion_family",
    }
    if not isinstance(efficacy, dict):
        audit.blocker(
            "efficacy_estimand_missing",
            "efficacy plan has no claim-grade efficacy metadata",
            "efficacy_plan",
        )
        efficacy = {}
    else:
        bad = [key for key, value in required_efficacy.items() if efficacy.get(key) != value]
        if set(efficacy.get("harm_channels") or ()) != {
            "action_rate",
            "foot_slip",
            "contact_impulse",
            "torque_saturation",
        }:
            bad.append("harm_channels")
        if bad:
            audit.blocker(
                "efficacy_estimand_invalid",
                f"claim-grade efficacy metadata differs on {sorted(set(bad))}",
                "efficacy_plan",
            )

    estimand = _label_estimand(manifest, efficacy)
    estimand_sha256 = sha256_of(estimand) if estimand is not None else None
    if estimand is None or plan.get("estimand_sha256") != estimand_sha256:
        audit.blocker(
            "efficacy_estimand_hash_mismatch",
            "recorded estimand_sha256 must hash the exact Gate-A deployment-utility estimand",
            "efficacy_plan",
        )

    objective = plan.get("objective")
    if not isinstance(objective, dict):
        audit.blocker(
            "j_eff_objective_missing", "efficacy plan has no J_eff objective", "efficacy_plan"
        )
    else:
        if objective.get("name") != "J_eff":
            audit.blocker("j_eff_name_invalid", "objective.name must be 'J_eff'", "efficacy_plan")
        if objective.get("aggregation") != "macro_mean_over_motion_families":
            audit.blocker(
                "j_eff_aggregation_invalid",
                "J_eff must be a macro-mean over motion families",
                "efficacy_plan",
            )
        conditions = objective.get("qualified_success_requires")
        if not isinstance(conditions, list) or set(conditions) != set(REQUIRED_QUALITY_CONDITIONS):
            audit.blocker(
                "j_eff_quality_conditions_incomplete",
                f"QSuccess must require exactly {list(REQUIRED_QUALITY_CONDITIONS)}",
                "efficacy_plan",
            )
        thresholds = objective.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != set(REQUIRED_QUALITY_THRESHOLDS):
            audit.blocker(
                "j_eff_thresholds_incomplete",
                f"frozen thresholds required for {list(REQUIRED_QUALITY_THRESHOLDS)}",
                "efficacy_plan",
            )
        elif not all(_is_finite_number(thresholds[name]) for name in REQUIRED_QUALITY_THRESHOLDS):
            audit.blocker(
                "j_eff_thresholds_invalid",
                "every J_eff quality threshold must be a finite number",
                "efficacy_plan",
            )
        if objective.get("thresholds_frozen") is not True or not objective.get("threshold_source"):
            audit.blocker(
                "j_eff_threshold_provenance_missing",
                "quality thresholds must be frozen with an explicit source",
                "efficacy_plan",
            )

    evaluation = plan.get("evaluation")
    dev_suite_sha256: str | None = None
    required_true = (
        "policy_frozen",
        "matched_motion_panel",
        "matched_physics_seeds",
        "evaluation_receipt_per_branch_horizon",
        "quality_outcomes_reported_separately",
        "performer_and_content_splits_reported_separately",
    )
    if not isinstance(evaluation, dict):
        audit.blocker(
            "deployment_evaluation_missing",
            "efficacy plan has no frozen-policy deployment evaluation block",
            "efficacy_plan",
        )
    else:
        missing_true = [key for key in required_true if evaluation.get(key) is not True]
        if missing_true:
            audit.blocker(
                "deployment_evaluation_incomplete",
                f"deployment evaluation must enable {missing_true}",
                "efficacy_plan",
            )
        if evaluation.get("learning_during_evaluation") is not False:
            audit.blocker(
                "deployment_evaluation_learns",
                "deployment evaluation must not update the policy",
                "efficacy_plan",
            )
        if evaluation.get("training_metric_fallback_allowed") is not False:
            audit.blocker(
                "training_metric_fallback_enabled",
                "claim-grade labels cannot fall back to training-side mean reward",
                "efficacy_plan",
            )
        if evaluation.get("final_test_accessible") is not False:
            audit.blocker(
                "final_test_not_sealed",
                "the frozen final test must remain inaccessible during screening",
                "efficacy_plan",
            )
        candidate = evaluation.get("dev_suite_sha256")
        if not _is_sha(candidate):
            audit.blocker(
                "dev_suite_hash_invalid",
                "evaluation.dev_suite_sha256 must be a SHA-256",
                "efficacy_plan",
            )
        else:
            dev_suite_sha256 = str(candidate)

    return plan, asset, estimand_sha256, dev_suite_sha256


def _validate_origin_maps(
    references: Any,
    manifest: _ManifestView,
    base_dir: Path,
    dev_suite_sha256: str | None,
    audit: _Audit,
) -> dict[str, _Origin]:
    """Validate the transitive receipts emitted by ``create_probe_origins.py``."""

    if not isinstance(references, dict):
        audit.blocker(
            "origin_map_missing",
            "preflight requires one hashed create_probe_origins map per manifest stage",
            "origins",
        )
        return {}
    expected_stages = set(manifest.stages)
    actual_stages = set(references)
    if actual_stages != expected_stages:
        audit.blocker(
            "origin_map_stage_coverage_inexact",
            f"origin-map stages missing={sorted(expected_stages - actual_stages)}, "
            f"extra={sorted(actual_stages - expected_stages)}",
            "origins",
        )

    origins: dict[str, _Origin] = {}
    for stage in manifest.stages:
        scope = f"origin-map:{stage}"
        asset = _verify_asset(references.get(stage), base_dir, audit, "origin_map", scope)
        if asset is None:
            continue
        payload = _read_json(asset.path, audit, "origin_map", scope)
        if payload is None:
            continue
        if payload.get("kind") != "practice_utility_probe_origin_map":
            audit.blocker("origin_map_kind_invalid", "origin-map kind is invalid", scope)
        if payload.get("schema_version") != 1:
            audit.blocker("origin_map_schema_invalid", "origin-map schema must be 1", scope)
        if payload.get("usable_for_manifest_selection") is not True:
            audit.blocker(
                "origin_map_not_usable",
                "origin creation reported blockers; this map cannot seed a manifest",
                scope,
            )
        if payload.get("seeds") != list(manifest.seeds):
            audit.blocker(
                "origin_map_seed_mismatch",
                f"origin map seeds {payload.get('seeds')!r} != manifest seeds {list(manifest.seeds)!r}",
                scope,
            )
        map_step = payload.get("origin_step")
        if not isinstance(map_step, int) or isinstance(map_step, bool) or map_step <= 0:
            audit.blocker("origin_map_step_invalid", "origin_step must be positive", scope)
            continue
        if not _is_sha(payload.get("source_checkpoint_sha256")):
            audit.blocker(
                "origin_map_source_hash_invalid",
                "origin map must bind the source checkpoint hash",
                scope,
            )

        expected_context_ids = [context_id for _, context_id, _ in manifest.contexts[stage]]
        common = payload.get("common_resident_context_ids")
        if not isinstance(common, list) or len(set(common)) != len(common):
            audit.blocker(
                "origin_map_common_contexts_invalid",
                "common resident context ids must be a unique list",
                scope,
            )
        elif not set(expected_context_ids).issubset(set(common)):
            missing = sorted(set(expected_context_ids) - set(common))
            audit.blocker(
                "manifest_not_from_origin_intersection",
                f"manifest selected contexts absent from the origin intersection: {missing}",
                scope,
            )
        if payload.get("num_common_resident_contexts") != (
            len(common) if isinstance(common, list) else None
        ):
            audit.blocker(
                "origin_map_common_count_mismatch",
                "num_common_resident_contexts does not match the serialized intersection",
                scope,
            )

        raw_origins = payload.get("origins")
        expected_seed_keys = {str(seed) for seed in manifest.seeds}
        if not isinstance(raw_origins, dict) or set(raw_origins) != expected_seed_keys:
            actual = set(raw_origins) if isinstance(raw_origins, dict) else set()
            audit.blocker(
                "origin_map_incomplete",
                f"origin map seed rows missing={sorted(expected_seed_keys - actual)}, "
                f"extra={sorted(actual - expected_seed_keys)}",
                scope,
            )
            continue

        for seed in manifest.seeds:
            row = raw_origins[str(seed)]
            origin_scope = f"origin:{stage}:s{seed}"
            if not isinstance(row, dict):
                audit.blocker("origin_row_invalid", "origin row must be an object", origin_scope)
                continue
            if row.get("seed") != seed or row.get("origin_step") != map_step:
                audit.blocker(
                    "origin_identity_mismatch",
                    f"origin row must bind seed {seed} and step {map_step}",
                    origin_scope,
                )
                continue
            if row.get("blockers") not in ([], None):
                audit.blocker(
                    "origin_creation_blocked",
                    f"origin creator recorded blockers: {row.get('blockers')}",
                    origin_scope,
                )
            if row.get("settled") is not True:
                audit.blocker(
                    "origin_not_settled",
                    "origin map does not explicitly mark this seed settled",
                    origin_scope,
                )
            _validate_origin_stability(row.get("stability"), audit, origin_scope)

            capsule = _verify_map_asset(
                row.get("capsule"),
                row.get("capsule_sha256"),
                asset.path.parent,
                audit,
                "capsule",
                origin_scope,
            )
            checkpoint = _verify_map_asset(
                row.get("checkpoint"),
                row.get("checkpoint_sha256"),
                asset.path.parent,
                audit,
                "checkpoint",
                origin_scope,
            )
            snapshot = _verify_map_asset(
                row.get("snapshot"),
                row.get("snapshot_sha256"),
                asset.path.parent,
                audit,
                "snapshot",
                origin_scope,
            )
            if capsule is None or checkpoint is None or snapshot is None:
                continue
            capsule_payload = _validate_full_capsule(
                capsule, manifest, map_step, dev_suite_sha256, audit, origin_scope
            )
            checkpoint_payload = _validate_paired_checkpoint(
                checkpoint, capsule, capsule_payload, map_step, audit, origin_scope
            )
            snapshot_payload = _validate_transitive_origin_snapshot(
                snapshot,
                row,
                stage,
                seed,
                map_step,
                manifest,
                expected_context_ids,
                audit,
                origin_scope,
            )
            if capsule_payload is None or checkpoint_payload is None or snapshot_payload is None:
                continue
            key = origin_key(stage, seed)
            origins[key] = _Origin(
                key=key,
                stage=stage,
                seed=seed,
                global_step=map_step,
                capsule=capsule,
                checkpoint=checkpoint,
                capsule_payload=capsule_payload,
                checkpoint_payload=checkpoint_payload,
                snapshot=snapshot,
                snapshot_payload=snapshot_payload,
            )
    if len(origins) == len(manifest.expected_origin_keys):
        audit.verified(
            "origin_map_complete",
            f"verified transitive full-state origins for all {len(origins)} stage-seed cells",
            "origins",
        )
    return origins


def _validate_origin_stability(stability: Any, audit: _Audit, scope: str) -> None:
    if not isinstance(stability, dict) or set(stability) < {"reward", "length"}:
        audit.blocker(
            "settled_evidence_missing",
            "origin map needs explicit reward and length trailing-window evidence",
            scope,
        )
        return
    bad: list[str] = []
    for metric in ("reward", "length"):
        evidence = stability.get(metric)
        if not isinstance(evidence, dict) or evidence.get("passes") is not True:
            bad.append(metric)
            continue
        for evidence_field in (
            "previous4",
            "last4",
            "relative_delta",
            "absolute_relative_limit",
        ):
            if not _is_finite_number(evidence.get(evidence_field)):
                bad.append(f"{metric}.{evidence_field}")
        relative = evidence.get("relative_delta")
        limit = evidence.get("absolute_relative_limit")
        if _is_finite_number(relative) and _is_finite_number(limit):
            if abs(float(relative)) > float(limit):
                bad.append(f"{metric}.decision")
    if bad:
        audit.blocker(
            "settled_evidence_invalid",
            f"origin trailing-window evidence failed or is incomplete: {sorted(set(bad))}",
            scope,
        )
    else:
        audit.verified(
            "settled_origin_verified",
            "explicit last-4 versus previous-4 reward/length stability evidence passes",
            scope,
        )


def _verify_map_asset(
    raw_path: Any,
    recorded_sha256: Any,
    base_dir: Path,
    audit: _Audit,
    code_prefix: str,
    scope: str,
) -> _Asset | None:
    return _verify_asset(
        {"path": raw_path, "sha256": recorded_sha256},
        base_dir,
        audit,
        code_prefix,
        scope,
    )


def _validate_transitive_origin_snapshot(
    asset: _Asset,
    origin_row: Mapping[str, Any],
    stage: str,
    seed: int,
    global_step: int,
    manifest: _ManifestView,
    expected_context_ids: Sequence[str],
    audit: _Audit,
    origin_scope: str,
) -> dict[str, Any] | None:
    scope = f"{origin_scope}/snapshot"
    payload = _read_json(asset.path, audit, "snapshot", scope)
    if payload is None:
        return None
    if payload.get("kind") != "practice_utility_sampler_snapshot":
        audit.blocker("snapshot_kind_invalid", "origin snapshot kind is invalid", scope)
    if payload.get("global_step") != global_step:
        audit.blocker(
            "snapshot_step_mismatch",
            f"snapshot step {payload.get('global_step')!r} != capsule/checkpoint step {global_step}",
            scope,
        )
    if not _is_sha(payload.get("distribution_sha256")):
        audit.blocker(
            "snapshot_distribution_hash_invalid", "snapshot distribution hash is invalid", scope
        )
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
        audit.blocker("snapshot_contexts_missing", "snapshot has no resident contexts", scope)
        return None
    if payload.get("num_active_bins") != len(contexts):
        audit.blocker(
            "snapshot_active_count_mismatch",
            "num_active_bins does not match serialized contexts",
            scope,
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in contexts:
        if not isinstance(row, dict):
            continue
        context_id = row.get("context_id")
        if isinstance(context_id, str):
            rows.setdefault(context_id, []).append(row)
    row_resident = origin_row.get("resident_context_ids")
    if not isinstance(row_resident, list) or sorted(rows) != sorted(row_resident):
        audit.blocker(
            "snapshot_origin_map_coverage_mismatch",
            "hashed snapshot context ids differ from its origin-map receipt",
            scope,
        )
    if origin_row.get("num_resident_contexts") != len(rows):
        audit.blocker(
            "snapshot_origin_map_count_mismatch",
            "origin map resident-context count differs from the hashed snapshot",
            scope,
        )
    expected_contexts = {cid: context for _, cid, context in manifest.contexts[stage]}
    missing = sorted(set(expected_context_ids) - set(rows))
    duplicates = sorted(cid for cid in expected_context_ids if len(rows.get(cid, ())) != 1)
    changed = []
    for context_id, expected in expected_contexts.items():
        members = rows.get(context_id, [])
        if len(members) == 1 and _canonical_context(members[0]) != expected:
            changed.append(context_id)
    if missing or duplicates or changed:
        audit.blocker(
            "snapshot_context_coverage_invalid",
            f"stage={stage}, seed={seed}: missing={missing}, duplicate={duplicates}, "
            f"changed={changed}",
            scope,
        )
    elif not any(item.scope == scope for item in audit.report.blockers):
        audit.verified(
            "snapshot_context_coverage_verified",
            f"all {len(expected_context_ids)} manifest contexts are exact residents in the "
            "hash-linked origin snapshot",
            scope,
        )
    return payload


def _validate_origins(
    raw_origins: Any,
    manifest: _ManifestView,
    base_dir: Path,
    dev_suite_sha256: str | None,
    audit: _Audit,
) -> dict[str, _Origin]:
    if not isinstance(raw_origins, dict):
        audit.blocker(
            "origin_map_missing",
            "preflight requires an origin map for every manifest (stage, seed)",
            "origins",
        )
        return {}

    expected = set(manifest.expected_origin_keys)
    actual = set(raw_origins)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        audit.blocker(
            "origin_map_incomplete",
            f"origin map is missing {missing}",
            "origins",
        )
    if extra:
        audit.blocker(
            "origin_map_extra",
            f"origin map contains undeclared origins {extra}",
            "origins",
        )

    origins: dict[str, _Origin] = {}
    for key in manifest.expected_origin_keys:
        entry = raw_origins.get(key)
        if not isinstance(entry, dict):
            continue
        stage, seed = _split_origin_key(key)
        scope = f"origin:{key}"
        if entry.get("stage") != stage or entry.get("seed") != seed:
            audit.blocker(
                "origin_identity_mismatch",
                f"entry must declare stage={stage!r}, seed={seed}",
                scope,
            )
            continue
        step = entry.get("global_step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            audit.blocker("origin_step_invalid", "origin global_step must be positive", scope)
            continue

        capsule = _verify_asset(entry.get("capsule"), base_dir, audit, "capsule", scope)
        checkpoint = _verify_asset(entry.get("checkpoint"), base_dir, audit, "checkpoint", scope)
        if capsule is None or checkpoint is None:
            continue
        capsule_payload = _validate_full_capsule(
            capsule, manifest, step, dev_suite_sha256, audit, scope
        )
        checkpoint_payload = _validate_paired_checkpoint(
            checkpoint, capsule, capsule_payload, step, audit, scope
        )
        if capsule_payload is None or checkpoint_payload is None:
            continue

        origin = _Origin(
            key=key,
            stage=stage,
            seed=seed,
            global_step=step,
            capsule=capsule,
            checkpoint=checkpoint,
            capsule_payload=capsule_payload,
            checkpoint_payload=checkpoint_payload,
        )
        _validate_settled_evidence(entry.get("settled_evidence"), origin, manifest, base_dir, audit)
        snapshot_asset, snapshot_payload = _validate_origin_snapshot(
            entry.get("snapshot"), origin, manifest, base_dir, audit
        )
        origin.snapshot = snapshot_asset
        origin.snapshot_payload = snapshot_payload
        origins[key] = origin

    if len(origins) == len(expected):
        audit.verified(
            "origin_map_complete",
            f"verified capsule/checkpoint origins for all {len(expected)} stage-seed cells",
            "origins",
        )
    return origins


def _validate_full_capsule(
    asset: _Asset,
    manifest: _ManifestView,
    expected_step: int,
    dev_suite_sha256: str | None,
    audit: _Audit,
    scope: str,
) -> dict[str, Any] | None:
    try:
        payload = BC.load_capsule(asset.path, restore_rng=False)
    except Exception as error:  # torch/pickle payloads have several failure types
        audit.blocker("capsule_invalid", f"cannot validate full capsule: {error}", scope)
        return None

    missing: list[str] = []
    model = payload.get("model_state")
    if not isinstance(model, dict) or not model.get("policy_state_dict"):
        missing.append("model_state.policy_state_dict")
    if not isinstance(model, dict) or not model.get("value_state_dict"):
        missing.append("model_state.value_state_dict")
    if not _nonempty_mapping(payload.get("optimizer_state")):
        missing.append("optimizer_state")
    trainer = payload.get("trainer_state")
    if not isinstance(trainer, dict) or trainer.get("trainer_state_obj") is None:
        missing.append("trainer_state.trainer_state_obj")
    if not _nonempty_mapping(payload.get("env_state")):
        missing.append("env_state")
    sampler = payload.get("native_sampler_state")
    if not isinstance(sampler, dict) or any(key not in sampler for key in BC.SAMPLER_KEYS):
        missing.append("native_sampler_state counters")
    rng = payload.get("rng")
    required_rng = (
        "python_state",
        "numpy_state",
        "torch_cpu_state",
        "torch_cuda_states",
        "counter_rng_enabled",
        "pair_id",
        "deterministic_flags",
    )
    if not isinstance(rng, dict) or any(key not in rng for key in required_rng):
        missing.append("complete RNG state")
    if missing:
        audit.blocker(
            "capsule_not_full",
            f"capsule is not a full continuation origin; missing {missing}",
            scope,
        )
        return None
    if payload.get("global_step") != expected_step or trainer.get("global_step") != expected_step:
        audit.blocker(
            "capsule_step_mismatch",
            f"capsule/trainer step must both equal origin step {expected_step}",
            scope,
        )
        return None

    provenance = payload.get("provenance")
    required_provenance = (
        "resolved_config_sha256",
        "motion_pool_manifest_sha256",
        "source_commit",
        "checkpoint_sha256",
    )
    if not isinstance(provenance, dict) or any(
        not provenance.get(key) for key in required_provenance
    ):
        audit.blocker(
            "capsule_provenance_incomplete",
            "capsule provenance must bind launch config, pool, commit, and source checkpoint",
            scope,
        )
        return None
    bad_hashes = [
        key
        for key in required_provenance
        if key != "source_commit" and not _is_sha(provenance.get(key))
    ]
    if bad_hashes or not _COMMIT.fullmatch(str(provenance.get("source_commit", ""))):
        audit.blocker(
            "capsule_provenance_hash_invalid",
            f"capsule provenance has invalid hashes/commit: {bad_hashes}",
            scope,
        )
        return None
    if provenance.get("motion_pool_manifest_sha256") != manifest.payload.get("pool_sha256"):
        audit.blocker(
            "capsule_pool_mismatch",
            "capsule motion-pool hash differs from the probe manifest",
            scope,
        )
        return None
    capsule_dev_suite = provenance.get("dev_suite_sha256")
    if capsule_dev_suite:
        if not _is_sha(capsule_dev_suite):
            audit.blocker(
                "capsule_dev_suite_hash_invalid",
                "capsule dev-suite provenance is present but is not a SHA-256",
                scope,
            )
            return None
        if dev_suite_sha256 is None or capsule_dev_suite != dev_suite_sha256:
            audit.blocker(
                "capsule_dev_suite_mismatch",
                "capsule names a dev suite that is not the frozen deployment-plan suite",
                scope,
            )
            return None
    elif dev_suite_sha256 is not None:
        audit.warning(
            "capsule_dev_suite_not_bound",
            "origin predates the deployment suite and does not bind it; the efficacy plan "
            "must remain independently frozen",
            scope,
        )
    audit.verified("full_capsule_verified", f"full capsule at global step {expected_step}", scope)
    return payload


def _validate_paired_checkpoint(
    asset: _Asset,
    capsule_asset: _Asset,
    capsule: dict[str, Any] | None,
    expected_step: int,
    audit: _Audit,
    scope: str,
) -> dict[str, Any] | None:
    if capsule is None:
        return None
    try:
        checkpoint = torch.load(asset.path, weights_only=False, map_location="cpu")
    except Exception as error:
        audit.blocker("checkpoint_invalid", f"cannot load paired checkpoint: {error}", scope)
        return None
    if not isinstance(checkpoint, dict):
        audit.blocker("checkpoint_invalid", "checkpoint payload is not a mapping", scope)
        return None

    required = (
        "policy_state_dict",
        "value_state_dict",
        "optimizer_state_dict",
        "state",
        "env_state_dict",
        "practice_utility",
    )
    missing = [key for key in required if checkpoint.get(key) is None]
    if missing:
        audit.blocker(
            "checkpoint_not_full",
            f"exported checkpoint is missing full-state fields {missing}",
            scope,
        )
        return None
    practice = checkpoint.get("practice_utility")
    if not isinstance(practice, dict):
        audit.blocker(
            "checkpoint_origin_link_missing", "checkpoint has no practice origin link", scope
        )
        return None
    if practice.get("global_step") != expected_step:
        audit.blocker(
            "checkpoint_step_mismatch",
            f"checkpoint step {practice.get('global_step')!r} != origin step {expected_step}",
            scope,
        )
        return None
    if practice.get("capsule_sha256") != capsule.get("capsule_sha256"):
        audit.blocker(
            "checkpoint_capsule_link_mismatch",
            "checkpoint does not carry the origin capsule's logical hash",
            scope,
        )
        return None

    model = capsule["model_state"]
    comparisons = {
        "policy_state_dict": (checkpoint["policy_state_dict"], model["policy_state_dict"]),
        "value_state_dict": (checkpoint["value_state_dict"], model["value_state_dict"]),
        "optimizer_state_dict": (checkpoint["optimizer_state_dict"], capsule["optimizer_state"]),
        "env_state_dict": (checkpoint["env_state_dict"], capsule["env_state"]),
    }
    mismatched = [
        name for name, (left, right) in comparisons.items() if not _states_equal(left, right)
    ]
    if mismatched:
        audit.blocker(
            "checkpoint_capsule_state_mismatch",
            f"checkpoint differs from its capsule in {mismatched}",
            scope,
        )
        return None
    state_step = _global_step_of(checkpoint.get("state"))
    if state_step is not None and state_step != expected_step:
        audit.blocker(
            "checkpoint_trainer_step_mismatch",
            f"checkpoint trainer state step {state_step} != {expected_step}",
            scope,
        )
        return None
    source = practice.get("source_capsule")
    if source and Path(str(source)).name != capsule_asset.path.name:
        audit.blocker(
            "checkpoint_source_capsule_mismatch",
            f"checkpoint names source capsule {source!r}, expected {capsule_asset.path.name!r}",
            scope,
        )
        return None
    audit.verified(
        "paired_checkpoint_verified",
        f"checkpoint state matches the full capsule at global step {expected_step}",
        scope,
    )
    return checkpoint


def _validate_settled_evidence(
    reference: Any,
    origin: _Origin,
    manifest: _ManifestView,
    base_dir: Path,
    audit: _Audit,
) -> None:
    scope = f"origin:{origin.key}/settled"
    asset = _verify_asset(reference, base_dir, audit, "settled_evidence", scope)
    if asset is None:
        return
    payload = _read_json(asset.path, audit, "settled_evidence", scope)
    if payload is None:
        return
    if payload.get("kind") != "practice_utility_settled_origin_evidence":
        audit.blocker("settled_evidence_kind_invalid", "settled evidence kind is invalid", scope)
    _validate_atomic_origin_link(payload, origin, manifest, audit, scope)
    if payload.get("settled") is not True:
        audit.blocker(
            "origin_not_settled", "settled evidence must explicitly set settled=true", scope
        )
    criterion = payload.get("criterion")
    if not isinstance(criterion, dict):
        audit.blocker("settled_criterion_missing", "settled evidence has no criterion", scope)
        return
    window = criterion.get("window_iterations")
    values = criterion.get("values")
    if not criterion.get("metric") or not criterion.get("decision_rule"):
        audit.blocker(
            "settled_criterion_incomplete",
            "settling criterion needs metric and decision_rule",
            scope,
        )
    if not isinstance(window, int) or isinstance(window, bool) or window < DEFAULT_EFFICACY_WINDOW:
        audit.blocker(
            "settled_window_too_short",
            f"settling evidence needs at least {DEFAULT_EFFICACY_WINDOW} iterations",
            scope,
        )
    if not isinstance(values, list) or len(values) < (window if isinstance(window, int) else 1):
        audit.blocker(
            "settled_values_incomplete",
            "settling evidence must contain the complete criterion window",
            scope,
        )
    elif not all(_is_finite_number(value) for value in values):
        audit.blocker("settled_values_invalid", "settling values must be finite numbers", scope)
    if not any(item.scope == scope for item in audit.report.blockers):
        audit.verified(
            "settled_origin_verified", "explicit settled-origin evidence verified", scope
        )


def _validate_origin_snapshot(
    reference: Any,
    origin: _Origin,
    manifest: _ManifestView,
    base_dir: Path,
    audit: _Audit,
) -> tuple[_Asset | None, dict[str, Any] | None]:
    scope = f"origin:{origin.key}/snapshot"
    asset = _verify_asset(reference, base_dir, audit, "snapshot", scope)
    if asset is None:
        return None, None
    payload = _read_json(asset.path, audit, "snapshot", scope)
    if payload is None:
        return asset, None
    if payload.get("kind") != "practice_utility_sampler_snapshot":
        audit.blocker("snapshot_kind_invalid", "origin snapshot kind is invalid", scope)
    _validate_atomic_origin_link(payload, origin, manifest, audit, scope)
    if not _is_sha(payload.get("distribution_sha256")):
        audit.blocker(
            "snapshot_distribution_hash_invalid", "snapshot distribution hash is invalid", scope
        )

    expected = [context_id for _, context_id, _ in manifest.contexts[origin.stage]]
    selected = payload.get("selected_context_ids")
    if selected != expected:
        audit.blocker(
            "snapshot_selected_contexts_mismatch",
            "selected_context_ids must exactly match manifest order for this stage",
            scope,
        )
    if payload.get("num_selected_contexts") != len(expected):
        audit.blocker(
            "snapshot_selected_count_mismatch",
            f"num_selected_contexts must equal {len(expected)}",
            scope,
        )
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
        audit.blocker("snapshot_contexts_missing", "snapshot has no resident contexts", scope)
        return asset, payload
    if payload.get("num_active_bins") != len(contexts):
        audit.blocker(
            "snapshot_active_count_mismatch",
            "num_active_bins does not match the serialized context rows",
            scope,
        )

    rows: dict[str, list[dict[str, Any]]] = {}
    for row in contexts:
        if not isinstance(row, dict) or not isinstance(row.get("context_id"), str):
            continue
        rows.setdefault(row["context_id"], []).append(row)
    expected_contexts = {
        context_id: context for _, context_id, context in manifest.contexts[origin.stage]
    }
    missing = sorted(set(expected_contexts) - set(rows))
    duplicate = sorted(context_id for context_id, members in rows.items() if len(members) > 1)
    changed = []
    for context_id, expected_context in expected_contexts.items():
        members = rows.get(context_id, [])
        if len(members) == 1:
            actual_context = _canonical_context(members[0].get("context", members[0]))
            if actual_context != expected_context:
                changed.append(context_id)
    if missing or duplicate or changed:
        audit.blocker(
            "snapshot_context_coverage_invalid",
            f"exact context coverage failed: missing={missing}, duplicate={duplicate}, changed={changed}",
            scope,
        )
    elif not any(item.scope == scope for item in audit.report.blockers):
        audit.verified(
            "snapshot_context_coverage_verified",
            f"all {len(expected)} selected contexts are exactly resident and atomically linked",
            scope,
        )
    return asset, payload


def _validate_atomic_origin_link(
    payload: dict[str, Any],
    origin: _Origin,
    manifest: _ManifestView,
    audit: _Audit,
    scope: str,
) -> None:
    expected = {
        "manifest_sha256": manifest.manifest_sha256,
        "stage": origin.stage,
        "seed": origin.seed,
        "global_step": origin.global_step,
        "origin_capsule_sha256": origin.capsule_sha256,
        "origin_capsule_file_sha256": origin.capsule.sha256,
        "origin_checkpoint_sha256": origin.checkpoint.sha256,
    }
    bad = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if bad:
        audit.blocker(
            "origin_atomic_link_mismatch",
            f"artifact is not atomically bound to the origin: {bad}",
            scope,
        )


def _validate_encoder(reference: Any, base_dir: Path, audit: _Audit) -> _Asset | None:
    asset = _verify_asset(reference, base_dir, audit, "encoder", "encoder")
    if asset is None:
        return None
    if not isinstance(reference, dict) or reference.get("frozen") is not True:
        audit.blocker("encoder_not_frozen", "encoder reference must set frozen=true", "encoder")
    if not isinstance(reference, dict) or reference.get("frozen_before_campaign") is not True:
        audit.blocker(
            "encoder_frozen_too_late",
            "encoder must be frozen before any campaign labels are observed",
            "encoder",
        )
    if not isinstance(reference, dict) or not reference.get("artifact_id"):
        audit.blocker(
            "encoder_id_missing", "encoder reference needs a stable artifact_id", "encoder"
        )
    if not any(item.scope == "encoder" for item in audit.report.blockers):
        audit.verified("frozen_encoder_verified", f"frozen encoder hash {asset.sha256}", "encoder")
    return asset


def _validate_proxy_features(
    reference: Any,
    manifest: _ManifestView,
    origins: Mapping[str, _Origin],
    encoder: _Asset | None,
    base_dir: Path,
    audit: _Audit,
) -> _Asset | None:
    asset = _verify_asset(reference, base_dir, audit, "proxy_feature", "proxy_features")
    if asset is None:
        return None
    payload = _read_json(asset.path, audit, "proxy_feature", "proxy_features")
    if payload is None:
        return asset
    try:
        index = _label_builder().load_proxy_feature_index(payload, manifest.payload)
    except ValueError as error:
        audit.blocker("proxy_feature_coverage_inexact", str(error), "proxy_features")
        return asset
    if encoder is None or payload.get("encoder_sha256") != encoder.sha256:
        audit.blocker(
            "proxy_feature_encoder_mismatch",
            "proxy table does not bind the verified frozen encoder hash",
            "proxy_features",
        )

    rows = payload.get("records") or []
    expected_snapshot_hashes = {
        (stage, seed): origin.snapshot.sha256
        for key, origin in origins.items()
        if origin.snapshot is not None
        for stage, seed in [(origin.stage, origin.seed)]
    }
    bad_links = []
    for row in rows:
        key = (str(row.get("stage")), int(row.get("seed")))
        if row.get("source_snapshot_sha256") != expected_snapshot_hashes.get(key):
            bad_links.append((*key, row.get("context_id")))
    if bad_links:
        audit.blocker(
            "proxy_feature_snapshot_links_invalid",
            f"{len(bad_links)} proxy rows do not bind their exact origin-snapshot hash",
            "proxy_features",
        )
    if not any(item.scope == "proxy_features" for item in audit.report.blockers):
        audit.verified(
            "proxy_feature_matrix_verified",
            f"verified frozen encoder/snapshot-linked features for all {len(index)} cells",
            "proxy_features",
        )
    return asset


def _validate_noise_floor(
    reference: Any,
    manifest: _ManifestView,
    efficacy_asset: _Asset | None,
    estimand_sha256: str | None,
    base_dir: Path,
    audit: _Audit,
) -> _Asset | None:
    asset = _verify_asset(reference, base_dir, audit, "noise_floor", "noise_floor")
    if asset is None:
        return None
    payload = _read_json(asset.path, audit, "noise_floor", "noise_floor")
    if payload is None:
        return asset
    long_horizon = max(manifest.horizons, key=manifest.horizons.get)  # type: ignore[arg-type]
    efficacy_plan = None
    if efficacy_asset is not None:
        efficacy_plan = _read_json(efficacy_asset.path, audit, "efficacy_plan", "noise_floor")
    efficacy = efficacy_plan.get("efficacy") if efficacy_plan else None
    try:
        values = _label_builder().validate_noise_floor(
            payload, manifest.payload, efficacy or {}, long_horizon
        )
    except (KeyError, TypeError, ValueError) as error:
        audit.blocker("noise_floor_estimand_mismatch", str(error), "noise_floor")
        return asset
    expected_fields = {
        "estimand_sha256": estimand_sha256,
        "efficacy_plan_sha256": efficacy_asset.sha256 if efficacy_asset else None,
        "epsilon": 0.0,
        "restart_protocol": "symmetric_fresh_restart",
        "origin_state": "settled",
        "replicate_design": "cross_seed_symmetric_restart",
        "randomness_contract": STOCHASTIC_RANDOMNESS_CONTRACT,
    }
    bad = [key for key, expected in expected_fields.items() if payload.get(key) != expected]
    if bad:
        audit.blocker(
            "noise_floor_estimand_mismatch",
            f"noise floor differs from the deployment J_eff estimand on {bad}",
            "noise_floor",
        )

    if len(values) < MIN_NOISE_REPLICATES:
        audit.blocker(
            "noise_floor_replicates_insufficient",
            f"same-estimand floor needs at least {MIN_NOISE_REPLICATES} utility deltas",
            "noise_floor",
        )
    elif statistics.stdev(values) <= 0:
        audit.blocker(
            "noise_floor_zero_variance",
            "same-estimand cross-seed floor is zero and cannot supply Gate A's denominator",
            "noise_floor",
        )
    if not any(item.scope == "noise_floor" for item in audit.report.blockers):
        audit.verified(
            "same_estimand_noise_floor_verified",
            "epsilon-zero settled-restart noise floor matches deployment J_eff at Gate A",
            "noise_floor",
        )
    return asset


def _validate_gates(
    reference: Any,
    manifest: _ManifestView,
    efficacy_plan: dict[str, Any] | None,
    proxy_asset: _Asset | None,
    noise_asset: _Asset | None,
    estimand_sha256: str | None,
    directional_asset: _Asset | None,
    directional_preregistration: dict[str, Any] | None,
    base_dir: Path,
    audit: _Audit,
) -> None:
    asset = _verify_asset(reference, base_dir, audit, "gate_preregistration", "gates")
    if asset is None:
        return
    gates = _read_json(asset.path, audit, "gate_preregistration", "gates")
    if gates is None:
        return
    if any(key in gates for key in ("gate_b_pass", "authorizes_estimator")):
        audit.blocker(
            "gate_b_vocabulary_ambiguous",
            "generic Gate-B fields conflate latent predictiveness with inverse authorization",
            "gates",
        )
    try:
        _label_builder().validate_preregistration(
            gates,
            manifest.payload,
            proxy_features_sha256=proxy_asset.sha256 if proxy_asset else "",
            noise_floor_sha256=noise_asset.sha256 if noise_asset else "",
            efficacy_window=DEFAULT_EFFICACY_WINDOW,
        )
    except (KeyError, TypeError, ValueError) as error:
        audit.blocker("gate_preregistration_invalid", str(error), "gates")
    if efficacy_plan is None or gates.get("efficacy") != efficacy_plan.get("efficacy"):
        audit.blocker(
            "gate_efficacy_plan_mismatch",
            "Gate preregistration efficacy metadata differs from the deployment plan",
            "gates",
        )
    if gates.get("estimand_sha256") != estimand_sha256:
        audit.blocker(
            "gate_estimand_hash_mismatch",
            "Gate preregistration does not bind the deployment utility estimand hash",
            "gates",
        )
    latent = gates.get("latent_proxy_audit")
    external_algorithm = (
        directional_preregistration.get("algorithm")
        if isinstance(directional_preregistration, dict)
        else None
    )
    external_logical_hash = (
        directional_preregistration.get("artifact_sha256")
        if isinstance(directional_preregistration, dict)
        else None
    )
    if not isinstance(latent, dict) or external_algorithm is None or directional_asset is None:
        audit.blocker(
            "gate_directional_preregistration_missing",
            "Gate B requires the verified external directional-calibration preregistration",
            "gates",
        )
    else:
        binding = {
            "embedded_algorithm": latent.get("directional_calibration") == external_algorithm,
            "logical_hash": latent.get("directional_calibration_preregistration_sha256")
            == external_logical_hash,
            "file_hash": latent.get("directional_calibration_preregistration_file_sha256")
            == directional_asset.sha256,
        }
        if not all(binding.values()):
            audit.blocker(
                "gate_directional_preregistration_mismatch",
                "Gate B does not bind the exact external directional algorithm/design artifact",
                "gates",
            )
        else:
            audit.verified(
                "gate_directional_preregistration_verified",
                "Gate B binds the external READY design and exact implemented algorithm",
                "gates",
            )
    if not any(item.scope == "gates" for item in audit.report.blockers):
        audit.verified(
            "gate_vocabulary_verified",
            "Gate A thresholds are explicit; latent predictiveness and inverse estimator "
            "authorization are separate preregistered decisions",
            "gates",
        )


def _verify_asset(
    reference: Any,
    base_dir: Path,
    audit: _Audit,
    code_prefix: str,
    scope: str,
    *,
    record_success: bool = True,
) -> _Asset | None:
    if not isinstance(reference, dict):
        audit.blocker(
            f"{code_prefix}_reference_missing",
            f"{code_prefix} requires a path and SHA-256 reference",
            scope,
        )
        return None
    raw_path = reference.get("path")
    recorded = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        audit.blocker(f"{code_prefix}_path_missing", f"{code_prefix}.path is missing", scope)
        return None
    if not _is_sha(recorded):
        audit.blocker(
            f"{code_prefix}_hash_invalid", f"{code_prefix}.sha256 is not a SHA-256", scope
        )
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        audit.blocker(f"{code_prefix}_missing", f"artifact does not exist: {path}", scope)
        return None
    actual = sha256_file(path)
    if actual != recorded:
        audit.blocker(
            f"{code_prefix}_hash_mismatch",
            f"recorded {recorded} != file hash {actual} for {path}",
            scope,
        )
        return None
    if record_success:
        audit.verified(f"{code_prefix}_hash_verified", f"verified {path} [{actual}]", scope)
    return _Asset(path, actual)


def _build_branch_specs(
    manifest: _ManifestView, origins: Mapping[str, _Origin]
) -> list[BranchSpec]:
    specs: list[BranchSpec] = []
    for stage in manifest.stages:
        for seed in manifest.seeds:
            key = origin_key(stage, seed)
            origin = origins[key]
            relative = dict(manifest.horizons)
            absolute = {label: origin.global_step + value for label, value in relative.items()}
            target = max(absolute.values())
            prefix = f"{manifest.campaign_id}_{stage}_s{seed}"
            control_id = f"{prefix}_control"
            common = {
                "stage": stage,
                "seed": seed,
                "origin_key": key,
                "origin_global_step": origin.global_step,
                "capsule_path": str(origin.capsule.path),
                "checkpoint_path": str(origin.checkpoint.path),
                "relative_horizons": relative,
                "absolute_horizons": absolute,
                "target_global_step": target,
                "manifest_sha256": manifest.manifest_sha256,
            }
            specs.append(BranchSpec(branch_id=control_id, role="control", **common))
            for index, context_id, _ in manifest.contexts[stage]:
                branch_id = f"{prefix}_c{index}_intervention"
                specs.append(
                    BranchSpec(
                        branch_id=branch_id,
                        role="intervention",
                        context_index=index,
                        context_id=context_id,
                        control_branch_id=control_id,
                        **common,
                    )
                )
    return specs


def _label_builder():
    """Load the claim-grade label contract lazily to avoid script import side effects."""

    from scripts.practice_utility import build_utility_labels

    return build_utility_labels


def _label_estimand(manifest: _ManifestView, efficacy: Mapping[str, Any]) -> dict[str, Any] | None:
    required = ("name", "units", "utility_units", "window")
    if any(key not in efficacy for key in required):
        return None
    horizon_label = max(manifest.horizons, key=manifest.horizons.get)  # type: ignore[arg-type]
    return {
        "name": efficacy["name"],
        "outcome_units": efficacy["units"],
        "utility_units": efficacy["utility_units"],
        "quantity": "dose_normalized_practice_utility",
        "normalization": "realized_extra_completed_kernel_steps",
        "horizon_label": horizon_label,
        "horizon_iterations": manifest.horizons[horizon_label],
        "window": efficacy["window"],
    }


def _missing_bundle_contract(audit: _Audit, *, include_bundle: bool) -> None:
    if include_bundle:
        audit.blocker("preflight_bundle_missing", "preflight bundle is required", "preflight")
    required = (
        (
            "passive_dose_plan_missing",
            "hash-bound v2 passive-dose plan with exact manifest context coverage",
            "dose_plan",
        ),
        (
            "live_passive_dose_smoke_missing",
            "hash-bound live-GPU passive-dose smoke receipt",
            "dose",
        ),
        (
            "shared_control_realized_dose_unimplemented",
            "shared controls remain claim-blocked until the live passive-dose smoke passes",
            "dose",
        ),
        (
            "directional_calibration_preregistration_missing",
            "external outcome-free directional-calibration algorithm/design artifact",
            "latent_direction",
        ),
        (
            "origin_map_missing",
            "per-(stage, seed) origin map with full capsule/checkpoint pairs",
            "origins",
        ),
        ("encoder_missing", "frozen, hashed encoder artifact", "encoder"),
        (
            "proxy_feature_coverage_missing",
            "hashed proxy features for every (stage, seed, context)",
            "proxy_features",
        ),
        (
            "efficacy_plan_missing",
            "frozen deployment-J_eff efficacy plan with quality thresholds",
            "efficacy_plan",
        ),
        (
            "same_estimand_noise_floor_missing",
            "epsilon-zero noise floor measured on the deployment-J_eff estimand",
            "noise_floor",
        ),
        (
            "gate_preregistration_missing",
            "explicit Gate-A thresholds and separate latent-proxy/estimator decisions",
            "gates",
        ),
        (
            "branch_specs_unavailable",
            "absolute branch horizons cannot be derived without verified origins",
            "branches",
        ),
    )
    for code, message, scope in required:
        audit.blocker(code, message, scope)


def _canonical_context(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    try:
        return ContextKey.from_dict(payload).to_dict()
    except (KeyError, TypeError, ValueError):
        return None


def _context_id(payload: Any) -> str:
    context = _canonical_context(payload)
    if context is None:
        return "invalid"
    return ContextKey.from_dict(context).context_id


def _split_origin_key(key: str) -> tuple[str, int]:
    stage, raw_seed = key.rsplit(":s", 1)
    return stage, int(raw_seed)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_positive_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) > 0


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _global_step_of(value: Any) -> int | None:
    if isinstance(value, Mapping):
        step = value.get("global_step")
    else:
        step = getattr(value, "global_step", None)
    return int(step) if isinstance(step, int) and not isinstance(step, bool) else None


def _states_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_states_equal(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except Exception:
        return False


__all__ = [
    "BranchSpec",
    "Finding",
    "PreflightReport",
    "audit_probe_campaign",
    "default_preflight_path",
    "manifest_claim_sha256",
    "origin_key",
    "sha256_file",
]
