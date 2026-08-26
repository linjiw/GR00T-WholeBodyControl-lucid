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
``(stage, seed)``.  :func:`audit_probe_campaign` preserves that frozen branch
count and emits deterministic screening branch specifications. The current
callback cannot attribute that shared control's realized completed steps to
each target kernel, so the audit remains blocked until passive per-context
dose instrumentation exists or a newly preregistered manifest uses independent
epsilon-zero controls. It also warns that confirmation evidence requires
independent paired controls.
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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


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
        audit.blocker(
            "shared_control_realized_dose_unimplemented",
            "the live shared control has no target kernels, so it records zero "
            "completed_kernel_steps and cannot support the preregistered realized-extra-dose "
            "denominator; implement passive per-context control dose or freeze a new "
            "independent-control campaign",
            "dose",
        )
        audit.blocker(
            "latent_directional_calibration_unimplemented",
            "Gate B preregisters nested-CV univariate calibration, but no leakage-free "
            "calibration implementation or immutable algorithm artifact exists; the label "
            "builder therefore emits rank-only diagnostics and keeps latent predictiveness "
            "and inverse estimator authorization incomplete",
            "latent_direction",
        )

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
