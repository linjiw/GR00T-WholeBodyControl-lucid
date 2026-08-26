"""Frozen contracts for passive, per-context realized-dose measurement.

A shared control does not need 24 live sampler overrides.  It needs one exact
histogram of executed reference-timeline bins and a frozen set of kernels to
project that histogram onto.  This module validates that outcome-blind plan
without importing Isaac or torch.
"""

# Ruff's force-sort setting conflicts with the repository's isort profile.
# ruff: noqa: I001

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gear_sonic.research.practice_utility.schema import ContextKey, sha256_of

PASSIVE_DOSE_PLAN_KIND = "practice_utility_passive_dose_plan"
PASSIVE_DOSE_PLAN_SCHEMA_VERSION = 2
PASSIVE_DOSE_RECEIPT_KIND = "practice_utility_passive_dose_receipt"
PASSIVE_DOSE_RECEIPT_SCHEMA_VERSION = 2
PASSIVE_DOSE_HOOK = "ManagerEnvWrapper.step.pre_transition_context"


@dataclass(frozen=True)
class PlannedContext:
    """One exact context whose kernel dose must be recoverable."""

    context: ContextKey

    @property
    def context_id(self) -> str:
        return self.context.context_id


@dataclass(frozen=True)
class PassiveDosePlan:
    """Validated, hash-bound passive-dose plan."""

    campaign_id: str
    manifest_sha256: str
    manifest_file_sha256: str
    source_manifest_path: str
    source_commit: str
    launcher_path: str
    launcher_sha256: str
    control_strategy: str
    kernel_radius_bins: int
    reference_bin_size_frames: int
    sigma_frames: float
    contexts_per_stage: dict[str, tuple[PlannedContext, ...]]
    logical_sha256: str
    file_sha256: str
    path: Path

    def contexts_for(self, stage: str) -> tuple[ContextKey, ...]:
        if stage not in self.contexts_per_stage:
            raise ValueError(
                f"passive dose plan has no stage {stage!r}; "
                f"available={sorted(self.contexts_per_stage)}"
            )
        return tuple(item.context for item in self.contexts_per_stage[stage])


def file_sha256(path: str | Path) -> str:
    """Hash an artifact without interpreting it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fields that define the passive-dose estimand."""

    contexts = payload.get("contexts_per_stage")
    if not isinstance(contexts, Mapping):
        contexts = {}
    frozen: dict[str, list[dict[str, Any]]] = {}
    for stage, rows in sorted(contexts.items()):
        if not isinstance(rows, list):
            rows = []
        frozen[str(stage)] = [
            {
                "context_id": row.get("context_id") if isinstance(row, Mapping) else None,
                "context": row.get("context") if isinstance(row, Mapping) else None,
            }
            for row in rows
        ]
    kernel = payload.get("kernel") if isinstance(payload.get("kernel"), Mapping) else {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    return {
        "campaign_id": payload.get("campaign_id"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "manifest_file_sha256": payload.get("manifest_file_sha256"),
        "provenance": provenance,
        "control_strategy": payload.get("control_strategy"),
        "measurement_hook": payload.get("measurement_hook"),
        "kernel": {
            "radius_bins": kernel.get("radius_bins"),
            "reference_bin_size_frames": kernel.get("reference_bin_size_frames"),
            "sigma_frames": kernel.get("sigma_frames"),
            "membership_normalization": kernel.get("membership_normalization"),
        },
        "contexts_per_stage": frozen,
    }


def logical_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_of(logical_payload(payload))


def load_passive_dose_plan(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_campaign_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> PassiveDosePlan:
    """Load and strictly validate a passive-dose plan."""

    path = Path(path).resolve()
    actual_file_sha256 = file_sha256(path)
    if expected_file_sha256 is not None and actual_file_sha256 != expected_file_sha256:
        raise ValueError(
            "passive dose plan file hash mismatch: "
            f"expected {expected_file_sha256}, got {actual_file_sha256}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read passive dose plan {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("passive dose plan must contain a JSON object")
    if payload.get("kind") != PASSIVE_DOSE_PLAN_KIND:
        raise ValueError("passive dose plan kind is invalid")
    if payload.get("schema_version") != PASSIVE_DOSE_PLAN_SCHEMA_VERSION:
        raise ValueError(f"passive dose plan schema must be {PASSIVE_DOSE_PLAN_SCHEMA_VERSION}")
    campaign_id = payload.get("campaign_id")
    manifest_sha256 = payload.get("manifest_sha256")
    manifest_file_sha256 = payload.get("manifest_file_sha256")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("passive dose plan requires campaign_id")
    if not _is_sha256(manifest_sha256):
        raise ValueError("passive dose plan requires a manifest SHA-256")
    if not _is_sha256(manifest_file_sha256):
        raise ValueError("passive dose plan requires a manifest file SHA-256")
    if expected_campaign_id is not None and campaign_id != expected_campaign_id:
        raise ValueError("passive dose plan campaign_id differs from the manifest")
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("passive dose plan is not bound to the frozen manifest")
    if (
        expected_manifest_file_sha256 is not None
        and manifest_file_sha256 != expected_manifest_file_sha256
    ):
        raise ValueError("passive dose plan is not bound to the frozen manifest bytes")
    if payload.get("control_strategy") != "shared_per_stage_seed":
        raise ValueError("passive dose plan must declare shared_per_stage_seed")
    if payload.get("measurement_hook") != PASSIVE_DOSE_HOOK:
        raise ValueError(f"passive dose plan must use {PASSIVE_DOSE_HOOK}")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("passive dose plan requires provenance")
    source_manifest = provenance.get("source_manifest")
    git = provenance.get("git")
    launcher = provenance.get("launcher")
    if not isinstance(source_manifest, dict):
        raise ValueError("passive dose plan requires source_manifest provenance")
    if (
        not isinstance(source_manifest.get("path"), str)
        or not source_manifest["path"]
        or source_manifest.get("logical_sha256") != manifest_sha256
        or source_manifest.get("file_sha256") != manifest_file_sha256
    ):
        raise ValueError("passive dose source_manifest provenance is inconsistent")
    if not isinstance(git, dict) or not _is_commit(git.get("sha")) or git.get("status_short") != []:
        raise ValueError("passive dose plan requires a clean 40-character Git identity")
    if (
        not isinstance(launcher, dict)
        or not isinstance(launcher.get("path"), str)
        or not launcher["path"]
        or not _is_sha256(launcher.get("sha256"))
    ):
        raise ValueError("passive dose plan requires launcher path and SHA-256 provenance")

    kernel = payload.get("kernel")
    if not isinstance(kernel, dict):
        raise ValueError("passive dose plan requires a kernel object")
    radius = kernel.get("radius_bins")
    reference_bin_size = kernel.get("reference_bin_size_frames")
    sigma = kernel.get("sigma_frames")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("passive dose kernel radius_bins must be non-negative")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or float(sigma) <= 0:
        raise ValueError("passive dose kernel sigma_frames must be positive")
    if (
        isinstance(reference_bin_size, bool)
        or not isinstance(reference_bin_size, int)
        or reference_bin_size <= 0
    ):
        raise ValueError("passive dose reference_bin_size_frames must be a positive integer")
    if float(sigma) != float(reference_bin_size):
        raise ValueError("passive dose sigma_frames must equal reference_bin_size_frames")
    if kernel.get("membership_normalization") != "peak_equals_one":
        raise ValueError("passive dose kernel membership_normalization must be peak_equals_one")

    raw_stages = payload.get("contexts_per_stage")
    if not isinstance(raw_stages, dict) or not raw_stages:
        raise ValueError("passive dose plan requires contexts_per_stage")
    contexts_per_stage: dict[str, tuple[PlannedContext, ...]] = {}
    all_contexts: list[ContextKey] = []
    for stage, rows in raw_stages.items():
        if not isinstance(stage, str) or not stage or not isinstance(rows, list) or not rows:
            raise ValueError("every passive dose stage must have a non-empty context list")
        parsed: list[PlannedContext] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("context"), dict):
                raise ValueError(f"passive dose context {stage}/{index} is malformed")
            try:
                context = ContextKey.from_dict(row["context"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"passive dose context {stage}/{index} is invalid") from error
            if row.get("context_id") != context.context_id:
                raise ValueError(f"passive dose context id mismatch at {stage}/{index}")
            if context.context_id in seen:
                raise ValueError(f"passive dose stage {stage!r} contains duplicate contexts")
            seen.add(context.context_id)
            parsed.append(PlannedContext(context))
            all_contexts.append(context)
        contexts_per_stage[stage] = tuple(parsed)

    derived_reference_bin_size = derive_reference_bin_size_frames(all_contexts)
    if reference_bin_size != derived_reference_bin_size:
        raise ValueError("passive dose reference_bin_size_frames differs from its frozen contexts")

    computed = logical_sha256(payload)
    if payload.get("dose_plan_sha256") != computed:
        raise ValueError("passive dose plan logical hash mismatch")
    return PassiveDosePlan(
        campaign_id=campaign_id,
        manifest_sha256=str(manifest_sha256),
        manifest_file_sha256=str(manifest_file_sha256),
        source_manifest_path=str(source_manifest["path"]),
        source_commit=str(git["sha"]),
        launcher_path=str(launcher["path"]),
        launcher_sha256=str(launcher["sha256"]),
        control_strategy="shared_per_stage_seed",
        kernel_radius_bins=radius,
        reference_bin_size_frames=reference_bin_size,
        sigma_frames=float(sigma),
        contexts_per_stage=contexts_per_stage,
        logical_sha256=computed,
        file_sha256=actual_file_sha256,
        path=path,
    )


def probe_manifest_claim_sha256(payload: Mapping[str, Any]) -> str:
    """Recompute the v1 probe manifest's claim-bearing logical hash."""

    raw_contexts = payload.get("contexts_per_stage")
    contexts = raw_contexts if isinstance(raw_contexts, Mapping) else {}
    frozen_contexts: dict[str, list[str]] = {}
    for stage, entries in sorted(contexts.items()):
        if not isinstance(entries, list):
            entries = []
        context_ids = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                context_ids.append("invalid")
                continue
            raw_context = entry.get("context")
            try:
                computed = ContextKey.from_dict(raw_context).context_id  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                computed = "invalid"
            context_ids.append(str(entry.get("context_id") or computed))
        frozen_contexts[str(stage)] = sorted(context_ids)
    seeds = payload.get("seeds")
    return sha256_of(
        {
            "campaign_id": payload.get("campaign_id"),
            "contexts": frozen_contexts,
            "seeds": sorted(seeds) if isinstance(seeds, list) else [],
            "epsilon": payload.get("epsilon"),
            "kernel_radius_bins": payload.get("kernel_radius_bins"),
            "horizons": payload.get("horizons") or {},
            "pool_sha256": payload.get("pool_sha256"),
            "split_sha256": payload.get("split_sha256"),
        }
    )


def build_passive_dose_plan_payload(
    manifest: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
    sigma_frames: float,
    created_at: str,
    source_manifest_path: str,
    source_commit: str,
    git_status_short: list[str],
    launcher_path: str,
    launcher_sha256: str,
) -> dict[str, Any]:
    """Create an outcome-blind v2 plan from one frozen v1 manifest.

    This reads only design fields (contexts, kernel, and lineage); it has no
    branch-result or label input.
    """

    if manifest.get("kind") != "practice_utility_probe_manifest":
        raise ValueError("source must be a practice_utility_probe_manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("source probe manifest schema must be 1")
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("source probe manifest requires campaign_id")
    claimed_manifest_sha256 = manifest.get("manifest_sha256")
    computed_manifest_sha256 = probe_manifest_claim_sha256(manifest)
    if claimed_manifest_sha256 != computed_manifest_sha256:
        raise ValueError("source probe manifest logical hash mismatch")
    if not _is_sha256(manifest_file_sha256):
        raise ValueError("manifest_file_sha256 must be a SHA-256")
    if not isinstance(source_manifest_path, str) or not source_manifest_path:
        raise ValueError("source_manifest_path is required")
    if not _is_commit(source_commit):
        raise ValueError("source_commit must be a 40-character lowercase Git SHA")
    if not isinstance(git_status_short, list) or not all(
        isinstance(entry, str) for entry in git_status_short
    ):
        raise ValueError("git_status_short must be a list of strings")
    if not isinstance(launcher_path, str) or not launcher_path:
        raise ValueError("launcher_path is required")
    if not _is_sha256(launcher_sha256):
        raise ValueError("launcher_sha256 must be a SHA-256")
    radius = manifest.get("kernel_radius_bins")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("source kernel_radius_bins must be non-negative")
    if isinstance(sigma_frames, bool) or not isinstance(sigma_frames, (int, float)):
        raise ValueError("sigma_frames must be positive")
    sigma_frames = float(sigma_frames)
    if sigma_frames <= 0:
        raise ValueError("sigma_frames must be positive")

    raw_stages = manifest.get("stages")
    raw_contexts = manifest.get("contexts_per_stage")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("source probe manifest requires stages")
    if not isinstance(raw_contexts, Mapping):
        raise ValueError("source probe manifest requires contexts_per_stage")
    stages = [str(stage) for stage in raw_stages]
    if len(set(stages)) != len(stages) or set(stages) != set(raw_contexts):
        raise ValueError("source stages and contexts_per_stage differ")

    contexts_per_stage: dict[str, list[dict[str, Any]]] = {}
    all_contexts: list[ContextKey] = []
    for stage in sorted(stages):
        rows = raw_contexts.get(stage)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"source stage {stage!r} requires contexts")
        parsed = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("context"), Mapping):
                raise ValueError(f"source context {stage}/{index} is malformed")
            try:
                context = ContextKey.from_dict(row["context"])  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"source context {stage}/{index} is invalid") from error
            if row.get("context_id") != context.context_id:
                raise ValueError(f"source context id mismatch at {stage}/{index}")
            if context.context_id in seen:
                raise ValueError(f"source stage {stage!r} contains duplicate contexts")
            seen.add(context.context_id)
            parsed.append({"context_id": context.context_id, "context": context.to_dict()})
            all_contexts.append(context)
        contexts_per_stage[stage] = sorted(parsed, key=lambda item: item["context_id"])

    reference_bin_size_frames = derive_reference_bin_size_frames(all_contexts)
    if sigma_frames != float(reference_bin_size_frames):
        raise ValueError(
            "sigma_frames must equal the manifest-derived reference bin size "
            f"({sigma_frames} != {reference_bin_size_frames})"
        )

    payload: dict[str, Any] = {
        "kind": PASSIVE_DOSE_PLAN_KIND,
        "schema_version": PASSIVE_DOSE_PLAN_SCHEMA_VERSION,
        "created_at": created_at,
        "campaign_id": campaign_id,
        "manifest_sha256": computed_manifest_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "provenance": {
            "source_manifest": {
                "path": source_manifest_path,
                "logical_sha256": computed_manifest_sha256,
                "file_sha256": manifest_file_sha256,
            },
            "git": {"sha": source_commit, "status_short": git_status_short},
            "launcher": {"path": launcher_path, "sha256": launcher_sha256},
        },
        "control_strategy": "shared_per_stage_seed",
        "measurement_hook": PASSIVE_DOSE_HOOK,
        "kernel": {
            "radius_bins": radius,
            "reference_bin_size_frames": reference_bin_size_frames,
            "sigma_frames": sigma_frames,
            "membership_normalization": "peak_equals_one",
        },
        "contexts_per_stage": contexts_per_stage,
    }
    payload["dose_plan_sha256"] = logical_sha256(payload)
    return payload


def derive_reference_bin_size_frames(contexts: list[ContextKey] | tuple[ContextKey, ...]) -> int:
    """Derive and validate the sampler's reference-timeline bin size.

    Nonzero bins identify the size exactly through
    ``bin_start_frame == bin_index * size``. Bin widths provide the fallback
    when a plan happens to select only bin zero, and validate terminal bins,
    whose final width may be shorter than the reference size.
    """

    if not contexts:
        raise ValueError("cannot derive reference bin size without contexts")
    indexed_candidates: set[int] = set()
    widths: list[int] = []
    for context in contexts:
        width = context.bin_end_frame - context.bin_start_frame
        widths.append(width)
        if context.bin_index == 0:
            if context.bin_start_frame != 0:
                raise ValueError("bin zero must start at reference frame zero")
            continue
        if context.bin_start_frame <= 0 or context.bin_start_frame % context.bin_index:
            raise ValueError(
                f"context {context.context_id} cannot identify an integer reference bin size"
            )
        indexed_candidates.add(context.bin_start_frame // context.bin_index)
    if len(indexed_candidates) > 1:
        raise ValueError(
            f"contexts imply inconsistent reference bin sizes {sorted(indexed_candidates)}"
        )
    reference = next(iter(indexed_candidates), max(widths))
    if reference <= 0:
        raise ValueError("derived reference bin size must be positive")
    for context, width in zip(contexts, widths):
        if context.bin_start_frame != context.bin_index * reference:
            raise ValueError(f"context {context.context_id} start does not match its bin index")
        if width <= 0 or width > reference:
            raise ValueError(
                f"context {context.context_id} width {width} exceeds reference bin size {reference}"
            )
    return reference


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)
    )
