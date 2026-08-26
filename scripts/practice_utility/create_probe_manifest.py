#!/usr/bin/env python3
"""Build a frozen, origin-aligned probe campaign.

A claim-bearing campaign is selected from settled sampler origins, not from a
loose snapshot.  ``create_probe_origins.py`` writes one hash-bound origin map
per policy stage.  This launcher verifies the map and every per-seed snapshot,
intersects resident ``ContextKey`` identities across the declared seeds, and
then performs outcome-blind stratified selection.

Snapshot bin bounds live on SONIC's resampled control timeline (normally 50
Hz), while offline clips retain their source rate (normally 30 Hz).  Feature
windows are converted by time before slicing the source clip, and both rates
and converted bounds are recorded with the candidate features.

Example
-------
    python scripts/practice_utility/create_probe_manifest.py \
        --origin-map late .../probe_origins_origin_map.json <sha256> \
        --pool-manifest .../pool_debug512.json \
        --split-manifest .../split_debug512_performer.json \
        --contexts-per-stage 24 --seeds 9300 9301 \
        --output .../probe_screen_v2_late.json

The legacy ``--snapshot STAGE=PATH`` route is retained only for explicitly
exploratory manifests and requires ``--exploratory``.
"""

# Ruff's import combiner disagrees with this repository's isort profile; keep
# the Makefile's isort layout and suppress only that overlap plus path-bootstrap imports.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import motion_pool as MP
from gear_sonic.research.practice_utility import probe_manifest as PM
from gear_sonic.research.practice_utility import proxy_features as PF
from gear_sonic.research.practice_utility.schema import ContextKey, sha256_of

DEFAULT_SNAPSHOT_TIMELINE_FPS = 50.0


@dataclass(frozen=True)
class OriginMapReference:
    """One stage-keyed origin-map asset with an expected byte hash."""

    stage: str
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class SnapshotSource:
    """One verified sampler snapshot bound to a stage and seed."""

    seed: int
    path: Path
    file_sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class OriginStage:
    """Verified inputs used to construct one manifest stage."""

    stage: str
    origin_step: int
    map_path: Path
    map_file_sha256: str
    map_payload_sha256: str
    snapshots: tuple[SnapshotSource, ...]
    common_context_ids: tuple[str, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--origin-map",
        action="append",
        nargs=3,
        metavar=("STAGE", "PATH", "SHA256"),
        help=("claim-grade origin map and expected file hash; repeat once per policy stage"),
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="legacy loose sampler snapshot; requires --exploratory",
    )
    parser.add_argument(
        "--intersect-origin-snapshots",
        action="store_true",
        help="intersect repeated loose snapshots (exploratory mode only)",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="allow the legacy loose-snapshot route; output is not claim-grade",
    )
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="separate creation receipt (default: OUTPUT.creation_receipt.json)",
    )
    parser.add_argument("--campaign-id", default="oracle_screen_v1")
    parser.add_argument("--contexts-per-stage", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--kernel-radius", type=int, default=1)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs=3,
        default=[8, 32, 128],
        metavar=("H_S", "H_M", "H_L"),
    )
    parser.add_argument("--selection-seed", type=int, default=20260818)
    parser.add_argument("--train-partition", default="adaptation")
    parser.add_argument(
        "--snapshot-timeline-fps",
        type=float,
        default=DEFAULT_SNAPSHOT_TIMELINE_FPS,
        help="rate of snapshot bin bounds before conversion to each source clip's FPS",
    )
    parser.add_argument(
        "--skip-motion-features",
        action="store_true",
        help="skip offline feature extraction (faster, coarser strata)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if args.snapshot_timeline_fps <= 0:
        parser.error("--snapshot-timeline-fps must be positive")
    if args.origin_map and args.snapshot:
        parser.error("use --origin-map or --snapshot, not both")
    if not args.origin_map and not args.snapshot:
        parser.error("claim-grade construction requires --origin-map")
    if args.snapshot and not args.exploratory:
        parser.error("loose --snapshot input requires --exploratory")
    if args.origin_map and args.exploratory:
        parser.error("--exploratory is only valid with loose --snapshot input")
    if args.intersect_origin_snapshots and not args.exploratory:
        parser.error("--intersect-origin-snapshots is exploratory-only")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    return output.splitlines()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return payload


def load_clip_index(pool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    motions = pool.get("motions")
    if not isinstance(motions, list):
        raise ValueError("pool manifest motions must be a list")
    clips: dict[str, dict[str, Any]] = {}
    for raw in motions:
        if not isinstance(raw, dict) or not isinstance(raw.get("motion_key"), str):
            raise ValueError("pool manifest contains an invalid motion row")
        key = raw["motion_key"]
        if key in clips:
            raise ValueError(f"pool manifest contains duplicate motion {key!r}")
        if not _is_sha256(raw.get("content_sha256")):
            raise ValueError(f"pool motion {key!r} has no valid content SHA-256")
        clips[key] = raw
    return clips


def verify_pool_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    """Verify the serialized pool hash against its logical claim-bearing content."""
    pool = _load_json(path, "pool manifest")
    if pool.get("kind") != "practice_utility_motion_pool" or pool.get("schema_version") != 1:
        raise ValueError("pool manifest has the wrong kind or schema version")
    clips = load_clip_index(pool)
    expected = sha256_of(
        {
            "source_root": pool.get("source_root"),
            "records": [
                {
                    "motion_key": key,
                    "content_sha256": clips[key].get("content_sha256"),
                }
                for key in sorted(clips)
            ],
        }
    )
    if pool.get("pool_sha256") != expected:
        raise ValueError(
            f"pool logical hash mismatch: serialized={pool.get('pool_sha256')!r}, "
            f"computed={expected}"
        )
    return pool, expected, file_sha256(path)


def verify_pool_source_bytes(pool: dict[str, Any]) -> dict[str, Any]:
    """Rescan the live pool and bind the exact files used for feature extraction."""
    source_root = pool.get("source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("pool manifest has no source_root")
    scan = MP.scan_pool(source_root)
    if pool.get("deduplicated"):
        scan = MP.drop_exact_duplicates(scan)
    if MP.pool_sha256(scan) != pool.get("pool_sha256"):
        raise ValueError("motion-pool source bytes differ from the frozen logical pool")

    serialized = load_clip_index(pool)
    live = {record.motion_key: record for record in scan.records}
    if set(serialized) != set(live):
        raise ValueError("rescanned motion keys differ from the frozen pool manifest")
    files: dict[str, dict[str, Any]] = {}
    for key in sorted(live):
        row = serialized[key]
        record = live[key]
        try:
            serialized_path = Path(row["path"]).resolve(strict=True)
            live_path = Path(record.path).resolve(strict=True)
        except (KeyError, FileNotFoundError) as error:
            raise ValueError(f"pool record {key!r} does not resolve to a source file") from error
        if serialized_path != live_path:
            raise ValueError(
                f"pool record {key!r} path is not its rescanned source file: "
                f"serialized={serialized_path}, rescanned={live_path}"
            )
        used_fields = {
            "content_sha256": record.content_sha256,
            "num_frames": record.num_frames,
            "fps": record.fps,
            "family": record.family,
        }
        mismatched = [field for field, value in used_fields.items() if row.get(field) != value]
        if mismatched:
            raise ValueError(
                f"pool record {key!r} differs from its rescanned source in {mismatched}"
            )
        files[key] = {
            "path": str(live_path),
            "file_sha256": file_sha256(live_path),
        }
    tree_sha256 = sha256_of(
        {
            "files": [
                {
                    "motion_key": key,
                    "path": files[key]["path"],
                    "file_sha256": files[key]["file_sha256"],
                }
                for key in sorted(files)
            ]
        }
    )
    return {
        "source_root": str(Path(source_root).resolve()),
        "pool_sha256": pool["pool_sha256"],
        "source_file_count": len(files),
        "source_files_sha256": tree_sha256,
        "files": files,
    }


def reverify_pool_source_bytes(binding: dict[str, Any]) -> None:
    """Refuse a manifest if a source file changed while selection was running."""
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("pool source-byte binding contains no files")
    for key, row in files.items():
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row.get("file_sha256"):
            raise ValueError(f"pool source file changed during manifest creation: {key!r}")


def verify_split_manifest(
    path: Path, *, pool_sha256: str, pool_motion_keys: set[str]
) -> tuple[dict[str, Any], str, str]:
    """Verify the split logical hash and its exact linkage to the selected pool."""
    split = _load_json(path, "split manifest")
    if (
        split.get("kind") != "practice_utility_group_disjoint_split"
        or split.get("schema_version") != 1
    ):
        raise ValueError("split manifest has the wrong kind or schema version")
    if split.get("pool_sha256") != pool_sha256:
        raise ValueError("split manifest does not bind the selected pool logical hash")
    assignment = split.get("assignment")
    if not isinstance(assignment, dict):
        raise ValueError("split manifest assignment must be an object")
    if set(assignment) != pool_motion_keys:
        missing = sorted(pool_motion_keys - set(assignment))
        extra = sorted(set(assignment) - pool_motion_keys)
        raise ValueError(
            f"split assignment does not exactly cover the pool: missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    expected = sha256_of(
        {
            "assignment": dict(sorted(assignment.items())),
            "linkage": split.get("linkage"),
            "seed": split.get("seed"),
            "pool_sha256": split.get("pool_sha256"),
        }
    )
    if split.get("split_sha256") != expected:
        raise ValueError(
            f"split logical hash mismatch: serialized={split.get('split_sha256')!r}, "
            f"computed={expected}"
        )
    return split, expected, file_sha256(path)


def parse_origin_map_references(items: list[list[str]]) -> dict[str, OriginMapReference]:
    """Parse and reject duplicate stage/path/hash origin-map references."""
    references: dict[str, OriginMapReference] = {}
    paths: set[Path] = set()
    for raw_stage, raw_path, expected_sha256 in items:
        stage = raw_stage.strip()
        path = Path(raw_path).resolve()
        if not stage:
            raise ValueError("origin-map stage cannot be empty")
        if stage in references:
            raise ValueError(f"duplicate origin map for stage {stage!r}")
        if path in paths:
            raise ValueError(f"origin map {path} was assigned to more than one stage")
        if not _is_sha256(expected_sha256):
            raise ValueError(f"origin-map hash for stage {stage!r} is not a SHA-256")
        references[stage] = OriginMapReference(stage, path, expected_sha256)
        paths.add(path)
    return references


def group_snapshot_paths(items: list[str]) -> dict[str, list[Path]]:
    """Parse repeatable exploratory ``STAGE=PATH`` arguments without overwriting."""
    grouped: dict[str, list[Path]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--snapshot expects STAGE=PATH, got {item!r}")
        stage, raw_path = item.split("=", 1)
        if not stage:
            raise SystemExit(f"--snapshot stage is empty in {item!r}")
        grouped.setdefault(stage, []).append(Path(raw_path))
    return grouped


def _validated_snapshot_rows(
    snapshot: dict[str, Any], clips: dict[str, dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    if (
        snapshot.get("kind") != "practice_utility_sampler_snapshot"
        or snapshot.get("schema_version") != 1
    ):
        raise ValueError(f"{label} has the wrong kind or schema version")
    contexts = snapshot.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"{label} contains no resident contexts")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in contexts:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} contains a non-object context")
        context = ContextKey.from_dict(raw)
        context_id = raw.get("context_id")
        if context_id != context.context_id:
            raise ValueError(f"{label} context hash mismatch for {context.motion_key!r}")
        if context_id in indexed:
            raise ValueError(f"{label} contains duplicate context {context_id}")
        record = clips.get(context.motion_key)
        if record is None:
            raise ValueError(f"{label} context motion {context.motion_key!r} is absent from pool")
        if context.motion_hash != record.get("content_sha256"):
            raise ValueError(
                f"{label} ContextKey motion hash mismatch for {context.motion_key!r}: "
                "snapshot is not bound to the selected pool"
            )
        indexed[context_id] = dict(raw)
    if snapshot.get("num_active_bins") not in (None, len(indexed)):
        raise ValueError(f"{label} num_active_bins does not match its context rows")
    return indexed


def intersect_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcome-blind intersection of already-validated replicate snapshots."""
    if not snapshots:
        raise ValueError("at least one snapshot is required")
    indexed: list[dict[str, dict[str, Any]]] = []
    for snapshot in snapshots:
        rows: dict[str, dict[str, Any]] = {}
        for raw in snapshot.get("contexts", []):
            row = dict(raw)
            context = ContextKey.from_dict(row)
            context_id = str(row.get("context_id") or context.context_id)
            if context_id in rows:
                raise ValueError(f"snapshot contains duplicate context {context_id}")
            if context.context_id != context_id:
                raise ValueError(f"snapshot context hash mismatch for {context_id}")
            rows[context_id] = row
        indexed.append(rows)

    common = set(indexed[0])
    for rows in indexed[1:]:
        common.intersection_update(rows)
    merged = []
    for context_id in sorted(common):
        versions = [rows[context_id] for rows in indexed]
        context = ContextKey.from_dict(versions[0])
        if any(ContextKey.from_dict(row) != context for row in versions[1:]):
            raise ValueError(f"context identity fields disagree for {context_id}")
        row = dict(versions[0])
        row["failure_rate"] = statistics.fmean(
            float(version.get("failure_rate", 0.0)) for version in versions
        )
        row["sampling_probability"] = statistics.fmean(
            float(version.get("sampling_probability", 0.0)) for version in versions
        )
        row["origin_count"] = len(versions)
        merged.append(row)
    return {
        "kind": "practice_utility_intersected_sampler_snapshot",
        "schema_version": 1,
        "num_origins": len(snapshots),
        "contexts": merged,
    }


def load_origin_stages(
    references: dict[str, OriginMapReference],
    *,
    seeds: list[int],
    clips: dict[str, dict[str, Any]],
    pool_sha256: str,
    split_sha256: str,
    snapshot_timeline_fps: float = DEFAULT_SNAPSHOT_TIMELINE_FPS,
) -> dict[str, OriginStage]:
    """Transitively verify origin maps and exactly one snapshot per declared seed."""
    expected_seed_keys = {str(seed) for seed in seeds}
    stages: dict[str, OriginStage] = {}
    all_map_hashes: set[str] = set()
    all_snapshot_paths: set[Path] = set()
    all_snapshot_hashes: set[str] = set()
    for stage, reference in references.items():
        actual_map_hash = file_sha256(reference.path)
        if actual_map_hash != reference.expected_sha256:
            raise ValueError(
                f"origin map {stage!r} file hash mismatch: expected "
                f"{reference.expected_sha256}, got {actual_map_hash}"
            )
        if actual_map_hash in all_map_hashes:
            raise ValueError("distinct stages cannot reuse one origin-map payload")
        all_map_hashes.add(actual_map_hash)
        payload = _load_json(reference.path, f"origin map {stage!r}")
        if (
            payload.get("kind") != "practice_utility_probe_origin_map"
            or payload.get("schema_version") != 1
        ):
            raise ValueError(f"origin map {stage!r} has the wrong kind or schema version")
        if payload.get("stage") != stage:
            raise ValueError(
                f"origin map stage mismatch: CLI={stage!r}, payload={payload.get('stage')!r}"
            )
        if payload.get("usable_for_manifest_selection") is not True:
            raise ValueError(f"origin map {stage!r} was not marked usable for selection")
        if payload.get("seeds") != seeds:
            raise ValueError(
                f"origin map {stage!r} seeds {payload.get('seeds')!r} "
                f"do not match declared seeds {seeds!r}"
            )
        if payload.get("motion_pool_manifest_sha256") != pool_sha256:
            raise ValueError(f"origin map {stage!r} does not bind the selected motion pool")
        if payload.get("dev_suite_sha256") != split_sha256:
            raise ValueError(f"origin map {stage!r} does not bind the selected split")
        experiment_id = payload.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError(f"origin map {stage!r} has no experiment_id")
        origin_step = payload.get("origin_step")
        if not isinstance(origin_step, int) or isinstance(origin_step, bool) or origin_step <= 0:
            raise ValueError(f"origin map {stage!r} has an invalid common origin_step")

        origins = payload.get("origins")
        if not isinstance(origins, dict) or set(origins) != expected_seed_keys:
            actual = set(origins) if isinstance(origins, dict) else set()
            raise ValueError(
                f"origin map {stage!r} must contain exactly one origin per declared seed; "
                f"missing={sorted(expected_seed_keys - actual)}, "
                f"extra={sorted(actual - expected_seed_keys)}"
            )

        sources: list[SnapshotSource] = []
        snapshot_paths: set[Path] = set()
        snapshot_hashes: set[str] = set()
        snapshot_contexts: list[set[str]] = []
        for seed in seeds:
            row = origins[str(seed)]
            if not isinstance(row, dict):
                raise ValueError(f"origin {stage!r}/seed={seed} must be an object")
            if row.get("seed") != seed or row.get("origin_step") != origin_step:
                raise ValueError(
                    f"origin {stage!r}/seed={seed} does not bind its seed and common step"
                )
            if row.get("settled") is not True or row.get("blockers") not in (None, []):
                raise ValueError(f"origin {stage!r}/seed={seed} is not a usable settled origin")
            raw_path = row.get("snapshot")
            expected_hash = row.get("snapshot_sha256")
            if not isinstance(raw_path, str) or not _is_sha256(expected_hash):
                raise ValueError(f"origin {stage!r}/seed={seed} lacks one hash-bound snapshot")
            path = Path(raw_path)
            if not path.is_absolute():
                path = (reference.path.parent / path).resolve()
            else:
                path = path.resolve()
            if path in snapshot_paths:
                raise ValueError(f"origin {stage!r} reuses one snapshot path across seeds")
            if path in all_snapshot_paths:
                raise ValueError(f"origin maps reuse snapshot path {path}")
            actual_hash = file_sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"snapshot hash mismatch for {stage!r}/seed={seed}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            if actual_hash in snapshot_hashes:
                raise ValueError(f"origin {stage!r} reuses one snapshot payload across seeds")
            if actual_hash in all_snapshot_hashes:
                raise ValueError("origin maps reuse one snapshot payload")
            snapshot = _load_json(path, f"snapshot {stage!r}/seed={seed}")
            if snapshot.get("snapshot_timeline_fps") != snapshot_timeline_fps:
                raise ValueError(
                    f"snapshot {stage!r}/seed={seed} timeline FPS "
                    f"{snapshot.get('snapshot_timeline_fps')!r} does not match "
                    f"the declared {snapshot_timeline_fps!r}"
                )
            if snapshot.get("global_step") != origin_step:
                raise ValueError(
                    f"snapshot {stage!r}/seed={seed} step {snapshot.get('global_step')!r} "
                    f"does not match common origin step {origin_step}"
                )
            if "seed" in snapshot and snapshot.get("seed") != seed:
                raise ValueError(f"snapshot {stage!r}/seed={seed} has a different embedded seed")
            expected_branch_id = f"{experiment_id}_s{seed}"
            if snapshot.get("branch_id") != expected_branch_id:
                raise ValueError(
                    f"snapshot {stage!r}/seed={seed} branch_id must be " f"{expected_branch_id!r}"
                )
            rows = _validated_snapshot_rows(snapshot, clips, f"snapshot {stage!r}/seed={seed}")
            resident_ids = row.get("resident_context_ids")
            if not isinstance(resident_ids, list) or sorted(resident_ids) != sorted(rows):
                raise ValueError(
                    f"origin-map resident ids do not match snapshot {stage!r}/seed={seed}"
                )
            if row.get("num_resident_contexts") != len(rows):
                raise ValueError(
                    f"origin-map resident count does not match snapshot {stage!r}/seed={seed}"
                )
            sources.append(SnapshotSource(seed, path, actual_hash, snapshot))
            snapshot_paths.add(path)
            snapshot_hashes.add(actual_hash)
            all_snapshot_paths.add(path)
            all_snapshot_hashes.add(actual_hash)
            snapshot_contexts.append(set(rows))

        if len(sources) != len(seeds):
            raise ValueError(f"origin map {stage!r} does not have one snapshot per seed")
        common = set.intersection(*snapshot_contexts)
        serialized_common = payload.get("common_resident_context_ids")
        if not isinstance(serialized_common, list) or len(serialized_common) != len(
            set(serialized_common)
        ):
            raise ValueError(f"origin map {stage!r} has invalid common resident ids")
        if set(serialized_common) != common:
            raise ValueError(f"origin map {stage!r} common ids do not equal snapshot intersection")
        if payload.get("num_common_resident_contexts") != len(common):
            raise ValueError(f"origin map {stage!r} common resident count is inconsistent")
        stages[stage] = OriginStage(
            stage=stage,
            origin_step=origin_step,
            map_path=reference.path,
            map_file_sha256=actual_map_hash,
            map_payload_sha256=sha256_of(payload),
            snapshots=tuple(sources),
            common_context_ids=tuple(sorted(common)),
        )
    return stages


def source_bin_bounds(
    start_frame: int,
    end_frame: int,
    *,
    source_fps: float,
    snapshot_timeline_fps: float,
    source_num_frames: int,
) -> tuple[int, int]:
    """Map a half-open snapshot interval to a covering source-clip interval."""
    if not 0 <= start_frame < end_frame:
        raise ValueError(f"invalid snapshot bin [{start_frame}, {end_frame})")
    if source_fps <= 0 or snapshot_timeline_fps <= 0:
        raise ValueError("source and snapshot timeline FPS must be positive")
    if source_num_frames < 2:
        raise ValueError("source clip needs at least two frames for bin features")
    scale = source_fps / snapshot_timeline_fps
    source_start = math.floor(start_frame * scale)
    source_end = math.ceil(end_frame * scale)
    source_start = min(source_start, source_num_frames - 2)
    source_end = min(max(source_end, source_start + 2), source_num_frames)
    return source_start, source_end


def build_candidates(
    snapshot: dict[str, Any],
    clips: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    partition: str,
    with_features: bool,
    snapshot_timeline_fps: float = DEFAULT_SNAPSHOT_TIMELINE_FPS,
):
    """Turn resident contexts into stratifiable candidates."""
    import joblib

    cache: dict[str, dict[str, Any]] = {}
    candidates, skipped_partition, skipped_missing = [], 0, 0

    for entry in snapshot["contexts"]:
        motion_key = entry["motion_key"]
        if assignment.get(motion_key) != partition:
            skipped_partition += 1
            continue
        record = clips.get(motion_key)
        if record is None:
            skipped_missing += 1
            continue

        regime, extras = "unknown", {}
        if with_features:
            if motion_key not in cache:
                cache[motion_key] = joblib.load(record["path"])[motion_key]
            clip = cache[motion_key]
            frames = int(clip["dof"].shape[0])
            source_fps = float(clip.get("fps", record.get("fps", 30.0)))
            recorded_fps = float(record.get("fps", source_fps))
            if not math.isclose(source_fps, recorded_fps, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"pool/source FPS mismatch for {motion_key!r}: "
                    f"manifest={recorded_fps}, clip={source_fps}"
                )
            start, end = source_bin_bounds(
                int(entry["bin_start_frame"]),
                int(entry["bin_end_frame"]),
                source_fps=source_fps,
                snapshot_timeline_fps=snapshot_timeline_fps,
                source_num_frames=frames,
            )
            extras = {
                "source_fps": source_fps,
                "snapshot_timeline_fps": snapshot_timeline_fps,
                "source_bin_start_frame": start,
                "source_bin_end_frame": end,
            }
            try:
                features = PF.features_for_bin(clip, start, end, fps=source_fps)
                regime = PF.contact_regime_proxy(features)
                extras = {
                    **features.as_proxy_features(),
                    **extras,
                }
            except ValueError:
                regime = "unknown"

        candidates.append(
            PM.ContextCandidate(
                context=ContextKey.from_dict(entry),
                failure_rate=float(entry.get("failure_rate", 0.0)),
                sampling_probability=float(entry.get("sampling_probability", 0.0)),
                family=record.get("family", "other"),
                contact_regime=regime,
                partition=partition,
                extras=extras,
            )
        )
    return candidates, skipped_partition, skipped_missing


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(payload, indent=2) + "\n")
    staging.replace(path)


def _default_receipt_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.creation_receipt.json")


def main(argv=None) -> int:
    args = parse_args(argv)
    claim_grade = bool(args.origin_map)
    receipt_path = args.receipt or _default_receipt_path(args.output)
    if receipt_path.resolve() == args.output.resolve():
        raise SystemExit("manifest and creation receipt must use distinct paths")
    existing = [path for path in (args.output, receipt_path) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"frozen output exists: {', '.join(str(path) for path in existing)}; "
            "pass --overwrite only if every downstream branch will be invalidated"
        )
    source_git_status = git_status()
    if claim_grade and source_git_status:
        raise SystemExit(
            "claim-grade manifest creation requires a clean committed tree; "
            f"git status has {len(source_git_status)} entries"
        )
    source_git_sha = git_sha()

    pool, pool_sha256, pool_file_sha256 = verify_pool_manifest(args.pool_manifest)
    pool_source_binding = verify_pool_source_bytes(pool) if claim_grade else None
    clips = load_clip_index(pool)
    split, split_sha256, split_file_sha256 = verify_split_manifest(
        args.split_manifest,
        pool_sha256=pool_sha256,
        pool_motion_keys=set(clips),
    )
    assignment = split["assignment"]

    origin_stages: dict[str, OriginStage] = {}
    stage_snapshots: dict[str, tuple[list[dict[str, Any]], list[Path]]] = {}
    if claim_grade:
        references = parse_origin_map_references(args.origin_map)
        origin_stages = load_origin_stages(
            references,
            seeds=args.seeds,
            clips=clips,
            pool_sha256=pool_sha256,
            split_sha256=split_sha256,
            snapshot_timeline_fps=args.snapshot_timeline_fps,
        )
        for stage, origin in origin_stages.items():
            stage_snapshots[stage] = (
                [source.payload for source in origin.snapshots],
                [source.path for source in origin.snapshots],
            )
    else:
        grouped_paths = group_snapshot_paths(args.snapshot)
        for stage, paths in grouped_paths.items():
            if len(paths) > 1 and not args.intersect_origin_snapshots:
                raise SystemExit(
                    f"stage {stage!r} has {len(paths)} loose snapshots; pass "
                    "--intersect-origin-snapshots so exploratory coverage is explicit"
                )
            snapshots = [_load_json(path, f"exploratory snapshot {stage!r}") for path in paths]
            for index, snapshot in enumerate(snapshots):
                _validated_snapshot_rows(snapshot, clips, f"exploratory snapshot {stage!r}/{index}")
            stage_snapshots[stage] = (snapshots, paths)

    candidates_per_stage: dict[str, list[PM.ContextCandidate]] = {}
    selection_counts: dict[str, dict[str, Any]] = {}
    for stage, (source_snapshots, paths) in stage_snapshots.items():
        snapshot = (
            intersect_snapshots(source_snapshots)
            if len(source_snapshots) > 1
            else source_snapshots[0]
        )
        candidates, skipped_partition, skipped_missing = build_candidates(
            snapshot,
            clips,
            assignment,
            args.train_partition,
            not args.skip_motion_features,
            args.snapshot_timeline_fps,
        )
        print(
            f"stage {stage!r}: {len(snapshot['contexts'])} common resident contexts "
            f"across {len(paths)} origin(s) -> {len(candidates)} candidates "
            f"({skipped_partition} outside {args.train_partition}, "
            f"{skipped_missing} not in pool)"
        )
        if not candidates:
            raise SystemExit(
                f"stage {stage!r} has no candidate contexts in the "
                f"{args.train_partition!r} partition; check origins and split"
            )
        regimes: dict[str, int] = {}
        for candidate in candidates:
            regimes[candidate.contact_regime] = regimes.get(candidate.contact_regime, 0) + 1
        print(f"           contact regimes: {dict(sorted(regimes.items()))}")
        candidates_per_stage[stage] = candidates
        selection_counts[stage] = {
            "source_origin_count": len(paths),
            "common_resident_contexts": len(snapshot["contexts"]),
            "partition_candidates": len(candidates),
            "skipped_outside_partition": skipped_partition,
            "skipped_missing_from_pool": skipped_missing,
        }

    horizons = dict(zip(("H_s", "H_m", "H_l"), args.horizons))
    notes = (
        "claim-grade: built outcome-blind from hash-verified origin-map snapshot intersections"
        if claim_grade
        else "EXPLORATORY: built from loose sampler snapshots"
    )
    manifest = PM.build_probe_manifest(
        campaign_id=args.campaign_id,
        candidates_per_stage=candidates_per_stage,
        num_contexts=args.contexts_per_stage,
        seeds=args.seeds,
        horizons=horizons,
        pool_sha256=pool_sha256,
        split_sha256=split_sha256,
        epsilon=args.epsilon,
        kernel_radius_bins=args.kernel_radius,
        selection_seed=args.selection_seed,
        notes=notes,
    )

    if pool_source_binding is not None:
        reverify_pool_source_bytes(pool_source_binding)
    manifest_payload = manifest.to_dict()
    _atomic_write_json(args.output, manifest_payload)
    manifest_file_sha256 = file_sha256(args.output)
    for stage, selected in manifest.contexts_per_stage.items():
        selection_counts[stage]["selected_contexts"] = len(selected)

    origin_receipt = {
        stage: {
            "path": str(origin.map_path.resolve()),
            "file_sha256": origin.map_file_sha256,
            "payload_sha256": origin.map_payload_sha256,
            "origin_step": origin.origin_step,
            "common_context_count": len(origin.common_context_ids),
            "snapshots": {
                str(source.seed): {
                    "path": str(source.path.resolve()),
                    "file_sha256": source.file_sha256,
                    "global_step": source.payload.get("global_step"),
                    "context_count": len(source.payload.get("contexts", [])),
                }
                for source in origin.snapshots
            },
        }
        for stage, origin in sorted(origin_stages.items())
    }
    receipt = {
        "kind": "practice_utility_probe_manifest_creation",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "campaign_id": manifest.campaign_id,
        "claim_grade_inputs": claim_grade,
        "git_sha": source_git_sha,
        "git_status_short": source_git_status,
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": file_sha256(Path(__file__)),
        "manifest": {
            "path": str(args.output.resolve()),
            "manifest_sha256": manifest.manifest_sha256,
            "file_sha256": manifest_file_sha256,
        },
        "inputs": {
            "pool_manifest": {
                "path": str(args.pool_manifest.resolve()),
                "file_sha256": pool_file_sha256,
                "pool_sha256": pool_sha256,
                "source_bytes": (
                    {
                        "source_root": pool_source_binding["source_root"],
                        "source_file_count": pool_source_binding["source_file_count"],
                        "source_files_sha256": pool_source_binding["source_files_sha256"],
                    }
                    if pool_source_binding is not None
                    else None
                ),
            },
            "split_manifest": {
                "path": str(args.split_manifest.resolve()),
                "file_sha256": split_file_sha256,
                "split_sha256": split_sha256,
                "pool_sha256": split["pool_sha256"],
            },
            "origin_maps": origin_receipt,
            "exploratory_snapshots": (
                {
                    stage: [
                        {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
                        for path in paths
                    ]
                    for stage, (_, paths) in sorted(stage_snapshots.items())
                }
                if not claim_grade
                else {}
            ),
        },
        "selection": {
            "selection_seed": args.selection_seed,
            "requested_contexts_per_stage": args.contexts_per_stage,
            "seeds": args.seeds,
            "train_partition": args.train_partition,
            "snapshot_timeline_fps": args.snapshot_timeline_fps,
            "motion_feature_bounds": "floor(start*source_fps/timeline_fps), "
            "ceil(end*source_fps/timeline_fps)",
            "counts_per_stage": selection_counts,
        },
        "verified": (
            [
                "origin-map byte hashes match the command-line references",
                "one distinct hash-verified snapshot exists per declared stage and seed",
                "origin-map common contexts equal the source-snapshot intersections",
                "pool and split logical hashes recompute and match",
                "pool source bytes and record paths rescan to the frozen logical pool",
                "the Git worktree was clean before claim-grade construction",
                "every snapshot ContextKey motion hash matches the frozen pool",
                "selection is outcome-blind and motion features use time-aligned source bounds",
            ]
            if claim_grade
            else ["pool and split logical hashes recompute and match"]
        ),
        "not_yet_verified": (
            [
                "counterfactual utility labels exceed the same-estimand noise floor",
                "frozen latent-gap features predict those labels",
            ]
            if claim_grade
            else ["exploratory loose snapshots are not claim-grade origins"]
        ),
    }
    _atomic_write_json(receipt_path, receipt)

    print(f"\ncampaign {manifest.campaign_id} [{manifest.manifest_sha256[:16]}]")
    print(
        f"  {manifest.num_branches} intervention branches + "
        f"{manifest.num_control_branches} shared controls"
    )
    for stage, coverage in manifest.coverage().items():
        print(
            f"  {stage:8s} quartiles={coverage['failure_quartiles']} "
            f"families={len(coverage['families'])} regimes={coverage['contact_regimes']}"
        )
    print(f"wrote {args.output}")
    print(f"receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
