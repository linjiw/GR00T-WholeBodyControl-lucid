#!/usr/bin/env python3
"""Create manifest-aligned, fully resumable origins for a probe campaign.

The historical ``probe_screen_v1_late`` manifest was selected from one live
resident sampler, but its stage capsule did not preserve enough trainer state
to seed the symmetric-restart estimand later validated at L0.  A different
full capsule cannot be substituted: resident motions are chosen while the
environment is constructed and are not serialized in a capsule.

This driver produces one full capsule, SONIC checkpoint, and sampler snapshot
*at the same global step* for every requested seed.  A later manifest must be
selected from the intersection of those snapshots.  The driver does not build
that manifest and does not inspect any utility outcome.

Dry-run is the default.  ``--execute`` performs the bounded continuations and
writes a receipt plus an origin-map JSON.
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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC
from gear_sonic.research.practice_utility import motion_pool as MP
from gear_sonic.research.practice_utility import run_log as RL
from scripts.practice_utility import run_latency_ab as LA
from scripts.practice_utility import run_throughput_probe as TP
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

CONTEXT_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeContextCallback"
CAPSULE_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"
TRAILING_WINDOW = 4
# Two-sigma versions of the settled-origin one-sigma reward/length floors.
REWARD_STABILITY_LIMIT = 2.0 * 0.0333
LENGTH_STABILITY_LIMIT = 2.0 * 0.0314
MIN_COMMON_CONTEXTS = 24
DEFAULT_POOL_MANIFEST = LUCID_ROOT / "manifests/pool_debug512.json"
DEFAULT_DEV_SUITE = LUCID_ROOT / "manifests/split_debug512_performer.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", default="late")
    parser.add_argument("--seeds", type=int, nargs="+", default=[9300, 9301])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--settle-iterations", type=int, default=12)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--motion-file", default="data/motion_lib_bones_seed/robot_filtered")
    parser.add_argument("--smpl-motion-file", default="data/motion_lib_bones_seed/smpl_filtered")
    parser.add_argument("--snapshot-timeline-fps", type=float, default=50.0)
    parser.add_argument("--pool-manifest", type=Path, default=DEFAULT_POOL_MANIFEST)
    parser.add_argument(
        "--dev-suite-manifest",
        type=Path,
        default=DEFAULT_DEV_SUITE,
        help="frozen split whose dev partition defines the deployment panel",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/probe_origins",
    )
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument(
        "--receipt-dir", type=Path, default=LUCID_ROOT / "manifests"
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.settle_iterations < 2 * TRAILING_WINDOW:
        parser.error(f"--settle-iterations must be at least {2 * TRAILING_WINDOW}")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if args.snapshot_timeline_fps <= 0:
        parser.error("--snapshot-timeline-fps must be positive")
    if not float(args.snapshot_timeline_fps).is_integer():
        parser.error("--snapshot-timeline-fps must be an integer-valued motion-lib target_fps")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Hash a JSON-normalized launch contract."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolved_directory(path: str | Path, *, label: str) -> Path:
    """Resolve one launch directory relative to the repository and require it exists."""
    requested = Path(path)
    candidate = requested if requested.is_absolute() else REPO / requested
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def directory_tree_binding(path: str | Path, *, label: str) -> dict[str, Any]:
    """Hash every regular file by relative name and bytes under a resolved directory."""
    resolved = _resolved_directory(path, label=label)
    files = sorted(
        (entry for entry in resolved.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(resolved).as_posix(),
    )
    if not files:
        raise ValueError(f"{label} contains no regular files: {resolved}")
    entries = [
        {
            "relative_path": entry.relative_to(resolved).as_posix(),
            "size_bytes": entry.stat().st_size,
            "sha256": sha256(entry),
        }
        for entry in files
    ]
    return {
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "tree_sha256": canonical_sha256({"files": entries}),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
    }


def validate_motion_sources(
    motion_file: str | Path,
    smpl_motion_file: str | Path,
    pool_manifest: Path,
    pool_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Bind the launch motion trees to the exact frozen pool support.

    The robot input must resolve through any symlink to the pool manifest's
    exact ``source_root``.  The SMPL tree has no independent historical
    manifest, so it is frozen here by resolved path and full tree hash and must
    carry exactly the same motion-key filenames as the robot pool.
    """
    pool = json.loads(pool_manifest.read_text())
    source_root = pool.get("source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("motion-pool manifest has no source_root")
    expected_robot_root = _resolved_directory(source_root, label="pool source_root")
    robot = directory_tree_binding(motion_file, label="--motion-file")
    actual_robot_root = Path(robot["resolved_path"])
    if actual_robot_root != expected_robot_root:
        raise ValueError(
            "--motion-file does not resolve to the pool manifest source_root: "
            f"actual={actual_robot_root}, expected={expected_robot_root}"
        )

    smpl = directory_tree_binding(smpl_motion_file, label="--smpl-motion-file")
    smpl_root = Path(smpl["resolved_path"])
    smpl_files = sorted(smpl_root.rglob("*.pkl"))
    smpl_keys = [entry.stem for entry in smpl_files]
    if len(set(smpl_keys)) != len(smpl_keys):
        raise ValueError("--smpl-motion-file contains duplicate motion-key filenames")
    expected_keys = set(pool_hashes)
    actual_keys = set(smpl_keys)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "--smpl-motion-file does not exactly cover the frozen robot pool: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    robot.update(
        pool_manifest_source_root=source_root,
        pool_manifest_source_root_resolved=str(expected_robot_root),
        motion_pool_manifest_sha256=pool.get("pool_sha256"),
    )
    smpl["paired_motion_keys_sha256"] = canonical_sha256({"motion_keys": sorted(actual_keys)})
    return {"robot": robot, "smpl": smpl}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a receipt atomically so interruption cannot leave valid-looking fragments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(payload, indent=2) + "\n")
    staging.replace(path)


def logical_manifest_hash(path: Path, *, kind: str, field: str) -> tuple[str, dict[str, Any]]:
    """Load one frozen manifest and return its claim-bearing logical hash."""
    payload = json.loads(path.read_text())
    if payload.get("kind") != kind:
        raise ValueError(f"{path} kind is {payload.get('kind')!r}, expected {kind!r}")
    value = payload.get(field)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} has no valid {field}")
    return value, payload


def validate_claim_inputs(pool_path: Path, dev_suite_path: Path) -> tuple[str, str, dict[str, str]]:
    """Validate logical manifests and the motion bytes they bind."""
    pool_sha256, pool = logical_manifest_hash(
        pool_path, kind="practice_utility_motion_pool", field="pool_sha256"
    )
    dev_suite_sha256, dev_suite = logical_manifest_hash(
        dev_suite_path,
        kind="practice_utility_group_disjoint_split",
        field="split_sha256",
    )
    recomputed_pool_sha256 = canonical_sha256(
        {
            "source_root": pool.get("source_root"),
            "records": [
                {
                    "motion_key": row.get("motion_key"),
                    "content_sha256": row.get("content_sha256"),
                }
                for row in sorted(
                    pool.get("motions") or [], key=lambda row: row.get("motion_key", "")
                )
            ],
        }
    )
    if recomputed_pool_sha256 != pool_sha256:
        raise ValueError("motion-pool logical hash does not match its serialized records")
    recomputed_split_sha256 = canonical_sha256(
        {
            "assignment": dict(sorted((dev_suite.get("assignment") or {}).items())),
            "linkage": dev_suite.get("linkage"),
            "seed": dev_suite.get("seed"),
            "pool_sha256": dev_suite.get("pool_sha256"),
        }
    )
    if recomputed_split_sha256 != dev_suite_sha256:
        raise ValueError("dev-suite logical hash does not match its serialized assignment")
    if dev_suite.get("pool_sha256") != pool_sha256:
        raise ValueError("dev-suite split is not linked to the selected motion pool")

    scan = MP.scan_pool(pool["source_root"])
    if pool.get("deduplicated"):
        scan = MP.drop_exact_duplicates(scan)
    if MP.pool_sha256(scan) != pool_sha256:
        raise ValueError("motion-pool source bytes differ from the frozen pool manifest")
    pool_hashes = {str(row["motion_key"]): str(row["content_sha256"]) for row in pool["motions"]}
    if len(pool_hashes) != len(pool["motions"]):
        raise ValueError("motion-pool manifest contains duplicate motion keys")
    return pool_sha256, dev_suite_sha256, pool_hashes


def origin_provenance(
    args: argparse.Namespace,
    *,
    seed: int,
    start_step: int,
    source_commit: str,
    pool_sha256: str,
    dev_suite_sha256: str,
    motion_sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind a capsule to every claim-bearing launch input.

    Hydra cannot hash its own resolved configuration without including the hash
    field recursively.  ``resolved_config_sha256`` therefore hashes this
    explicitly versioned, provenance-redacted launch contract.  The receipt
    records that basis alongside the digest.
    """
    contract = {
        "kind": "practice_utility_origin_launch_contract",
        "schema_version": 2,
        "stage": args.stage,
        "seed": seed,
        "num_envs": args.num_envs,
        "source_step": start_step,
        "settle_iterations": args.settle_iterations,
        "target_step": start_step + args.settle_iterations,
        "experiment_config": args.exp,
        "motion_file": args.motion_file,
        "smpl_motion_file": args.smpl_motion_file,
        "motion_sources": motion_sources,
        "motion_lib_target_fps": args.snapshot_timeline_fps,
        "snapshot_timeline_fps": args.snapshot_timeline_fps,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "motion_pool_manifest_sha256": pool_sha256,
        "dev_suite_sha256": dev_suite_sha256,
        "source_commit": source_commit,
        "callbacks": {
            "context": CONTEXT_CALLBACK,
            "capsule": CAPSULE_CALLBACK,
        },
    }
    provenance = {
        "resolved_config_sha256": canonical_sha256(contract),
        "motion_pool_manifest_sha256": pool_sha256,
        "dev_suite_sha256": dev_suite_sha256,
        "source_commit": source_commit,
        "checkpoint_sha256": sha256(args.checkpoint),
    }
    return provenance, contract


def checkpoint_global_step(path: Path) -> int:
    """Read the absolute trainer step carried by a SONIC checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state")
    step = (
        state.get("global_step") if isinstance(state, dict) else getattr(state, "global_step", None)
    )
    if step is None:
        practice = payload.get("practice_utility") or {}
        step = practice.get("global_step")
    if step is None:
        raise ValueError(f"checkpoint {path} does not record a global step")
    return int(step)


def validate_exported_checkpoint(
    path: Path,
    *,
    capsule: dict[str, Any],
    target_step: int,
) -> list[str]:
    """Verify the exported checkpoint points back to the exact full capsule."""
    blockers: list[str] = []
    if not path.is_file():
        return ["exported SONIC checkpoint missing"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        step = checkpoint_global_step(path)
    except (KeyError, TypeError, ValueError) as error:
        blockers.append(f"exported checkpoint has no valid trainer step: {error}")
    else:
        if step != target_step:
            blockers.append(f"exported checkpoint step {step} differs from target {target_step}")
    link = payload.get("practice_utility")
    if not isinstance(link, dict):
        blockers.append("exported checkpoint has no capsule provenance link")
    else:
        if link.get("capsule_sha256") != capsule.get("capsule_sha256"):
            blockers.append("exported checkpoint logical capsule hash mismatch")
        if link.get("global_step") != target_step:
            blockers.append("exported checkpoint capsule link has the wrong step")
    for key in (
        "policy_state_dict",
        "value_state_dict",
        "optimizer_state_dict",
        "state",
        "env_state_dict",
    ):
        if payload.get(key) is None:
            blockers.append(f"exported checkpoint is missing {key}")
    return blockers


def _scalar_at(value: Any, index: int) -> float:
    selected = value[index]
    return float(selected.item() if hasattr(selected, "item") else selected)


def canonicalize_snapshot_contexts(
    snapshot: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Collapse identical with-replacement rows without merging sampler statistics.

    SONIC's active-bin draw is with replacement, so one ``ContextKey`` may occupy
    several resident slots.  Those copies are valid only when their complete
    serialized rows agree.  The slot probabilities add, but the global sampler
    counters describe the underlying bin and therefore remain a single copy.
    """
    blockers: list[str] = []
    rows = snapshot.get("contexts")
    if not isinstance(rows, list) or not rows:
        return {}, ["snapshot contains no resident contexts"]

    raw_count = len(rows)
    declared_count = snapshot.get("num_active_bins")
    if (
        not isinstance(declared_count, int)
        or isinstance(declared_count, bool)
        or declared_count != raw_count
    ):
        blockers.append(
            "snapshot num_active_bins does not match the raw serialized context row count"
        )

    canonical: dict[str, dict[str, Any]] = {}
    exemplars: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(rows):
        if not isinstance(raw, dict):
            blockers.append(f"snapshot context row {position} is not an object")
            continue
        context_id = raw.get("context_id")
        if not isinstance(context_id, str) or not context_id:
            blockers.append(f"snapshot context row {position} has no context_id")
            continue
        probability = raw.get("sampling_probability")
        if isinstance(probability, bool):
            blockers.append(f"snapshot context {context_id} has invalid sampling_probability")
            continue
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            blockers.append(f"snapshot context {context_id} has invalid sampling_probability")
            continue
        if not math.isfinite(probability) or probability < 0.0:
            blockers.append(f"snapshot context {context_id} has invalid sampling_probability")
            continue

        if context_id in canonical:
            if raw != exemplars[context_id]:
                blockers.append(
                    f"snapshot duplicate context {context_id} has conflicting serialized rows"
                )
                continue
            row = canonical[context_id]
            row["sampling_probability"] = math.fsum(
                (float(row["sampling_probability"]), probability)
            )
            row["resident_multiplicity"] = int(row["resident_multiplicity"]) + 1
            continue

        exemplars[context_id] = dict(raw)
        row = dict(raw)
        row["sampling_probability"] = probability
        row["resident_multiplicity"] = 1
        canonical[context_id] = row
    return canonical, blockers


def validate_snapshot_against_capsule_and_pool(
    snapshot: dict[str, Any],
    capsule: dict[str, Any],
    pool_hashes: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Check that the snapshot names the capsule's counters and frozen pool exactly."""
    rows, blockers = canonicalize_snapshot_contexts(snapshot)
    sampler = capsule.get("native_sampler_state")
    if not isinstance(sampler, dict):
        return rows, blockers + ["capsule has no native sampler state"]
    episodes = sampler.get("adp_samp_num_episodes")
    failures = sampler.get("adp_samp_num_failures")
    if episodes is None or failures is None:
        return rows, blockers + ["capsule sampler state has no episode/failure counters"]

    for row in rows.values():
        motion_key = row.get("motion_key")
        expected_hash = pool_hashes.get(str(motion_key))
        if expected_hash is None:
            blockers.append(f"snapshot motion {motion_key!r} is absent from the frozen pool")
        elif row.get("motion_hash") != expected_hash:
            blockers.append(f"snapshot motion hash differs from frozen pool for {motion_key!r}")
        try:
            global_bin = int(row["global_bin_id"])
            expected_episodes = _scalar_at(episodes, global_bin)
            expected_failures = _scalar_at(failures, global_bin)
            actual_episodes = float(row["num_episodes"])
            actual_failures = float(row["num_failures"])
        except (IndexError, KeyError, TypeError, ValueError) as error:
            blockers.append(f"snapshot sampler counter row is invalid: {error}")
            continue
        if not math.isclose(actual_episodes, expected_episodes, rel_tol=0.0, abs_tol=0.0):
            blockers.append(f"snapshot/capsule episode counter mismatch at bin {global_bin}")
        if not math.isclose(actual_failures, expected_failures, rel_tol=0.0, abs_tol=0.0):
            blockers.append(f"snapshot/capsule failure counter mismatch at bin {global_bin}")
    return rows, blockers


def build_command(
    args: argparse.Namespace,
    *,
    seed: int,
    start_step: int,
    experiment_id: str,
    provenance: dict[str, str],
) -> tuple[list[str], dict[str, Path]]:
    target_step = start_step + args.settle_iterations
    branch_id = f"{experiment_id}_s{seed}"
    output_dir = args.artifact_root / experiment_id / f"seed_{seed}"
    paths = {
        "output_dir": output_dir,
        "snapshot": output_dir / "origin_snapshot.json",
        "capsule": output_dir / "capsules" / f"{branch_id}_origin.capsule.pt",
        "checkpoint": output_dir / "origin_checkpoint.pt",
    }
    command = [
        sys.executable,
        str(REPO / "gear_sonic" / "train_agent_trl.py"),
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        "+resume=true",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={seed}",
        f"++algo.config.num_learning_iterations={target_step}",
        "++algo.config.save_interval=100000",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
        "++manager_env.commands.motion.motion_lib_cfg.target_fps="
        f"{args.snapshot_timeline_fps:g}",
        f"++callbacks.practice_context._target_={CONTEXT_CALLBACK}",
        "++callbacks.practice_context.enabled=true",
        "++callbacks.practice_context.role=control",
        f"++callbacks.practice_context.pair_id={branch_id}",
        f"++callbacks.practice_context.branch_id={branch_id}",
        f"++callbacks.practice_context.dose_report_dir={output_dir}",
        f"++callbacks.practice_context.snapshot_path={paths['snapshot']}",
        f"++callbacks.practice_context.snapshot_at_step={target_step}",
        f"++callbacks.practice_context.snapshot_timeline_fps={args.snapshot_timeline_fps}",
        f"++callbacks.practice_context.manifest_path={args.pool_manifest}",
        f"++callbacks.practice_capsule._target_={CAPSULE_CALLBACK}",
        "++callbacks.practice_capsule.enabled=true",
        f"++callbacks.practice_capsule.capsule_dir={output_dir / 'capsules'}",
        f"++callbacks.practice_capsule.pair_id={branch_id}",
        "++callbacks.practice_capsule.role=control",
        f"++callbacks.practice_capsule.branch_id={branch_id}",
        f"++callbacks.practice_capsule.horizons.origin={target_step}",
        "++callbacks.practice_capsule.provenance={"
        + ",".join(f"{key}:{value}" for key, value in sorted(provenance.items()))
        + "}",
    ]
    return command, paths


def trailing_stability(run: RL.RunLog, metric: str) -> dict[str, float | bool | None]:
    """Compare the last four iterations with the preceding four."""
    values = [value for _, value in sorted(run.series(metric).items())]
    if len(values) < 2 * TRAILING_WINDOW:
        return {"previous4": None, "last4": None, "relative_delta": None, "passes": False}
    previous = statistics.fmean(values[-2 * TRAILING_WINDOW : -TRAILING_WINDOW])
    latest = statistics.fmean(values[-TRAILING_WINDOW:])
    relative = None if previous == 0 else (latest - previous) / abs(previous)
    limit = REWARD_STABILITY_LIMIT if metric == "Mean rewards" else LENGTH_STABILITY_LIMIT
    return {
        "previous4": previous,
        "last4": latest,
        "relative_delta": relative,
        "absolute_relative_limit": limit,
        "passes": relative is not None and abs(relative) <= limit,
    }


def validate_origin(
    *,
    paths: dict[str, Path],
    log_path: Path,
    start_step: int,
    target_step: int,
    settle_iterations: int,
    expected_provenance: dict[str, str],
    pool_hashes: dict[str, str],
    snapshot_timeline_fps: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    origin: dict[str, Any] = {
        "origin_step": target_step,
        "source_step": start_step,
        "capsule": str(paths["capsule"]),
        "snapshot": str(paths["snapshot"]),
        "checkpoint": str(paths["checkpoint"]),
    }
    if not paths["capsule"].is_file():
        blockers.append("origin capsule missing")
        return origin, blockers
    if not paths["snapshot"].is_file():
        blockers.append("origin sampler snapshot missing")
        return origin, blockers

    capsule = BC.load_capsule(
        paths["capsule"],
        expected_provenance=BC.Provenance(**expected_provenance),
        restore_rng=False,
    )
    origin["capsule_sha256"] = sha256(paths["capsule"])
    if int(capsule["global_step"]) != target_step:
        blockers.append("capsule global step differs from preregistered target")
    trainer_obj = (capsule.get("trainer_state") or {}).get("trainer_state_obj")
    if trainer_obj is None:
        blockers.append("capsule has no full trainer state")
    if not (capsule.get("optimizer_state") or {}):
        blockers.append("capsule has no optimizer state")
    if (capsule.get("rng") or {}).get("counter_rng_enabled") is not False:
        blockers.append(
            "capsule incorrectly claims production channel-keyed counter RNG; "
            "this campaign uses a measured stochastic restart estimand"
        )

    snapshot = json.loads(paths["snapshot"].read_text())
    origin["snapshot_sha256"] = sha256(paths["snapshot"])
    if int(snapshot.get("global_step", -1)) != target_step:
        blockers.append("snapshot and capsule were not captured at the same step")
    if snapshot.get("snapshot_timeline_fps") != snapshot_timeline_fps:
        blockers.append("snapshot sampler-timeline FPS differs from preregistration")
    canonical_rows, snapshot_blockers = validate_snapshot_against_capsule_and_pool(
        snapshot, capsule, pool_hashes
    )
    blockers.extend(snapshot_blockers)
    context_ids = sorted(canonical_rows)
    raw_context_count = len(snapshot.get("contexts", []))
    origin.update(
        resident_context_ids=context_ids,
        num_resident_contexts=len(context_ids),
        num_active_context_rows=raw_context_count,
        num_duplicate_active_context_rows=raw_context_count - len(context_ids),
    )

    checkpoint_blockers = validate_exported_checkpoint(
        paths["checkpoint"], capsule=capsule, target_step=target_step
    )
    blockers.extend(checkpoint_blockers)
    if paths["checkpoint"].is_file():
        origin["checkpoint_sha256"] = sha256(paths["checkpoint"])

    run = RL.parse_run_log(log_path)
    actual_indices = [iteration.index for iteration in run.iterations]
    expected_indices = list(range(start_step + 1, target_step + 1))
    if actual_indices != expected_indices:
        blockers.append(
            "parsed continuation interval differs from preregistration: "
            f"actual={actual_indices}, expected={expected_indices}"
        )
    stability = {
        "reward": trailing_stability(run, "Mean rewards"),
        "length": trailing_stability(run, "Mean length"),
    }
    if not all(item["passes"] for item in stability.values()):
        blockers.append("origin did not meet the frozen operational stability rule")

    origin.update(
        stability=stability,
        settled=all(item["passes"] for item in stability.values()),
        continuation_iteration_indices=actual_indices,
    )
    return origin, blockers


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.pool_manifest.is_file():
        raise FileNotFoundError(args.pool_manifest)
    if not args.dev_suite_manifest.is_file():
        raise FileNotFoundError(args.dev_suite_manifest)
    pool_sha256, dev_suite_sha256, pool_hashes = validate_claim_inputs(
        args.pool_manifest, args.dev_suite_manifest
    )
    motion_sources = validate_motion_sources(
        args.motion_file,
        args.smpl_motion_file,
        args.pool_manifest,
        pool_hashes,
    )
    start_step = checkpoint_global_step(args.checkpoint)
    target_step = start_step + args.settle_iterations
    source_commit = TP.git_sha()
    git_status = TP.git_status()
    created_at = datetime.now().astimezone()
    stamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
    experiment_id = f"probe_origins_ne{args.num_envs}_{stamp}"
    provenance_and_contracts = {
        seed: origin_provenance(
            args,
            seed=seed,
            start_step=start_step,
            source_commit=source_commit,
            pool_sha256=pool_sha256,
            dev_suite_sha256=dev_suite_sha256,
            motion_sources=motion_sources,
        )
        for seed in args.seeds
    }
    planned = {
        seed: build_command(
            args,
            seed=seed,
            start_step=start_step,
            experiment_id=experiment_id,
            provenance=provenance_and_contracts[seed][0],
        )
        for seed in args.seeds
    }
    for seed, (command, _) in planned.items():
        print(f"[seed {seed}]\n" + "\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    preregistration_path = args.receipt_dir / f"{experiment_id}_preregistration.json"
    origin_map_path = args.receipt_dir / f"{experiment_id}_origin_map.json"
    commands = {str(seed): command for seed, (command, _) in planned.items()}
    runtime: dict[str, Any] = {}
    origins: dict[str, Any] = {}
    blockers: list[str] = []
    receipt = {
        "kind": "practice_utility_probe_origin_creation",
        "schema_version": 1,
        "status": "preregistered",
        "created_at": created_at.isoformat(),
        "experiment_id": experiment_id,
        "git_sha": source_commit,
        "git_status_short": git_status,
        "launcher_sha256": sha256(Path(__file__)),
        "preregistered_before_run": {
            "source_step": start_step,
            "settle_iterations": args.settle_iterations,
            "target_step": target_step,
            "trailing_window": TRAILING_WINDOW,
            "reward_absolute_relative_delta_lte": REWARD_STABILITY_LIMIT,
            "length_absolute_relative_delta_lte": LENGTH_STABILITY_LIMIT,
            "stability_interpretation": (
                "operational last-4-versus-previous-4 drift rule; not a stationarity test"
            ),
            "manifest_rule": "select only from the intersection of all origin snapshots",
            "minimum_common_contexts": MIN_COMMON_CONTEXTS,
            "snapshot_timeline_fps": args.snapshot_timeline_fps,
            "randomness_contract": "stochastic_potential_outcomes_no_channelwise_crn",
            "required_iteration_indices": list(range(start_step + 1, target_step + 1)),
        },
        "config": {
            "stage": args.stage,
            "num_envs": args.num_envs,
            "seeds": args.seeds,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "pool_manifest": str(args.pool_manifest.resolve()),
            "pool_manifest_file_sha256": sha256(args.pool_manifest),
            "motion_pool_manifest_sha256": pool_sha256,
            "motion_sources": motion_sources,
            "motion_lib_target_fps": args.snapshot_timeline_fps,
            "dev_suite_manifest": str(args.dev_suite_manifest.resolve()),
            "dev_suite_manifest_file_sha256": sha256(args.dev_suite_manifest),
            "dev_suite_sha256": dev_suite_sha256,
            "source_commit": source_commit,
            "resolved_config_sha256_basis": (
                "canonical practice_utility_origin_launch_contract v2 with provenance field redacted"
            ),
            "launch_contracts": {
                str(seed): contract for seed, (_, contract) in provenance_and_contracts.items()
            },
            "launch_contract_sha256": {
                str(seed): provenance["resolved_config_sha256"]
                for seed, (provenance, _) in provenance_and_contracts.items()
            },
        },
        "commands": commands,
        "runtime": runtime,
        "origin_map": str(origin_map_path),
        "origins_completed": origins,
        "verified": [],
        "not_yet_verified": [
            "bounded origin continuations have not started",
            "a new probe manifest selected from the common resident contexts",
            "outcome-blind per-context frozen latent-gap features",
            "same-estimand Gate A noise floor and deployment J_eff evaluation",
        ],
    }
    preregistration = {
        "kind": "practice_utility_probe_origin_preregistration",
        "schema_version": 1,
        "frozen": True,
        "created_at": created_at.isoformat(),
        "experiment_id": experiment_id,
        "git_sha": source_commit,
        "git_status_short": git_status,
        "launcher_sha256": receipt["launcher_sha256"],
        "preregistered_before_run": receipt["preregistered_before_run"],
        "config": receipt["config"],
        "commands": commands,
    }
    # This immutable sidecar precedes the first capacity query or GPU
    # subprocess. The mutable execution receipt hash-binds it thereafter.
    atomic_write_json(preregistration_path, preregistration)
    receipt.update(
        preregistration=str(preregistration_path),
        preregistration_sha256=sha256(preregistration_path),
    )
    atomic_write_json(receipt_path, receipt)

    if git_status:
        blockers.append(
            "claim-grade origin creation requires a clean committed tree; "
            f"git status had {len(git_status)} entries"
        )
        receipt.update(status="blocked_preflight", not_yet_verified=blockers)
        atomic_write_json(receipt_path, receipt)
        print(f"receipt {receipt_path}")
        return 1

    receipt.update(
        status="running",
        execution_started_at=datetime.now().astimezone().isoformat(),
        not_yet_verified=["origin continuations are running"],
    )
    atomic_write_json(receipt_path, receipt)

    try:
        for seed, (command, paths) in planned.items():
            seed_key = str(seed)
            paths["output_dir"].mkdir(parents=True, exist_ok=False)
            log_path = args.log_dir / f"{experiment_id}_s{seed}.log"
            try:
                arm_runtime = LA.run_arm(command, log_path, args.min_free_mib)
            except Exception as error:  # capacity/subprocess setup failures still get receipts
                arm_runtime = {
                    "exit_code": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            arm_runtime["log_path"] = str(log_path)
            runtime[seed_key] = arm_runtime
            seed_blockers: list[str] = []
            if arm_runtime.get("exit_code") != 0:
                seed_blockers.append(
                    f"training subprocess exit_code={arm_runtime.get('exit_code')!r}"
                )
            if arm_runtime.get("exit_code") == 0:
                if not paths["capsule"].is_file():
                    seed_blockers.append("training completed without the preregistered capsule")
                else:
                    try:
                        BC.export_sonic_checkpoint(paths["capsule"], paths["checkpoint"])
                    except Exception as error:
                        seed_blockers.append(
                            f"SONIC checkpoint export failed: {type(error).__name__}: {error}"
                        )
            try:
                origin, validation_blockers = validate_origin(
                    paths=paths,
                    log_path=log_path,
                    start_step=start_step,
                    target_step=target_step,
                    settle_iterations=args.settle_iterations,
                    expected_provenance=provenance_and_contracts[seed][0],
                    pool_hashes=pool_hashes,
                    snapshot_timeline_fps=args.snapshot_timeline_fps,
                )
                seed_blockers.extend(validation_blockers)
            except Exception as error:
                origin = {
                    "origin_step": target_step,
                    "source_step": start_step,
                    "capsule": str(paths["capsule"]),
                    "snapshot": str(paths["snapshot"]),
                    "checkpoint": str(paths["checkpoint"]),
                }
                seed_blockers.append(f"origin validation raised {type(error).__name__}: {error}")
            origin.update(
                seed=seed,
                stage=args.stage,
                num_envs=args.num_envs,
                blockers=seed_blockers,
            )
            origins[seed_key] = origin
            blockers.extend(f"seed {seed}: {reason}" for reason in seed_blockers)
            receipt.update(runtime=runtime, origins_completed=origins)
            atomic_write_json(receipt_path, receipt)
    except BaseException as error:
        receipt.update(
            status="interrupted",
            runtime=runtime,
            origins_completed=origins,
            not_yet_verified=blockers
            + [f"execution interrupted by {type(error).__name__}: {error}"],
        )
        atomic_write_json(receipt_path, receipt)
        raise

    common_contexts = sorted(
        set.intersection(
            *(set(origin.get("resident_context_ids", [])) for origin in origins.values())
        )
        if len(origins) == len(args.seeds)
        else set()
    )
    if len(common_contexts) < MIN_COMMON_CONTEXTS:
        blockers.append(
            f"origin snapshots share {len(common_contexts)} contexts; "
            f"at least {MIN_COMMON_CONTEXTS} are required for the frozen screen size"
        )
    origin_map = {
        "kind": "practice_utility_probe_origin_map",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "stage": args.stage,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_step": start_step,
        "origin_step": target_step,
        "num_envs": args.num_envs,
        "motion_pool_manifest_sha256": pool_sha256,
        "motion_sources": motion_sources,
        "motion_lib_target_fps": args.snapshot_timeline_fps,
        "snapshot_timeline_fps": args.snapshot_timeline_fps,
        "dev_suite_sha256": dev_suite_sha256,
        "source_commit": source_commit,
        "randomness_contract": "stochastic_potential_outcomes_no_channelwise_crn",
        "preregistration": str(preregistration_path),
        "preregistration_sha256": sha256(preregistration_path),
        "seeds": args.seeds,
        "origins": origins,
        "common_resident_context_ids": common_contexts,
        "num_common_resident_contexts": len(common_contexts),
        "usable_for_manifest_selection": not blockers,
    }
    atomic_write_json(origin_map_path, origin_map)

    receipt.update(
        status="complete" if not blockers else "blocked",
        completed_at=datetime.now().astimezone().isoformat(),
        runtime=runtime,
        origins_completed=origins,
        origin_map_sha256=sha256(origin_map_path),
        verified=(
            [
                "each origin produced a full capsule and exactly linked SONIC checkpoint",
                "each pool-bound sampler snapshot matches capsule counters at the same step",
                "robot and SMPL motion trees match the preregistered resolved paths and hashes",
                "snapshot timeline FPS equals the live motion-lib target_fps forced by Hydra",
                "each log contains the exact preregistered continuation interval",
                "each origin passed the preregistered operational stability rule",
                "common resident contexts were computed without utility outcomes",
                "capsules truthfully record that channel-wise counter RNG is not integrated",
            ]
            if not blockers
            else []
        ),
        not_yet_verified=blockers
        + [
            "a new probe manifest selected from the common resident contexts",
            "outcome-blind per-context frozen latent-gap features",
            "same-estimand Gate A noise floor and deployment J_eff evaluation",
        ],
    )
    atomic_write_json(receipt_path, receipt)
    print(f"origin map {origin_map_path}")
    print(f"receipt {receipt_path}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
