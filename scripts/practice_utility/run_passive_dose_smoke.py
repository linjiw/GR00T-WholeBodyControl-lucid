#!/usr/bin/env python3
"""Run one hash-bound, claim-mode passive-dose control continuation.

This is a deliberately narrow bridge between the CPU passive-dose contracts
and the full probe campaign.  It resumes one settled origin, runs only the
manifest's short horizon, and verifies the exact shared-control dose receipt.
It does not launch an intervention, evaluate utility, or assemble labels.

The default is a read-only dry run.  ``--execute`` requires a clean committed
tree, writes an immutable preregistration before the first GPU query, passes a
strict idle-GPU gate, and then launches exactly one continuation.
"""

# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for path-bootstrap imports.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC
from gear_sonic.research.practice_utility import dose_plan as DP
from gear_sonic.research.practice_utility.schema import ContextKey, sha256_of
from scripts.practice_utility import create_probe_origins as CPO
from scripts.practice_utility import run_throughput_probe as TP

CONTEXT_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeContextCallback"
RESUME_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleResumeCallback"
SMOKE_KIND = "practice_utility_live_passive_dose_smoke"
DEFAULT_EXP = "manager/universal_token/all_modes/sonic_release"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GPU_GROWTH_MIB = 512.0


class SmokeError(RuntimeError):
    """A fail-closed smoke validation or execution error."""


class GpuNotIdleError(SmokeError):
    """The preregistered idle-GPU gate did not pass."""

    def __init__(self, message: str, samples: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.samples = list(samples or [])


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.path.stat().st_size,
        }


@dataclass(frozen=True)
class PreparedSmoke:
    """Fully validated, outcome-blind launch contract."""

    run_id: str
    stage: str
    seed: int
    manifest: dict[str, Any]
    manifest_asset: Asset
    origin_map: dict[str, Any]
    origin_map_asset: Asset
    dose_plan: DP.PassiveDosePlan
    origin: dict[str, Any]
    capsule_asset: Asset
    checkpoint_asset: Asset
    snapshot_asset: Asset
    origin_step: int
    short_horizon: int
    target_step: int
    num_envs: int
    num_steps_per_env: int
    source_config_asset: Asset
    launcher_asset: Asset
    callback_asset: Asset
    implementation_assets: dict[str, Asset]
    motion_source_bindings: dict[str, dict[str, Any]]
    source_commit: str
    source_tree_status: tuple[str, ...]
    branch_id: str
    pair_id: str
    artifact_dir: Path
    runtime_checkpoint_path: Path
    dose_dir: Path
    log_path: Path
    preregistration_path: Path
    status_path: Path
    smoke_path: Path
    dose_receipt_path: Path
    command: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--origin-map", type=Path, required=True)
    parser.add_argument("--origin-map-sha256", required=True)
    parser.add_argument("--dose-plan", type=Path, required=True)
    parser.add_argument("--dose-plan-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp", default=DEFAULT_EXP)
    parser.add_argument(
        "--run-id",
        help="collision-exclusive identifier (default includes stage, seed, and UTC time)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/artifacts/passive_dose_smoke"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/outputs"),
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/manifests"),
    )
    parser.add_argument(
        "--gpu-lock",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/.passive_dose_smoke.gpu.lock"),
    )
    parser.add_argument("--min-free-mib", type=int, default=28000)
    parser.add_argument("--max-gpu-util-pct", type=float, default=5.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-sample-seconds", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for name in ("manifest_sha256", "origin_map_sha256", "dose_plan_sha256"):
        if not SHA256.fullmatch(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    if args.run_id is not None and not RUN_ID.fullmatch(args.run_id):
        parser.error("--run-id must contain 1-96 filename-safe characters")
    if args.min_free_mib <= 0 or args.max_gpu_util_pct < 0:
        parser.error("GPU thresholds must be non-negative and min-free-mib positive")
    if args.idle_samples < 1 or args.idle_sample_seconds < 0:
        parser.error("idle sampling requires at least one sample and non-negative spacing")
    return args


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id(stage: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"passive_dose_smoke_{stage}_s{seed}_{stamp}"


def read_json_asset(path: Path, expected_sha256: str, label: str) -> tuple[Asset, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SmokeError(f"{label} is not a file: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise SmokeError(f"{label} file hash mismatch: expected {expected_sha256}, got {actual}")
    try:
        payload = json.loads(resolved.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise SmokeError(f"{label} must contain a JSON object")
    return Asset(resolved, actual), payload


def asset_from_binding(path: Any, expected_sha256: Any, label: str) -> Asset:
    if not isinstance(path, str) or not SHA256.fullmatch(str(expected_sha256)):
        raise SmokeError(f"{label} path/hash binding is malformed")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise SmokeError(f"{label} is not a file: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise SmokeError(f"{label} file hash mismatch: expected {expected_sha256}, got {actual}")
    return Asset(resolved, actual)


def validate_motion_source_bindings(origin_map: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Rehash the exact robot/SMPL trees frozen by the settled-origin receipt."""

    sources = origin_map.get("motion_sources")
    if not isinstance(sources, Mapping):
        raise SmokeError("origin map has no frozen motion-source bindings")
    verified: dict[str, dict[str, Any]] = {}
    required = ("requested_path", "resolved_path", "tree_sha256", "file_count", "total_bytes")
    for name in ("robot", "smpl"):
        expected = sources.get(name)
        if not isinstance(expected, Mapping):
            raise SmokeError(f"origin map has no {name} motion-source binding")
        requested_path = expected.get("requested_path")
        if not isinstance(requested_path, str) or not requested_path:
            raise SmokeError(f"origin map {name} motion source has no requested_path")
        try:
            actual = CPO.directory_tree_binding(
                requested_path, label=f"origin-map {name} motion source"
            )
        except (OSError, ValueError) as error:
            raise SmokeError(f"cannot verify {name} motion source: {error}") from error
        mismatched = [key for key in required if expected.get(key) != actual.get(key)]
        if mismatched:
            raise SmokeError(
                f"{name} motion source differs from the settled origin on {mismatched}"
            )
        verified[name] = actual
    return verified


def source_config(exp: str) -> tuple[Asset, int]:
    base = (REPO / "gear_sonic" / "config" / "exp").resolve()
    path = (base / f"{exp}.yaml").resolve(strict=True)
    if path != base and base not in path.parents:
        raise SmokeError("experiment config resolves outside gear_sonic/config/exp")
    payload = yaml.safe_load(path.read_text())
    try:
        value = payload["algo"]["config"]["num_steps_per_env"]
    except (KeyError, TypeError) as error:
        raise SmokeError(
            f"source config does not freeze algo.config.num_steps_per_env: {path}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SmokeError("source num_steps_per_env must be a positive integer")
    return Asset(path, file_sha256(path)), value


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    return [line for line in output.splitlines() if line]


def _validate_manifest(payload: dict[str, Any]) -> None:
    if (
        payload.get("kind") != "practice_utility_probe_manifest"
        or payload.get("schema_version") != 1
    ):
        raise SmokeError("manifest kind/schema is invalid")
    computed = DP.probe_manifest_claim_sha256(payload)
    if payload.get("manifest_sha256") != computed:
        raise SmokeError("manifest logical hash mismatch")


def _validate_origin(
    manifest: Mapping[str, Any],
    origin_map: Mapping[str, Any],
    *,
    stage: str,
    seed: int,
) -> tuple[dict[str, Any], Asset, Asset, Asset, dict[str, Any]]:
    if (
        origin_map.get("kind") != "practice_utility_probe_origin_map"
        or origin_map.get("schema_version") != 1
    ):
        raise SmokeError("origin map kind/schema is invalid")
    if origin_map.get("usable_for_manifest_selection") is not True:
        raise SmokeError("origin map is not usable for manifest selection")
    if origin_map.get("stage") != stage:
        raise SmokeError("origin map stage differs from requested stage")
    if seed not in origin_map.get("seeds", []):
        raise SmokeError("requested seed is absent from the origin map")
    origins = origin_map.get("origins")
    if not isinstance(origins, dict) or not isinstance(origins.get(str(seed)), dict):
        raise SmokeError("origin map has no requested seed record")
    origin = dict(origins[str(seed)])
    if origin.get("settled") is not True or origin.get("blockers") != []:
        raise SmokeError("selected origin is not settled and blocker-free")
    if origin.get("stage", stage) != stage or origin.get("seed", seed) != seed:
        raise SmokeError("selected origin stage/seed lineage differs")
    step = origin.get("origin_step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise SmokeError("selected origin_step must be positive")
    if step != origin_map.get("origin_step"):
        raise SmokeError("selected origin_step differs from the origin map")

    capsule = asset_from_binding(origin.get("capsule"), origin.get("capsule_sha256"), "capsule")
    checkpoint = asset_from_binding(
        origin.get("checkpoint"), origin.get("checkpoint_sha256"), "checkpoint"
    )
    snapshot = asset_from_binding(origin.get("snapshot"), origin.get("snapshot_sha256"), "snapshot")
    try:
        capsule_payload = BC.load_capsule(capsule.path, restore_rng=False)
    except (BC.CapsuleIntegrityError, OSError, RuntimeError, ValueError) as error:
        raise SmokeError(f"capsule integrity validation failed: {error}") from error
    if capsule_payload.get("global_step") != step:
        raise SmokeError("capsule global_step differs from the origin map")

    try:
        checkpoint_payload = torch.load(checkpoint.path, weights_only=False, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise SmokeError(f"checkpoint cannot be loaded: {error}") from error
    checkpoint_link = checkpoint_payload.get("practice_utility", {})
    expected_capsule_path = str(capsule.path)
    if (
        checkpoint_link.get("source_capsule") != expected_capsule_path
        or checkpoint_link.get("capsule_sha256") != capsule_payload.get("capsule_sha256")
        or checkpoint_link.get("global_step") != step
    ):
        raise SmokeError("checkpoint does not link exactly to the selected capsule")

    try:
        snapshot_payload = json.loads(snapshot.path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"origin snapshot cannot be read: {error}") from error
    if snapshot_payload.get("global_step") != step:
        raise SmokeError("snapshot global_step differs from the selected origin")
    rows = snapshot_payload.get("contexts")
    if not isinstance(rows, list):
        raise SmokeError("origin snapshot has no contexts")
    snapshot_contexts: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SmokeError("origin snapshot contains a malformed context")
        context_id = row.get("context_id")
        if not isinstance(context_id, str):
            raise SmokeError("origin snapshot context has no context_id")
        try:
            computed = ContextKey.from_dict(row).context_id
        except (KeyError, TypeError, ValueError) as error:
            raise SmokeError("origin snapshot contains an invalid context") from error
        if context_id != computed:
            raise SmokeError("origin snapshot context_id mismatch")
        prior = snapshot_contexts.get(context_id)
        context_fields = ContextKey.from_dict(row).to_dict()
        if prior is not None and prior != context_fields:
            raise SmokeError("origin snapshot has conflicting duplicate contexts")
        snapshot_contexts[context_id] = context_fields

    manifest_rows = manifest.get("contexts_per_stage", {}).get(stage)
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise SmokeError("manifest has no contexts for requested stage")
    for row in manifest_rows:
        if not isinstance(row, dict) or not isinstance(row.get("context"), dict):
            raise SmokeError("manifest context is malformed")
        context = ContextKey.from_dict(row["context"])
        if row.get("context_id") != context.context_id:
            raise SmokeError("manifest context_id mismatch")
        if snapshot_contexts.get(context.context_id) != context.to_dict():
            raise SmokeError("manifest context is absent or differs in the selected snapshot")

    common = set(origin_map.get("common_resident_context_ids") or [])
    expected_ids = {row["context_id"] for row in manifest_rows}
    if not expected_ids <= common:
        raise SmokeError("manifest contexts are not all in the origin-map intersection")
    if manifest.get("pool_sha256") != origin_map.get("motion_pool_manifest_sha256"):
        raise SmokeError("manifest motion pool differs from the origin map")
    if manifest.get("split_sha256") != origin_map.get("dev_suite_sha256"):
        raise SmokeError("manifest deployment split differs from the origin map")
    return origin, capsule, checkpoint, snapshot, capsule_payload


def _build_command(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    origin_map: Mapping[str, Any],
    motion_source_bindings: Mapping[str, Mapping[str, Any]],
    dose_plan: DP.PassiveDosePlan,
    capsule: Asset,
    checkpoint_path: Path,
    origin_step: int,
    target_step: int,
    num_envs: int,
    num_steps_per_env: int,
    branch_id: str,
    pair_id: str,
    dose_dir: Path,
) -> list[str]:
    robot = (motion_source_bindings.get("robot") or {}).get("resolved_path")
    smpl = (motion_source_bindings.get("smpl") or {}).get("resolved_path")
    target_fps = origin_map.get("motion_lib_target_fps")
    if not isinstance(robot, str) or not isinstance(smpl, str):
        raise SmokeError("verified robot and SMPL motion source paths are unavailable")
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)):
        raise SmokeError("origin map does not freeze motion_lib_target_fps")
    lineage = (
        "{campaign_id:"
        f"{manifest['campaign_id']},manifest_sha256:{manifest['manifest_sha256']},"
        f"manifest_file_sha256:{dose_plan.manifest_file_sha256}"
        "}"
    )
    return [
        sys.executable,
        str(REPO / "gear_sonic" / "train_agent_trl.py"),
        f"+exp={args.exp}",
        f"checkpoint={checkpoint_path}",
        "+resume=true",
        f"num_envs={num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={target_step}",
        "++algo.config.save_interval=100000",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={robot}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={smpl}",
        f"++manager_env.commands.motion.motion_lib_cfg.target_fps={float(target_fps):g}",
        f"++callbacks.practice_resume._target_={RESUME_CALLBACK}",
        "++callbacks.practice_resume.enabled=true",
        f"++callbacks.practice_resume.capsule_path={capsule.path}",
        f"++callbacks.practice_context._target_={CONTEXT_CALLBACK}",
        "++callbacks.practice_context.enabled=true",
        "++callbacks.practice_context.role=control",
        f"++callbacks.practice_context.pair_id={pair_id}",
        f"++callbacks.practice_context.branch_id={branch_id}",
        "++callbacks.practice_context.epsilon=0.0",
        f"++callbacks.practice_context.kernel_radius_bins={manifest['kernel_radius_bins']}",
        f"++callbacks.practice_context.dose_report_dir={dose_dir}",
        "++callbacks.practice_context.dose_report_frequency=0",
        "++callbacks.practice_context.claim_mode=true",
        f"++callbacks.practice_context.dose_plan_path={dose_plan.path}",
        f"++callbacks.practice_context.dose_plan_sha256={dose_plan.file_sha256}",
        f"++callbacks.practice_context.dose_plan_stage={args.stage}",
        f"++callbacks.practice_context.dose_report_horizons.H_s={target_step}",
        f"++callbacks.practice_context.dose_origin_global_step={origin_step}",
        f"++callbacks.practice_context.dose_num_steps_per_iteration={num_steps_per_env}",
        f"++callbacks.practice_context.dose_num_envs={num_envs}",
        f"++callbacks.practice_context.dose_lineage={lineage}",
    ]


def prepare(args: argparse.Namespace) -> PreparedSmoke:
    run_id = args.run_id or default_run_id(args.stage, args.seed)
    if not RUN_ID.fullmatch(run_id):
        raise SmokeError("generated run id is not filename-safe")
    manifest_asset, manifest = read_json_asset(args.manifest, args.manifest_sha256, "manifest")
    _validate_manifest(manifest)
    if args.stage not in manifest.get("stages", []):
        raise SmokeError("requested stage is absent from the manifest")
    if args.seed not in manifest.get("seeds", []):
        raise SmokeError("requested seed is absent from the manifest")
    origin_map_asset, origin_map = read_json_asset(
        args.origin_map, args.origin_map_sha256, "origin map"
    )
    motion_source_bindings = validate_motion_source_bindings(origin_map)
    origin, capsule, checkpoint, snapshot, _ = _validate_origin(
        manifest, origin_map, stage=args.stage, seed=args.seed
    )
    plan = DP.load_passive_dose_plan(
        args.dose_plan,
        expected_file_sha256=args.dose_plan_sha256,
        expected_campaign_id=manifest["campaign_id"],
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_manifest_file_sha256=manifest_asset.sha256,
    )
    plan.contexts_for(args.stage)
    manifest_ids = {row["context_id"] for row in manifest["contexts_per_stage"][args.stage]}
    if {context.context_id for context in plan.contexts_for(args.stage)} != manifest_ids:
        raise SmokeError("passive dose plan context coverage differs from the manifest")

    short = manifest.get("horizons", {}).get("H_s")
    if isinstance(short, bool) or not isinstance(short, int) or short <= 0:
        raise SmokeError("manifest must freeze a positive integer H_s")
    origin_step = int(origin["origin_step"])
    target_step = origin_step + short
    num_envs = origin.get("num_envs", origin_map.get("num_envs"))
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise SmokeError("origin map must freeze a positive environment count")
    if num_envs != origin_map.get("num_envs"):
        raise SmokeError("selected origin environment count differs from the origin map")
    config_asset, num_steps_per_env = source_config(args.exp)
    callback_path = REPO / "gear_sonic" / "research" / "practice_utility" / "callbacks.py"
    callback_asset = Asset(callback_path.resolve(strict=True), file_sha256(callback_path))
    launcher_path = Path(__file__).resolve(strict=True)
    launcher_asset = Asset(launcher_path, file_sha256(launcher_path))
    implementation_paths = {
        "sampler_adapter": REPO / "gear_sonic/research/practice_utility/sampler_adapter.py",
        "dose_plan": REPO / "gear_sonic/research/practice_utility/dose_plan.py",
        "schema": REPO / "gear_sonic/research/practice_utility/schema.py",
        "intervention": REPO / "gear_sonic/research/practice_utility/intervention.py",
        "branch_capsule": REPO / "gear_sonic/research/practice_utility/branch_capsule.py",
        "train_agent": REPO / "gear_sonic/train_agent_trl.py",
        "ppo_trainer": REPO / "gear_sonic/trl/trainer/ppo_trainer.py",
        "manager_env_wrapper": REPO / "gear_sonic/envs/wrapper/manager_env_wrapper.py",
    }
    implementation_assets = {
        name: Asset(path.resolve(strict=True), file_sha256(path))
        for name, path in implementation_paths.items()
    }
    commit = git_sha()
    tree_status = tuple(git_status())

    artifact_dir = args.artifact_root.resolve() / run_id
    runtime_checkpoint = artifact_dir / "origin_checkpoint.pt"
    dose_dir = artifact_dir / "dose"
    log_path = args.log_dir.resolve() / f"{run_id}.log"
    prefix = args.receipt_dir.resolve() / run_id
    preregistration = prefix.with_name(f"{prefix.name}.preregistration.json")
    status = prefix.with_name(f"{prefix.name}.status.json")
    smoke = prefix.with_name(f"{prefix.name}.smoke.json")
    pair_id = f"{run_id}_shared"
    branch_id = f"{pair_id}_control"
    dose_receipt = dose_dir / f"dose_{branch_id}_H_s_step{target_step:06d}.json"
    command = _build_command(
        args=args,
        manifest=manifest,
        origin_map=origin_map,
        motion_source_bindings=motion_source_bindings,
        dose_plan=plan,
        capsule=capsule,
        checkpoint_path=runtime_checkpoint,
        origin_step=origin_step,
        target_step=target_step,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        branch_id=branch_id,
        pair_id=pair_id,
        dose_dir=dose_dir,
    )
    return PreparedSmoke(
        run_id=run_id,
        stage=args.stage,
        seed=args.seed,
        manifest=manifest,
        manifest_asset=manifest_asset,
        origin_map=origin_map,
        origin_map_asset=origin_map_asset,
        dose_plan=plan,
        origin=origin,
        capsule_asset=capsule,
        checkpoint_asset=checkpoint,
        snapshot_asset=snapshot,
        origin_step=origin_step,
        short_horizon=short,
        target_step=target_step,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        source_config_asset=config_asset,
        launcher_asset=launcher_asset,
        callback_asset=callback_asset,
        implementation_assets=implementation_assets,
        motion_source_bindings=motion_source_bindings,
        source_commit=commit,
        source_tree_status=tree_status,
        branch_id=branch_id,
        pair_id=pair_id,
        artifact_dir=artifact_dir,
        runtime_checkpoint_path=runtime_checkpoint,
        dose_dir=dose_dir,
        log_path=log_path,
        preregistration_path=preregistration,
        status_path=status,
        smoke_path=smoke,
        dose_receipt_path=dose_receipt,
        command=tuple(command),
    )


def preregistration_payload(prepared: PreparedSmoke, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "practice_utility_passive_dose_smoke_preregistration",
        "schema_version": 1,
        "created_at": now_iso(),
        "immutable": True,
        "outcome_blind": True,
        "run_id": prepared.run_id,
        "source_commit": prepared.source_commit,
        "source_tree_clean": not prepared.source_tree_status,
        "source_tree_status": list(prepared.source_tree_status),
        "launcher": {
            "path": str(prepared.launcher_asset.path),
            "sha256": prepared.launcher_asset.sha256,
        },
        "implementation": {
            "launcher": prepared.launcher_asset.to_dict(),
            "callback": prepared.callback_asset.to_dict(),
            "source_config": prepared.source_config_asset.to_dict(),
            "sources": {
                name: asset.to_dict()
                for name, asset in sorted(prepared.implementation_assets.items())
            },
        },
        "inputs": {
            "manifest": prepared.manifest_asset.to_dict(),
            "origin_map": prepared.origin_map_asset.to_dict(),
            "passive_dose_plan": {
                **Asset(prepared.dose_plan.path, prepared.dose_plan.file_sha256).to_dict(),
                "logical_sha256": prepared.dose_plan.logical_sha256,
            },
            "capsule": prepared.capsule_asset.to_dict(),
            "checkpoint": prepared.checkpoint_asset.to_dict(),
            "localized_checkpoint": {
                "path": str(prepared.runtime_checkpoint_path),
                "sha256": prepared.checkpoint_asset.sha256,
                "source_path": str(prepared.checkpoint_asset.path),
                "publication": "exclusive_byte_copy",
            },
            "snapshot": prepared.snapshot_asset.to_dict(),
            "motion_sources": prepared.motion_source_bindings,
            "callback": prepared.callback_asset.to_dict(),
        },
        "design": {
            "campaign_id": prepared.manifest["campaign_id"],
            "manifest_sha256": prepared.manifest["manifest_sha256"],
            "stage": prepared.stage,
            "seed": prepared.seed,
            "role": "control",
            "epsilon": 0.0,
            "origin_global_step": prepared.origin_step,
            "relative_horizon": {"H_s": prepared.short_horizon},
            "absolute_horizon": {"H_s": prepared.target_step},
            "num_envs": prepared.num_envs,
            "num_steps_per_env": prepared.num_steps_per_env,
            "num_steps_source_config": prepared.source_config_asset.to_dict(),
            "expected_completed_env_steps": (
                prepared.short_horizon * prepared.num_steps_per_env * prepared.num_envs
            ),
        },
        "gpu_gate": {
            "minimum_free_mib": args.min_free_mib,
            "maximum_utilization_pct": args.max_gpu_util_pct,
            "samples": args.idle_samples,
            "sample_seconds": args.idle_sample_seconds,
            "require_zero_compute_processes": True,
        },
        "command": list(prepared.command),
        "command_sha256": sha256_of({"argv": list(prepared.command)}),
        "outputs": {
            "artifact_dir": str(prepared.artifact_dir),
            "localized_checkpoint": str(prepared.runtime_checkpoint_path),
            "log": str(prepared.log_path),
            "status": str(prepared.status_path),
            "smoke": str(prepared.smoke_path),
            "dose_receipt": str(prepared.dose_receipt_path),
        },
        "postflight_requirements": [
            "claim-mode H_s dose receipt is complete and hash-valid",
            "completed totals equal H_s * source num_steps_per_env * num_envs",
            "completion hook has nonzero exact calls with no drops",
            "dose registry is stable and every planned context appears exactly once",
            "control has epsilon zero and no intervention context",
            "callback and launcher are bound to the clean source commit",
        ],
        "scope_exclusions": [
            "no intervention branch",
            "no deployment evaluation or utility label",
            "no paired no-callback trajectory comparison",
        ],
    }
    payload["preregistration_sha256"] = sha256_of(payload)
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise SmokeError(f"refusing to overwrite stale partial receipt: {partial}")
    partial.write_bytes(_json_bytes(payload))
    partial.replace(path)


def reserve_outputs(prepared: PreparedSmoke) -> None:
    claimed = [
        prepared.artifact_dir,
        prepared.log_path,
        prepared.preregistration_path,
        prepared.status_path,
        prepared.smoke_path,
    ]
    existing = [str(path) for path in claimed if path.exists()]
    if existing:
        raise SmokeError("collision with existing smoke outputs: " + ", ".join(existing))
    prepared.artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    prepared.artifact_dir.mkdir(exist_ok=False)
    prepared.dose_dir.mkdir()
    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.preregistration_path.parent.mkdir(parents=True, exist_ok=True)


def materialize_runtime_checkpoint(prepared: PreparedSmoke) -> Asset:
    """Copy the frozen checkpoint so SONIC resume writes only inside this smoke run."""

    try:
        with prepared.checkpoint_asset.path.open(
            "rb"
        ) as source, prepared.runtime_checkpoint_path.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
    except OSError as error:
        raise SmokeError(f"cannot localize the origin checkpoint: {error}") from error
    actual = file_sha256(prepared.runtime_checkpoint_path)
    if actual != prepared.checkpoint_asset.sha256:
        raise SmokeError(
            "localized checkpoint differs from its frozen source: "
            f"expected {prepared.checkpoint_asset.sha256}, got {actual}"
        )
    return Asset(prepared.runtime_checkpoint_path, actual)


def verify_prelaunch_identity(
    prepared: PreparedSmoke, *, require_runtime_checkpoint: bool = False
) -> None:
    """Recheck every frozen byte and Git identity immediately before launch."""

    current_commit = git_sha()
    current_status = tuple(git_status())
    if current_commit != prepared.source_commit:
        raise SmokeError(
            f"source HEAD changed after preparation: {prepared.source_commit} -> {current_commit}"
        )
    if current_status != prepared.source_tree_status:
        raise SmokeError(
            "source tree status changed after preparation: "
            f"{list(prepared.source_tree_status)} -> {list(current_status)}"
        )
    if current_status:
        raise SmokeError("live smoke requires a clean committed source tree")
    frozen_assets = (
        ("launcher", prepared.launcher_asset.path, prepared.launcher_asset.sha256),
        ("callback", prepared.callback_asset.path, prepared.callback_asset.sha256),
        ("source config", prepared.source_config_asset.path, prepared.source_config_asset.sha256),
        ("manifest", prepared.manifest_asset.path, prepared.manifest_asset.sha256),
        ("origin map", prepared.origin_map_asset.path, prepared.origin_map_asset.sha256),
        ("passive dose plan", prepared.dose_plan.path, prepared.dose_plan.file_sha256),
        ("capsule", prepared.capsule_asset.path, prepared.capsule_asset.sha256),
        ("checkpoint", prepared.checkpoint_asset.path, prepared.checkpoint_asset.sha256),
        ("snapshot", prepared.snapshot_asset.path, prepared.snapshot_asset.sha256),
    ) + tuple(
        (name, asset.path, asset.sha256)
        for name, asset in sorted(prepared.implementation_assets.items())
    )
    for label, path, expected in frozen_assets:
        try:
            actual = file_sha256(path)
        except OSError as error:
            raise SmokeError(f"frozen {label} became unreadable: {path}") from error
        if actual != expected:
            raise SmokeError(
                f"frozen {label} changed after preparation: expected {expected}, got {actual}"
            )
    current_motion_sources = validate_motion_source_bindings(prepared.origin_map)
    if current_motion_sources != prepared.motion_source_bindings:
        raise SmokeError("motion-source tree bindings changed after preparation")
    if require_runtime_checkpoint:
        if not prepared.runtime_checkpoint_path.is_file():
            raise SmokeError("localized runtime checkpoint is missing")
        actual = file_sha256(prepared.runtime_checkpoint_path)
        if actual != prepared.checkpoint_asset.sha256:
            raise SmokeError(
                "localized runtime checkpoint changed before launch: "
                f"expected {prepared.checkpoint_asset.sha256}, got {actual}"
            )


def compute_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise GpuNotIdleError(f"cannot parse compute process row: {line!r}")
        rows.append(
            {"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": float(parts[2])}
        )
    return rows


def gpu_idle_sample() -> dict[str, Any]:
    return {"gpu": TP.gpu_snapshot(), "compute_processes": compute_processes()}


def audit_gpu_idle(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index in range(args.idle_samples):
        sample = gpu_idle_sample()
        samples.append(sample)
        gpu = sample["gpu"]
        if gpu["free_mib"] < args.min_free_mib:
            blockers.append(
                f"sample {index}: free {gpu['free_mib']:.0f} MiB < {args.min_free_mib} MiB"
            )
        if gpu["gpu_util_pct"] > args.max_gpu_util_pct:
            blockers.append(
                f"sample {index}: utilization {gpu['gpu_util_pct']:.0f}% > "
                f"{args.max_gpu_util_pct:.0f}%"
            )
        if sample["compute_processes"]:
            blockers.append(
                f"sample {index}: active compute PIDs "
                f"{[row['pid'] for row in sample['compute_processes']]}"
            )
        if index + 1 < args.idle_samples:
            time.sleep(args.idle_sample_seconds)
    if blockers:
        raise GpuNotIdleError("GPU idle gate failed: " + "; ".join(blockers), samples)
    return samples


@contextmanager
def gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GpuNotIdleError(f"another passive-dose smoke holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_command(
    command: tuple[str, ...], log_path: Path, initial_gpu: dict[str, float]
) -> dict[str, Any]:
    samples = [initial_gpu]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(1.0):
            try:
                samples.append(TP.gpu_snapshot())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    with log_path.open("xb") as handle:
        exit_code = subprocess.call(
            list(command), cwd=REPO, env=_runtime_env(), stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.monotonic() - started
    stop.set()
    thread.join(timeout=5.0)
    return {
        "exit_code": exit_code,
        "wall_seconds": elapsed,
        "gpu": TP.summarize_gpu(samples),
        "cuda_memory_growth_mib": max(row["used_mib"] for row in samples) - initial_gpu["used_mib"],
    }


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TMPDIR", "/data/robotixx/lucid-sonic/tmp")
    env.setdefault("WANDB_MODE", "offline")
    return env


def _validate_context_projections(
    prepared: PreparedSmoke,
    dose: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Recompute every passive projection from the emitted completed-bin histogram."""

    raw_completed = dose.get("per_bin_completed")
    rows = dose.get("context_doses")
    registry_sha256 = dose.get("dose_registry_sha256_at_report")
    if (
        not isinstance(raw_completed, Mapping)
        or not isinstance(rows, list)
        or not SHA256.fullmatch(str(registry_sha256))
    ):
        return False, []
    completed_by_bin: dict[int, float] = {}
    try:
        for raw_key, raw_value in raw_completed.items():
            key = int(raw_key)
            value = float(raw_value)
            if str(key) != str(raw_key) or not math.isfinite(value) or value < 0:
                return False, []
            completed_by_bin[key] = value
    except (TypeError, ValueError):
        return False, []

    planned = {
        context.context_id: context.to_dict()
        for context in prepared.dose_plan.contexts_for(prepared.stage)
    }
    actual_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return False, actual_ids
        context_id = row.get("context_id")
        if not isinstance(context_id, str) or row.get("context") != planned.get(context_id):
            return False, actual_ids
        if context_id in actual_ids:
            return False, actual_ids
        actual_ids.append(context_id)
        if row.get("kernel_radius_bins") != prepared.dose_plan.kernel_radius_bins:
            return False, actual_ids
        try:
            sigma = float(row.get("sigma_frames"))
            completed = float(row.get("completed_kernel_steps"))
        except (TypeError, ValueError):
            return False, actual_ids
        if (
            not math.isclose(sigma, prepared.dose_plan.sigma_frames, rel_tol=0.0, abs_tol=1e-12)
            or not math.isfinite(completed)
            or completed < 0
        ):
            return False, actual_ids
        raw_membership = row.get("membership_by_global_bin")
        if not isinstance(raw_membership, Mapping) or not raw_membership:
            return False, actual_ids
        membership: dict[int, float] = {}
        try:
            for raw_key, raw_value in raw_membership.items():
                key = int(raw_key)
                value = float(raw_value)
                if (
                    str(key) != str(raw_key)
                    or not math.isfinite(value)
                    or not 0.0 < value <= 1.0 + 1e-12
                ):
                    return False, actual_ids
                membership[key] = value
        except (TypeError, ValueError):
            return False, actual_ids
        if not math.isclose(max(membership.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            return False, actual_ids
        reconstructed = math.fsum(
            completed_by_bin.get(bin_id, 0.0) * weight for bin_id, weight in membership.items()
        )
        if not math.isclose(reconstructed, completed, rel_tol=0.0, abs_tol=1e-6):
            return False, actual_ids
        expected_hash = sha256_of(
            {
                "context_id": context_id,
                "kernel_radius_bins": prepared.dose_plan.kernel_radius_bins,
                "sigma_frames": sigma,
                "membership_by_global_bin": {
                    str(key): value for key, value in sorted(membership.items())
                },
                "dose_registry_sha256": registry_sha256,
            }
        )
        if row.get("kernel_membership_sha256") != expected_hash:
            return False, actual_ids
    return set(actual_ids) == set(planned) and len(actual_ids) == len(planned), actual_ids


def verify_dose_receipt(prepared: PreparedSmoke, runtime: Mapping[str, Any]) -> dict[str, Any]:
    path = prepared.dose_receipt_path
    blockers: list[str] = []
    if not path.is_file():
        return {"valid": False, "blockers": [f"missing exact H_s dose receipt: {path}"]}
    try:
        dose = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"valid": False, "blockers": [f"cannot read dose receipt: {error}"]}
    if not isinstance(dose, dict):
        return {"valid": False, "blockers": ["dose receipt is not a JSON object"]}

    receipt_hash = dose.get("receipt_payload_sha256")
    body = dict(dose)
    body.pop("receipt_payload_sha256", None)
    expected_total = prepared.short_horizon * prepared.num_steps_per_env * prepared.num_envs
    expected_hooks = prepared.short_horizon * prepared.num_steps_per_env
    expected_observations = expected_hooks * prepared.num_envs
    exact = {
        "kind_schema": dose.get("kind") == DP.PASSIVE_DOSE_RECEIPT_KIND
        and dose.get("schema_version") == DP.PASSIVE_DOSE_RECEIPT_SCHEMA_VERSION,
        "status": dose.get("status") == "complete" and dose.get("valid_for_claim") is True,
        "receipt_hash": isinstance(receipt_hash, str) and receipt_hash == sha256_of(body),
        "control": dose.get("role") == "control"
        and dose.get("epsilon") == 0.0
        and dose.get("context_id") == "native"
        and dose.get("armed") is False
        and dose.get("never_armed") is True,
        "branch": dose.get("branch_id") == prepared.branch_id
        and dose.get("pair_id") == prepared.pair_id,
        "horizon": dose.get("global_step") == prepared.target_step
        and dose.get("horizon_label") == "H_s",
        "positive_exact_total": dose.get("completed_env_steps") == expected_total
        and expected_total > 0
        and dose.get("expected_env_steps") == expected_total,
        "exact_hook_calls": dose.get("completion_hook_calls") == expected_hooks
        and dose.get("expected_completion_hook_calls") == expected_hooks
        and expected_hooks > 0,
        "exact_observations": dose.get("completion_observations") == expected_observations
        and dose.get("termination_observations") == expected_observations
        and dose.get("expected_completion_observations") == expected_observations,
        "no_drops": dose.get("dropped_completion_batches") == 0,
        "registry_stable": dose.get("registry_stable") is True
        and dose.get("dose_registry_sha256_at_install")
        == dose.get("dose_registry_sha256_at_report")
        and SHA256.fullmatch(str(dose.get("dose_registry_sha256_at_report", ""))) is not None,
        "atomic": not path.with_suffix(path.suffix + ".partial").exists(),
    }
    per_bin = dose.get("per_bin_completed")
    try:
        per_bin_total = math.fsum(float(value) for value in per_bin.values())
    except (AttributeError, TypeError, ValueError):
        per_bin_total = -1.0
    exact["per_bin_total"] = math.isclose(per_bin_total, expected_total, rel_tol=0.0, abs_tol=1e-6)

    plan_link = dose.get("passive_dose_plan")
    exact["dose_plan"] = isinstance(plan_link, dict) and plan_link == {
        "path": str(prepared.dose_plan.path),
        "file_sha256": prepared.dose_plan.file_sha256,
        "logical_sha256": prepared.dose_plan.logical_sha256,
        "stage": prepared.stage,
    }
    exact["lineage"] = dose.get("lineage") == {
        "campaign_id": prepared.manifest["campaign_id"],
        "manifest_sha256": prepared.manifest["manifest_sha256"],
        "manifest_file_sha256": prepared.manifest_asset.sha256,
    }
    callback = dose.get("implementation")
    exact["callback_hash"] = (
        isinstance(callback, dict)
        and callback.get("callback_sha256") == prepared.callback_asset.sha256
    )

    expected_contexts = sorted(
        context.context_id for context in prepared.dose_plan.contexts_for(prepared.stage)
    )
    projections_valid, actual_contexts = _validate_context_projections(prepared, dose)
    actual_contexts = sorted(actual_contexts)
    exact["context_coverage"] = projections_valid and actual_contexts == expected_contexts
    coverage_hash = sha256_of({"stage": prepared.stage, "context_ids": expected_contexts})
    failed = sorted(name for name, passed in exact.items() if passed is not True)
    blockers.extend(f"dose postflight failed {name}" for name in failed)
    cuda_growth = runtime.get("cuda_memory_growth_mib")
    cuda_verified = isinstance(cuda_growth, (int, float)) and cuda_growth >= GPU_GROWTH_MIB
    if not cuda_verified:
        blockers.append(f"CUDA memory growth was not observed at >= {GPU_GROWTH_MIB:g} MiB")
    return {
        "valid": not blockers,
        "checks": exact,
        "expected_context_ids_sha256": coverage_hash,
        "actual_context_ids_sha256": sha256_of(
            {"stage": prepared.stage, "context_ids": actual_contexts}
        ),
        "expected_completed_env_steps": expected_total,
        "dose_receipt": {"path": str(path), "sha256": file_sha256(path)},
        "cuda_execution_verified": cuda_verified,
        "blockers": blockers,
    }


def smoke_payload(
    prepared: PreparedSmoke,
    postflight: Mapping[str, Any],
    preregistration: Asset | None,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    dose_reference = postflight.get("dose_receipt")
    complete = postflight.get("valid") is True
    dose_checks = postflight.get("checks", {})
    log_asset = (
        Asset(prepared.log_path, file_sha256(prepared.log_path))
        if prepared.log_path.is_file()
        else None
    )
    checks = {
        "passive_completion_exact": bool(
            complete
            and dose_checks.get("positive_exact_total") is True
            and dose_checks.get("exact_hook_calls") is True
            and dose_checks.get("exact_observations") is True
            and dose_checks.get("per_bin_total") is True
        ),
        "epsilon_zero_control": bool(complete and dose_checks.get("control") is True),
        "dose_registry_stable": bool(complete and dose_checks.get("registry_stable") is True),
        "exact_context_projection": bool(
            complete
            and dose_checks.get("context_coverage") is True
            and postflight.get("actual_context_ids_sha256")
            == postflight.get("expected_context_ids_sha256")
        ),
        "cuda_execution_verified": postflight.get("cuda_execution_verified") is True,
        "atomic_receipt_no_partial": bool(complete and dose_checks.get("atomic") is True),
        "immutable_preregistration_bound": bool(
            preregistration is not None
            and preregistration.path.is_file()
            and file_sha256(preregistration.path) == preregistration.sha256
        ),
        "successful_runtime_and_log": bool(
            complete and runtime.get("exit_code") == 0 and log_asset is not None
        ),
    }
    smoke_blockers = list(postflight.get("blockers", []))
    smoke_blockers.extend(
        f"smoke check failed {name}" for name, passed in checks.items() if passed is not True
    )
    payload: dict[str, Any] = {
        "kind": SMOKE_KIND,
        "schema_version": 1,
        "created_at": now_iso(),
        "status": "complete" if all(checks.values()) else "blocked",
        "campaign_id": prepared.manifest["campaign_id"],
        "manifest_sha256": prepared.manifest["manifest_sha256"],
        "manifest_file_sha256": prepared.manifest_asset.sha256,
        "measurement_hook": DP.PASSIVE_DOSE_HOOK,
        "execution_mode": "live_gpu_simulator",
        "device_type": "cuda",
        "stage": prepared.stage,
        "seed": prepared.seed,
        "source_commit": prepared.source_commit,
        "source_tree_status": list(prepared.source_tree_status),
        "implementation": {
            "launcher": prepared.launcher_asset.to_dict(),
            "callback": prepared.callback_asset.to_dict(),
            "source_config": prepared.source_config_asset.to_dict(),
            "sources": {
                name: asset.to_dict()
                for name, asset in sorted(prepared.implementation_assets.items())
            },
        },
        "preregistration": preregistration.to_dict() if preregistration is not None else None,
        "command_sha256": sha256_of({"argv": list(prepared.command)}),
        "runtime": {
            "exit_code": runtime.get("exit_code"),
            "wall_seconds": runtime.get("wall_seconds"),
            "cuda_memory_growth_mib": runtime.get("cuda_memory_growth_mib"),
            "gpu": runtime.get("gpu"),
            "log": log_asset.to_dict() if log_asset is not None else None,
        },
        "passive_dose_plan": {
            "file_sha256": prepared.dose_plan.file_sha256,
            "logical_sha256": prepared.dose_plan.logical_sha256,
        },
        "origin": {
            "origin_map_file_sha256": prepared.origin_map_asset.sha256,
            "global_step": prepared.origin_step,
            "capsule_file_sha256": prepared.capsule_asset.sha256,
            "checkpoint_file_sha256": prepared.checkpoint_asset.sha256,
            "snapshot_file_sha256": prepared.snapshot_asset.sha256,
        },
        "dose_receipt": dose_reference,
        "context_coverage_sha256": postflight.get("actual_context_ids_sha256"),
        "checks": checks,
        "verification_basis": {
            "passive_hook": "live exact H_s completed-step receipt",
            "control_identity": "receipt observes role=control and epsilon=0",
            "projection": "hash-bound exact context coverage over a stable global registry",
        },
        "not_yet_verified": [
            "native step return preservation in the live process",
            "native distribution bitwise identity against a paired no-callback reference",
            "callback wrapper removal observed after on_train_end",
            "paired no-callback stochastic trajectory equivalence",
            "campaign utility efficacy or any treatment effect",
        ],
        "blockers": smoke_blockers,
    }
    payload["smoke_sha256"] = sha256_of(payload)
    return payload


def error_payload(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def initial_status(prepared: PreparedSmoke, preregistration: Asset) -> dict[str, Any]:
    return {
        "kind": "practice_utility_passive_dose_smoke_status",
        "schema_version": 1,
        "updated_at": now_iso(),
        "run_id": prepared.run_id,
        "state": "preregistered",
        "preregistration": preregistration.to_dict(),
        "gpu_idle_samples": [],
        "runtime": None,
        "postflight": None,
        "smoke": None,
        "error": None,
    }


def execute(prepared: PreparedSmoke, args: argparse.Namespace) -> int:
    verify_prelaunch_identity(prepared)
    reserve_outputs(prepared)
    localized_checkpoint = materialize_runtime_checkpoint(prepared)
    prereg = preregistration_payload(prepared, args)
    prereg["inputs"]["localized_checkpoint"] = {
        **localized_checkpoint.to_dict(),
        "source_path": str(prepared.checkpoint_asset.path),
        "publication": "exclusive_byte_copy",
    }
    prereg.pop("preregistration_sha256", None)
    prereg["preregistration_sha256"] = sha256_of(prereg)
    write_exclusive(prepared.preregistration_path, prereg)
    prereg_asset = Asset(prepared.preregistration_path, file_sha256(prepared.preregistration_path))
    status = initial_status(prepared, prereg_asset)
    atomic_write(prepared.status_path, status)
    try:
        with gpu_lock(args.gpu_lock.resolve()):
            # The immutable preregistration now exists. Recheck after receipt
            # I/O and immediately before the first GPU query so neither code
            # nor an input can change between preparation and launch.
            verify_prelaunch_identity(prepared, require_runtime_checkpoint=True)
            samples = audit_gpu_idle(args)
            status["state"] = "running"
            status["gpu_idle_samples"] = samples
            status["updated_at"] = now_iso()
            atomic_write(prepared.status_path, status)
            runtime = run_command(prepared.command, prepared.log_path, samples[-1]["gpu"])
        status["runtime"] = runtime
        if runtime.get("exit_code") != 0:
            raise SmokeError(f"simulator continuation exited {runtime.get('exit_code')}")
        postflight = verify_dose_receipt(prepared, runtime)
        status["postflight"] = postflight
        smoke = smoke_payload(prepared, postflight, prereg_asset, runtime)
        atomic_write(prepared.smoke_path, smoke)
        status["smoke"] = {
            "path": str(prepared.smoke_path),
            "sha256": file_sha256(prepared.smoke_path),
        }
        if postflight.get("valid") is not True or smoke["status"] != "complete":
            raise SmokeError("live passive-dose postflight remained claim-blocked")
        status["state"] = "complete"
        status["updated_at"] = now_iso()
        atomic_write(prepared.status_path, status)
        return 0
    except GpuNotIdleError as error:
        status["state"] = "blocked"
        status["gpu_idle_samples"] = error.samples
        status["error"] = error_payload(error)
        status["updated_at"] = now_iso()
        atomic_write(prepared.status_path, status)
        return 2
    except BaseException as error:  # write evidence even for unexpected simulator failures
        status["state"] = "failed"
        status["error"] = error_payload(error)
        status["updated_at"] = now_iso()
        atomic_write(prepared.status_path, status)
        return 1


def dry_run_payload(prepared: PreparedSmoke, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "execution_mode": "dry_run",
        "ready_for_execute": not prepared.source_tree_status,
        "source_tree_status": list(prepared.source_tree_status),
        "preregistration_preview": preregistration_payload(prepared, args),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepared = prepare(args)
        print(json.dumps(dry_run_payload(prepared, args), indent=2, sort_keys=True))
        if not args.execute:
            print("dry run; pass --execute after committing this exact clean source tree")
            return 0
        return execute(prepared, args)
    except (BC.CapsuleIntegrityError, FileNotFoundError, OSError, SmokeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
