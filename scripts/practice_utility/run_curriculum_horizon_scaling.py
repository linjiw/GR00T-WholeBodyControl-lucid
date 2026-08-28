#!/usr/bin/env python3
"""Preregister and run the exact LUCID horizon-scaling training matrix.

This is the claim-facing orchestration layer around
``run_curriculum_comparison.py``.  The older driver remains useful for a small,
single-budget experiment; this layer adds the properties required by the paper
campaign: one frozen matrix, exclusive paths, hash-bound inputs, incremental
receipts, verified retry, and a strict idle-GPU gate.

The default invocation is a dry run that *reserves* the campaign and writes its
immutable preregistration.  Execute that exact campaign in a later invocation
with ``--resume --execute``.  No GPU query occurs while creating the dry run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402
from scripts.practice_utility import (  # noqa: E402
    run_curriculum_comparison as CC,
    run_latency_ab as LA,
    run_throughput_probe as TP,
)

SCHEMA_VERSION = 1
CAMPAIGN_KIND = "lucid_curriculum_horizon_scaling_preregistration"
STATUS_KIND = "lucid_curriculum_horizon_scaling_status"
INDEX_KIND = "lucid_curriculum_horizon_scaling_training_index"
MODES = ("lucid", "fixed", "off")
HORIZON_MATRIX = (
    (32, (8603, 8604)),
    (64, (8600, 8601, 8602, 8603, 8604)),
    (128, (8600, 8601, 8602, 8603, 8604)),
    (256, (8600, 8601, 8602, 8603, 8604)),
)
HISTORICAL_32_SEEDS = (8600, 8601, 8602)
EXPECTED_TERMS = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

HASH_BOUND_CODE = {
    "horizon_scaling_launcher": "scripts/practice_utility/run_curriculum_horizon_scaling.py",
    "single_budget_launcher": "scripts/practice_utility/run_curriculum_comparison.py",
    "delay_launcher": "scripts/practice_utility/train_with_delay.py",
    "arm_runtime": "scripts/practice_utility/run_latency_ab.py",
    "gpu_runtime": "scripts/practice_utility/run_throughput_probe.py",
    "actuator_patch": "gear_sonic/research/practice_utility/actuator_patch.py",
    "branch_capsule": "gear_sonic/research/practice_utility/branch_capsule.py",
    "capsule_callback": "gear_sonic/research/practice_utility/callbacks.py",
    "curriculum_callback": "gear_sonic/research/practice_utility/dr_curriculum.py",
    "dr_scaling": "gear_sonic/research/practice_utility/dr_scaling.py",
    "reset_safe_events": "gear_sonic/research/practice_utility/events_reset_safe.py",
    "observer": "gear_sonic/research/practice_utility/observer.py",
}
HASH_BOUND_CONFIG = {
    "curriculum_callback_config": "gear_sonic/config/callbacks/lucid_curriculum.yaml",
    "curriculum_event_config": "gear_sonic/config/manager_env/events/tracking/lucid_curriculum.yaml",
    "joint_default_event": "gear_sonic/config/manager_env/events/terms/add_joint_default_pos.yaml",
    "base_com_event": "gear_sonic/config/manager_env/events/terms/base_com.yaml",
    "material_event": "gear_sonic/config/manager_env/events/terms/physics_material.yaml",
    "push_event": "gear_sonic/config/manager_env/events/terms/push_robot.yaml",
    "latency_event": "gear_sonic/config/manager_env/events/terms/randomize_action_delay.yaml",
    "mass_event": "gear_sonic/config/manager_env/events/terms/randomize_rigid_body_mass.yaml",
}


class CampaignError(RuntimeError):
    """The campaign cannot proceed without changing its frozen contract."""


class GpuNotIdleError(CampaignError):
    """The shared GPU does not satisfy the preregistered idle gate."""


class CampaignInterrupted(BaseException):
    """Raised on SIGTERM so the mutable receipt can record interruption."""


@dataclass(frozen=True)
class BranchSpec:
    budget_iterations: int
    seed: int
    mode: str
    order_index: int
    branch_id: str

    @property
    def spec_sha256(self) -> str:
        return sha256_json(asdict(self))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_bytes(path: Path, data: bytes, *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes(target, canonical_json(payload), exclusive=True)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        _write_bytes(staging, canonical_json(payload), exclusive=True)
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="curriculum_horizon_scaling_v1_20260826")
    parser.add_argument(
        "--checkpoint",
        default=(
            "logs_rl/TRL_G1_Track/manager/universal_token/all_modes/"
            "sonic_release_test-20260818_141446/model_step_000024.pt"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--delta-target", type=float, default=0.778)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--integral-max", type=float, default=1.0)
    parser.add_argument("--return-floor", type=float, default=8.0)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--encoder",
        default=str(LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"),
    )
    parser.add_argument(
        "--historical-32-receipt",
        type=Path,
        default=LUCID_ROOT / "manifests/curriculum_comparison_ne128_20260820_143058.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/curriculum_horizon_scaling",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=LUCID_ROOT / "manifests",
    )
    parser.add_argument("--min-free-mib", type=int, default=28000)
    parser.add_argument("--max-gpu-util-pct", type=float, default=5.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-sample-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not CAMPAIGN_ID.fullmatch(args.campaign_id):
        parser.error("campaign-id must be 1-96 safe filename characters")
    if args.num_envs <= 0:
        parser.error("num-envs must be positive")
    if args.warmup_iterations < 0 or args.warmup_iterations >= min(b for b, _ in HORIZON_MATRIX):
        parser.error("warmup-iterations must be nonnegative and below every budget")
    if args.min_free_mib <= 0:
        parser.error("min-free-mib must be positive")
    if args.max_gpu_util_pct < 0:
        parser.error("max-gpu-util-pct must be nonnegative")
    if args.idle_samples < 1 or args.idle_sample_seconds < 0:
        parser.error("idle sampling requires at least one sample and a nonnegative interval")
    if args.execute and not args.resume:
        parser.error(
            "--execute requires --resume so the preregistration is frozen in a prior dry run"
        )
    return args


def campaign_paths(args: argparse.Namespace) -> dict[str, Any]:
    campaign_root = args.artifact_root.resolve() / args.campaign_id
    prefix = args.receipt_dir.resolve() / args.campaign_id
    budget_receipts = {
        str(budget): str(prefix.with_name(f"{prefix.name}.b{budget:04d}.training.json"))
        for budget, _ in HORIZON_MATRIX
    }
    return {
        "campaign_root": str(campaign_root),
        "preregistration": str(prefix.with_name(f"{prefix.name}.preregistration.json")),
        "status": str(prefix.with_name(f"{prefix.name}.status.json")),
        "index": str(prefix.with_name(f"{prefix.name}.training_index.json")),
        "budget_receipts": budget_receipts,
        "combined_32_receipt": str(prefix.with_name(f"{prefix.name}.b0032.combined.training.json")),
        "gpu_lock": str(args.artifact_root.resolve() / ".horizon_scaling_gpu.lock"),
    }


def build_specs(campaign_id: str) -> list[BranchSpec]:
    specs: list[BranchSpec] = []
    order_index = 0
    for budget, seeds in HORIZON_MATRIX:
        for seed_index, seed in enumerate(seeds):
            for mode in CC.arm_order(list(MODES), seed_index):
                specs.append(
                    BranchSpec(
                        budget_iterations=budget,
                        seed=seed,
                        mode=mode,
                        order_index=order_index,
                        branch_id=f"{campaign_id}_b{budget:04d}_s{seed}_{mode}",
                    )
                )
                order_index += 1
    return specs


def cc_args(args: argparse.Namespace, budget: int, checkpoint: str | None = None) -> Any:
    return SimpleNamespace(
        checkpoint=args.checkpoint if checkpoint is None else checkpoint,
        num_envs=args.num_envs,
        iterations=budget,
        warmup_iterations=args.warmup_iterations,
        max_delay=args.max_delay,
        delta_target=args.delta_target,
        kp=args.kp,
        ki=args.ki,
        alpha=args.alpha,
        integral_max=args.integral_max,
        return_floor=args.return_floor,
        exp=args.exp,
        encoder=args.encoder,
    )


def attempt_paths(campaign_root: Path, spec: BranchSpec, attempt: int) -> dict[str, Path]:
    branch_root = (
        campaign_root / f"budget_{spec.budget_iterations:04d}" / f"seed_{spec.seed}" / spec.mode
    )
    root = branch_root / f"attempt_{attempt:03d}"
    return {
        "root": root,
        "artifact_dir": root / "artifacts",
        "log": root / "training.log",
    }


def branch_command(
    args: argparse.Namespace,
    spec: BranchSpec,
    campaign_root: Path,
    attempt: int,
) -> list[str]:
    paths = attempt_paths(campaign_root, spec, attempt)
    return CC.build_command(
        cc_args(args, spec.budget_iterations),
        spec.mode,
        spec.seed,
        spec.branch_id,
        paths["artifact_dir"],
    )


def resolved_file(path: str | Path, *, base: Path = REPO) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise CampaignError(f"required input is not a file: {candidate}")
    return candidate


def bind_file(path: str | Path, *, base: Path = REPO) -> dict[str, Any]:
    requested = Path(path).expanduser()
    resolved = resolved_file(requested, base=base)
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def git_identity() -> dict[str, Any]:
    status = TP.git_status()
    if status:
        raise CampaignError(f"claim campaign requires a clean tree; dirty entries: {status}")
    return {"sha": TP.git_sha(), "status_short": status}


def collect_input_bindings(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolved_file(args.checkpoint)
    source_config = checkpoint.parent / "config.yaml"
    bindings: dict[str, Any] = {
        "checkpoint": bind_file(checkpoint),
        "encoder": bind_file(args.encoder),
        "source_resolved_config": bind_file(source_config),
    }
    for name, relative in {**HASH_BOUND_CODE, **HASH_BOUND_CONFIG}.items():
        bindings[name] = bind_file(REPO / relative)
    return bindings


def normalized_command(command: Sequence[str]) -> list[str]:
    if len(command) < 2:
        raise CampaignError("training command is truncated")
    normalized = ["<python>", str(Path(command[1]).resolve())]
    for token in command[2:]:
        if "=" not in token:
            normalized.append(token)
            continue
        key, value = token.split("=", 1)
        if key.lstrip("+") == "checkpoint":
            value = str(resolved_file(value))
        normalized.append(f"{key}={value}")
    return normalized


def command_value(command: Sequence[str], key: str) -> str:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    matches = [token.split("=", 1)[1] for token in command if token.startswith(prefixes)]
    if len(matches) != 1:
        raise CampaignError(f"expected one {key!r} override, found {len(matches)}")
    return matches[0]


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and float(left) == float(right)
    )


def audit_historical_32(args: argparse.Namespace) -> dict[str, Any]:
    path = resolved_file(args.historical_32_receipt)
    payload = read_json(path)
    config = payload.get("config", {})
    expected_config = {
        "num_envs": args.num_envs,
        "iterations": 32,
        "warmup_iterations": args.warmup_iterations,
        "max_delay_steps": args.max_delay,
    }
    errors = []
    if payload.get("kind") != "lucid_three_arm_training_comparison":
        errors.append("unexpected receipt kind")
    for key, expected in expected_config.items():
        actual = config.get(key)
        if actual != expected:
            errors.append(f"config {key}={actual!r}, expected {expected!r}")
    if tuple(config.get("seeds", [])) != HISTORICAL_32_SEEDS:
        errors.append("historical receipt does not contain exactly seeds 8600-8602")
    if tuple(config.get("modes", [])) != MODES:
        errors.append("historical receipt does not contain lucid/fixed/off in frozen order")
    controller = config.get("controller", {})
    for key in ("delta_target", "kp", "ki", "alpha", "integral_max", "return_floor"):
        if not _same_number(controller.get(key), getattr(args, key)):
            errors.append(f"controller {key} differs from the horizon campaign")
    source_checkpoint = resolved_file(config.get("checkpoint", ""))
    if source_checkpoint != resolved_file(args.checkpoint):
        errors.append("historical source checkpoint path differs")

    arms_by_key: dict[tuple[int, str], tuple[str, dict[str, Any]]] = {}
    for branch_id, arm in payload.get("arms", {}).items():
        key = (int(arm.get("seed", -1)), str(arm.get("mode")))
        if key in arms_by_key:
            errors.append(f"duplicate historical arm {key}")
        arms_by_key[key] = (branch_id, arm)

    artifact_bindings: dict[str, Any] = {}
    command_hashes: dict[str, str] = {}
    for seed in HISTORICAL_32_SEEDS:
        for mode in MODES:
            key = (seed, mode)
            if key not in arms_by_key:
                errors.append(f"missing historical arm seed={seed} mode={mode}")
                continue
            branch_id, arm = arms_by_key[key]
            runtime = payload.get("runtime", {}).get(branch_id, {})
            if runtime.get("exit_code") != 0 or not arm.get("complete"):
                errors.append(f"historical arm did not complete: {branch_id}")
            if not arm.get("checkpoint_exported") or arm.get("actuator_groups_swapped") != 5:
                errors.append(f"historical arm failed actuator/checkpoint audit: {branch_id}")
            if set(arm.get("scalable_terms", [])) != EXPECTED_TERMS:
                errors.append(f"historical arm has wrong scalable terms: {branch_id}")
            if mode == "lucid" and not arm.get("mean_return_observed"):
                errors.append(f"historical LUCID return guard was not live: {branch_id}")

            checkpoint = resolved_file(arm.get("checkpoint", ""))
            capsule = resolved_file(arm.get("capsule", ""))
            artifact_bindings[branch_id] = {
                "checkpoint": bind_file(checkpoint),
                "capsule": bind_file(capsule),
            }
            recorded = payload.get("commands", {}).get(branch_id)
            if not isinstance(recorded, list):
                errors.append(f"historical command missing: {branch_id}")
                continue
            old_checkpoint = command_value(recorded, "checkpoint")
            expected = CC.build_command(
                cc_args(args, 32, checkpoint=old_checkpoint),
                mode,
                seed,
                branch_id,
                checkpoint.parent,
            )
            if normalized_command(recorded) != normalized_command(expected):
                errors.append(
                    f"historical command/config differs from current builder: {branch_id}"
                )
            command_hashes[branch_id] = sha256_json(normalized_command(recorded))

    if errors:
        raise CampaignError(
            "historical 32-iteration compatibility audit failed: " + "; ".join(errors)
        )
    return {
        "receipt_path": str(path),
        "receipt_sha256": file_sha256(path),
        "git_sha": payload.get("git_sha"),
        "git_status_short": payload.get("git_status_short", []),
        "launcher_sha256": payload.get("launcher_sha256"),
        "current_launcher_sha256": file_sha256(REPO / HASH_BOUND_CODE["single_budget_launcher"]),
        "seeds": list(HISTORICAL_32_SEEDS),
        "command_config_compatible": True,
        "same_launcher_bytes": payload.get("launcher_sha256")
        == file_sha256(REPO / HASH_BOUND_CODE["single_budget_launcher"]),
        "command_sha256_by_branch": command_hashes,
        "artifact_bindings": artifact_bindings,
        "limitation": (
            "historical commands and frozen configuration match the current builder, but the "
            "historical dirty-run launcher bytes have a different SHA256; the combined 32-step "
            "cell therefore retains two explicit code-lineage strata"
        ),
    }


def preregistration_payload(
    args: argparse.Namespace,
    paths: Mapping[str, Any],
    specs: Sequence[BranchSpec],
    git: Mapping[str, Any],
    inputs: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_root = Path(paths["campaign_root"])
    branches = []
    for spec in specs:
        command = branch_command(args, spec, campaign_root, attempt=1)
        branches.append(
            {
                **asdict(spec),
                "spec_sha256": spec.spec_sha256,
                "first_attempt_command": command,
                "first_attempt_command_sha256": sha256_json(normalized_command(command)),
            }
        )
    payload = {
        "kind": CAMPAIGN_KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "campaign_id": args.campaign_id,
        "git": dict(git),
        "paths": dict(paths),
        "matrix": {
            "new_training": [
                {"budget_iterations": budget, "seeds": list(seeds)}
                for budget, seeds in HORIZON_MATRIX
            ],
            "historical_32_seeds": list(HISTORICAL_32_SEEDS),
            "modes": list(MODES),
            "new_branch_count": len(specs),
            "new_iteration_count": sum(
                budget * len(seeds) * len(MODES) for budget, seeds in HORIZON_MATRIX
            ),
        },
        "training_config": {
            "checkpoint": str(resolved_file(args.checkpoint)),
            "encoder": str(resolved_file(args.encoder)),
            "num_envs": args.num_envs,
            "warmup_iterations": args.warmup_iterations,
            "max_delay_steps": args.max_delay,
            "max_delay_ms": args.max_delay * 5,
            "delta_target": args.delta_target,
            "kp": args.kp,
            "ki": args.ki,
            "alpha": args.alpha,
            "integral_max": args.integral_max,
            "return_floor": args.return_floor,
            "exp": args.exp,
            "event_preset": "tracking/lucid_curriculum",
        },
        "gpu_gate": {
            "minimum_free_mib": args.min_free_mib,
            "maximum_utilization_pct": args.max_gpu_util_pct,
            "samples": args.idle_samples,
            "sample_interval_seconds": args.idle_sample_seconds,
            "require_zero_compute_processes": True,
            "applied_before_every_branch": True,
            "cooperative_campaign_lock": paths["gpu_lock"],
        },
        "hypotheses": {
            "primary": (
                "At 256 iterations, LUCID full-DR deployment success is non-inferior "
                "to fixed DR while clean deployment success is superior"
            ),
            "mechanistic": "fixed DR clean deployment success decreases with training budget",
            "outcomes_not_in_this_training_campaign": (
                "all claims require separately frozen clean/full-DR evaluator receipts"
            ),
            "noninferiority_margin": None,
            "noninferiority_margin_status": (
                "must be frozen in the evaluator preregistration before any deployment outcome "
                "is opened; this training preregistration does not invent a margin"
            ),
        },
        "input_bindings": dict(inputs),
        "historical_32": dict(historical),
        "branches": branches,
        "verified": [
            "the exact 51-branch, 6912-iteration new-training matrix is frozen",
            "checkpoint, encoder, resolved config, research launchers, and DR configs are hash-bound",
            "historical 32-step commands/configs and output artifacts passed compatibility audit",
            "campaign and receipt paths are reserved exclusively before any GPU query",
        ],
        "not_yet_verified": [
            "any new GPU branch or training outcome",
            "frozen-policy clean/full-DR deployment efficacy",
            "the numerical non-inferiority margin and deployment analysis plan",
            "historical and new 32-step arms share commands/config but not identical launcher bytes",
        ],
    }
    payload["preregistration_sha256"] = sha256_json(payload)
    return payload


def verify_preregistration(payload: Mapping[str, Any]) -> None:
    recorded = payload.get("preregistration_sha256")
    unhashed = dict(payload)
    unhashed.pop("preregistration_sha256", None)
    if recorded != sha256_json(unhashed):
        raise CampaignError("preregistration content hash does not match; it was modified")


def initial_status(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": STATUS_KIND,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": preregistration["campaign_id"],
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "state": "preregistered",
        "last_error": None,
        "branches": {
            row["branch_id"]: {
                "spec_sha256": row["spec_sha256"],
                "state": "pending",
                "attempts": [],
            }
            for row in preregistration["branches"]
        },
    }


def prereg_specs(payload: Mapping[str, Any]) -> list[BranchSpec]:
    return [
        BranchSpec(
            budget_iterations=int(row["budget_iterations"]),
            seed=int(row["seed"]),
            mode=str(row["mode"]),
            order_index=int(row["order_index"]),
            branch_id=str(row["branch_id"]),
        )
        for row in payload["branches"]
    ]


def verify_current_inputs(preregistration: Mapping[str, Any]) -> None:
    current_git = git_identity()
    if current_git != preregistration["git"]:
        raise CampaignError("git identity differs from the immutable preregistration")
    for name, binding in preregistration["input_bindings"].items():
        path = resolved_file(binding["resolved_path"])
        if path.stat().st_size != binding["size_bytes"] or file_sha256(path) != binding["sha256"]:
            raise CampaignError(f"hash-bound input changed after preregistration: {name}")
    historical = preregistration["historical_32"]
    receipt = resolved_file(historical["receipt_path"])
    if file_sha256(receipt) != historical["receipt_sha256"]:
        raise CampaignError("historical 32-step receipt changed after preregistration")
    for branch_id, artifacts in historical["artifact_bindings"].items():
        for label, binding in artifacts.items():
            path = resolved_file(binding["resolved_path"])
            if (
                path.stat().st_size != binding["size_bytes"]
                or file_sha256(path) != binding["sha256"]
            ):
                raise CampaignError(f"historical {label} changed for {branch_id}")


def status_counts(status: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        state: 0 for state in ("pending", "running", "blocked", "failed", "interrupted", "complete")
    }
    for branch in status["branches"].values():
        state = branch["state"]
        counts[state] = counts.get(state, 0) + 1
    return counts


def completed_records(
    status: Mapping[str, Any], specs: Sequence[BranchSpec]
) -> dict[str, dict[str, Any]]:
    output = {}
    for spec in specs:
        branch = status["branches"][spec.branch_id]
        if branch["state"] == "complete":
            output[spec.branch_id] = branch["completed"]
    return output


def training_receipt(
    preregistration: Mapping[str, Any],
    status: Mapping[str, Any],
    specs: Sequence[BranchSpec],
    budget: int,
    *,
    historical_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = [spec for spec in specs if spec.budget_iterations == budget]
    records = completed_records(status, selected)
    arms = {branch_id: record["arm"] for branch_id, record in records.items()}
    runtime = {branch_id: record["runtime"] for branch_id, record in records.items()}
    commands = {branch_id: record["command"] for branch_id, record in records.items()}
    artifact_hashes = {
        branch_id: record["artifact_hashes"] for branch_id, record in records.items()
    }
    expected_seeds = sorted({spec.seed for spec in selected})
    lineage = [preregistration["preregistration_sha256"]]
    if historical_payload is not None:
        for branch_id, arm in historical_payload["arms"].items():
            arms[branch_id] = arm
            runtime[branch_id] = historical_payload["runtime"][branch_id]
            commands[branch_id] = historical_payload["commands"][branch_id]
        expected_seeds = sorted(set(expected_seeds) | set(HISTORICAL_32_SEEDS))
        lineage.append(preregistration["historical_32"]["receipt_sha256"])

    expected = len(expected_seeds) * len(MODES)
    complete = len(arms) == expected and all(arm.get("complete") for arm in arms.values())
    mode_summary = CC.aggregate(arms, list(MODES)) if arms else {}
    receipt = {
        "kind": "lucid_three_arm_training_comparison",
        "schema_version": 2,
        "created_at": now_iso(),
        "campaign_id": preregistration["campaign_id"],
        "experiment_id": f"{preregistration['campaign_id']}_b{budget:04d}",
        "status": "complete" if complete else "in_progress",
        "claim_grade_training_index": complete,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "lineage_sha256": lineage,
        "git_sha": preregistration["git"]["sha"],
        "git_status_short": preregistration["git"]["status_short"],
        "launcher_sha256": preregistration["input_bindings"]["horizon_scaling_launcher"]["sha256"],
        "git_lineage": [
            {
                "stratum": "new_horizon_campaign",
                "git_sha": preregistration["git"]["sha"],
                "git_status_short": preregistration["git"]["status_short"],
            }
        ],
        "launcher_lineage": [
            {
                "stratum": "new_horizon_campaign",
                "horizon_scaling_launcher_sha256": preregistration["input_bindings"][
                    "horizon_scaling_launcher"
                ]["sha256"],
                "single_budget_launcher_sha256": preregistration["input_bindings"][
                    "single_budget_launcher"
                ]["sha256"],
            }
        ],
        "config": {
            "checkpoint": preregistration["training_config"]["checkpoint"],
            "encoder": preregistration["training_config"]["encoder"],
            "num_envs": preregistration["training_config"]["num_envs"],
            "iterations": budget,
            "warmup_iterations": preregistration["training_config"]["warmup_iterations"],
            "seeds": expected_seeds,
            "modes": list(MODES),
            "event_preset": "tracking/lucid_curriculum",
            "max_delay_steps": preregistration["training_config"]["max_delay_steps"],
            "max_delay_ms": preregistration["training_config"]["max_delay_ms"],
            "controller": {
                key: preregistration["training_config"][key]
                for key in (
                    "delta_target",
                    "kp",
                    "ki",
                    "alpha",
                    "integral_max",
                    "return_floor",
                )
            },
        },
        "commands": commands,
        "runtime": runtime,
        "arms": arms,
        "artifact_sha256": artifact_hashes,
        "mode_summary": mode_summary,
        "training_comparison": CC.comparisons(mode_summary) if mode_summary else {},
        "verified": (
            [
                "every indexed arm completed and its artifacts were content-hashed",
                "all five live actuator groups used delayed actuators",
                "all six DR channels were runtime-scalable",
                "LUCID received SONIC objective/rewards for its return guard",
            ]
            if complete
            else []
        ),
        "not_yet_verified": [
            *([] if complete else [f"{expected - len(arms)} training arms"]),
            "frozen-policy clean/full-DR deployment evaluation",
            *(
                [preregistration["historical_32"]["limitation"]]
                if historical_payload is not None
                else []
            ),
        ],
    }
    if historical_payload is not None:
        historical = preregistration["historical_32"]
        receipt["git_lineage"].append(
            {
                "stratum": "historical_32_seeds_8600_8602",
                "git_sha": historical["git_sha"],
                "git_status_short": historical["git_status_short"],
            }
        )
        receipt["launcher_lineage"].append(
            {
                "stratum": "historical_32_seeds_8600_8602",
                "single_budget_launcher_sha256": historical["launcher_sha256"],
                "same_as_current_single_budget_launcher": historical["same_launcher_bytes"],
            }
        )
        receipt["historical_32_artifact_bindings"] = historical["artifact_bindings"]
        receipt["historical_32_command_sha256_by_branch"] = historical[
            "command_sha256_by_branch"
        ]
    return receipt


def index_receipt(
    preregistration: Mapping[str, Any],
    status: Mapping[str, Any],
    receipt_hashes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = status_counts(status)
    complete = counts.get("complete") == preregistration["matrix"]["new_branch_count"]
    return {
        "kind": INDEX_KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "campaign_id": preregistration["campaign_id"],
        "state": status["state"],
        "preregistration_path": preregistration["paths"]["preregistration"],
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "status_path": preregistration["paths"]["status"],
        "status_counts": counts,
        "matrix": preregistration["matrix"],
        "training_receipts": dict(receipt_hashes),
        "evaluator_training_receipt_by_budget": {
            "32": preregistration["paths"]["combined_32_receipt"],
            **{
                str(budget): preregistration["paths"]["budget_receipts"][str(budget)]
                for budget in (64, 128, 256)
            },
        },
        "verified": (
            [
                "all 51 new branches completed with verified artifact hashes",
                "budget-specific receipts expose the legacy evaluator arms/config interface",
                "the 32-step evaluator receipt combines three sealed historical and two new seeds",
            ]
            if complete
            else []
        ),
        "not_yet_verified": [
            *([] if complete else ["the full preregistered training matrix"]),
            "any deployment-evaluation claim",
            "a frozen numerical non-inferiority margin",
        ],
    }


def sync_receipts(
    preregistration: Mapping[str, Any],
    status: dict[str, Any],
    *,
    initialize: bool = False,
) -> None:
    paths = preregistration["paths"]
    specs = prereg_specs(preregistration)
    historical_payload = read_json(preregistration["historical_32"]["receipt_path"])
    status["updated_at"] = now_iso()
    writer = write_json_exclusive if initialize else write_json_atomic
    writer(paths["status"], status)

    hashes: dict[str, dict[str, Any]] = {}
    for budget, _ in HORIZON_MATRIX:
        payload = training_receipt(preregistration, status, specs, budget)
        path = paths["budget_receipts"][str(budget)]
        writer(path, payload)
        hashes[str(budget)] = {
            "path": path,
            "sha256": file_sha256(path),
            "status": payload["status"],
            "seeds": payload["config"]["seeds"],
        }

    combined = training_receipt(
        preregistration,
        status,
        specs,
        32,
        historical_payload=historical_payload,
    )
    writer(paths["combined_32_receipt"], combined)
    hashes["32_combined"] = {
        "path": paths["combined_32_receipt"],
        "sha256": file_sha256(paths["combined_32_receipt"]),
        "status": combined["status"],
        "seeds": combined["config"]["seeds"],
    }

    index = index_receipt(preregistration, status, hashes)
    writer(paths["index"], index)


def create_campaign(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = campaign_paths(args)
    campaign_root = Path(paths["campaign_root"])
    campaign_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        campaign_root.mkdir()
    except FileExistsError as error:
        raise CampaignError(
            f"campaign path already exists: {campaign_root}; use --resume for the frozen campaign"
        ) from error

    try:
        git = git_identity()
        inputs = collect_input_bindings(args)
        historical = audit_historical_32(args)
        specs = build_specs(args.campaign_id)
        preregistration = preregistration_payload(args, paths, specs, git, inputs, historical)
        write_json_exclusive(paths["preregistration"], preregistration)
        status = initial_status(preregistration)
        sync_receipts(preregistration, status, initialize=True)
    except BaseException:
        # The exclusive directory is deliberately retained as a collision tombstone.
        raise
    return preregistration, status


def load_campaign(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = campaign_paths(args)
    preregistration_path = Path(paths["preregistration"])
    if not preregistration_path.is_file():
        raise CampaignError(f"no preregistration exists for --resume: {preregistration_path}")
    preregistration = read_json(preregistration_path)
    verify_preregistration(preregistration)
    if preregistration.get("campaign_id") != args.campaign_id:
        raise CampaignError("campaign-id differs from preregistration")
    if preregistration.get("paths") != paths:
        raise CampaignError("CLI output paths differ from preregistration")
    expected_config = preregistration["training_config"]
    current = {
        "checkpoint": str(resolved_file(args.checkpoint)),
        "encoder": str(resolved_file(args.encoder)),
        "num_envs": args.num_envs,
        "warmup_iterations": args.warmup_iterations,
        "max_delay_steps": args.max_delay,
        "max_delay_ms": args.max_delay * 5,
        "delta_target": args.delta_target,
        "kp": args.kp,
        "ki": args.ki,
        "alpha": args.alpha,
        "integral_max": args.integral_max,
        "return_floor": args.return_floor,
        "exp": args.exp,
        "event_preset": "tracking/lucid_curriculum",
    }
    if current != expected_config:
        raise CampaignError("resume training arguments differ from immutable preregistration")
    historical_path = str(resolved_file(args.historical_32_receipt))
    if historical_path != preregistration["historical_32"]["receipt_path"]:
        raise CampaignError("historical 32-step receipt differs from immutable preregistration")
    current_gpu_gate = {
        "minimum_free_mib": args.min_free_mib,
        "maximum_utilization_pct": args.max_gpu_util_pct,
        "samples": args.idle_samples,
        "sample_interval_seconds": args.idle_sample_seconds,
        "require_zero_compute_processes": True,
        "applied_before_every_branch": True,
        "cooperative_campaign_lock": paths["gpu_lock"],
    }
    if current_gpu_gate != preregistration["gpu_gate"]:
        raise CampaignError("resume GPU gate differs from immutable preregistration")
    verify_current_inputs(preregistration)
    status = read_json(paths["status"])
    if status.get("preregistration_sha256") != preregistration["preregistration_sha256"]:
        raise CampaignError("status receipt belongs to a different preregistration")
    verify_completed_branches(preregistration, status)
    return preregistration, status


def verify_artifact_hashes(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    required = {"training_log", "observer", "curriculum", "capsule", "checkpoint"}
    if set(bindings) != required:
        raise CampaignError(
            f"completed branch artifact set differs: {sorted(bindings)} != {sorted(required)}"
        )
    for label, binding in bindings.items():
        path = resolved_file(binding["path"])
        if path.stat().st_size != binding["size_bytes"] or file_sha256(path) != binding["sha256"]:
            raise CampaignError(f"completed branch artifact changed: {label} at {path}")


def verify_completed_branch(spec: BranchSpec, record: Mapping[str, Any]) -> None:
    if record.get("spec_sha256") != spec.spec_sha256:
        raise CampaignError(f"completed record has wrong spec hash: {spec.branch_id}")
    arm = record.get("arm", {})
    runtime = record.get("runtime", {})
    if (
        runtime.get("exit_code") != 0
        or not arm.get("complete")
        or arm.get("iterations_parsed") != spec.budget_iterations
        or arm.get("actuator_groups_swapped") != 5
        or not arm.get("checkpoint_exported")
        or set(arm.get("scalable_terms", [])) != EXPECTED_TERMS
        or (spec.mode == "lucid" and not arm.get("mean_return_observed"))
    ):
        raise CampaignError(f"completed record fails mechanics audit: {spec.branch_id}")
    verify_artifact_hashes(record.get("artifact_hashes", {}))
    capsule = BC.load_capsule(arm["capsule"], restore_rng=False)
    if (
        capsule["branch_id"] != spec.branch_id
        or int(capsule["global_step"]) != spec.budget_iterations
    ):
        raise CampaignError(f"capsule identity/step mismatch: {spec.branch_id}")


def verify_completed_branches(
    preregistration: Mapping[str, Any], status: Mapping[str, Any]
) -> None:
    specs = prereg_specs(preregistration)
    expected = {spec.branch_id: spec for spec in specs}
    if set(status.get("branches", {})) != set(expected):
        raise CampaignError("status branch set differs from preregistration")
    for branch_id, branch in status["branches"].items():
        if branch.get("spec_sha256") != expected[branch_id].spec_sha256:
            raise CampaignError(f"status spec hash mismatch: {branch_id}")
        if branch.get("state") == "complete":
            verify_completed_branch(expected[branch_id], branch["completed"])


def compute_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    processes = []
    for line in output.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise GpuNotIdleError(f"cannot parse compute-process row: {line!r}")
        processes.append(
            {"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": float(parts[2])}
        )
    return processes


def gpu_idle_sample() -> dict[str, Any]:
    return {"gpu": TP.gpu_snapshot(), "compute_processes": compute_processes()}


def audit_gpu_idle(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    blockers = []
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
            pids = [row["pid"] for row in sample["compute_processes"]]
            blockers.append(f"sample {index}: active compute PIDs {pids}")
        if index + 1 < args.idle_samples:
            time.sleep(args.idle_sample_seconds)
    if blockers:
        raise GpuNotIdleError("GPU idle gate failed: " + "; ".join(blockers))
    return samples


@contextmanager
def gpu_campaign_lock(path: str | Path) -> Iterator[None]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GpuNotIdleError(f"another horizon campaign holds {target}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def artifact_binding(path: Path) -> dict[str, Any]:
    resolved = resolved_file(path)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def summarize_completed_attempt(
    args: argparse.Namespace,
    spec: BranchSpec,
    command: list[str],
    paths: Mapping[str, Path],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    artifact_dir = paths["artifact_dir"]
    observer = artifact_dir / f"observer_{spec.branch_id}.jsonl"
    curriculum_path = artifact_dir / f"curriculum_{spec.branch_id}.jsonl"
    capsule_path = artifact_dir / "capsules" / f"{spec.branch_id}_final.capsule.pt"
    checkpoint_path = artifact_dir / "final_checkpoint.pt"
    arm = LA.summarize_arm(paths["log"], observer, spec.budget_iterations)
    curriculum = CC.read_jsonl(curriculum_path)
    if runtime["exit_code"] == 0 and capsule_path.is_file():
        BC.export_sonic_checkpoint(capsule_path, checkpoint_path)
    arm.update(
        {
            "seed": spec.seed,
            "mode": spec.mode,
            "budget_iterations": spec.budget_iterations,
            "branch_id": spec.branch_id,
            "curriculum_path": str(curriculum_path),
            "curriculum_rows": len(curriculum),
            "final_lambda": curriculum[-1].get("lambda") if curriculum else None,
            "final_integral": curriculum[-1].get("integral") if curriculum else None,
            "mean_return_observed": any(row.get("mean_return") is not None for row in curriculum),
            "return_guard_trips": sum(bool(row.get("guard_tripped")) for row in curriculum),
            "scalable_terms": curriculum[-1].get("scalable_terms", []) if curriculum else [],
            "capsule": str(capsule_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_exported": checkpoint_path.is_file(),
        }
    )
    mechanics_ok = (
        runtime["exit_code"] == 0
        and arm["complete"]
        and arm["actuator_groups_swapped"] == 5
        and arm["checkpoint_exported"]
        and set(arm["scalable_terms"]) == EXPECTED_TERMS
        and (spec.mode != "lucid" or arm["mean_return_observed"])
    )
    required = {
        "training_log": paths["log"],
        "observer": observer,
        "curriculum": curriculum_path,
        "capsule": capsule_path,
        "checkpoint": checkpoint_path,
    }
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        mechanics_ok = False
    hashes = {label: artifact_binding(path) for label, path in required.items() if path.is_file()}
    if mechanics_ok:
        capsule = BC.load_capsule(capsule_path, restore_rng=False)
        mechanics_ok = (
            capsule["branch_id"] == spec.branch_id
            and int(capsule["global_step"]) == spec.budget_iterations
        )
    return {
        "spec_sha256": spec.spec_sha256,
        "command": command,
        "runtime": runtime,
        "arm": arm,
        "artifact_hashes": hashes,
        "mechanics_ok": mechanics_ok,
        "missing_artifacts": missing,
    }


def format_error(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def run_one_branch(
    args: argparse.Namespace,
    preregistration: Mapping[str, Any],
    status: dict[str, Any],
    spec: BranchSpec,
) -> bool:
    branch = status["branches"][spec.branch_id]
    attempt_number = len(branch["attempts"]) + 1
    paths = attempt_paths(Path(preregistration["paths"]["campaign_root"]), spec, attempt_number)
    paths["root"].mkdir(parents=True, exist_ok=False)
    paths["artifact_dir"].mkdir()
    command = branch_command(
        args, spec, Path(preregistration["paths"]["campaign_root"]), attempt_number
    )
    attempt: dict[str, Any] = {
        "attempt": attempt_number,
        "state": "running",
        "started_at": now_iso(),
        "paths": {name: str(path) for name, path in paths.items()},
        "command": command,
    }
    branch["attempts"].append(attempt)
    branch["state"] = "running"
    status["state"] = "running"
    status["last_error"] = None
    sync_receipts(preregistration, status)
    try:
        attempt["gpu_idle_samples"] = audit_gpu_idle(args)
        sync_receipts(preregistration, status)
        runtime = LA.run_arm(command, paths["log"], args.min_free_mib)
        completed = summarize_completed_attempt(args, spec, command, paths, runtime)
        attempt["finished_at"] = now_iso()
        attempt["runtime"] = runtime
        if not completed["mechanics_ok"]:
            error = CampaignError(
                f"branch mechanics audit failed: {spec.branch_id}; "
                f"missing={completed['missing_artifacts']}"
            )
            attempt["state"] = "failed"
            attempt["error"] = format_error(error)
            branch["state"] = "failed"
            status["state"] = "failed"
            status["last_error"] = attempt["error"]
            sync_receipts(preregistration, status)
            return False
        attempt["state"] = "complete"
        branch["state"] = "complete"
        branch["completed"] = completed
        sync_receipts(preregistration, status)
        return True
    except GpuNotIdleError as error:
        attempt["state"] = "blocked"
        attempt["finished_at"] = now_iso()
        attempt["error"] = format_error(error)
        branch["state"] = "blocked"
        status["state"] = "blocked"
        status["last_error"] = attempt["error"]
        sync_receipts(preregistration, status)
        raise
    except (KeyboardInterrupt, CampaignInterrupted) as error:
        attempt["state"] = "interrupted"
        attempt["finished_at"] = now_iso()
        attempt["error"] = format_error(error)
        branch["state"] = "interrupted"
        status["state"] = "interrupted"
        status["last_error"] = attempt["error"]
        sync_receipts(preregistration, status)
        raise
    except BaseException as error:
        attempt["state"] = "failed"
        attempt["finished_at"] = now_iso()
        attempt["error"] = format_error(error)
        branch["state"] = "failed"
        status["state"] = "failed"
        status["last_error"] = attempt["error"]
        sync_receipts(preregistration, status)
        raise


def recover_stale_running(status: dict[str, Any]) -> None:
    for branch in status["branches"].values():
        if branch["state"] != "running":
            continue
        branch["state"] = "interrupted"
        if branch["attempts"] and branch["attempts"][-1]["state"] == "running":
            branch["attempts"][-1]["state"] = "interrupted"
            branch["attempts"][-1]["finished_at"] = now_iso()
            branch["attempts"][-1]["error"] = {
                "type": "StaleRunningAttempt",
                "message": "previous orchestrator stopped without finalizing this attempt",
                "traceback": "",
            }


def run_campaign(
    args: argparse.Namespace,
    preregistration: Mapping[str, Any],
    status: dict[str, Any],
) -> int:
    recover_stale_running(status)
    sync_receipts(preregistration, status)
    specs = prereg_specs(preregistration)
    try:
        with gpu_campaign_lock(preregistration["paths"]["gpu_lock"]):
            for spec in specs:
                if status["branches"][spec.branch_id]["state"] == "complete":
                    continue
                if not run_one_branch(args, preregistration, status, spec):
                    return 1
    except GpuNotIdleError as error:
        if status["state"] != "blocked":
            status["state"] = "blocked"
            status["last_error"] = format_error(error)
            sync_receipts(preregistration, status)
        raise
    status["state"] = "complete"
    status["last_error"] = None
    sync_receipts(preregistration, status)
    return 0


def _sigterm(_signum: int, _frame: Any) -> None:
    raise CampaignInterrupted("received SIGTERM")


def print_plan(preregistration: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    summary = {
        "campaign_id": preregistration["campaign_id"],
        "preregistration": preregistration["paths"]["preregistration"],
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "training_index": preregistration["paths"]["index"],
        "matrix": preregistration["matrix"],
        "status": status["state"],
        "status_counts": status_counts(status),
        "first_branch": preregistration["branches"][0],
        "last_branch": preregistration["branches"][-1],
    }
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.resume:
            preregistration, status = load_campaign(args)
        else:
            preregistration, status = create_campaign(args)
        print_plan(preregistration, status)
        if not args.execute:
            print("dry run frozen; inspect receipts, then pass --resume --execute")
            return 0
        signal.signal(signal.SIGTERM, _sigterm)
        return run_campaign(args, preregistration, status)
    except GpuNotIdleError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (KeyboardInterrupt, CampaignInterrupted):
        return 130
    except CampaignError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
