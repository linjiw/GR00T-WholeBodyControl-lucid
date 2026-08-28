#!/usr/bin/env python3
"""Verify two SONIC branches restarted from one capsule are identical."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC  # noqa: E402
from gear_sonic.research.practice_utility import run_log as RL
from scripts.practice_utility import run_latency_ab as LA  # noqa: E402
from scripts.practice_utility import run_resume_equivalence as RE
from scripts.practice_utility import run_throughput_probe as TP
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--total-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8500)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument(
        "--receipt-dir", type=Path, default=LUCID_ROOT / "manifests"
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def build_command(args) -> list[str]:
    return RE.build_resume_command(args, args.checkpoint, args.capsule)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    print("[branch_a/branch_b identical command]\n" + "\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0
    if not args.checkpoint.is_file() or not args.capsule.is_file():
        raise FileNotFoundError("checkpoint and capsule must both exist")

    capsule_payload = BC.load_capsule(args.capsule, restore_rng=False)
    split_step = int(capsule_payload["global_step"])
    expected_iterations = args.total_iterations - split_step
    if expected_iterations < 1:
        raise ValueError("total iterations must be greater than the capsule step")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"restart_pair_equivalence_ne{args.num_envs}_{stamp}"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, Any] = {}
    parsed: dict[str, RL.RunLog] = {}
    commands = {}
    for label in ("branch_a", "branch_b"):
        commands[label] = list(command)
        log_path = args.log_dir / f"{experiment_id}_{label}.log"
        runtime[label] = LA.run_arm(command, log_path, args.min_free_mib)
        runtime[label]["log_path"] = str(log_path)
        parsed[label] = RL.parse_run_log(log_path)

    comparison = RL.compare_runs(parsed["branch_a"], parsed["branch_b"], tolerance=0.0).to_dict()
    restored_marker = f"restored capsule RNG at step {split_step}"
    mechanism_ok = all(
        runtime[label]["exit_code"] == 0
        and len(parsed[label].iterations) == expected_iterations
        and restored_marker in Path(runtime[label]["log_path"]).read_text(errors="replace")
        for label in runtime
    )
    equivalent = bool(mechanism_ok and comparison["passes"])
    receipt = {
        "kind": "lucid_symmetric_restart_equivalence",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "launcher_sha256": source_sha256(),
        "config": {
            "num_envs": args.num_envs,
            "seed": args.seed,
            "capsule_global_step": split_step,
            "absolute_total_iterations": args.total_iterations,
            "expected_iterations_per_branch": expected_iterations,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "capsule": str(args.capsule.resolve()),
            "capsule_sha256": sha256(args.capsule),
            "identity_tolerance": 0.0,
            "estimand_boundary": "two symmetric fresh resets from one capsule",
        },
        "commands": commands,
        "runtime": runtime,
        "comparison": {**comparison, "restart_pair_equivalent": equivalent},
        "verified": (
            [
                "both branches restored the same capsule RNG immediately before reset",
                "both branches parsed every requested post-capsule iteration",
                "all printed parity metrics match exactly after symmetric restart",
            ]
            if equivalent
            else []
        ),
        "not_yet_verified": (
            []
            if equivalent
            else ["symmetric restart identity did not pass; causal branch gate remains blocked"]
        ),
        "unsupported_contract": (
            "uninterrupted-vs-resumed live-trajectory identity; SONIC does not checkpoint "
            "simulator and episode state"
        ),
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["comparison"], indent=2))
    print(f"receipt {receipt_path}")
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
