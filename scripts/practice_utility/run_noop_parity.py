#!/usr/bin/env python3
"""Compare native SONIC with research callbacks installed but disabled."""

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

from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from scripts.practice_utility import run_latency_ab as LA  # noqa: E402
from scripts.practice_utility import run_throughput_probe as TP
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

OBSERVER = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"
CURRICULUM = "gear_sonic.research.practice_utility.dr_curriculum.LucidCurriculumCallback"
ARMS = ("native", "research_disabled")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LUCID_ROOT / "outputs",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=LUCID_ROOT / "manifests",
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def build_command(args, arm: str) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm {arm!r}")
    command = [
        sys.executable,
        str(REPO / "gear_sonic" / "train_agent_trl.py"),
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++algo.config.save_interval=100000",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/motion_lib_bones_seed/smpl_filtered",
    ]
    if arm == "research_disabled":
        command.extend(
            [
                f"++callbacks.practice_observer._target_={OBSERVER}",
                "++callbacks.practice_observer.enabled=false",
                f"++callbacks.lucid_curriculum._target_={CURRICULUM}",
                "++callbacks.lucid_curriculum.enabled=false",
            ]
        )
    return command


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def series(run: RL.RunLog) -> dict[str, dict[int, float]]:
    return {key: run.series(key) for key in RL.PARITY_KEYS}


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"noop_parity_ne{args.num_envs}_{stamp}"
    commands = {arm: build_command(args, arm) for arm in ARMS}
    for arm, command in commands.items():
        print(f"[{arm}]\n" + "\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, Any] = {}
    runs: dict[str, RL.RunLog] = {}
    for arm in ARMS:
        log_path = args.log_dir / f"{experiment_id}_{arm}.log"
        runtime[arm] = LA.run_arm(commands[arm], log_path, args.min_free_mib)
        runtime[arm]["log_path"] = str(log_path)
        runs[arm] = RL.parse_run_log(log_path)

    comparison = RL.compare_runs(runs["native"], runs["research_disabled"], tolerance=0.0).to_dict()
    complete = all(
        runtime[arm]["exit_code"] == 0 and len(runs[arm].iterations) == args.iterations
        for arm in ARMS
    )
    verified = complete and comparison["passes"]
    receipt = {
        "kind": "lucid_research_disabled_noop_parity",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "launcher_sha256": source_sha256(),
        "config": {
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "arm_order": list(ARMS),
            "tolerance": 0.0,
        },
        "commands": commands,
        "runtime": runtime,
        "arms": {
            arm: {
                "iterations_parsed": len(runs[arm].iterations),
                "training": series(runs[arm]),
            }
            for arm in ARMS
        },
        "comparison": comparison,
        "verified": (
            [
                "both arms exited 0 and parsed every requested iteration",
                "disabled observer and curriculum callbacks are exact no-ops on printed training metrics",
            ]
            if verified
            else []
        ),
        "not_yet_verified": ["resume equivalence"],
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"receipt {receipt_path}")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
