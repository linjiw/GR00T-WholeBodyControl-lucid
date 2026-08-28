#!/usr/bin/env python3
"""Verify that lucid and off arms are identical until LUCID changes lambda."""

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
MODES = ("lucid", "off")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8300)
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--encoder",
        default=str(LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/warmup_parity",
    )
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
    args = parser.parse_args(argv)
    if args.iterations < args.warmup_iterations + 2:
        parser.error("iterations must include warmup plus two post-warmup rollouts")
    return args


def build_command(args, mode: str, branch_id: str) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}")
    output_dir = args.artifact_root / branch_id
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "train_with_delay.py"),
        "--max-delay",
        str(args.max_delay),
        "--",
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={args.seed}",
        "manager_env/events=tracking/lucid_curriculum",
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++algo.config.save_interval=100000",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/motion_lib_bones_seed/smpl_filtered",
        f"++callbacks.practice_observer._target_={OBSERVER}",
        "++callbacks.practice_observer.enabled=true",
        f"++callbacks.practice_observer.encoder_path={args.encoder}",
        f"++callbacks.practice_observer.branch_id={branch_id}",
        f"++callbacks.practice_observer.output_dir={output_dir}",
        f"++callbacks.lucid_curriculum._target_={CURRICULUM}",
        "++callbacks.lucid_curriculum.enabled=true",
        f'++callbacks.lucid_curriculum.mode="{mode}"',
        f"++callbacks.lucid_curriculum.observer_branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.output_dir={output_dir}",
        "++callbacks.lucid_curriculum.delta_target=0.778",
        "++callbacks.lucid_curriculum.alpha=0.05",
        "++callbacks.lucid_curriculum.return_floor=1.0",
        f"++callbacks.lucid_curriculum.warmup_iterations={args.warmup_iterations}",
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare_prefix(left: RL.RunLog, right: RL.RunLog, count: int) -> dict[str, Any]:
    left_prefix = RL.RunLog(left.path, left.iterations[:count])
    right_prefix = RL.RunLog(right.path, right.iterations[:count])
    return RL.compare_runs(left_prefix, right_prefix, tolerance=0.0).to_dict()


def summarize_arm(log_path: Path, artifact_dir: Path, branch_id: str) -> dict[str, Any]:
    log = RL.parse_run_log(log_path)
    curriculum_path = artifact_dir / f"curriculum_{branch_id}.jsonl"
    observer_path = artifact_dir / f"observer_{branch_id}.jsonl"
    return {
        "log": log,
        "curriculum_path": str(curriculum_path),
        "curriculum": read_jsonl(curriculum_path),
        "observer_path": str(observer_path),
        "observer": read_jsonl(observer_path),
    }


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"curriculum_warmup_parity_ne{args.num_envs}_{stamp}"
    branch_ids = {mode: f"{experiment_id}_{mode}" for mode in MODES}
    commands = {mode: build_command(args, mode, branch_ids[mode]) for mode in MODES}
    for mode, command in commands.items():
        print(f"[{mode}]\n" + "\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    parsed: dict[str, RL.RunLog] = {}
    for mode in MODES:
        branch_id = branch_ids[mode]
        log_path = args.log_dir / f"{branch_id}.log"
        runtime[mode] = LA.run_arm(commands[mode], log_path, args.min_free_mib)
        summary = summarize_arm(log_path, args.artifact_root / branch_id, branch_id)
        parsed[mode] = summary.pop("log")
        summary["iterations_parsed"] = len(parsed[mode].iterations)
        arms[mode] = summary
        runtime[mode]["log_path"] = str(log_path)

    treatment_free_rollouts = args.warmup_iterations + 1
    prefix = compare_prefix(parsed["lucid"], parsed["off"], treatment_free_rollouts)
    warmup_rows_ok = all(
        len(arms[mode]["curriculum"]) >= args.warmup_iterations + 1
        and all(
            row.get("warmup_hold") is True and row.get("lambda") == 0.0
            for row in arms[mode]["curriculum"][: args.warmup_iterations]
        )
        for mode in MODES
    )
    treatment_rows_ok = (
        arms["lucid"]["curriculum"][args.warmup_iterations].get("lambda", 0.0) > 0.0
        and arms["off"]["curriculum"][args.warmup_iterations].get("lambda") == 0.0
        if warmup_rows_ok
        else False
    )
    complete = all(
        runtime[mode]["exit_code"] == 0 and len(parsed[mode].iterations) == args.iterations
        for mode in MODES
    )
    verified = complete and warmup_rows_ok and treatment_rows_ok and prefix["passes"]

    serializable_arms = {}
    for mode in MODES:
        serializable_arms[mode] = {
            **arms[mode],
            "training": {key: parsed[mode].series(key) for key in RL.PARITY_KEYS},
        }
    receipt = {
        "kind": "lucid_curriculum_warmup_parity",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "launcher_sha256": source_sha256(),
        "config": {
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "treatment_free_rollouts": treatment_free_rollouts,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "event_preset": "tracking/lucid_curriculum",
            "arm_order": list(MODES),
        },
        "commands": commands,
        "runtime": runtime,
        "arms": serializable_arms,
        "comparison_before_treatment": prefix,
        "verified": (
            [
                "both arms exited 0 and parsed every requested iteration",
                "lucid and off applied lambda=0 throughout warmup",
                "all printed training metrics matched exactly before treatment",
                "lucid raised lambda after warmup while off remained at zero",
            ]
            if verified
            else []
        ),
        "not_yet_verified": [
            "multi-seed curriculum efficacy",
            "resume equivalence",
            "held-out checkpoint evaluation",
        ],
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(prefix, indent=2))
    print(f"receipt {receipt_path}")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
