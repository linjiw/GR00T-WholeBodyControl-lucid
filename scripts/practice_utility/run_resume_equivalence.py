#!/usr/bin/env python3
"""Run the preregistered uninterrupted-vs-capsule-resume hygiene check."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC  # noqa: E402
from gear_sonic.research.practice_utility import run_log as RL
from scripts.practice_utility import run_latency_ab as LA  # noqa: E402
from scripts.practice_utility import run_throughput_probe as TP

CAPSULE_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"
RESUME_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleResumeCallback"
REWARD_FLOOR = 0.0333
LENGTH_FLOOR = 0.0314
TRAILING_WINDOW = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--total-iterations", type=int, default=20)
    parser.add_argument("--split-iteration", type=int, default=10)
    parser.add_argument("--seed", type=int, default=8500)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/artifacts/resume_equivalence"),
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
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.split_iteration < args.total_iterations:
        parser.error("split iteration must lie strictly inside the full run")
    if args.total_iterations - args.split_iteration < TRAILING_WINDOW:
        parser.error("resumed segment must contain at least four iterations")
    return args


def common_command(args, checkpoint: str, iterations: int) -> list[str]:
    return [
        sys.executable,
        str(REPO / "gear_sonic" / "train_agent_trl.py"),
        f"+exp={args.exp}",
        f"checkpoint={checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={iterations}",
        "++algo.config.save_interval=100000",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/motion_lib_bones_seed/smpl_filtered",
    ]


def build_full_command(args, capsule_dir: Path, branch_id: str) -> list[str]:
    command = common_command(args, args.checkpoint, args.total_iterations)
    command.extend(
        [
            f"++callbacks.practice_capsule._target_={CAPSULE_CALLBACK}",
            "++callbacks.practice_capsule.enabled=true",
            f"++callbacks.practice_capsule.capsule_dir={capsule_dir}",
            "++callbacks.practice_capsule.pair_id=resume_equivalence",
            "++callbacks.practice_capsule.role=control",
            f"++callbacks.practice_capsule.branch_id={branch_id}",
            f"++callbacks.practice_capsule.horizons.split={args.split_iteration}",
        ]
    )
    return command


def build_resume_command(args, checkpoint: Path, capsule: Path) -> list[str]:
    # SONIC restores ``global_step`` from the checkpoint and the HF flow callback
    # compares it to max_steps. Keep the absolute target here: setting only the
    # remaining count would make step split+1 immediately exceed max_steps.
    command = common_command(args, str(checkpoint), args.total_iterations)
    command.extend(
        [
            "+resume=true",
            f"++callbacks.practice_resume._target_={RESUME_CALLBACK}",
            "++callbacks.practice_resume.enabled=true",
            f"++callbacks.practice_resume.capsule_path={capsule}",
        ]
    )
    return command


def trailing_mean(run: RL.RunLog, metric: str) -> float | None:
    values = [value for _, value in sorted(run.series(metric).items())]
    if len(values) < TRAILING_WINDOW:
        return None
    return statistics.fmean(values[-TRAILING_WINDOW:])


def relative_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left == 0:
        return None
    return (right - left) / abs(left)


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"resume_equivalence_ne{args.num_envs}_{stamp}"
    artifact_dir = args.artifact_root / experiment_id
    capsule_dir = artifact_dir / "capsules"
    branch_id = f"{experiment_id}_full"
    full_command = build_full_command(args, capsule_dir, branch_id)
    print("[uninterrupted]\n" + "\n".join(full_command))
    if not args.execute:
        print("resume command is generated after capsule export")
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    full_log = args.log_dir / f"{experiment_id}_full.log"
    runtime: dict[str, Any] = {
        "uninterrupted": LA.run_arm(full_command, full_log, args.min_free_mib)
    }
    runtime["uninterrupted"]["log_path"] = str(full_log)

    capsule = capsule_dir / f"{branch_id}_split.capsule.pt"
    exported = artifact_dir / "split_checkpoint.pt"
    if runtime["uninterrupted"]["exit_code"] == 0 and capsule.exists():
        BC.export_sonic_checkpoint(capsule, exported)
    resume_command = build_resume_command(args, exported, capsule)
    print("[resumed]\n" + "\n".join(resume_command))
    resume_log = args.log_dir / f"{experiment_id}_resumed.log"
    if exported.exists():
        runtime["resumed"] = LA.run_arm(resume_command, resume_log, args.min_free_mib)
    else:
        runtime["resumed"] = {
            "exit_code": None,
            "error": "split capsule was not produced/exported",
        }
    runtime["resumed"]["log_path"] = str(resume_log)

    full = RL.parse_run_log(full_log)
    resumed = (
        RL.parse_run_log(resume_log) if resume_log.exists() else RL.RunLog(str(resume_log), [])
    )
    exact = RL.compare_runs(
        full,
        resumed,
        tolerance=0.0,
        skip_iterations=args.split_iteration + 1,
    ).to_dict()
    trailing = {}
    for metric in ("Mean rewards", "Mean length", "Mean entropy"):
        left = trailing_mean(full, metric)
        right = trailing_mean(resumed, metric)
        trailing[metric] = {
            "uninterrupted": left,
            "resumed": right,
            "relative_delta": relative_delta(left, right),
        }
    reward_delta = trailing["Mean rewards"]["relative_delta"]
    length_delta = trailing["Mean length"]["relative_delta"]
    equivalent = bool(
        reward_delta is not None
        and length_delta is not None
        and abs(reward_delta) <= REWARD_FLOOR
        and abs(length_delta) <= LENGTH_FLOOR
    )
    resume_text = resume_log.read_text(errors="replace") if resume_log.exists() else ""
    mechanism_ok = bool(
        runtime["uninterrupted"]["exit_code"] == 0
        and runtime["resumed"]["exit_code"] == 0
        and len(full.iterations) == args.total_iterations
        and len(resumed.iterations) == args.total_iterations - args.split_iteration
        and f"restored capsule RNG at step {args.split_iteration}" in resume_text
    )
    verified = []
    if mechanism_ok:
        verified.extend(
            [
                "uninterrupted and resumed arms exited 0 and parsed every requested iteration",
                "capsule exported policy/value/optimizer/trainer/environment state at the split",
                "resumed arm restored capsule RNG immediately before its fresh environment reset",
            ]
        )
    if equivalent:
        verified.append(
            "resumed last-4 reward and length remain within settled-origin one-sigma floors"
        )
    not_yet = []
    if not mechanism_ok:
        not_yet.append("resume mechanism incomplete; no equivalence verdict")
    elif not equivalent:
        not_yet.append(
            "resume equivalence failed the preregistered 3.33% reward / 3.14% length rule"
        )
    receipt = {
        "kind": "lucid_capsule_resume_equivalence",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "launcher_sha256": source_sha256(),
        "config": {
            "num_envs": args.num_envs,
            "total_iterations": args.total_iterations,
            "split_iteration": args.split_iteration,
            "resumed_iterations": args.total_iterations - args.split_iteration,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "trailing_window": TRAILING_WINDOW,
            "equivalence_rule": {
                "reward_absolute_relative_delta_lte": REWARD_FLOOR,
                "length_absolute_relative_delta_lte": LENGTH_FLOOR,
                "basis": "settled-origin last-4 cross-seed one-sigma floors",
            },
        },
        "commands": {
            "uninterrupted": full_command,
            "resumed": resume_command,
        },
        "artifacts": {
            "capsule": str(capsule),
            "exported_checkpoint": str(exported),
        },
        "runtime": runtime,
        "comparison": {
            "exact_post_split": exact,
            "last4": trailing,
            "resume_equivalent": equivalent,
            "restart_note": (
                "SONIC resets all episodes at train start, so exact post-split identity is not expected"
            ),
        },
        "verified": verified,
        "not_yet_verified": not_yet,
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["comparison"], indent=2))
    print(f"receipt {receipt_path}")
    return 0 if mechanism_ok and equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
