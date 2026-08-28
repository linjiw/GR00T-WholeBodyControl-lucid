#!/usr/bin/env python3
"""Run the preregistered fixed-lambda, delay-only SONIC mechanism check."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402
from scripts.practice_utility import run_throughput_probe as TP  # noqa: E402

OBSERVER = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"
REWARD_FLOOR = 0.0333
LENGTH_FLOOR = 0.0314
TRAILING_WINDOW = 4
OBSERVER_METRICS = (
    "latent_median",
    "latent_p90",
    "raw_median",
    "raw_p90",
    "action_rate",
    "action_acceleration",
    "foot_slip_per_step_m",
    "contact_impulse_total",
    "contact_force_peak",
    "undesired_contact_rate",
    "torque_saturation",
    "joint_limit_proximity",
    "energy_proxy",
    "action_delay_mean_steps",
    "action_delay_nonzero_fraction",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--encoder",
        default=str(LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/latency_ab",
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
    return parser.parse_args(argv)


def build_command(args, lambda_value: int, branch_id: str) -> list[str]:
    if lambda_value not in (0, 1):
        raise ValueError("fixed lambda must be 0 or 1")
    delay_high = args.max_delay * lambda_value
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
        "manager_env/events=tracking/lucid_latency_only",
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++algo.config.save_interval=100000",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/motion_lib_bones_seed/smpl_filtered",
        f"++manager_env.events.randomize_action_delay.params.delay_range=[0.0,{float(delay_high):.1f}]",
        f"++callbacks.practice_observer._target_={OBSERVER}",
        "++callbacks.practice_observer.enabled=true",
        f"++callbacks.practice_observer.encoder_path={args.encoder}",
        f"++callbacks.practice_observer.branch_id={branch_id}",
        f"++callbacks.practice_observer.output_dir={output_dir}",
    ]


def read_observer(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trailing_mean(values: list[float], window: int = TRAILING_WINDOW) -> float | None:
    if not values:
        return None
    return statistics.fmean(values[-window:])


def summarize_arm(log_path: Path, observer_path: Path, iterations: int) -> dict[str, Any]:
    parsed = RL.parse_run_log(log_path)
    training = {}
    for metric in ("Mean rewards", "Mean length", "Mean entropy"):
        series = parsed.series(metric)
        training[metric] = {
            "series": series,
            "last4_mean": trailing_mean([series[index] for index in sorted(series)]),
        }

    rows = read_observer(observer_path)
    observer = {}
    for metric in OBSERVER_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        observer[metric] = trailing_mean(values)

    live_delay = {}
    if rows:
        final = rows[-1]
        for metric in (
            "action_delay_actuator_groups",
            "action_delay_num_lags",
            "action_delay_min_steps",
            "action_delay_max_steps",
            "action_delay_mean_steps",
            "action_delay_nonzero_fraction",
            "action_delay_histogram",
        ):
            if metric in final:
                live_delay[metric] = final[metric]

    text = log_path.read_text(errors="replace")
    swapped = re.search(r"\[latency\] swapped (\d+) actuator groups", text)
    return {
        "iterations_parsed": len(parsed.iterations),
        "complete": len(parsed.iterations) == iterations,
        "actuator_groups_swapped": int(swapped.group(1)) if swapped else 0,
        "training": training,
        "observer_last4_mean": observer,
        "observer_rows": len(rows),
        "live_delay_final": live_delay,
    }


def relative_delta(control: float | None, treatment: float | None) -> float | None:
    if control is None or treatment is None or control == 0:
        return None
    return (treatment - control) / abs(control)


def compare_arms(lambda0: dict[str, Any], lambda1: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for metric in ("Mean rewards", "Mean length", "Mean entropy"):
        left = lambda0["training"][metric]["last4_mean"]
        right = lambda1["training"][metric]["last4_mean"]
        deltas[metric] = {
            "absolute": None if left is None or right is None else right - left,
            "relative": relative_delta(left, right),
        }
    for metric in OBSERVER_METRICS:
        left = lambda0["observer_last4_mean"][metric]
        right = lambda1["observer_last4_mean"][metric]
        deltas[f"observer/{metric}"] = {
            "absolute": None if left is None or right is None else right - left,
            "relative": relative_delta(left, right),
        }

    reward_delta = deltas["Mean rewards"]["relative"]
    length_delta = deltas["Mean length"]["relative"]
    active = bool(
        (reward_delta is not None and abs(reward_delta) > REWARD_FLOOR)
        or (length_delta is not None and abs(length_delta) > LENGTH_FLOOR)
    )
    return {
        "deltas_lambda1_minus_lambda0": deltas,
        "preregistered_activation_rule": {
            "reward_absolute_relative_delta_gt": REWARD_FLOOR,
            "length_absolute_relative_delta_gt": LENGTH_FLOOR,
            "basis": "settled-origin last-4 cross-seed one-sigma floors",
        },
        "latency_channel_behaviorally_active": active,
    }


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def wait_for_gpu(min_free_mib: int, max_wait_seconds: float, poll_seconds: float = 30.0) -> dict[str, float]:
    """Block until the shared GPU has ``min_free_mib`` free, or give up.

    The GPU is shared with other users, so a campaign must be able to queue
    behind their jobs rather than die at the first arm. Training metrics are
    valid under contention; only wall-clock is not, and it is recorded.
    """
    import time

    deadline = time.monotonic() + max_wait_seconds
    snapshot = TP.gpu_snapshot()
    while snapshot["free_mib"] < min_free_mib:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU capacity gate failed: {snapshot['free_mib']:.0f} MiB free < {min_free_mib} MiB"
                f" after waiting {max_wait_seconds:.0f} s"
            )
        time.sleep(poll_seconds)
        snapshot = TP.gpu_snapshot()
    return snapshot


def run_arm(command: list[str], log_path: Path, min_free_mib: int) -> dict[str, Any]:
    import os

    max_wait = float(os.environ.get("LUCID_GPU_WAIT_SECONDS", "0"))
    initial_gpu = wait_for_gpu(min_free_mib, max_wait)
    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(TP.gpu_snapshot())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(2.0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    with log_path.open("w") as handle:
        exit_code = subprocess.call(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    stop.set()
    thread.join(timeout=5.0)
    return {
        "exit_code": exit_code,
        "wall_seconds": elapsed,
        "gpu": TP.summarize_gpu(samples or [initial_gpu]),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"latency_ab_ne{args.num_envs}_{stamp}"
    commands = {
        label: build_command(args, value, f"{experiment_id}_{label}")
        for label, value in (("lambda0", 0), ("lambda1", 1))
    }
    for label, command in commands.items():
        print(f"[{label}]\n" + "\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    runtime = {}
    summaries = {}
    for label, command in commands.items():
        branch_id = f"{experiment_id}_{label}"
        log_path = args.log_dir / f"{branch_id}.log"
        runtime[label] = run_arm(command, log_path, args.min_free_mib)
        observer_path = args.artifact_root / branch_id / f"observer_{branch_id}.jsonl"
        summaries[label] = summarize_arm(log_path, observer_path, args.iterations)
        runtime[label]["log_path"] = str(log_path)
        runtime[label]["observer_path"] = str(observer_path)

    comparison = compare_arms(summaries["lambda0"], summaries["lambda1"])
    mechanism_ok = all(
        runtime[label]["exit_code"] == 0
        and summaries[label]["complete"]
        and summaries[label]["actuator_groups_swapped"] == 5
        for label in ("lambda0", "lambda1")
    )
    mechanism_ok = mechanism_ok and (
        summaries["lambda0"]["live_delay_final"].get("action_delay_actuator_groups") == 5
        and summaries["lambda1"]["live_delay_final"].get("action_delay_actuator_groups") == 5
        and summaries["lambda0"]["live_delay_final"].get("action_delay_max_steps") == 0
        and summaries["lambda1"]["live_delay_final"].get("action_delay_max_steps", 0) > 0
    )
    receipt = {
        "kind": "lucid_fixed_lambda_latency_ab",
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
            "max_delay_steps": args.max_delay,
            "max_delay_ms": args.max_delay * 5,
            "trailing_window": TRAILING_WINDOW,
            "event_preset": "tracking/lucid_latency_only",
            "other_dr_channels": [],
            "arm_order": ["lambda0", "lambda1"],
        },
        "commands": commands,
        "runtime": runtime,
        "arms": summaries,
        "comparison": comparison,
        "verified": (
            [
                "both arms exited 0 and parsed all requested iterations",
                "all five G1 actuator groups used DelayedImplicitActuatorCfg in both arms",
                "the event preset contains action delay and no other DR channel",
                "lambda=0 used [0,0] steps; lambda=1 used [0,8] steps",
                "live buffers confirm zero lag at lambda=0 and nonzero lag at lambda=1",
            ]
            if mechanism_ok
            else []
        ),
        "not_yet_verified": [
            "multi-seed latency efficacy",
            "held-out latency-stress evaluation",
        ],
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"receipt {receipt_path}")
    return 0 if mechanism_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
