#!/usr/bin/env python3
"""Run a bounded SONIC throughput probe and write an auditable receipt.

The first PPO iteration is warm-up (CUDA graph capture and allocator setup), so
the reported steady-state metric is the median over subsequent iterations.  Two
variants are supported: ``native`` measures the normal continuation path, while
``observer`` enables the frozen LUCID gap and physical-quality probe used by the
screening campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402

OBSERVER = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", "observer"), default="observer")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--motion-file", default="data/motion_lib_bones_seed/robot_filtered")
    parser.add_argument("--smpl-motion-file", default="data/motion_lib_bones_seed/smpl_filtered")
    parser.add_argument(
        "--encoder",
        default="/data/robotixx/lucid-sonic/artifacts/lucid_encoder_debug512.pt",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/artifacts/throughput_idle"),
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
    return parser.parse_args(argv)


def gpu_snapshot() -> dict[str, float]:
    query = "memory.total,memory.used,memory.free,utilization.gpu,utilization.memory"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [float(value.strip()) for value in output.split(",")]
    keys = ("total_mib", "used_mib", "free_mib", "gpu_util_pct", "memory_util_pct")
    return dict(zip(keys, values, strict=True))


def summarize_gpu(samples: list[dict[str, float]]) -> dict[str, float | int]:
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "start_free_mib": samples[0]["free_mib"],
        "min_free_mib": min(sample["free_mib"] for sample in samples),
        "max_used_mib": max(sample["used_mib"] for sample in samples),
        "max_gpu_util_pct": max(sample["gpu_util_pct"] for sample in samples),
    }


def build_command(args, run_id: str) -> list[str]:
    overrides = [
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++algo.config.save_interval=100000",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
    ]
    if args.variant == "observer":
        output_dir = args.artifact_root / run_id
        overrides.extend(
            [
                f"++callbacks.practice_observer._target_={OBSERVER}",
                "++callbacks.practice_observer.enabled=true",
                f"++callbacks.practice_observer.encoder_path={args.encoder}",
                f"++callbacks.practice_observer.branch_id={run_id}",
                f"++callbacks.practice_observer.output_dir={output_dir}",
            ]
        )
    return [sys.executable, str(REPO / "gear_sonic" / "train_agent_trl.py"), *overrides]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    return [line for line in output.splitlines() if line]


def launcher_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = f"throughput_idle_{args.variant}_ne{args.num_envs}_{timestamp}"
    command = build_command(args, run_id)
    print("\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{run_id}.log"
    receipt_path = args.receipt_dir / f"{run_id}.json"

    initial_gpu = gpu_snapshot()
    if initial_gpu["free_mib"] < args.min_free_mib:
        raise SystemExit(
            f"GPU capacity gate failed: {initial_gpu['free_mib']:.0f} MiB free < {args.min_free_mib} MiB"
        )

    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(gpu_snapshot())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(2.0)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started_wall = datetime.now().astimezone()
    started = time.monotonic()
    with log_path.open("w") as handle:
        code = subprocess.call(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    stop.set()
    monitor_thread.join(timeout=5.0)

    parsed = RL.parse_run_log(log_path)
    throughput = RL.throughput_report(parsed, args.num_envs)
    complete = code == 0 and len(parsed.iterations) == args.iterations
    receipt = {
        "kind": "lucid_idle_gpu_throughput",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": git_sha(),
        "git_status_short": git_status(),
        "launcher_sha256": launcher_sha256(),
        "config": {
            "variant": args.variant,
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "motion_file": args.motion_file,
            "smpl_motion_file": args.smpl_motion_file,
            "first_iteration_excluded_from_steady_state": True,
        },
        "command": command,
        "started_at": started_wall.isoformat(),
        "wall_seconds": elapsed,
        "exit_code": code,
        "log_path": str(log_path),
        "gpu": summarize_gpu(samples or [initial_gpu]),
        "throughput": throughput,
        "learning_curve": {key: parsed.series(key) for key in ("Mean rewards", "Mean length")},
        "verified": (
            [
                f"bounded {args.variant} continuation exited 0",
                f"parsed all {args.iterations} requested iterations",
                "steady-state median excludes the first warm-up iteration",
            ]
            if complete
            else []
        ),
        "not_yet_verified": [
            "whether this single-seed throughput persists across repeated launches",
            "campaign efficacy or any scientific treatment effect",
        ],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(throughput, indent=2))
    print(f"receipt {receipt_path}")
    print(f"log {log_path}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
