#!/usr/bin/env python3
"""Measure peak GPU memory against ``num_envs`` for a from-scratch run.

The user's constraint on the from-scratch probe is a memory budget, not an
environment count: training must fit in 8 GB. That makes ``num_envs`` a
*measured* quantity rather than a chosen one, and it has to be measured on the
configuration that will actually run -- same event preset, same delayed
actuator, same research callbacks -- because each of those allocates.

So this walks a ladder of environment counts, runs a handful of real iterations
at each, samples ``nvidia-smi`` throughout, and stops at the first count whose
peak exceeds the budget. It reports the largest count that fit, together with
the throughput at that count, which is what sizes the campaign.

Peak is read as *total device* usage, not the process's own allocation: the
budget the user set is a budget on the card, and anything else resident counts
against it. The desktop's ~1.2 GB is therefore included, and reported
separately so the number can be read either way.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402
from scripts.practice_utility import run_throughput_probe as TP  # noqa: E402

LAUNCHER = REPO / "scripts" / "practice_utility" / "run_curriculum_comparison.py"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", type=int, nargs="+", default=[256, 512, 768, 1024, 1536])
    parser.add_argument("--budget-mib", type=int, default=8000)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument(
        "--motion-file",
        default=str(LUCID_ROOT / "pools/subsets/train016/robot_filtered"),
    )
    parser.add_argument("--smpl-motion-file", default="dummy")
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def sample_peak(command: list[str], log_path: Path) -> dict[str, Any]:
    """Run one bounded training job, sampling device memory throughout."""
    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(TP.gpu_snapshot())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1.0)

    baseline = TP.gpu_snapshot()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    with log_path.open("w") as handle:
        code = subprocess.call(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    stop.set()
    thread.join(timeout=5.0)
    peak = max((s["used_mib"] for s in samples), default=baseline["used_mib"])
    return {
        "exit_code": code,
        "wall_seconds": elapsed,
        "samples": len(samples),
        "baseline_used_mib": baseline["used_mib"],
        "peak_used_mib": peak,
        "peak_over_baseline_mib": peak - baseline["used_mib"],
        "max_util_pct": max((s["gpu_util_pct"] for s in samples), default=0.0),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)

    rungs: list[dict[str, Any]] = []
    largest_fit: int | None = None
    for num_envs in sorted(args.ladder):
        command = [
            sys.executable, str(LAUNCHER),
            "--from-scratch",
            "--num-envs", str(num_envs),
            "--iterations", str(args.iterations),
            "--warmup-iterations", str(args.warmup_iterations),
            "--seeds", str(args.seed),
            "--modes", "off",
            "--motion-file", args.motion_file,
            "--smpl-motion-file", args.smpl_motion_file,
            "--min-free-mib", "2000",
            "--execute",
        ]
        print(f"--- num_envs={num_envs}", flush=True)
        if not args.execute:
            print("  " + " ".join(command))
            continue
        log_path = args.log_dir / f"vram_ladder_ne{num_envs}_{timestamp}.log"
        measured = sample_peak(command, log_path)
        # The launcher writes the training log under its own artifact tree; the
        # per-iteration timings we want are in the branch log it produced.
        branch_logs = sorted(
            args.log_dir.glob(f"curriculum_comparison_ne{num_envs}_*_s{args.seed}_off.log")
        )
        throughput: dict[str, Any] = {}
        if branch_logs:
            parsed = RL.parse_run_log(branch_logs[-1])
            throughput = RL.throughput_report(parsed, num_envs)
        rung = {
            "num_envs": num_envs,
            "fits_budget": measured["peak_used_mib"] <= args.budget_mib,
            **measured,
            "throughput": throughput,
            "log_path": str(log_path),
        }
        rungs.append(rung)
        status = "fits" if rung["fits_budget"] else "OVER BUDGET"
        secs = throughput.get("median_iteration_seconds")
        print(
            f"  peak {measured['peak_used_mib']:.0f} MiB "
            f"(+{measured['peak_over_baseline_mib']:.0f} over a {measured['baseline_used_mib']:.0f} MiB baseline)"
            f"  exit={measured['exit_code']}  {status}"
            + (f"  {secs:.2f} s/iter" if secs else ""),
            flush=True,
        )
        if rung["fits_budget"] and measured["exit_code"] == 0:
            largest_fit = num_envs
        elif not rung["fits_budget"]:
            print("  budget exceeded; stopping the ladder", flush=True)
            break

    chosen = next((r for r in rungs if r["num_envs"] == largest_fit), None)
    receipt = {
        "kind": "lucid_vram_ladder",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "config": {
            "budget_mib": args.budget_mib,
            "ladder": sorted(args.ladder),
            "iterations": args.iterations,
            "motion_file": args.motion_file,
            "smpl_motion_file": args.smpl_motion_file,
            "from_scratch": True,
            "events": "tracking/lucid_curriculum, curriculum mode=off (lambda=0)",
            "peak_is": "total device memory, so anything else resident counts against the budget",
        },
        "rungs": rungs,
        "largest_fitting_num_envs": largest_fit,
        "chosen": chosen,
        "verified": [
            "each rung ran the real launcher, with the delayed actuator and research "
            "callbacks the campaign will use",
            "peak sampled once per second for the whole subprocess lifetime",
        ],
        "not_yet_verified": [
            "that peak memory over 8 iterations bounds peak over thousands -- allocator "
            "growth and motion resampling could push it higher later",
            "anything about learning; this measures memory and throughput only",
        ],
    }
    out = args.receipt_dir / f"vram_ladder_{timestamp}.json"
    out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"\nlargest num_envs within {args.budget_mib} MiB: {largest_fit}")
    print(f"receipt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
