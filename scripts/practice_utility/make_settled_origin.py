#!/usr/bin/env python3
"""Produce a settled branch origin on this host, and receipt what it is.

Every LUCID branch must start from a *settled* checkpoint rather than the cold
release model: §23.2 measured the cross-seed efficacy floor at 10.62% from a
cold origin and 3.33% from a settled one, so two thirds of the apparent noise
was restart transient, not signal. The original origin
(``sonic_release_test-20260818_141446/model_step_000024.pt``) was produced on
the first host and cannot be moved -- checkpoints are not in the data root and
were never exported -- so a second host has to make its own.

This script makes that explicit rather than incidental. It runs the stock SONIC
continuation (``+exp=.../sonic_release``, stock ``level0_4`` events, no research
callbacks, no curriculum) for a fixed number of iterations from the released
checkpoint, then records the resulting checkpoint's sha256 alongside the exact
command, the learning curve, and a settling diagnostic. Any campaign that pins
the origin can therefore prove which origin it used and how that origin was
made.

What this does *not* claim: that the resulting origin is numerically the same
policy as the first host's step-24 origin. It is a **new branch lineage**. The
within-campaign comparisons that matter (every arm shares this origin) are
unaffected; cross-host comparisons to pre-2026-08-28 receipts are replications,
not identities, and must be reported as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

#: Where SONIC's stock config puts a ``sonic_release`` continuation.
RUN_ROOT = REPO / "logs_rl/TRL_G1_Track/manager/universal_token/all_modes"
RUN_GLOB = "sonic_release_test-*"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="sonic_release/last.pt")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument(
        "--iterations",
        type=int,
        default=24,
        help="origin depth; 24 matches the first host's settled origin",
    )
    parser.add_argument("--seed", type=int, default=8500)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--motion-file", default="data/motion_lib_bones_seed/robot_filtered")
    parser.add_argument("--smpl-motion-file", default="data/motion_lib_bones_seed/smpl_filtered")
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def build_command(args) -> list[str]:
    return [
        sys.executable,
        str(REPO / "gear_sonic" / "train_agent_trl.py"),
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={args.iterations}",
        # The checkpoint cadence lives on the model-save callback, not on
        # ``algo.config.save_interval`` -- that key exists but nothing reads it,
        # which is why a run asking for it silently saves nothing.
        f"++callbacks.model_save.save_frequency={args.iterations}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
    ]


def free_mib() -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return float(output.strip().splitlines()[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settling(series: dict[int, float], window: int = 4) -> dict[str, Any]:
    """Trailing-window settling diagnostic for one metric series.

    Not the cross-seed noise floor -- that needs replicate runs. This only says
    whether the run has stopped climbing steeply, which is the property that
    makes an origin usable: ``last`` vs the window before it.
    """
    ordered = [series[k] for k in sorted(series)]
    if len(ordered) < 2 * window:
        return {"insufficient_iterations": True, "count": len(ordered)}
    last = ordered[-window:]
    prior = ordered[-2 * window : -window]
    last_mean = sum(last) / window
    prior_mean = sum(prior) / window
    return {
        "window": window,
        "last_window_mean": last_mean,
        "prior_window_mean": prior_mean,
        "relative_change": (
            (last_mean - prior_mean) / abs(prior_mean) if prior_mean else None
        ),
    }


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    return [line for line in output.splitlines() if line]


def existing_runs() -> set[Path]:
    return set(RUN_ROOT.glob(RUN_GLOB)) if RUN_ROOT.exists() else set()


def main(argv=None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    print("\n".join(command))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    available = free_mib()
    if available < args.min_free_mib:
        raise SystemExit(f"GPU gate failed: {available:.0f} MiB free < {args.min_free_mib}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = f"settled_origin_ne{args.num_envs}_{timestamp}"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{run_id}.log"

    before = existing_runs()
    started_wall = datetime.now().astimezone()
    started = time.monotonic()
    with log_path.open("w") as handle:
        code = subprocess.call(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started

    new_runs = sorted(existing_runs() - before)
    run_dir = new_runs[-1] if new_runs else None
    origin = run_dir / f"model_step_{args.iterations:06d}.pt" if run_dir else None
    found = origin is not None and origin.exists()

    parsed = RL.parse_run_log(log_path)
    rewards = parsed.series("Mean rewards")
    lengths = parsed.series("Mean length")
    receipt = {
        "kind": "lucid_settled_origin",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": run_id,
        "git_sha": git_sha(),
        "git_status_short": git_status(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {
            "source_checkpoint": str((REPO / args.checkpoint).resolve()),
            "source_checkpoint_sha256": sha256_file(REPO / args.checkpoint),
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "seed": args.seed,
            "exp": args.exp,
            "events": "stock (tracking/level0_4); no curriculum, no research callbacks",
            "motion_file": args.motion_file,
            "smpl_motion_file": args.smpl_motion_file,
        },
        "command": command,
        "started_at": started_wall.isoformat(),
        "wall_seconds": elapsed,
        "exit_code": code,
        "log_path": str(log_path),
        "origin": {
            "run_dir": str(run_dir) if run_dir else None,
            "checkpoint": str(origin) if found else None,
            "checkpoint_sha256": sha256_file(origin) if found else None,
            "size_bytes": origin.stat().st_size if found else None,
            "repo_relative": str(origin.relative_to(REPO)) if found else None,
        },
        "learning_curve": {"Mean rewards": rewards, "Mean length": lengths},
        "settling": {
            "Mean rewards": settling(rewards),
            "Mean length": settling(lengths),
        },
        "verified": (
            [
                f"stock continuation exited 0 after {args.iterations} iterations",
                f"parsed {len(parsed.iterations)} iterations from the training log",
                "origin checkpoint exists and its sha256 is recorded",
            ]
            if code == 0 and found
            else []
        ),
        "not_yet_verified": [
            "that this origin is the same policy as the first host's step-24 origin "
            "(it is not; this starts a new branch lineage)",
            "the cross-seed efficacy noise floor on this host (needs replicate runs)",
            "any treatment effect",
        ],
    }
    receipt_path = args.receipt_dir / f"{run_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt {receipt_path}")
    if not found:
        print("ORIGIN NOT FOUND", file=sys.stderr)
        return 1
    print(f"origin {origin}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
