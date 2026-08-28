#!/usr/bin/env python3
"""Measure the paired-branch noise floor with epsilon = 0 replicate pairs.

This is the measurement that must come before any campaign is sized, because it
sets the resolution of everything downstream. Gate A asks whether
between-context utility variation exceeds paired branch noise; without a
measured floor, that question has no denominator and any apparent utility
difference could be branch noise wearing a context label.

Design: each pair runs the *same* configuration from the *same* checkpoint with
the *same* seed, and differs only in that one arm is armed at ``epsilon = 0``.
The armed arm therefore traverses the full intervention code path -- kernel
construction, distribution override, dose accounting -- while changing the
distribution not at all. Any difference between the arms is pure branch noise:
RNG divergence plus GPU non-determinism.

An epsilon = 0 arm, not a second unarmed control, is the point. Two unarmed
controls would measure only run-to-run noise and would silently exclude any
noise the intervention machinery itself introduces.

Replicate pairs use different seeds, so the reported floor is a distribution
rather than a single number.

Example
-------
    python scripts/practice_utility/run_noise_floor.py \\
        --pairs 3 --num-envs 256 --iterations 12 --execute
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeContextCallback"

#: Metrics whose paired spread defines the floor. Training-side, so they are
#: available without a separate evaluation pass; a dev-suite J_eff floor is a
#: strictly later and more expensive measurement.
FLOOR_METRICS = ("Mean rewards", "Mean length", "Mean entropy")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--checkpoint", default="sonic_release/last.pt")
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--context-json", type=Path, required=False,
                        help="context for the armed arm; defaults to the first "
                             "context in --snapshot")
    parser.add_argument("--snapshot", type=Path, required=False)
    parser.add_argument("--out-dir", type=Path,
                        default=LUCID_ROOT / "artifacts/noise_floor")
    parser.add_argument("--log-dir", type=Path,
                        default=LUCID_ROOT / "outputs")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def pick_context(args) -> dict:
    if args.context_json:
        return json.loads(args.context_json.read_text())
    if not args.snapshot:
        raise SystemExit("supply --context-json or --snapshot")
    snapshot = json.loads(args.snapshot.read_text())
    if not snapshot.get("contexts"):
        raise SystemExit(f"{args.snapshot} has no contexts")
    # Any resident context works: at epsilon = 0 it changes nothing. Take the
    # median-probability one so the armed arm is not an outlier in any respect.
    ordered = sorted(snapshot["contexts"], key=lambda e: e["sampling_probability"])
    chosen = ordered[len(ordered) // 2]
    return {
        k: chosen[k]
        for k in ("motion_key", "motion_hash", "bin_index", "bin_start_frame",
                  "bin_end_frame", "perturbation_group", "severity_level", "encoder_mode")
    }


def build_command(args, seed, role, context, branch_dir) -> list[str]:
    overrides = [
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=True", "use_wandb=false",
        f"seed={seed}",
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
        "data/motion_lib_bones_seed/smpl_filtered",
        f"++callbacks.practice_context._target_={CALLBACK}",
        "++callbacks.practice_context.enabled=true",
        f"++callbacks.practice_context.role={role}",
        f"++callbacks.practice_context.pair_id=floor_s{seed}",
        f"++callbacks.practice_context.branch_id=floor_s{seed}_{role}",
        "++callbacks.practice_context.epsilon=0.0",
        f"++callbacks.practice_context.dose_report_dir={branch_dir}",
        "++callbacks.practice_context.dose_report_frequency=4",
    ]
    if role == "intervention":
        for key, value in context.items():
            overrides.append(f"++callbacks.practice_context.context.{key}={value}")
    return [sys.executable, str(REPO / "gear_sonic" / "train_agent_trl.py"), *overrides]


def run(command, log_path) -> tuple[int, float]:
    env = dict(os.environ)
    env.setdefault("TMPDIR", str(LUCID_ROOT / "tmp"))
    env.setdefault("WANDB_MODE", "offline")
    started = time.time()
    with open(log_path, "w") as handle:
        code = subprocess.call(command, cwd=str(REPO), stdout=handle,
                               stderr=subprocess.STDOUT, env=env)
    return code, time.time() - started


def summarize(results, iterations) -> dict:
    """Paired differences per metric, plus the cross-seed divergence floor."""
    cross_seed = cross_seed_spread(results)
    floor: dict[str, dict] = {}
    for metric in FLOOR_METRICS:
        deltas, finals = [], []
        for entry in results:
            control = entry["control_series"].get(metric, {})
            armed = entry["intervention_series"].get(metric, {})
            shared = sorted(set(control) & set(armed))
            if not shared:
                continue
            last = shared[-1]
            deltas.append(armed[last] - control[last])
            finals.append(control[last])
        if not deltas:
            continue
        floor[metric] = {
            "num_pairs": len(deltas),
            "paired_deltas": deltas,
            "mean_delta": statistics.fmean(deltas),
            "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "max_abs_delta": max(abs(d) for d in deltas),
            "control_level_mean": statistics.fmean(finals),
            "relative_sd": (
                statistics.stdev(deltas) / abs(statistics.fmean(finals))
                if len(deltas) > 1 and statistics.fmean(finals) else None
            ),
        }
    return {
        "kind": "practice_utility_noise_floor",
        "schema_version": 1,
        "iterations_per_branch": iterations,
        "metrics": floor,
        "cross_seed_control_spread": cross_seed,
        "interpretation": (
            "Two different floors, and they answer different questions.\n"
            "epsilon=0 same-seed paired delta (metrics.*.sd_delta) is MACHINERY "
            "noise: it shows whether arming the intervention path perturbs a run "
            "that should be unchanged. Near-zero here is the end-to-end "
            "confirmation of the epsilon=0 identity guarantee.\n"
            "It does NOT bound the noise a real intervention faces. Once epsilon>0 "
            "changes which bins are sampled, RNG consumption diverges and "
            "trajectories separate; an epsilon=0 pair cannot exhibit that by "
            "construction. cross_seed_control_spread estimates that divergence "
            "component from control runs at different seeds, and it is the floor "
            "Gate A should be judged against."
        ),
    }


def cross_seed_spread(results) -> dict:
    """Spread across control runs at different seeds.

    This is the practically relevant floor. An epsilon=0 pair shares its whole
    random stream, so it cannot show the divergence a real intervention causes
    the moment it samples a different bin. Different-seed controls do exhibit
    exactly that divergence while having no treatment effect at all.
    """
    spread: dict[str, dict] = {}
    for metric in FLOOR_METRICS:
        finals = []
        for entry in results:
            series = entry.get("control_series", {}).get(metric, {})
            if series:
                finals.append(series[max(series)])
        if len(finals) < 2:
            continue
        mean = statistics.fmean(finals)
        sd = statistics.stdev(finals)
        spread[metric] = {
            "num_seeds": len(finals),
            "final_values": finals,
            "mean": mean,
            "sd": sd,
            "relative_sd": sd / abs(mean) if mean else None,
        }
    return spread


def main(argv=None) -> int:
    args = parse_args(argv)
    context = pick_context(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"epsilon=0 noise floor: {args.pairs} pairs x 2 branches, "
          f"{args.iterations} iterations at num_envs={args.num_envs}")
    print(f"armed arm context: {context['motion_key']} bin {context['bin_index']}")

    results = []
    for index in range(args.pairs):
        seed = args.base_seed + index
        entry = {"seed": seed}
        for role in ("control", "intervention"):
            branch_dir = args.out_dir / f"floor_s{seed}_{role}"
            branch_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.log_dir / f"floor_s{seed}_{role}.log"
            command = build_command(args, seed, role, context, branch_dir)
            if not args.execute:
                print(f"  [dry] {role} seed {seed} -> {log_path}")
                continue
            print(f"  running {role} seed {seed} ...", flush=True)
            code, elapsed = run(command, log_path)
            log = RL.parse_run_log(log_path)
            entry[f"{role}_exit"] = code
            entry[f"{role}_seconds"] = round(elapsed, 1)
            entry[f"{role}_iterations"] = len(log.iterations)
            entry[f"{role}_series"] = {m: log.series(m) for m in FLOOR_METRICS}
            print(f"    exit {code} in {elapsed:.0f}s, {len(log.iterations)} iterations")
        results.append(entry)

    if not args.execute:
        print("\ndry run; pass --execute")
        return 0

    report = summarize(results, args.iterations)
    report["context"] = context
    report["num_envs"] = args.num_envs
    report["pairs"] = results
    path = args.out_dir / "noise_floor_report.json"
    path.write_text(json.dumps(report, indent=2))

    print("\nMACHINERY noise (epsilon = 0, same seed -- shared random stream):")
    for metric, stats in report["metrics"].items():
        rel = f"{stats['relative_sd']:.4f}" if stats["relative_sd"] is not None else "n/a"
        print(f"  {metric:16s} n={stats['num_pairs']} sd={stats['sd_delta']:.6f} "
              f"max|d|={stats['max_abs_delta']:.6f} level={stats['control_level_mean']:.4f} "
              f"rel_sd={rel}")
    print("\nDIVERGENCE noise (control runs, different seeds -- the floor Gate A faces):")
    for metric, stats in report["cross_seed_control_spread"].items():
        rel = f"{stats['relative_sd']:.4f}" if stats["relative_sd"] is not None else "n/a"
        print(f"  {metric:16s} n={stats['num_seeds']} sd={stats['sd']:.6f} "
              f"mean={stats['mean']:.4f} rel_sd={rel}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
