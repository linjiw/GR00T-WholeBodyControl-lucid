#!/usr/bin/env python3
"""Launch one branch of a paired continuation from a frozen probe manifest.

A branch is a SONIC training continuation with the practice-utility callbacks
attached. The manifest decides everything that could otherwise drift between
the two arms -- context, dose, kernel radius, horizons, seeds -- so this script
only translates a manifest entry into Hydra overrides and runs it.

The control arm is *shared* per (stage, seed): its distribution does not depend
on which context is being probed, so one control serves every intervention at
that stage and seed. That is the single largest cost saving in the campaign,
and it is sound for screening; confirmation runs use independent paired
controls instead.

Prints the command by default. Pass ``--execute`` to run it.

Example
-------
    python scripts/practice_utility/run_branch.py \\
        --manifest .../probe_oracle_screen.json \\
        --stage middle --seed 0 --context-index 3 --role intervention \\
        --checkpoint sonic_release/last.pt --num-envs 64 --execute
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeContextCallback"
CAPSULE_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=["control", "intervention"], required=True)
    parser.add_argument("--context-index", type=int,
                        help="index into the stage's frozen context list "
                             "(required for an intervention branch)")
    parser.add_argument("--checkpoint", default="sonic_release/last.pt")
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--motion-file", default="data/motion_lib_bones_seed/robot_filtered")
    parser.add_argument("--smpl-motion-file", default="data/motion_lib_bones_seed/smpl_filtered")
    parser.add_argument("--artifact-dir", type=Path,
                        default=Path("/data/robotixx/lucid-sonic/artifacts"))
    parser.add_argument("--pool-manifest", type=Path, default=None)
    parser.add_argument("--epsilon-override", type=float, default=None,
                        help="override the manifest dose; use only for an eps=0 "
                             "noise-floor branch")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def build_overrides(args, manifest) -> tuple[str, list[str]]:
    stages = manifest["contexts_per_stage"]
    if args.stage not in stages:
        raise SystemExit(f"stage {args.stage!r} not in manifest; have {sorted(stages)}")
    if args.seed not in manifest["seeds"]:
        raise SystemExit(f"seed {args.seed} not in manifest; have {manifest['seeds']}")

    horizons = manifest["horizons"]
    epsilon = manifest["epsilon"] if args.epsilon_override is None else args.epsilon_override
    pair_id = f"{manifest['campaign_id']}_{args.stage}_s{args.seed}"
    context = None

    if args.role == "intervention":
        if args.context_index is None:
            raise SystemExit("--context-index is required for an intervention branch")
        entries = stages[args.stage]
        if not 0 <= args.context_index < len(entries):
            raise SystemExit(
                f"--context-index {args.context_index} out of range "
                f"(stage {args.stage!r} has {len(entries)} contexts)"
            )
        entry = entries[args.context_index]
        context = entry["context"]
        # The pair id binds a control to its intervention: both arms of one pair
        # must share it, or the common-random-number streams will not match.
        pair_id = f"{pair_id}_c{args.context_index}"

    branch_id = f"{pair_id}_{args.role}"
    branch_dir = args.artifact_dir / manifest["campaign_id"] / branch_id

    overrides = [
        f"+exp={args.exp}",
        f"checkpoint={args.checkpoint}",
        f"num_envs={args.num_envs}",
        "headless=True",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={max(horizons.values())}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
        f"++callbacks.practice_context._target_={CALLBACK}",
        "++callbacks.practice_context.enabled=true",
        f"++callbacks.practice_context.role={args.role}",
        f"++callbacks.practice_context.pair_id={pair_id}",
        f"++callbacks.practice_context.branch_id={branch_id}",
        f"++callbacks.practice_context.epsilon={epsilon}",
        f"++callbacks.practice_context.kernel_radius_bins={manifest['kernel_radius_bins']}",
        f"++callbacks.practice_context.dose_report_dir={branch_dir}",
        "++callbacks.practice_context.dose_report_frequency=8",
        f"++callbacks.practice_context.snapshot_path={branch_dir / 'snapshot.json'}",
        f"++callbacks.practice_capsule._target_={CAPSULE_CALLBACK}",
        "++callbacks.practice_capsule.enabled=true",
        f"++callbacks.practice_capsule.capsule_dir={branch_dir / 'capsules'}",
        f"++callbacks.practice_capsule.pair_id={pair_id}",
        f"++callbacks.practice_capsule.role={args.role}",
        f"++callbacks.practice_capsule.branch_id={branch_id}",
    ]
    for label, horizon in horizons.items():
        overrides.append(f"++callbacks.practice_capsule.horizons.{label}={horizon}")
    if args.pool_manifest:
        overrides.append(f"++callbacks.practice_context.manifest_path={args.pool_manifest}")
    if context is not None:
        for key, value in context.items():
            overrides.append(f"++callbacks.practice_context.context.{key}={value}")

    return branch_id, overrides


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    branch_id, overrides = build_overrides(args, manifest)

    command = [sys.executable, str(REPO / "gear_sonic" / "train_agent_trl.py"), *overrides]
    print(f"branch  {branch_id}")
    print(f"manifest {manifest['manifest_sha256'][:16]}  role {args.role}")
    print("\n" + " \\\n  ".join(shlex.quote(part) for part in command) + "\n")

    if not args.execute:
        print("dry run; pass --execute to launch")
        return 0

    env = dict(os.environ)
    env.setdefault("TMPDIR", "/data/robotixx/lucid-sonic/tmp")
    env.setdefault("WANDB_MODE", "offline")
    return subprocess.call(command, cwd=str(REPO), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
