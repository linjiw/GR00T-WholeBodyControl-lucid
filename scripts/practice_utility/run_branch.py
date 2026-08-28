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

Prints the command by default. This low-level translator is not the
receipt-producing claim launcher: ``--execute`` is available only with an
explicit ``--exploratory`` acknowledgement until it consumes a ready campaign
preflight and its exact ``BranchSpec``.

Example
-------
    python scripts/practice_utility/run_branch.py \\
        --manifest .../probe_oracle_screen.json \\
        --stage middle --seed 0 --context-index 3 --role intervention \\
        --checkpoint .../settled_origin.pt --capsule .../settled_origin.capsule.pt \\
        --num-envs 64 --execute
"""

# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for mixed import/from-import blocks.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeContextCallback"
CAPSULE_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"
RESUME_CALLBACK = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleResumeCallback"
OBSERVER_CALLBACK = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"
DEFAULT_ENCODER = LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=["control", "intervention"], required=True)
    parser.add_argument(
        "--context-index",
        type=int,
        help="index into the stage's frozen context list " "(required for an intervention branch)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="SONIC checkpoint exported from the same settled capsule",
    )
    parser.add_argument(
        "--capsule",
        required=True,
        type=Path,
        help="settled branch capsule whose full RNG stream seeds this fresh restart",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=DEFAULT_ENCODER,
        help="frozen temporal encoder used for branch latent-gap telemetry",
    )
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--motion-file", default="data/motion_lib_bones_seed/robot_filtered")
    parser.add_argument("--smpl-motion-file", default="data/motion_lib_bones_seed/smpl_filtered")
    parser.add_argument(
        "--artifact-dir", type=Path, default=LUCID_ROOT / "artifacts"
    )
    parser.add_argument("--pool-manifest", type=Path, default=None)
    parser.add_argument(
        "--epsilon-override",
        type=float,
        default=None,
        help="override the manifest dose; use only for an eps=0 " "noise-floor branch",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "allow low-level execution without a ready preflight/BranchSpec; "
            "outputs are not claim-grade"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _global_step(value, *, source: str) -> int:
    """Return a validated positive trainer step for a settled origin."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} global_step must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(
            f"{source} global_step must be positive; claim-bearing branches require "
            "a settled, non-cold origin"
        )
    return value


def capsule_global_step(path: Path) -> int:
    """Load and integrity-check a capsule without disturbing the caller's RNG."""
    payload = BC.load_capsule(path, restore_rng=False)
    step = _global_step(payload.get("global_step"), source="capsule")

    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise ValueError(f"capsule trainer_state is not a mapping: {path}")
    trainer_step = _global_step(trainer_state.get("global_step"), source="capsule trainer")
    if trainer_step != step:
        raise ValueError(
            f"capsule global_step mismatch at {path}: payload={step}, trainer={trainer_step}"
        )
    return step


def checkpoint_global_step(path: Path) -> int:
    """Read the trainer step SONIC will restore from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state")
    value = (
        state.get("global_step") if isinstance(state, dict) else getattr(state, "global_step", None)
    )
    step = _global_step(value, source="checkpoint trainer")

    practice_metadata = payload.get("practice_utility")
    if isinstance(practice_metadata, dict) and practice_metadata.get("global_step") is not None:
        metadata_step = _global_step(
            practice_metadata["global_step"], source="checkpoint practice metadata"
        )
        if metadata_step != step:
            raise ValueError(
                f"checkpoint global_step mismatch at {path}: trainer={step}, "
                f"practice metadata={metadata_step}"
            )
    return step


def validate_resume_origin(checkpoint: Path, capsule: Path) -> int:
    """Require one matched settled checkpoint/capsule origin for both arms."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if not capsule.is_file():
        raise FileNotFoundError(f"capsule does not exist: {capsule}")
    capsule_step = capsule_global_step(capsule)
    checkpoint_step = checkpoint_global_step(checkpoint)
    if checkpoint_step != capsule_step:
        raise ValueError(
            "checkpoint/capsule global_step mismatch: "
            f"checkpoint={checkpoint_step}, capsule={capsule_step}"
        )
    return capsule_step


def continuation_horizons(manifest: dict) -> dict[str, int]:
    """Validate the manifest's post-capsule continuation lengths."""
    horizons = manifest.get("horizons")
    if not isinstance(horizons, dict) or not horizons:
        raise ValueError("manifest horizons must be a non-empty mapping")
    for label, horizon in horizons.items():
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"manifest continuation horizon {label!r} must be a positive integer")
    return horizons


def build_overrides(args, manifest, capsule_step: int) -> tuple[str, list[str]]:
    stages = manifest["contexts_per_stage"]
    if args.stage not in stages:
        raise SystemExit(f"stage {args.stage!r} not in manifest; have {sorted(stages)}")
    if args.seed not in manifest["seeds"]:
        raise SystemExit(f"seed {args.seed} not in manifest; have {manifest['seeds']}")

    capsule_step = _global_step(capsule_step, source="capsule")
    horizons = continuation_horizons(manifest)
    absolute_horizons = {
        label: capsule_step + continuation_length for label, continuation_length in horizons.items()
    }
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
        "+resume=true",
        f"num_envs={args.num_envs}",
        "headless=True",
        "use_wandb=false",
        f"seed={args.seed}",
        f"++algo.config.num_learning_iterations={max(absolute_horizons.values())}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
        f"++callbacks.practice_resume._target_={RESUME_CALLBACK}",
        "++callbacks.practice_resume.enabled=true",
        f"++callbacks.practice_resume.capsule_path={args.capsule}",
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
        f"++callbacks.practice_observer._target_={OBSERVER_CALLBACK}",
        "++callbacks.practice_observer.enabled=true",
        f"++callbacks.practice_observer.encoder_path={args.encoder}",
        f"++callbacks.practice_observer.output_dir={branch_dir}",
        f"++callbacks.practice_observer.branch_id={branch_id}",
        f"++callbacks.practice_capsule._target_={CAPSULE_CALLBACK}",
        "++callbacks.practice_capsule.enabled=true",
        f"++callbacks.practice_capsule.capsule_dir={branch_dir / 'capsules'}",
        f"++callbacks.practice_capsule.pair_id={pair_id}",
        f"++callbacks.practice_capsule.role={args.role}",
        f"++callbacks.practice_capsule.branch_id={branch_id}",
    ]
    for label, horizon in absolute_horizons.items():
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
    capsule_step = validate_resume_origin(args.checkpoint, args.capsule)
    branch_id, overrides = build_overrides(args, manifest, capsule_step)

    command = [sys.executable, str(REPO / "gear_sonic" / "train_agent_trl.py"), *overrides]
    print(f"branch  {branch_id}")
    print(f"manifest {manifest['manifest_sha256'][:16]}  role {args.role}")
    print(
        f"origin  step {capsule_step}; continuation "
        f"{max(continuation_horizons(manifest).values())} iterations"
    )
    print("\n" + " \\\n  ".join(shlex.quote(part) for part in command) + "\n")

    if not args.execute:
        print("dry run; low-level execution requires --execute --exploratory")
        return 0

    if not args.exploratory:
        raise SystemExit(
            "claim-grade execution is blocked: this low-level runner does not yet "
            "consume a ready preflight report and exact BranchSpec; use the future "
            "campaign launcher, or pass --exploratory for non-claim diagnostics"
        )

    env = dict(os.environ)
    env.setdefault("TMPDIR", str(LUCID_ROOT / "tmp"))
    env.setdefault("WANDB_MODE", "offline")
    return subprocess.call(command, cwd=str(REPO), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
