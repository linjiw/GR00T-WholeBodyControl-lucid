#!/usr/bin/env python3
"""Continue LUCID and fixed-DR policies through a matched full-DR terminal phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import branch_capsule as BC  # noqa: E402
from scripts.practice_utility import run_curriculum_comparison as CC  # noqa: E402
from scripts.practice_utility import run_latency_ab as LA
from scripts.practice_utility import run_throughput_probe as TP

OBSERVER = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"
CURRICULUM = "gear_sonic.research.practice_utility.dr_curriculum.LucidCurriculumCallback"
CAPSULE = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"
RESUME = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleResumeCallback"
MODES = ("lucid", "fixed")
EXPECTED_TERMS = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-training-receipt",
        type=Path,
        default=Path(
            "/data/robotixx/lucid-sonic/manifests/"
            "curriculum_comparison_ne128_20260820_143058.json"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--continuation-iterations", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8600, 8601, 8602])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--encoder",
        default="/data/robotixx/lucid-sonic/artifacts/lucid_encoder_debug512.pt",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/artifacts/curriculum_consolidation"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("/data/robotixx/lucid-sonic/outputs"))
    parser.add_argument(
        "--receipt-dir", type=Path, default=Path("/data/robotixx/lucid-sonic/manifests")
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.continuation_iterations < 4:
        parser.error("continuation must include at least four iterations for trailing metrics")
    return args


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state")
    if isinstance(state, dict):
        value = state.get("global_step")
    else:
        value = getattr(state, "global_step", None)
    if value is None:
        raise ValueError(f"checkpoint has no trainer global_step: {path}")
    return int(value)


def source_index(receipt: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    index = {}
    for arm in receipt["arms"].values():
        mode = arm.get("mode")
        if mode in MODES:
            index[(int(arm["seed"]), mode)] = arm
    return index


def rotated(items: list[str], offset: int) -> list[str]:
    if not items:
        return []
    offset %= len(items)
    return items[offset:] + items[:offset]


def build_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    capsule: Path,
    seed: int,
    branch_id: str,
    artifact_dir: Path,
    target_step: int,
) -> list[str]:
    """Build one full-state resume with lambda pinned to one for consolidation."""
    capsule_dir = artifact_dir / "capsules"
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "train_with_delay.py"),
        "--max-delay",
        str(args.max_delay),
        "--",
        f"+exp={args.exp}",
        f"checkpoint={checkpoint}",
        "+resume=true",
        f"num_envs={args.num_envs}",
        "headless=true",
        "use_wandb=false",
        f"seed={seed}",
        "manager_env/events=tracking/lucid_curriculum",
        f"++algo.config.num_learning_iterations={target_step}",
        "++algo.config.save_interval=100000",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/motion_lib_bones_seed/smpl_filtered",
        f"++callbacks.practice_resume._target_={RESUME}",
        "++callbacks.practice_resume.enabled=true",
        f"++callbacks.practice_resume.capsule_path={capsule}",
        f"++callbacks.practice_observer._target_={OBSERVER}",
        "++callbacks.practice_observer.enabled=true",
        f"++callbacks.practice_observer.encoder_path={args.encoder}",
        f"++callbacks.practice_observer.branch_id={branch_id}",
        f"++callbacks.practice_observer.output_dir={artifact_dir}",
        f"++callbacks.lucid_curriculum._target_={CURRICULUM}",
        "++callbacks.lucid_curriculum.enabled=true",
        "++callbacks.lucid_curriculum.mode=fixed",
        f"++callbacks.lucid_curriculum.observer_branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.output_dir={artifact_dir}",
        "++callbacks.lucid_curriculum.initial_lambda=1.0",
        "++callbacks.lucid_curriculum.fixed_lambda=1.0",
        "++callbacks.lucid_curriculum.warmup_iterations=0",
        f"++callbacks.practice_capsule._target_={CAPSULE}",
        "++callbacks.practice_capsule.enabled=true",
        f"++callbacks.practice_capsule.capsule_dir={capsule_dir}",
        f"++callbacks.practice_capsule.pair_id=consolidation_seed_{seed}",
        "++callbacks.practice_capsule.role=control",
        f"++callbacks.practice_capsule.branch_id={branch_id}",
        f"++callbacks.practice_capsule.horizons.final={target_step}",
    ]


def aggregate(arms: dict[str, dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    summary = CC.aggregate(arms, modes)
    for mode in modes:
        members = [arm for arm in arms.values() if arm["mode"] == mode]
        source_lambdas = [float(arm["source_lambda"]) for arm in members]
        summary[mode]["source_lambda_mean"] = statistics.fmean(source_lambdas)
        summary[mode]["terminal_lambda_mean"] = statistics.fmean(
            float(arm["final_lambda"]) for arm in members
        )
    return summary


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    source_receipt = load_json(args.source_training_receipt)
    index = source_index(source_receipt)
    modes = list(dict.fromkeys(args.modes))

    specs = []
    for seed_index, seed in enumerate(args.seeds):
        for mode in rotated(modes, seed_index):
            source = index.get((seed, mode))
            if source is None:
                raise KeyError(f"source receipt has no seed={seed}, mode={mode}")
            checkpoint = Path(source["checkpoint"]).resolve()
            capsule = Path(source["capsule"]).resolve()
            if not checkpoint.is_file() or not capsule.is_file():
                raise FileNotFoundError(
                    f"missing source checkpoint/capsule for seed={seed}, mode={mode}"
                )
            capsule_step = int(BC.load_capsule(capsule, restore_rng=False)["global_step"])
            model_step = checkpoint_step(checkpoint)
            if capsule_step != model_step:
                raise ValueError(
                    f"source checkpoint/capsule step mismatch for seed={seed}, mode={mode}: "
                    f"{model_step} != {capsule_step}"
                )
            specs.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "source": source,
                    "checkpoint": checkpoint,
                    "capsule": capsule,
                    "source_step": model_step,
                    "target_step": model_step + args.continuation_iterations,
                }
            )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"curriculum_consolidation_ne{args.num_envs}_{stamp}"
    experiment_root = args.artifact_root / experiment_id
    commands = {}
    for spec in specs:
        branch_id = f"{experiment_id}_s{spec['seed']}_{spec['mode']}"
        artifact_dir = experiment_root / f"seed_{spec['seed']}" / spec["mode"]
        spec["branch_id"] = branch_id
        spec["artifact_dir"] = artifact_dir
        commands[branch_id] = build_command(
            args,
            checkpoint=spec["checkpoint"],
            capsule=spec["capsule"],
            seed=spec["seed"],
            branch_id=branch_id,
            artifact_dir=artifact_dir,
            target_step=spec["target_step"],
        )
        print(f"[{branch_id}]\n" + "\n".join(commands[branch_id]))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    source_hashes = {
        spec["branch_id"]: {
            "checkpoint": file_sha256(spec["checkpoint"]),
            "capsule": file_sha256(spec["capsule"]),
        }
        for spec in specs
    }
    arms: dict[str, dict[str, Any]] = {}
    runtime: dict[str, dict[str, Any]] = {}

    def make_receipt() -> dict[str, Any]:
        complete = len(arms) == len(specs)
        mechanics_ok = complete and all(
            arm["complete"]
            and arm["resumed_source_state"]
            and arm["actuator_groups_swapped"] == 5
            and arm["checkpoint_exported"]
            and arm["checkpoint_step"] == arm["target_step"]
            and arm["curriculum_rows"] == args.continuation_iterations
            and arm["all_terminal_lambda_one"]
            and set(arm["scalable_terms"]) == EXPECTED_TERMS
            for arm in arms.values()
        )
        mode_summary = aggregate(arms, modes) if complete else {}
        comparison = CC.comparisons(mode_summary) if complete else {}
        return {
            "kind": "lucid_terminal_full_dr_consolidation",
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment_id": experiment_id,
            "git_sha": TP.git_sha(),
            "git_status_short": TP.git_status(),
            "launcher_sha256": source_sha256(),
            "source_training_receipt": str(args.source_training_receipt.resolve()),
            "source_training_experiment_id": source_receipt.get("experiment_id"),
            "config": {
                "checkpoint": source_receipt["config"]["checkpoint"],
                "num_envs": args.num_envs,
                "continuation_iterations": args.continuation_iterations,
                "absolute_target_steps": sorted({spec["target_step"] for spec in specs}),
                "seeds": args.seeds,
                "modes": modes,
                "arm_order": [
                    {"seed": seed, "modes": rotated(modes, index)}
                    for index, seed in enumerate(args.seeds)
                ],
                "event_preset": "tracking/lucid_curriculum",
                "terminal_schedule": "lambda=1 for every continuation rollout",
                "max_delay_steps": args.max_delay,
                "max_delay_ms": args.max_delay * 5,
                "estimand": (
                    "frozen-policy robustness after equal 16-iteration full-DR continuation; "
                    "training reward is diagnostic only"
                ),
            },
            "source_hashes": source_hashes,
            "commands": commands,
            "runtime": runtime,
            "arms": arms,
            "mode_summary": mode_summary,
            "training_comparison": comparison,
            "verified": (
                [
                    "all branches restored their full source checkpoint and capsule RNG state",
                    "all branches completed equal compute through the absolute target step",
                    "all continuation rollouts used lambda=1 over all six DR channels",
                    "all five live actuator groups used the corrected delayed actuator",
                    "every branch exported a SONIC-compatible final checkpoint",
                ]
                if mechanics_ok
                else []
            ),
            "not_yet_verified": [
                *([] if complete else ["the full requested training matrix"]),
                "frozen-policy clean/full-DR/60 ms efficacy; training curves are not the metric",
                "episode-masked physical-quality comparison",
                "robustness beyond the trained 0-40 ms latency support",
            ],
        }

    receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")
    try:
        for spec in specs:
            branch_id = spec["branch_id"]
            artifact_dir = spec["artifact_dir"]
            artifact_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.log_dir / f"{branch_id}.log"
            runtime[branch_id] = LA.run_arm(commands[branch_id], log_path, args.min_free_mib)
            observer_path = artifact_dir / f"observer_{branch_id}.jsonl"
            arm = LA.summarize_arm(log_path, observer_path, args.continuation_iterations)
            curriculum_path = artifact_dir / f"curriculum_{branch_id}.jsonl"
            curriculum = CC.read_jsonl(curriculum_path)
            capsule = artifact_dir / "capsules" / f"{branch_id}_final.capsule.pt"
            checkpoint = artifact_dir / "final_checkpoint.pt"
            if runtime[branch_id]["exit_code"] == 0 and capsule.is_file():
                BC.export_sonic_checkpoint(capsule, checkpoint)
            text = log_path.read_text(errors="replace")
            final_step = checkpoint_step(checkpoint) if checkpoint.is_file() else None
            arm.update(
                {
                    "seed": spec["seed"],
                    "mode": spec["mode"],
                    "variant": f"{spec['mode']}_terminal_full_dr",
                    "branch_id": branch_id,
                    "source_checkpoint": str(spec["checkpoint"]),
                    "source_capsule": str(spec["capsule"]),
                    "source_step": spec["source_step"],
                    "source_lambda": spec["source"].get("final_lambda"),
                    "target_step": spec["target_step"],
                    "resumed_source_state": (
                        f"restored capsule RNG at step {spec['source_step']}" in text
                    ),
                    "curriculum_path": str(curriculum_path),
                    "curriculum_rows": len(curriculum),
                    "all_terminal_lambda_one": bool(curriculum)
                    and all(float(row.get("lambda", -1.0)) == 1.0 for row in curriculum),
                    "final_lambda": curriculum[-1].get("lambda") if curriculum else None,
                    "scalable_terms": (
                        curriculum[-1].get("scalable_terms", []) if curriculum else []
                    ),
                    "capsule": str(capsule),
                    "checkpoint": str(checkpoint),
                    "checkpoint_exported": checkpoint.is_file(),
                    "checkpoint_step": final_step,
                }
            )
            arms[branch_id] = arm
            runtime[branch_id]["log_path"] = str(log_path)
            runtime[branch_id]["observer_path"] = str(observer_path)
            receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")
    finally:
        receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")

    receipt = make_receipt()
    print(json.dumps(receipt["mode_summary"], indent=2))
    print(f"receipt {receipt_path}")
    return 0 if receipt["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
