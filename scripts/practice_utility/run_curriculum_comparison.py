#!/usr/bin/env python3
"""Train and compare SONIC LUCID, fixed-DR, and no-DR branches."""

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
from scripts.practice_utility import run_latency_ab as LA  # noqa: E402
from scripts.practice_utility import run_throughput_probe as TP
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

OBSERVER = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"
CURRICULUM = "gear_sonic.research.practice_utility.dr_curriculum.LucidCurriculumCallback"
CAPSULE = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"
#: Arm name -> (curriculum mode, anchor ratio, yoked source arm). The TACE arms
#: pin a fixed cohort of envs to the full envelope (see tace.py); the yoked arm
#: replays its source arm's lambda trajectory for the same seed with no feedback.
ARMS: dict[str, tuple[str, float, str | None]] = {
    "lucid": ("lucid", 0.0, None),
    "fixed": ("fixed", 0.0, None),
    "off": ("off", 0.0, None),
    "ta_lucid_25": ("lucid", 0.25, None),
    "ta_lucid_50": ("lucid", 0.50, None),
    "ta_yoked_25": ("yoked", 0.25, "ta_lucid_25"),
    "ta_yoked_50": ("yoked", 0.50, "ta_lucid_50"),
    # Cross-seed yoking: seed s replays the schedule learned on the *next* seed.
    # Same-seed yoking is bit-identical to its source (deterministic simulator),
    # so it cannot test online feedback; this can.
    "ta_yoked_25x": ("yoked", 0.25, "ta_lucid_25"),
    "ta_yoked_50x": ("yoked", 0.50, "ta_lucid_50"),
}
CROSS_SEED_ARMS = {"ta_yoked_25x", "ta_yoked_50x"}
#: Channel-attribution arms: fixed intensity with per-term overrides.
NON_LATENCY_TERMS = (
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_rigid_body_mass",
)
ARM_TERM_OVERRIDES: dict[str, dict[str, float]] = {
    "fixed_nolat": {"randomize_action_delay": 0.0},
    "fixed_latonly": {term: 0.0 for term in NON_LATENCY_TERMS},
    # Latency-only *curriculum* arms. The untrained origin is already robust to
    # the five non-latency channels (60.5% at the full envelope, 56.2% at 1.25x
    # it) and has zero margin at a pinned 60 ms; training the full envelope
    # destroys it, and channel attribution says latency carries 89% of that
    # damage. So the only axis with headroom is the one that must be approached
    # gently -- which is what these arms are: a gap-gated, stratified,
    # relatively-guarded curriculum on latency alone.
    "lucid_latonly_s4_rg": {term: 0.0 for term in NON_LATENCY_TERMS},
    "ta_lucid_50_latonly_s4_rg": {term: 0.0 for term in NON_LATENCY_TERMS},
}
ARMS.update({"fixed_nolat": ("fixed", 0.0, None), "fixed_latonly": ("fixed", 0.0, None)})
ARMS.update(
    {
        "lucid_latonly_s4_rg": ("lucid", 0.0, None),
        "ta_lucid_50_latonly_s4_rg": ("lucid", 0.50, None),
    }
)

#: LUCID-S arms. ``spread_strata = K`` splits the focus cohort into K intensity
#: strata so the training mixture spans ``(0, lambda]`` rather than the single
#: point ``lambda``; ``return_guard = "relative"`` replaces the absolute return
#: floor, which the 128-iteration horizon study showed is not scale-stable.
#: The two are separate arms as well as a combined one, because a combined-only
#: result cannot say which change did the work.
ARMS.update(
    {
        "lucid_s4": ("lucid", 0.0, None),
        "lucid_rg": ("lucid", 0.0, None),
        "lucid_s4_rg": ("lucid", 0.0, None),
        "ta_lucid_50_s4_rg": ("lucid", 0.50, None),
    }
)
ARM_SPREAD_STRATA: dict[str, int] = {
    "lucid_s4": 4,
    "lucid_s4_rg": 4,
    "ta_lucid_50_s4_rg": 4,
}
#: Per-channel *ceilings*. Unlike ARM_TERM_OVERRIDES, which pins a channel at a
#: constant, a cap lets the curriculum still schedule the channel up to its own
#: limit -- the one thing a scalar lambda cannot express. These arms exist for
#: the case where channel attribution names a single destructive channel; the
#: cap value is set at launch by --latency-cap and recorded in the receipt.
CAP_ARMS = ("lucid_latcap_s4_rg", "ta_lucid_50_latcap_s4_rg")
ARMS.update(
    {
        "lucid_latcap_s4_rg": ("lucid", 0.0, None),
        "ta_lucid_50_latcap_s4_rg": ("lucid", 0.50, None),
    }
)
ARM_RETURN_GUARD: dict[str, str] = {
    "lucid_rg": "relative",
    "lucid_s4_rg": "relative",
    "ta_lucid_50_s4_rg": "relative",
    "lucid_latcap_s4_rg": "relative",
    "ta_lucid_50_latcap_s4_rg": "relative",
}
ARM_SPREAD_STRATA.update({arm: 4 for arm in CAP_ARMS})
LATONLY_ARMS = ("lucid_latonly_s4_rg", "ta_lucid_50_latonly_s4_rg")
ARM_SPREAD_STRATA.update({arm: 4 for arm in LATONLY_ARMS})
ARM_RETURN_GUARD.update({arm: "relative" for arm in LATONLY_ARMS})
MODES = tuple(ARMS)
TRAINING_METRICS = ("Mean rewards", "Mean length", "Mean entropy")
QUALITY_METRICS = (
    "latent_p90",
    "foot_slip_per_step_m",
    "torque_saturation",
    "energy_proxy",
    "action_delay_mean_steps",
    "action_delay_nonzero_fraction",
)
#: Repo-relative motion paths. Callers that synthesise an ``args`` namespace
#: (the horizon orchestrator) need not carry them, so they are read with a
#: default rather than as required attributes.
DEFAULT_MOTION_FILE = "data/motion_lib_bones_seed/robot_filtered"
DEFAULT_SMPL_MOTION_FILE = "data/motion_lib_bones_seed/smpl_filtered"
REWARD_FLOOR = 0.0333
LENGTH_FLOOR = 0.0314


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="branch origin; omit together with --from-scratch to train a fresh policy",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help=(
            "train from a fresh initialisation instead of continuing a checkpoint. "
            "Fine-tuning the released policy is destructive at this scale -- plain "
            "no-DR continuation costs 23 profile-AUC points against the untrained "
            "origin -- so an arm comparison that starts there is measuring damage, "
            "not learning."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=None,
        help=(
            "extra iteration counts at which to export a capsule, for a convergence "
            "curve measured along one trajectory rather than across separate runs"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8600, 8601, 8602])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["lucid", "fixed", "off"])
    parser.add_argument(
        "--yoked-source-receipt",
        type=Path,
        default=None,
        help="take yoked schedules from this earlier training receipt instead of this run",
    )
    parser.add_argument(
        "--consolidation-fraction",
        type=float,
        default=0.0,
        help="TACE arms only: final fraction of the budget with every env on the full envelope",
    )
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--delta-target", type=float, default=0.778)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--integral-max", type=float, default=1.0)
    parser.add_argument("--return-floor", type=float, default=8.0)
    parser.add_argument(
        "--latency-cap",
        type=float,
        default=0.5,
        help="cap arms: ceiling on the actuation-latency channel's share of lambda",
    )
    parser.add_argument(
        "--return-relative-drop",
        type=float,
        default=0.25,
        help="relative-guard arms: fractional fall below the trailing best that counts as harm",
    )
    parser.add_argument(
        "--return-window",
        type=int,
        default=8,
        help="relative-guard arms: how many epochs of its own history an arm is judged against",
    )
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--terminations",
        default=None,
        help=(
            "termination preset, e.g. tracking/base or tracking/eval. The stock training "
            "preset (tracking/base_adaptive_strict_ori_foot_xyz) is STRICTER than the eval "
            "preset -- 0.15 m position and 0.2 rad orientation, plus a 0.2 m foot term -- "
            "which is right for a competent policy and fatal from scratch, where 93%% of "
            "episodes die on tracking error in ~0.25 s and essentially none reach time-out."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="stream metrics to this Weights & Biases project; omit to stay offline",
    )
    parser.add_argument(
        "--motion-file",
        default=DEFAULT_MOTION_FILE,
        help="motion_lib pool every arm trains on",
    )
    parser.add_argument(
        "--smpl-motion-file",
        default=DEFAULT_SMPL_MOTION_FILE,
        help=(
            "SMPL pack for the SMPL observation encoder. 'dummy' substitutes zeros, "
            "which is also what a missing path does; hosts without the 32 GB pack "
            "must pass it explicitly so the receipt records the difference."
        ),
    )
    parser.add_argument(
        "--encoder",
        default=str(LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/curriculum_comparison",
    )
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument(
        "--receipt-dir", type=Path, default=LUCID_ROOT / "manifests"
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= args.warmup_iterations:
        parser.error("iterations must exceed warmup iterations")
    if args.from_scratch and args.checkpoint:
        parser.error("--from-scratch and --checkpoint are mutually exclusive")
    if not args.from_scratch and not args.checkpoint:
        parser.error("pass --checkpoint, or --from-scratch to train a fresh policy")
    for horizon in args.horizons or ():
        if not 0 < horizon <= args.iterations:
            parser.error(f"horizon {horizon} must be in (0, {args.iterations}]")
    return args


def arm_order(modes: list[str], seed_index: int) -> list[str]:
    """Rotate arm order by seed to avoid confounding mode with wall-clock order."""
    offset = seed_index % len(modes)
    return modes[offset:] + modes[:offset]


def build_command(
    args,
    mode: str,
    seed: int,
    branch_id: str,
    artifact_dir: Path,
    yoked_schedule: Path | None = None,
) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}")
    curriculum_mode, anchor_ratio, source = ARMS[mode]
    if curriculum_mode == "yoked" and yoked_schedule is None:
        raise ValueError(f"arm {mode!r} needs the {source!r} schedule for seed {seed}")
    capsule_dir = artifact_dir / "capsules"
    tace = (
        [
            f"++callbacks.lucid_curriculum.anchor_ratio={anchor_ratio}",
            f"++callbacks.lucid_curriculum.anchor_seed={seed}",
            f"++callbacks.lucid_curriculum.consolidation_fraction={args.consolidation_fraction}",
        ]
        if anchor_ratio > 0.0
        else []
    )
    yoked = (
        [f"++callbacks.lucid_curriculum.yoked_schedule_path={yoked_schedule}"]
        if curriculum_mode == "yoked"
        else []
    )
    overrides = [
        f"++callbacks.lucid_curriculum.term_lambda_overrides.{term}={value}"
        for term, value in ARM_TERM_OVERRIDES.get(mode, {}).items()
    ]
    caps = (
        [f"++callbacks.lucid_curriculum.term_lambda_caps.randomize_action_delay={args.latency_cap}"]
        if mode in CAP_ARMS
        else []
    )
    strata = ARM_SPREAD_STRATA.get(mode, 1)
    guard = ARM_RETURN_GUARD.get(mode, "absolute")
    spread = [f"++callbacks.lucid_curriculum.spread_strata={strata}"] if strata > 1 else []
    if strata > 1 and anchor_ratio == 0.0:
        # Strata need the cohort machinery, which the callback only installs
        # when it has a seed to draw the partition from.
        spread.append(f"++callbacks.lucid_curriculum.anchor_seed={seed}")
    relative_guard = (
        [
            f"++callbacks.lucid_curriculum.return_guard={guard}",
            f"++callbacks.lucid_curriculum.return_relative_drop={args.return_relative_drop}",
            f"++callbacks.lucid_curriculum.return_window={args.return_window}",
        ]
        if guard != "absolute"
        else []
    )
    origin = [] if getattr(args, "from_scratch", False) else [f"checkpoint={args.checkpoint}"]
    horizons = [
        f"++callbacks.practice_capsule.horizons.h{h:04d}={h}"
        for h in sorted(set(getattr(args, "horizons", None) or ()))
    ]
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "train_with_delay.py"),
        "--max-delay",
        str(args.max_delay),
        "--",
        f"+exp={args.exp}",
        *origin,
        f"num_envs={args.num_envs}",
        "headless=true",
        *(
            ["use_wandb=true", f"project_name={args.wandb_project}"]
            if getattr(args, "wandb_project", None)
            else ["use_wandb=false"]
        ),
        f"seed={seed}",
        "manager_env/events=tracking/lucid_curriculum",
        *(
            [f"manager_env/terminations={args.terminations}"]
            if getattr(args, "terminations", None)
            else []
        ),
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++algo.config.save_interval=100000",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file="
        f"{getattr(args, 'motion_file', DEFAULT_MOTION_FILE)}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
        f"{getattr(args, 'smpl_motion_file', DEFAULT_SMPL_MOTION_FILE)}",
        f"++callbacks.practice_observer._target_={OBSERVER}",
        "++callbacks.practice_observer.enabled=true",
        f"++callbacks.practice_observer.encoder_path={args.encoder}",
        f"++callbacks.practice_observer.branch_id={branch_id}",
        f"++callbacks.practice_observer.output_dir={artifact_dir}",
        f"++callbacks.lucid_curriculum._target_={CURRICULUM}",
        "++callbacks.lucid_curriculum.enabled=true",
        f"++callbacks.lucid_curriculum.mode={curriculum_mode}",
        *tace,
        *yoked,
        *overrides,
        *caps,
        *spread,
        *relative_guard,
        f"++callbacks.lucid_curriculum.observer_branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.output_dir={artifact_dir}",
        "++callbacks.lucid_curriculum.initial_lambda=0.0",
        "++callbacks.lucid_curriculum.fixed_lambda=1.0",
        f"++callbacks.lucid_curriculum.delta_target={args.delta_target}",
        f"++callbacks.lucid_curriculum.kp={args.kp}",
        f"++callbacks.lucid_curriculum.ki={args.ki}",
        f"++callbacks.lucid_curriculum.alpha={args.alpha}",
        f"++callbacks.lucid_curriculum.integral_max={args.integral_max}",
        f"++callbacks.lucid_curriculum.return_floor={args.return_floor}",
        f"++callbacks.lucid_curriculum.warmup_iterations={args.warmup_iterations}",
        f"++callbacks.practice_capsule._target_={CAPSULE}",
        "++callbacks.practice_capsule.enabled=true",
        f"++callbacks.practice_capsule.capsule_dir={capsule_dir}",
        f"++callbacks.practice_capsule.pair_id=curriculum_seed_{seed}",
        "++callbacks.practice_capsule.role=control",
        f"++callbacks.practice_capsule.branch_id={branch_id}",
        f"++callbacks.practice_capsule.horizons.final={args.iterations}",
        *horizons,
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trailing(values: dict[int, float], window: int = 4) -> float | None:
    ordered = [values[index] for index in sorted(values)]
    return statistics.fmean(ordered[-window:]) if len(ordered) >= window else None


def aggregate(arms: dict[str, dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    result = {}
    for mode in modes:
        members = [arm for arm in arms.values() if arm["mode"] == mode]
        metrics = {}
        for metric in TRAINING_METRICS:
            values = [arm["training"][metric]["last4_mean"] for arm in members]
            values = [float(value) for value in values if value is not None]
            metrics[metric] = {
                "per_seed": {
                    str(arm["seed"]): arm["training"][metric]["last4_mean"] for arm in members
                },
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        for metric in QUALITY_METRICS:
            values = [arm["observer_last4_mean"].get(metric) for arm in members]
            values = [float(value) for value in values if value is not None]
            metrics[f"observer/{metric}"] = {
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        final_lambdas = [arm.get("final_lambda") for arm in members]
        final_lambdas = [float(value) for value in final_lambdas if value is not None]
        result[mode] = {
            "num_seeds": len(members),
            "metrics": metrics,
            "final_lambda_mean": statistics.fmean(final_lambdas) if final_lambdas else None,
            "all_complete": all(arm["complete"] for arm in members),
        }
    return result


def relative(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left == 0:
        return None
    return (right - left) / abs(left)


def comparisons(summary: dict[str, Any]) -> dict[str, Any]:
    if "lucid" not in summary:
        return {}
    output = {}
    for other in ("fixed", "off"):
        if other not in summary:
            continue
        pair = {}
        for metric, floor in (("Mean rewards", REWARD_FLOOR), ("Mean length", LENGTH_FLOOR)):
            lucid = summary["lucid"]["metrics"][metric]["mean"]
            reference = summary[other]["metrics"][metric]["mean"]
            delta = relative(reference, lucid)
            pair[metric] = {
                "lucid": lucid,
                other: reference,
                "relative_lucid_minus_other": delta,
                "outside_settled_noise_floor": delta is not None and abs(delta) > floor,
            }
        output[f"lucid_vs_{other}"] = pair
    return output


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"curriculum_comparison_ne{args.num_envs}_{stamp}"
    modes = list(dict.fromkeys(args.modes))
    source_receipt = json.loads(args.yoked_source_receipt.read_text()) if args.yoked_source_receipt else None
    for mode in modes:
        source = ARMS[mode][2]
        if source is not None and source not in modes and source_receipt is None:
            raise SystemExit(f"arm {mode!r} requires its source arm {source!r} in --modes or --yoked-source-receipt")
    run_specs = []
    for seed_index, seed in enumerate(args.seeds):
        ordered = arm_order(modes, seed_index)
        # A yoked arm replays its source's schedule, so it must run after it.
        ordered = [m for m in ordered if ARMS[m][2] is None] + [m for m in ordered if ARMS[m][2]]
        for mode in ordered:
            branch_id = f"{experiment_id}_s{seed}_{mode}"
            artifact_dir = args.artifact_root / experiment_id / f"seed_{seed}" / mode
            run_specs.append((seed, mode, branch_id, artifact_dir))

    def schedule_for(seed: int, mode: str) -> Path | None:
        source = ARMS[mode][2]
        if source is None:
            return None
        source_seed = seed
        if mode in CROSS_SEED_ARMS:
            seeds = list(args.seeds)
            source_seed = seeds[(seeds.index(seed) + 1) % len(seeds)]
        if source_receipt is not None:
            for arm in source_receipt["arms"].values():
                if arm["mode"] == source and int(arm["seed"]) == source_seed:
                    return Path(arm["curriculum_path"])
            raise SystemExit(f"no {source!r} seed {source_seed} in {args.yoked_source_receipt}")
        source_branch = f"{experiment_id}_s{source_seed}_{source}"
        return args.artifact_root / experiment_id / f"seed_{source_seed}" / source / f"curriculum_{source_branch}.jsonl"

    commands = {
        branch_id: build_command(args, mode, seed, branch_id, artifact_dir, schedule_for(seed, mode))
        for seed, mode, branch_id, artifact_dir in run_specs
    }
    for seed, mode, branch_id, _ in run_specs:
        print(f"[seed={seed} mode={mode} branch={branch_id}]")
        print("\n".join(commands[branch_id]))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    runtime = {}
    arms: dict[str, dict[str, Any]] = {}
    for seed, mode, branch_id, artifact_dir in run_specs:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{branch_id}.log"
        runtime[branch_id] = LA.run_arm(commands[branch_id], log_path, args.min_free_mib)
        observer_path = artifact_dir / f"observer_{branch_id}.jsonl"
        arm = LA.summarize_arm(log_path, observer_path, args.iterations)
        curriculum_path = artifact_dir / f"curriculum_{branch_id}.jsonl"
        curriculum = read_jsonl(curriculum_path)
        capsule = artifact_dir / "capsules" / f"{branch_id}_final.capsule.pt"
        checkpoint = artifact_dir / "final_checkpoint.pt"
        if runtime[branch_id]["exit_code"] == 0 and capsule.is_file():
            BC.export_sonic_checkpoint(capsule, checkpoint)
        arm.update(
            {
                "seed": seed,
                "mode": mode,
                "branch_id": branch_id,
                "curriculum_path": str(curriculum_path),
                "curriculum_rows": len(curriculum),
                "final_lambda": curriculum[-1].get("lambda") if curriculum else None,
                "final_integral": curriculum[-1].get("integral") if curriculum else None,
                "mean_return_observed": any(
                    row.get("mean_return") is not None for row in curriculum
                ),
                "return_guard_trips": sum(bool(row.get("guard_tripped")) for row in curriculum),
                "scalable_terms": curriculum[-1].get("scalable_terms", []) if curriculum else [],
                "arm_spec": {
                    "curriculum_mode": ARMS[mode][0],
                    "anchor_ratio": ARMS[mode][1],
                    "yoked_source": ARMS[mode][2],
                    "yoked_cross_seed": mode in CROSS_SEED_ARMS,
                    "term_lambda_overrides": ARM_TERM_OVERRIDES.get(mode, {}),
                    "spread_strata": ARM_SPREAD_STRATA.get(mode, 1),
                    "return_guard": ARM_RETURN_GUARD.get(mode, "absolute"),
                    "term_lambda_caps": (
                        {"randomize_action_delay": args.latency_cap} if mode in CAP_ARMS else {}
                    ),
                    "yoked_schedule_path": str(schedule_for(seed, mode)) if ARMS[mode][2] else None,
                },
                "tace_final": curriculum[-1].get("tace") if curriculum else None,
                "consolidation_rows": sum(bool(row.get("consolidation")) for row in curriculum),
                "capsule": str(capsule),
                "checkpoint": str(checkpoint),
                "checkpoint_exported": checkpoint.is_file(),
            }
        )
        arms[branch_id] = arm
        runtime[branch_id]["log_path"] = str(log_path)
        runtime[branch_id]["observer_path"] = str(observer_path)

    mode_summary = aggregate(arms, modes)
    comparison = comparisons(mode_summary)
    expected_terms = {
        "add_joint_default_pos",
        "base_com",
        "physics_material",
        "push_robot",
        "randomize_action_delay",
        "randomize_rigid_body_mass",
    }
    mechanics_ok = all(
        runtime[branch_id]["exit_code"] == 0
        and arm["complete"]
        and arm["actuator_groups_swapped"] == 5
        and arm["checkpoint_exported"]
        and set(arm["scalable_terms"]) == expected_terms
        for branch_id, arm in arms.items()
    )
    if "lucid" in modes:
        lucid_arms = [arm for arm in arms.values() if arm["mode"] == "lucid"]
        mechanics_ok = mechanics_ok and all(arm["mean_return_observed"] for arm in lucid_arms)
    for arm in arms.values():
        ratio = ARMS[arm["mode"]][1]
        if ratio > 0.0:
            tace = arm.get("tace_final") or {}
            expected = round(ratio * args.num_envs)
            mechanics_ok = mechanics_ok and tace.get("num_anchor") == expected

    receipt = {
        "kind": "lucid_three_arm_training_comparison",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "launcher_sha256": source_sha256(),
        "config": {
            "checkpoint": (
                None if args.from_scratch else str(Path(args.checkpoint).resolve())
            ),
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "seeds": args.seeds,
            "modes": modes,
            "arm_order": [
                {"seed": seed, "modes": arm_order(modes, index)}
                for index, seed in enumerate(args.seeds)
            ],
            "event_preset": "tracking/lucid_curriculum",
            "termination_preset": (
                args.terminations or "tracking/base_adaptive_strict_ori_foot_xyz (exp default)"
            ),
            "wandb_project": args.wandb_project,
            "from_scratch": bool(args.from_scratch),
            "capsule_horizons": sorted(set(args.horizons or ())) + [args.iterations],
            "motion_file": args.motion_file,
            "smpl_motion_file": args.smpl_motion_file,
            "arms": {mode: ARMS[mode] for mode in modes},
            "consolidation_fraction": args.consolidation_fraction,
            "max_delay_steps": args.max_delay,
            "max_delay_ms": args.max_delay * 5,
            "controller": {
                "delta_target": args.delta_target,
                "kp": args.kp,
                "ki": args.ki,
                "alpha": args.alpha,
                "integral_max": args.integral_max,
                "return_floor": args.return_floor,
                "calibration": (
                    "manuscript mu+3sigma lambda=0 target; integral contribution "
                    "capped at ki*integral_max=0.02"
                ),
            },
            "training_noise_floors": {"reward": REWARD_FLOOR, "length": LENGTH_FLOOR},
        },
        "commands": commands,
        "runtime": runtime,
        "arms": arms,
        "mode_summary": mode_summary,
        "training_comparison": comparison,
        "verified": (
            [
                "every branch completed and exported a final SONIC-compatible checkpoint",
                "all five live actuator groups used delayed actuators",
                "all six DR channels were runtime-scalable",
                "LUCID received SONIC objective/rewards for its return guard",
            ]
            if mechanics_ok
            else []
        ),
        "not_yet_verified": [
            "held-out ID-clean, OOD-heavy, and 60 ms checkpoint evaluation",
            "training curves alone do not establish final policy generalization",
            *("three-seed confirmation" for _ in [0] if len(args.seeds) < 3),
        ],
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"mode_summary": mode_summary, "training_comparison": comparison}, indent=2))
    print(f"receipt {receipt_path}")
    return 0 if mechanics_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
