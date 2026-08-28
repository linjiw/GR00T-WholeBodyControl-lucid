#!/usr/bin/env python3
"""Evaluate frozen LUCID curriculum checkpoints under matched deployment DR."""

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

from scripts.practice_utility import run_latency_ab as LA  # noqa: E402
from scripts.practice_utility import run_throughput_probe as TP
from gear_sonic.research.practice_utility.paths import LUCID_ROOT, relocate  # noqa: E402

MODES = (
    "lucid", "fixed", "off", "origin", "fixed_nolat", "fixed_latonly",
    "ta_lucid_25", "ta_lucid_50", "ta_yoked_25", "ta_yoked_50",
    "ta_yoked_25x", "ta_yoked_50x",
    "lucid_s4", "lucid_rg", "lucid_s4_rg", "ta_lucid_50_s4_rg",
    "lucid_latcap_s4_rg", "ta_lucid_50_latcap_s4_rg",
)
PRESETS = {
    "id_clean": "tracking/lucid_eval_clean",
    "dr_full": "tracking/lucid_curriculum",
    "latency_60ms": "tracking/lucid_eval_latency_60ms",
    # Robustness-profile cells: full latency envelope with the five non-latency
    # channels scaled to a fraction of their training maximum.
    "dr_025": "tracking/lucid_curriculum",
    "dr_050": "tracking/lucid_curriculum",
    "dr_075": "tracking/lucid_curriculum",
    # Past the training envelope. A deployment claim is about conditions the
    # randomization did not anticipate, so the profile must not stop at 1.
    "dr_125": "tracking/lucid_curriculum",
    "dr_150": "tracking/lucid_curriculum",
}
PRESET_DR_SCALE = {
    "dr_025": 0.25,
    "dr_050": 0.5,
    "dr_075": 0.75,
    "dr_125": 1.25,
    "dr_150": 1.5,
}
CALLBACK = "gear_sonic.research.practice_utility.eval_callback.PracticeRobustnessEvalCallback"
SUMMARY_METRICS = (
    "success_rate",
    "progress_rate",
    "mpjpe_g",
    "mpjpe_l",
    "foot_slip_per_step_m",
    "undesired_contact_rate",
    "torque_saturation",
    "energy_proxy",
)
HIGHER_IS_BETTER = {"success_rate", "progress_rate"}
EXPECTED_DR_TERMS = {
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
        "--training-receipt",
        type=Path,
        default=LUCID_ROOT / "manifests/curriculum_comparison_ne128_20260820_143058.json",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8600, 8601, 8602])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=None,
        help="defaults to the arms present in the training receipt",
    )
    parser.add_argument(
        "--presets", nargs="+", choices=tuple(PRESETS), default=["id_clean", "dr_full", "latency_60ms"]
    )
    parser.add_argument("--eval-seed-base", type=int, default=8700)
    parser.add_argument("--max-delay", type=int, default=12)
    parser.add_argument(
        "--training-config",
        type=Path,
        help="resolved SONIC config.yaml; defaults to the source checkpoint's config",
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=LUCID_ROOT / "manifests/pool_debug512.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=LUCID_ROOT / "manifests/split_debug512_content.json",
    )
    parser.add_argument("--partition", default="dev")
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=LUCID_ROOT / "pools/debug512/content_dev",
    )
    parser.add_argument(
        "--smpl-motion-file",
        default="data/motion_lib_bones_seed/smpl_filtered",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/curriculum_robustness_eval",
    )
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument(
        "--receipt-dir", type=Path, default=LUCID_ROOT / "manifests"
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def materialize_suite(
    pool_manifest: Path,
    split_manifest: Path,
    partition: str,
    suite_root: Path,
) -> dict[str, Any]:
    """Create a stable symlink-only motion panel and verify every target."""
    pool = load_json(pool_manifest)
    split = load_json(split_manifest)
    if split["pool_sha256"] != pool["pool_sha256"]:
        raise ValueError("pool and split manifests do not match")
    selected = {key for key, assigned in split["assignment"].items() if assigned == partition}
    motion_by_key = {row["motion_key"]: row for row in pool["motions"]}
    missing = sorted(selected - motion_by_key.keys())
    if missing:
        raise ValueError(f"split keys missing from pool: {missing[:3]}")

    motion_dir = suite_root / "robot_filtered"
    motion_dir.mkdir(parents=True, exist_ok=True)
    for key in sorted(selected):
        source = relocate(motion_by_key[key]["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        link = motion_dir / f"{key}.pkl"
        if link.is_symlink():
            if link.resolve() != source:
                raise ValueError(f"existing suite link points elsewhere: {link}")
        elif link.exists():
            raise ValueError(f"suite path is not a symlink: {link}")
        else:
            link.symlink_to(source)

    actual = {path.stem for path in motion_dir.glob("*.pkl")}
    extras = sorted(actual - selected)
    if extras:
        raise ValueError(f"suite contains motions outside frozen partition: {extras[:3]}")
    return {
        "motion_file": str(motion_dir.resolve()),
        "motion_count": len(selected),
        "motion_keys_sha256": hashlib.sha256(
            ("\n".join(sorted(selected)) + "\n").encode()
        ).hexdigest(),
        "pool_sha256": pool["pool_sha256"],
        "split_sha256": split["split_sha256"],
        "split_linkage": split["linkage"],
        "partition": partition,
    }


def checkpoint_index(training_receipt: dict[str, Any]) -> dict[tuple[int, str], Path]:
    index = {}
    for arm in training_receipt["arms"].values():
        index[(int(arm["seed"]), arm["mode"])] = Path(arm["checkpoint"]).resolve()
    return index


def ensure_checkpoint_configs(
    checkpoints: list[Path], training_receipt: dict[str, Any], explicit: Path | None = None
) -> dict[str, Any]:
    """Expose the resolved architecture config beside each exported checkpoint."""
    source = explicit
    if source is None:
        source_checkpoint = Path(training_receipt["config"]["checkpoint"]).resolve()
        source = source_checkpoint.parent / "config.yaml"
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"resolved training config not found: {source}")
    source_hash = file_sha256(source)
    installed = []
    for checkpoint in sorted(set(checkpoints)):
        destination = checkpoint.parent / "config.yaml"
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or file_sha256(destination) != source_hash:
                raise ValueError(f"checkpoint has a different config.yaml: {destination}")
        else:
            destination.symlink_to(source)
        installed.append(str(destination))
    return {"source": str(source), "sha256": source_hash, "installed": installed}


def rotated(items: list[str], offset: int) -> list[str]:
    if not items:
        return []
    offset %= len(items)
    return items[offset:] + items[:offset]


def build_command(
    args: argparse.Namespace,
    checkpoint: Path,
    mode: str,
    preset: str,
    eval_seed: int,
    branch_id: str,
    output_dir: Path,
    motion_file: str,
) -> list[str]:
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "eval_with_delay.py"),
        "--max-delay",
        str(args.max_delay),
        "--",
        f"checkpoint={checkpoint}",
        f"+num_envs={args.num_envs}",
        "+headless=true",
        "+use_wandb=false",
        f"+seed={eval_seed}",
        f"+manager_env/events={PRESETS[preset]}",
        "+use_encoder=g1",
        "+eval_callbacks=[practice_eval]",
        "+run_eval_loop=false",
        "++manager_env.config.train_only_events=[]",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
        f"++callbacks.practice_eval._target_={CALLBACK}",
        "++callbacks.practice_eval.eval_frequency=1",
        "++callbacks.practice_eval.eval_only=true",
        f"++callbacks.practice_eval.output_dir={output_dir}",
        f"++callbacks.practice_eval.preset_id={preset}",
        f"++callbacks.practice_eval.branch_id={branch_id}",
        *(
            [f"++callbacks.practice_eval.non_latency_dr_scale={PRESET_DR_SCALE[preset]}"]
            if preset in PRESET_DR_SCALE
            else []
        ),
    ]


def scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    all_dict = metrics.get("eval/all_metrics_dict", {})
    return {
        "success_rate": scalar(metrics.get("eval/success/success_rate")),
        "progress_rate": scalar(metrics.get("eval/success/progress_rate")),
        "mpjpe_g": scalar(metrics.get("eval/all/mpjpe_g")),
        "mpjpe_l": scalar(metrics.get("eval/all/mpjpe_l")),
        "foot_slip_per_step_m": scalar(metrics.get("eval/quality/foot_slip_per_step_m")),
        "undesired_contact_rate": scalar(metrics.get("eval/quality/undesired_contact_rate")),
        "torque_saturation": scalar(metrics.get("eval/quality/torque_saturation")),
        "energy_proxy": scalar(metrics.get("eval/quality/energy_proxy")),
        "quality_missing_signals": metrics.get("eval/quality/missing_signals", []),
        "active_dr_terms": metrics.get("eval/protocol/active_dr_terms", []),
        "dr_ranges": metrics.get("eval/protocol/dr_ranges", {}),
        "delay": {
            key.removeprefix("eval/delay/"): value
            for key, value in metrics.items()
            if key.startswith("eval/delay/")
        },
        "motion_count": len(all_dict.get("motion_keys", [])),
        "failed_count": len(metrics.get("failed_keys", [])),
    }


def aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for preset in PRESETS:
        preset_runs = [run for run in runs.values() if run["preset"] == preset and run["complete"]]
        if not preset_runs:
            continue
        grouped[preset] = {}
        for mode in MODES:
            members = [run for run in preset_runs if run["mode"] == mode]
            if not members:
                continue
            metric_summary = {}
            for metric in SUMMARY_METRICS:
                per_seed = {
                    str(run["checkpoint_seed"]): run["summary"].get(metric) for run in members
                }
                values = [float(value) for value in per_seed.values() if value is not None]
                metric_summary[metric] = {
                    "per_checkpoint_seed": per_seed,
                    "mean": statistics.fmean(values) if values else None,
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                }
            grouped[preset][mode] = {"num_runs": len(members), "metrics": metric_summary}
    return grouped


def paired_comparisons(summary: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for preset, modes in summary.items():
        comparisons[preset] = {}
        pairs = [
            (treatment, other)
            for treatment in modes
            for other in ("fixed", "off", "lucid")
            if other in modes and treatment != other and treatment not in ("fixed", "off")
        ]
        for treatment, other in pairs:
            metrics = {}
            for metric in SUMMARY_METRICS:
                lucid = modes[treatment]["metrics"][metric]["per_checkpoint_seed"]
                reference = modes[other]["metrics"][metric]["per_checkpoint_seed"]
                common = sorted(set(lucid) & set(reference))
                deltas = {
                    seed: float(lucid[seed]) - float(reference[seed])
                    for seed in common
                    if lucid[seed] is not None and reference[seed] is not None
                }
                values = list(deltas.values())
                metrics[metric] = {
                    "treatment_minus_reference_per_seed": deltas,
                    "lucid_minus_reference_per_seed": deltas,
                    "mean_delta": statistics.fmean(values) if values else None,
                    "favorable_direction": "positive" if metric in HIGHER_IS_BETTER else "negative",
                }
            comparisons[preset][f"{treatment}_vs_{other}"] = metrics
    return comparisons


def delay_matches(preset: str, summary: dict[str, Any]) -> bool:
    delay = summary.get("delay", {})
    if delay.get("action_delay_actuator_groups") != 5:
        return False
    if preset == "id_clean":
        return delay.get("action_delay_max_steps") == 0
    if preset == "dr_full" or preset in PRESET_DR_SCALE:
        return (
            delay.get("action_delay_min_steps", -1) >= 0
            and delay.get("action_delay_max_steps") == 8
            and delay.get("action_delay_nonzero_fraction", 0) > 0
        )
    return delay.get("action_delay_min_steps") == 12 and delay.get("action_delay_max_steps") == 12


def main(argv=None) -> int:
    args = parse_args(argv)
    training_receipt = load_json(args.training_receipt)
    suite = materialize_suite(
        args.pool_manifest, args.split_manifest, args.partition, args.suite_root
    )
    checkpoints = checkpoint_index(training_receipt)
    receipt_modes = list(dict.fromkeys(arm["mode"] for arm in training_receipt["arms"].values()))
    modes = list(dict.fromkeys(args.modes)) if args.modes else receipt_modes
    presets = list(dict.fromkeys(args.presets))
    specs = []
    for seed_index, checkpoint_seed in enumerate(args.seeds):
        eval_seed = args.eval_seed_base + seed_index
        for preset in rotated(presets, seed_index):
            for mode in rotated(modes, seed_index):
                checkpoint = checkpoints.get((checkpoint_seed, mode))
                if checkpoint is None or not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"checkpoint missing for seed={checkpoint_seed} mode={mode}"
                    )
                specs.append((checkpoint_seed, eval_seed, mode, preset, checkpoint))
    training_config = ensure_checkpoint_configs(
        [row[4] for row in specs], training_receipt, args.training_config
    )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"curriculum_robustness_ne{args.num_envs}_{stamp}"
    experiment_root = args.artifact_root / experiment_id
    commands = {}
    output_dirs = {}
    for checkpoint_seed, eval_seed, mode, preset, checkpoint in specs:
        branch_id = f"{experiment_id}_s{checkpoint_seed}_{mode}_{preset}"
        output_dir = experiment_root / f"seed_{checkpoint_seed}" / mode / preset
        output_dirs[branch_id] = output_dir
        commands[branch_id] = build_command(
            args,
            checkpoint,
            mode,
            preset,
            eval_seed,
            branch_id,
            output_dir,
            suite["motion_file"],
        )
        print(f"[{branch_id}]\n" + "\n".join(commands[branch_id]))
    if not args.execute:
        print(json.dumps({"suite": suite, "num_runs": len(specs)}, indent=2))
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes_before = {
        str(checkpoint): file_sha256(checkpoint)
        for checkpoint in sorted(set(row[4] for row in specs))
    }
    runs: dict[str, dict[str, Any]] = {}
    receipt_path = args.receipt_dir / f"{experiment_id}.json"

    def make_receipt() -> dict[str, Any]:
        summary = aggregate(runs)
        complete = len(runs) == len(specs) and all(run["complete"] for run in runs.values())
        mechanisms = complete and all(
            delay_matches(run["preset"], run["summary"])
            and set(run["summary"].get("active_dr_terms", [])) == EXPECTED_DR_TERMS
            for run in runs.values()
        )
        hashes_after = {path: file_sha256(Path(path)) for path in checkpoint_hashes_before}
        frozen = checkpoint_hashes_before == hashes_after
        return {
            "kind": "lucid_frozen_checkpoint_robustness_evaluation",
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment_id": experiment_id,
            "git_sha": TP.git_sha(),
            "git_status_short": TP.git_status(),
            "launcher_sha256": file_sha256(Path(__file__)),
            "training_receipt": str(args.training_receipt.resolve()),
            "training_experiment_id": training_receipt.get("experiment_id"),
            "protocol": {
                "estimand": "frozen-policy robustness; training reward is excluded",
                "num_envs": args.num_envs,
                "checkpoint_seeds": args.seeds,
                "evaluation_seed_by_checkpoint_seed": {
                    str(seed): args.eval_seed_base + index for index, seed in enumerate(args.seeds)
                },
                "modes": modes,
                "presets": {
                    "id_clean": "six channels collapsed to LUCID lambda=0 nominal",
                    "dr_full": "fresh draws from the complete six-channel training envelope",
                    "latency_60ms": (
                        "full five non-latency DR channels plus fixed 60 ms latency, "
                        "beyond the 0-40 ms train range"
                    ),
                },
                "max_delay_capacity_steps": args.max_delay,
                "physics_step_ms": 5,
                "suite": suite,
                "resolved_training_config": training_config,
                "motion_generalization_claim": (
                    "none: the frozen dev panel was included in the 512-motion training pool; "
                    "this is a fresh-physics and latency robustness evaluation"
                ),
                "primary_outcomes": ["success_rate", "progress_rate", "mpjpe_g", "mpjpe_l"],
                "secondary_batch_diagnostics": [
                    "foot_slip_per_step_m",
                    "undesired_contact_rate",
                    "torque_saturation",
                    "energy_proxy",
                ],
                "no_learning": True,
            },
            "commands": commands,
            "runs": runs,
            "mode_summary": summary,
            "paired_comparisons": paired_comparisons(summary),
            "checkpoint_sha256_before": checkpoint_hashes_before,
            "checkpoint_sha256_after": hashes_after,
            "verified": (
                [
                    "all frozen checkpoints completed the matched motion panel",
                    "all six DR terms remained active in evaluation",
                    "all runs used five delayed-actuator groups with the prescribed live lag",
                    "checkpoint hashes were unchanged by evaluation",
                    "mode comparisons use matched checkpoint and evaluation seeds",
                ]
                if mechanisms and frozen
                else []
            ),
            "not_yet_verified": [
                *([] if complete else ["the full requested evaluation matrix"]),
                "unseen-motion generalization",
                (
                    "episode-masked physical-quality comparison; current batch diagnostics "
                    "include auto-reset environments after their scored motion terminates"
                ),
                "hardware transfer or real-world safety",
                "statistical significance beyond three checkpoint seeds",
            ],
        }

    try:
        for checkpoint_seed, eval_seed, mode, preset, checkpoint in specs:
            branch_id = f"{experiment_id}_s{checkpoint_seed}_{mode}_{preset}"
            output_dir = output_dirs[branch_id]
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.log_dir / f"{branch_id}.log"
            runtime = LA.run_arm(commands[branch_id], log_path, args.min_free_mib)
            metrics_path = output_dir / "metrics_eval.json"
            metrics = load_json(metrics_path) if metrics_path.is_file() else {}
            summary = summarize_metrics(metrics) if metrics else {}
            run_complete = (
                runtime["exit_code"] == 0
                and metrics_path.is_file()
                and summary.get("motion_count") == suite["motion_count"]
            )
            runs[branch_id] = {
                "checkpoint_seed": checkpoint_seed,
                "evaluation_seed": eval_seed,
                "mode": mode,
                "preset": preset,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hashes_before[str(checkpoint)],
                "metrics_path": str(metrics_path),
                "log_path": str(log_path),
                "runtime": runtime,
                "summary": summary,
                "complete": run_complete,
            }
            receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")
            print(json.dumps({"branch_id": branch_id, "summary": summary}, indent=2), flush=True)
    finally:
        receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")

    receipt = make_receipt()
    print(json.dumps(receipt["mode_summary"], indent=2))
    print(f"receipt {receipt_path}")
    return 0 if receipt["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
