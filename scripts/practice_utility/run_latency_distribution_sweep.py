#!/usr/bin/env python3
"""Preregister and run a frozen-policy sweep over synthetic latency processes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.practice_utility import run_curriculum_robustness_eval as CR
from scripts.practice_utility import run_latency_ab as LA
from scripts.practice_utility import run_throughput_probe as TP
from gear_sonic.research.practice_utility.paths import LUCID_ROOT, relocate  # noqa: E402

MODES = ("lucid", "fixed", "off")
EXPECTED_RESET_TERMS = CR.EXPECTED_DR_TERMS
EXPECTED_JITTER_TERMS = EXPECTED_RESET_TERMS | {"randomize_action_delay_interval"}

PROCESS_SPECS = (
    {
        "id": "fixed_00",
        "kind": "reset",
        "range": [0, 0],
        "distribution": "uniform",
        "coupling": "common",
    },
    {
        "id": "fixed_04",
        "kind": "reset",
        "range": [4, 4],
        "distribution": "uniform",
        "coupling": "common",
    },
    {
        "id": "fixed_08",
        "kind": "reset",
        "range": [8, 8],
        "distribution": "uniform",
        "coupling": "common",
    },
    {
        "id": "fixed_12",
        "kind": "reset",
        "range": [12, 12],
        "distribution": "uniform",
        "coupling": "common",
    },
    {
        "id": "episode_uniform_common_08",
        "kind": "reset",
        "range": [0, 8],
        "distribution": "uniform",
        "coupling": "common",
    },
    {
        "id": "episode_uniform_common_12",
        "kind": "reset",
        "range": [0, 12],
        "distribution": "uniform",
        "coupling": "common",
    },
    {
        "id": "episode_uniform_independent_08",
        "kind": "reset",
        "range": [0, 8],
        "distribution": "uniform",
        "coupling": "independent",
    },
    {
        "id": "jitter_uniform_common_08",
        "kind": "jitter",
        "range": [0, 8],
        "distribution": "uniform",
        "coupling": "common",
        "interval_range_s": [0.2, 0.5],
    },
    {
        "id": "jitter_uniform_common_12",
        "kind": "jitter",
        "range": [0, 12],
        "distribution": "uniform",
        "coupling": "common",
        "interval_range_s": [0.2, 0.5],
    },
    {
        "id": "burst_common_12_p10",
        "kind": "burst",
        "range": [0, 12],
        "distribution": "two_point",
        "coupling": "common",
        "high_probability": 0.1,
        "interval_range_s": [0.2, 0.5],
    },
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("discovery", "confirmation"), default="discovery")
    parser.add_argument("--discovery-receipt", type=Path)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--training-receipt", type=Path, default=CR.parse_args([]).training_receipt)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--discovery-seed", type=int, default=8600)
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", default=[8600, 8601, 8602])
    parser.add_argument("--eval-seed-base", type=int)
    parser.add_argument("--discovery-target-motions", type=int, default=18)
    parser.add_argument("--panel-salt", default="lucid-latency-surface-v1")
    parser.add_argument("--max-confirmation-cells", type=int, default=2)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--pool-manifest", type=Path, default=CR.parse_args([]).pool_manifest)
    parser.add_argument("--split-manifest", type=Path, default=CR.parse_args([]).split_manifest)
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=LUCID_ROOT / "pools/debug512/latency_surface",
    )
    parser.add_argument("--smpl-motion-file", default="data/motion_lib_bones_seed/smpl_filtered")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/latency_distribution_sweep",
    )
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument(
        "--receipt-dir", type=Path, default=LUCID_ROOT / "manifests"
    )
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument(
        "--capacity-wait-minutes",
        type=float,
        default=0.0,
        help="wait this long for the free-memory gate before failing",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def hash_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def build_cells() -> list[dict[str, Any]]:
    cells = []
    for scale in (0.0, 1.0):
        for process in PROCESS_SPECS:
            cell = dict(process)
            cell["non_latency_dr_scale"] = scale
            cell["cell_id"] = f"dr{int(scale * 100):03d}_{process['id']}"
            cell["training_matched"] = (
                scale == 1.0 and process["id"] == "episode_uniform_independent_08"
            )
            cells.append(cell)
    for process_id in ("episode_uniform_common_08", "jitter_uniform_common_08"):
        process = next(row for row in PROCESS_SPECS if row["id"] == process_id)
        cell = dict(process)
        cell["non_latency_dr_scale"] = 0.5
        cell["cell_id"] = f"dr050_{process_id}"
        cell["training_matched"] = False
        cells.append(cell)
    return cells


def select_panel_keys(
    pool: dict[str, Any],
    split: dict[str, Any],
    target: int,
    salt: str,
) -> dict[str, list[str]]:
    """Make an outcome-blind, canonical-content-disjoint dev sub-split."""
    records = {row["motion_key"]: row for row in pool["motions"]}
    dev_keys = sorted(key for key, part in split["assignment"].items() if part == "dev")
    groups: dict[str, list[str]] = {}
    for key in dev_keys:
        groups.setdefault(records[key]["canonical_name"], []).append(key)
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{salt}:{group}".encode()).hexdigest(),
    )
    discovery_groups = []
    count = 0
    for group in ordered:
        if count >= target:
            break
        discovery_groups.append(group)
        count += len(groups[group])
    discovery = sorted(key for group in discovery_groups for key in groups[group])
    confirmation = sorted(set(dev_keys) - set(discovery))
    return {"discovery": discovery, "confirmation": confirmation}


def materialize_panel(args: argparse.Namespace, role: str) -> dict[str, Any]:
    pool, split = CR.load_json(args.pool_manifest), CR.load_json(args.split_manifest)
    if pool["pool_sha256"] != split["pool_sha256"]:
        raise ValueError("pool and split manifests do not match")
    panels = select_panel_keys(pool, split, args.discovery_target_motions, args.panel_salt)
    selected = panels[role]
    records = {row["motion_key"]: row for row in pool["motions"]}
    motion_dir = args.suite_root / role / "robot_filtered"
    motion_dir.mkdir(parents=True, exist_ok=True)
    for key in selected:
        source = relocate(records[key]["path"]).resolve()
        link = motion_dir / f"{key}.pkl"
        if link.is_symlink():
            if link.resolve() != source:
                raise ValueError(f"panel link points elsewhere: {link}")
        elif link.exists():
            raise ValueError(f"panel entry is not a symlink: {link}")
        else:
            link.symlink_to(source)
    actual = {path.stem for path in motion_dir.glob("*.pkl")}
    if actual != set(selected):
        raise ValueError("materialized panel does not exactly match preregistered keys")
    other_role = "confirmation" if role == "discovery" else "discovery"
    return {
        "role": role,
        "motion_file": str(motion_dir.resolve()),
        "motion_count": len(selected),
        "motion_keys": selected,
        "motion_keys_sha256": hash_lines(selected),
        "canonical_groups": sorted({records[key]["canonical_name"] for key in selected}),
        "canonical_group_count": len({records[key]["canonical_name"] for key in selected}),
        "canonical_group_overlap_with_other_panel": sorted(
            {records[key]["canonical_name"] for key in selected}
            & {records[key]["canonical_name"] for key in panels[other_role]}
        ),
        "pool_sha256": pool["pool_sha256"],
        "split_sha256": split["split_sha256"],
        "selection": f"SHA-256 order of canonical content groups with salt {args.panel_salt!r}",
    }


def delay_overrides(cell: dict[str, Any]) -> list[str]:
    values = cell["range"]
    reset_range = [0, 0] if cell["kind"] == "burst" else values
    reset_distribution = "uniform" if cell["kind"] == "burst" else cell["distribution"]
    overrides = [
        f"++manager_env.events.randomize_action_delay.params.delay_range=[{reset_range[0]},{reset_range[1]}]",
        f"++manager_env.events.randomize_action_delay.params.distribution={reset_distribution}",
        f"++manager_env.events.randomize_action_delay.params.coupling={cell['coupling']}",
    ]
    if cell["kind"] in ("jitter", "burst"):
        interval = cell["interval_range_s"]
        overrides.extend(
            [
                "++manager_env.events.randomize_action_delay_interval.params."
                f"delay_range=[{values[0]},{values[1]}]",
                "++manager_env.events.randomize_action_delay_interval.params."
                f"distribution={cell['distribution']}",
                "++manager_env.events.randomize_action_delay_interval.params."
                f"coupling={cell['coupling']}",
                "++manager_env.events.randomize_action_delay_interval."
                f"interval_range_s=[{interval[0]},{interval[1]}]",
            ]
        )
        if cell["kind"] == "burst":
            overrides.append(
                "++manager_env.events.randomize_action_delay_interval.params."
                f"high_probability={cell['high_probability']}"
            )
    return overrides


def build_command(
    args: argparse.Namespace,
    checkpoint: Path,
    mode: str,
    cell: dict[str, Any],
    eval_seed: int,
    branch_id: str,
    output_dir: Path,
    motion_file: str,
) -> list[str]:
    preset = (
        "tracking/lucid_eval_latency_jitter"
        if cell["kind"] in ("jitter", "burst")
        else "tracking/lucid_curriculum"
    )
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "eval_with_delay.py"),
        "--max-delay",
        "12",
        "--",
        f"checkpoint={checkpoint}",
        f"+num_envs={args.num_envs}",
        "+headless=true",
        "+use_wandb=false",
        f"+seed={eval_seed}",
        f"+manager_env/events={preset}",
        "+use_encoder=g1",
        "+eval_callbacks=[practice_eval]",
        "+run_eval_loop=false",
        "++manager_env.config.train_only_events=[]",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
        f"++callbacks.practice_eval._target_={CR.CALLBACK}",
        "++callbacks.practice_eval.eval_frequency=1",
        "++callbacks.practice_eval.eval_only=true",
        f"++callbacks.practice_eval.output_dir={output_dir}",
        f"++callbacks.practice_eval.preset_id={cell['cell_id']}",
        f"++callbacks.practice_eval.branch_id={branch_id}",
        "++callbacks.practice_eval.non_latency_dr_scale=" f"{cell['non_latency_dr_scale']}",
        *delay_overrides(cell),
    ]


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = CR.summarize_metrics(metrics)
    all_metrics = metrics.get("eval/all_metrics_dict", {})
    keys = all_metrics.get("motion_keys", [])
    terminated = all_metrics.get("terminated", [])
    progress = all_metrics.get("progress", [])
    summary["motion_outcomes"] = {
        key: {"success": not bool(failed), "progress": float(prog)}
        for key, failed, prog in zip(keys, terminated, progress)
    }
    summary["non_latency_dr_scale"] = metrics.get("eval/protocol/non_latency_dr_scale")
    summary["dr_scale_report"] = metrics.get("eval/protocol/dr_scale_report")
    return summary


def aggregate(cells: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for cell in cells:
        cell_runs = [
            run for run in runs.values() if run["cell_id"] == cell["cell_id"] and run["complete"]
        ]
        result[cell["cell_id"]] = {"cell": cell, "modes": {}}
        for mode in MODES:
            members = [run for run in cell_runs if run["mode"] == mode]
            metrics = {}
            for metric in CR.SUMMARY_METRICS:
                per_seed = {
                    str(run["checkpoint_seed"]): run["summary"].get(metric) for run in members
                }
                values = [float(value) for value in per_seed.values() if value is not None]
                metrics[metric] = {
                    "per_checkpoint_seed": per_seed,
                    "mean": statistics.fmean(values) if values else None,
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                }
            result[cell["cell_id"]]["modes"][mode] = {
                "num_runs": len(members),
                "metrics": metrics,
            }
        result[cell["cell_id"]]["paired"] = paired_cell(result[cell["cell_id"]]["modes"])
    return result


def paired_cell(modes: dict[str, Any]) -> dict[str, Any]:
    paired = {}
    if "lucid" not in modes:
        return paired
    for reference in ("fixed", "off"):
        if reference not in modes:
            continue
        paired[reference] = {}
        for metric in CR.SUMMARY_METRICS:
            lucid = modes["lucid"]["metrics"][metric]["per_checkpoint_seed"]
            other = modes[reference]["metrics"][metric]["per_checkpoint_seed"]
            common = sorted(set(lucid) & set(other))
            deltas = {
                seed: float(lucid[seed]) - float(other[seed])
                for seed in common
                if lucid[seed] is not None and other[seed] is not None
            }
            paired[reference][metric] = {
                "per_checkpoint_seed": deltas,
                "mean_delta": statistics.fmean(deltas.values()) if deltas else None,
                "positive_seed_count": sum(value > 0 for value in deltas.values()),
            }
    return paired


def select_confirmation_cells(summary: dict[str, Any], limit: int = 2) -> list[str]:
    """Apply the preregistered discovery rule; no metric-dependent fallback."""
    eligible = []
    for cell_id, row in summary.items():
        paired = row.get("paired", {})
        if not all(reference in paired for reference in ("fixed", "off")):
            continue
        success = [paired[ref]["success_rate"]["mean_delta"] for ref in ("fixed", "off")]
        progress = [paired[ref]["progress_rate"]["mean_delta"] for ref in ("fixed", "off")]
        if any(value is None for value in success + progress):
            continue
        if min(success) >= 0.0 and min(progress) >= 0.01:
            eligible.append((min(success), min(progress), cell_id))
    eligible.sort(reverse=True)
    return [cell_id for _, _, cell_id in eligible[:limit]]


def mechanism_matches(cell: dict[str, Any], summary: dict[str, Any]) -> bool:
    delay = summary.get("delay", {})
    expected_terms = (
        EXPECTED_JITTER_TERMS if cell["kind"] in ("jitter", "burst") else EXPECTED_RESET_TERMS
    )
    if set(summary.get("active_dr_terms", [])) != expected_terms:
        return False
    if delay.get("action_delay_actuator_groups") != 5:
        return False
    if delay.get("action_delay_process_assignments", 0) <= 0:
        return False
    if summary.get("non_latency_dr_scale") != cell["non_latency_dr_scale"]:
        return False
    if (
        cell["coupling"] == "common"
        and delay.get("action_delay_process_cross_group_equal_fraction") != 1.0
    ):
        return False
    if (
        cell["coupling"] == "independent"
        and cell["range"][0] != cell["range"][1]
        and delay.get("action_delay_process_cross_group_equal_fraction", 1.0) >= 1.0
    ):
        return False
    if cell["kind"] == "reset" and cell["range"][0] == cell["range"][1]:
        return (
            delay.get("action_delay_min_steps") == cell["range"][0]
            and delay.get("action_delay_max_steps") == cell["range"][1]
        )
    if cell["kind"] in ("jitter", "burst"):
        counts = delay.get("action_delay_process_distribution_counts", {})
        return counts.get(cell["distribution"], 0) > 0
    return True


def wait_for_capacity(min_free_mib: int, wait_minutes: float, poll_seconds: float = 30.0) -> None:
    """Wait boundedly for a shared GPU without touching other users' processes."""
    deadline = time.monotonic() + max(0.0, wait_minutes) * 60.0
    while True:
        snapshot = TP.gpu_snapshot()
        if snapshot["free_mib"] >= min_free_mib:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"GPU capacity gate failed: {snapshot['free_mib']:.0f} MiB free "
                f"< {min_free_mib} MiB"
            )
        print(
            f"[capacity] {snapshot['free_mib']:.0f} MiB free < {min_free_mib} MiB; "
            f"waiting up to {remaining / 60.0:.1f} more minutes",
            flush=True,
        )
        time.sleep(min(poll_seconds, remaining, 60.0))


def main(argv=None) -> int:
    args = parse_args(argv)
    resumed = CR.load_json(args.resume_receipt) if args.resume_receipt else None
    if args.stage == "confirmation" and args.discovery_receipt is None:
        raise ValueError("--discovery-receipt is required for confirmation")
    if resumed is not None and resumed.get("stage") != args.stage:
        raise ValueError("resume receipt stage does not match --stage")
    training = CR.load_json(args.training_receipt)
    all_cells = build_cells()
    if args.stage == "discovery":
        cells = all_cells
        seeds = [args.discovery_seed]
        panel_role = "discovery"
        eval_seed_base = args.eval_seed_base or 9100
    else:
        discovery = CR.load_json(args.discovery_receipt)
        selected = discovery.get("selected_confirmation_cells", [])
        cells = [cell for cell in all_cells if cell["cell_id"] in selected]
        seeds = list(args.confirmation_seeds)
        panel_role = "confirmation"
        eval_seed_base = args.eval_seed_base or 9200
    if args.cells:
        requested = set(args.cells)
        cells = [cell for cell in cells if cell["cell_id"] in requested]
        missing = requested - {cell["cell_id"] for cell in cells}
        if missing:
            raise ValueError(f"unknown or unavailable cells: {sorted(missing)}")
    if not cells:
        raise ValueError("no cells selected")

    panel = materialize_panel(args, panel_role)
    if panel["canonical_group_overlap_with_other_panel"]:
        raise ValueError("discovery and confirmation panels leak canonical content")
    checkpoints = CR.checkpoint_index(training)
    checkpoint_paths = []
    modes = list(dict.fromkeys(args.modes))
    for seed in seeds:
        for mode in modes:
            checkpoint = checkpoints.get((seed, mode))
            if checkpoint is None or not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint missing for seed={seed} mode={mode}")
            checkpoint_paths.append(checkpoint)
    training_config = CR.ensure_checkpoint_configs(checkpoint_paths, training)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = (
        resumed["experiment_id"]
        if resumed is not None
        else f"latency_distribution_{args.stage}_ne{args.num_envs}_{stamp}"
    )
    experiment_root = args.artifact_root / experiment_id
    specs = []
    commands, output_dirs = {}, {}
    for seed_index, seed in enumerate(seeds):
        eval_seed = eval_seed_base + seed_index
        for cell_index, cell in enumerate(cells):
            for mode in CR.rotated(modes, seed_index + cell_index):
                checkpoint = checkpoints[(seed, mode)]
                branch_id = f"{experiment_id}_s{seed}_{mode}_{cell['cell_id']}"
                output_dir = experiment_root / f"seed_{seed}" / mode / cell["cell_id"]
                command = build_command(
                    args,
                    checkpoint,
                    mode,
                    cell,
                    eval_seed,
                    branch_id,
                    output_dir,
                    panel["motion_file"],
                )
                specs.append((seed, eval_seed, mode, cell, checkpoint, branch_id))
                commands[branch_id] = command
                output_dirs[branch_id] = output_dir

    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes_before = {
        str(path): CR.file_sha256(path) for path in sorted(set(checkpoint_paths))
    }
    preregistration = {
        "kind": "lucid_latency_distribution_preregistration",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "stage": args.stage,
        "source_training_receipt": str(args.training_receipt.resolve()),
        "source_training_experiment_id": training["experiment_id"],
        "discovery_receipt": (
            str(args.discovery_receipt.resolve()) if args.discovery_receipt else None
        ),
        "panel": panel,
        "cells": cells,
        "checkpoint_seeds": seeds,
        "modes": modes,
        "evaluation_seed_by_checkpoint_seed": {
            str(seed): eval_seed_base + index for index, seed in enumerate(seeds)
        },
        "selection_rule": {
            "eligible": (
                "LUCID success is no worse than both references and LUCID progress exceeds "
                "both by at least 0.01 on discovery"
            ),
            "ranking": "descending minimum success margin, then minimum progress margin",
            "maximum_cells": args.max_confirmation_cells,
            "confirmation": (
                "disjoint canonical-content panel, all three retained checkpoint seeds, "
                "fresh matched physics seeds; report every selected cell"
            ),
        },
        "primary_outcomes": ["success_rate", "progress_rate"],
        "secondary_outcomes": list(CR.SUMMARY_METRICS[2:]),
        "interpretation": (
            "discovery characterizes the response surface and may generate hypotheses; only "
            "the preregistered disjoint confirmation can support an advantage claim"
        ),
        "commands": commands,
        "checkpoint_sha256_before": checkpoint_hashes_before,
        "resolved_training_config": training_config,
    }
    if resumed is not None:
        prior_commands = resumed["protocol"]["commands"]
        if commands != prior_commands:
            raise ValueError("resume command matrix differs from the preregistered campaign")
        preregistration = resumed["protocol"]
        prereg_path = Path(resumed["preregistration"])
        prereg_hash = CR.file_sha256(prereg_path)
        if prereg_hash != resumed["preregistration_sha256"]:
            raise ValueError("resume preregistration hash changed")
    else:
        prereg_path = args.receipt_dir / f"{experiment_id}_preregistration.json"
        prereg_path.write_text(json.dumps(preregistration, indent=2) + "\n")
        prereg_hash = CR.file_sha256(prereg_path)
    print(f"preregistration {prereg_path} sha256={prereg_hash}")
    print(f"runs {len(specs)}; panel motions {panel['motion_count']}")
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = dict(resumed.get("runs", {})) if resumed else {}
    receipt_path = args.resume_receipt or (args.receipt_dir / f"{experiment_id}.json")

    def make_receipt() -> dict[str, Any]:
        summary = aggregate(cells, runs)
        complete = len(runs) == len(specs) and all(run["complete"] for run in runs.values())
        mechanisms = complete and all(
            mechanism_matches(run["cell"], run["summary"]) for run in runs.values()
        )
        hashes_after = {path: CR.file_sha256(Path(path)) for path in checkpoint_hashes_before}
        selected = (
            select_confirmation_cells(summary, args.max_confirmation_cells)
            if args.stage == "discovery" and complete
            else []
        )
        return {
            "kind": "lucid_latency_distribution_sweep",
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment_id": experiment_id,
            "stage": args.stage,
            "git_sha": TP.git_sha(),
            "git_status_short": TP.git_status(),
            "launcher_sha256": CR.file_sha256(Path(__file__)),
            "preregistration": str(prereg_path),
            "preregistration_sha256": prereg_hash,
            "protocol": preregistration,
            "runs": runs,
            "cell_summary": summary,
            "selected_confirmation_cells": selected,
            "checkpoint_sha256_after": hashes_after,
            "verified": (
                [
                    "all frozen-policy runs completed the preregistered panel",
                    "live aggregate telemetry matched every requested latency process",
                    "the five non-latency DR channels were scaled independently of latency",
                    "checkpoint hashes were unchanged",
                ]
                if mechanisms and hashes_after == checkpoint_hashes_before
                else []
            ),
            "not_yet_verified": [
                *(
                    ["holdout confirmation of discovery-selected cells"]
                    if args.stage == "discovery"
                    else []
                ),
                "unseen-motion generalization (training used all debug512 motions)",
                "hardware latency distribution or real-world safety",
            ],
        }

    try:
        for seed, eval_seed, mode, cell, checkpoint, branch_id in specs:
            if branch_id in runs and runs[branch_id].get("complete"):
                print(f"[resume] keeping completed branch {branch_id}", flush=True)
                continue
            output_dir = output_dirs[branch_id]
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.log_dir / f"{branch_id}.log"
            wait_for_capacity(args.min_free_mib, args.capacity_wait_minutes)
            runtime = LA.run_arm(commands[branch_id], log_path, args.min_free_mib)
            metrics_path = output_dir / "metrics_eval.json"
            metrics = CR.load_json(metrics_path) if metrics_path.is_file() else {}
            summary = summarize_metrics(metrics) if metrics else {}
            run_complete = (
                runtime["exit_code"] == 0
                and metrics_path.is_file()
                and summary.get("motion_count") == panel["motion_count"]
            )
            runs[branch_id] = {
                "checkpoint_seed": seed,
                "evaluation_seed": eval_seed,
                "mode": mode,
                "cell_id": cell["cell_id"],
                "cell": cell,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hashes_before[str(checkpoint)],
                "metrics_path": str(metrics_path),
                "log_path": str(log_path),
                "runtime": runtime,
                "summary": summary,
                "complete": run_complete,
            }
            receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")
            primary = {key: summary.get(key) for key in ("success_rate", "progress_rate")}
            print(json.dumps({"branch_id": branch_id, **primary}, indent=2), flush=True)
    finally:
        receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")

    receipt = make_receipt()
    print(
        json.dumps(
            {"selected_confirmation_cells": receipt["selected_confirmation_cells"]}, indent=2
        )
    )
    print(f"receipt {receipt_path}")
    return 0 if receipt["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
