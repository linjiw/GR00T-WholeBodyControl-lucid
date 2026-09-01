"""Fail-closed contract and arithmetic tests for the Tier-2 support screen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from gear_sonic.research.practice_utility import dr_scaling as DS
from scripts.practice_utility import analyze_support_screen as A


@pytest.fixture(autouse=True)
def compact_training_history(monkeypatch):
    """Keep fixtures small while exercising the same all-row production audit."""
    monkeypatch.setattr(A, "EXPECTED_TRAINING_ITERATIONS", 12)
    monkeypatch.setattr(A, "EXPECTED_WARMUP_ITERATIONS", 2)


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def mutate_json(path: Path, mutation) -> None:
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest_keys(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def baseline_ranges() -> dict:
    return {
        "add_joint_default_pos": {"pos_distribution_params": [-0.01, 0.01]},
        "base_com": {
            "com_range": {
                "x": [-0.025, 0.025],
                "y": [-0.05, 0.05],
                "z": [-0.05, 0.05],
            }
        },
        "physics_material": {
            "static_friction_range": [0.3, 1.6],
            "dynamic_friction_range": [0.3, 1.2],
            "restitution_range": [0.0, 0.5],
        },
        "push_robot": {
            "velocity_range": {
                "x": [-0.5, 0.5],
                "y": [-0.5, 0.5],
                "z": [-0.2, 0.2],
                "yaw": [-0.78, 0.78],
            }
        },
        "randomize_action_delay": {"delay_range": [0.0, 8.0]},
        "randomize_rigid_body_mass": {"mass_distribution_params": [0.8, 1.5]},
    }


def make_tace() -> dict:
    counts = {f"focus_s{index}": count for index, count in enumerate(A.EXPECTED_STRATUM_SIZES)}
    dispatch = {}
    for term in sorted(A.EXPECTED_SCALABLE_TERMS):
        baseline = baseline_ranges()[term]
        params = []
        for dose in A.EXPECTED_STRATUM_LAMBDAS[:-1]:
            value = DS.scaled_term_params(baseline, dose, allow_extrapolation=True)
            value, _ = DS.clamp_params_physical(value)
            params.append(value)
        dispatch[term] = {
            "term": term,
            "num_strata": 8,
            "stratum_params": [*params, None],
            "env_counts": counts,
            "anchor_params": baseline,
        }
    return {
        "num_anchor": 0,
        "num_focus": 1024,
        "anchor_ratio": 0.0,
        "num_strata": 8,
        "stratum_sizes": A.EXPECTED_STRATUM_SIZES,
        "stratum_lambdas": A.EXPECTED_STRATUM_LAMBDAS,
        "consolidating": False,
        "dispatch": dispatch,
    }


def write_curriculum(path: Path, role: str) -> Path:
    fixed_lambda = 1.0 if role in ("historical_fixed", "fresh_fixed") else 1.5
    rows = []
    for step in range(1, A.EXPECTED_TRAINING_ITERATIONS + 1):
        row = {
            "global_step": step,
            "mode": "fixed",
            "lambda": fixed_lambda,
            "gap_quantile": None,
            "scalable_terms": sorted(A.EXPECTED_SCALABLE_TERMS),
        }
        if step <= A.EXPECTED_WARMUP_ITERATIONS:
            row["warmup_hold"] = True
        else:
            if role in ("fixed_150", "fixed_u150"):
                row.update(allow_extrapolation=True, physical_clamp=["physics_material"])
            if role == "fixed_u150":
                row["tace"] = make_tace()
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def write_training(tmp_path: Path, role: str, git_sha: str, trainer_sha: str):
    mode = A.ROLE_MODE[role]
    directory = tmp_path / "training" / role
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "final_checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint:{role}".encode())
    (directory / "config.yaml").write_text(f"model: {role}\n")
    curriculum = write_curriculum(directory / "curriculum.jsonl", role)
    capsule = directory / "final.capsule.pt"
    capsule.write_bytes(f"capsule:{role}".encode())
    branch = f"s8600_{mode}"
    fixed_lambda = 1.0 if role == "fresh_fixed" else 1.5
    extrapolation = role != "fresh_fixed"
    max_delay = 8 if role == "fresh_fixed" else 12
    arm = {
        "branch_id": branch,
        "seed": 8600,
        "mode": mode,
        "complete": True,
        "iterations_parsed": A.EXPECTED_TRAINING_ITERATIONS,
        "curriculum_rows": A.EXPECTED_TRAINING_ITERATIONS,
        "curriculum_path": str(curriculum.resolve()),
        "final_lambda": fixed_lambda,
        "actuator_groups_swapped": 5,
        "consolidation_rows": 0,
        "live_delay_final": {
            "action_delay_actuator_groups": 5,
            "action_delay_num_lags": 5120,
            "action_delay_min_steps": 0,
            "action_delay_max_steps": max_delay,
            "action_delay_nonzero_fraction": 0.9,
            "action_delay_histogram": [0] * max_delay + [5120],
        },
        "checkpoint_exported": True,
        "checkpoint": str(checkpoint.resolve()),
        "capsule": str(capsule.resolve()),
        "scalable_terms": sorted(A.EXPECTED_SCALABLE_TERMS),
        "arm_spec": {
            "curriculum_mode": "fixed",
            "anchor_ratio": 0.0,
            "anchor_seed": 8600 if role == "fixed_u150" else None,
            "yoked_source": None,
            "yoked_cross_seed": False,
            "term_lambda_overrides": {},
            "spread_strata": 8 if role == "fixed_u150" else 1,
            "stratum_sizes": A.EXPECTED_STRATUM_SIZES if role == "fixed_u150" else None,
            "stratum_lambdas": A.EXPECTED_STRATUM_LAMBDAS if role == "fixed_u150" else None,
            "top_fraction": 0.75 if role == "fixed_u150" else None,
            "return_guard": "absolute",
            "fixed_lambda": fixed_lambda,
            "allow_extrapolation": extrapolation,
            "physical_clamp": ["physics_material"] if extrapolation else None,
            "signal": "gap",
            "margin": None,
            "term_lambda_caps": {},
            "max_delay_steps": 12,
        },
        "tace_final": make_tace() if role == "fixed_u150" else None,
        "expand_contract": {"passed": True, "errors": []} if role == "fixed_u150" else None,
    }
    receipt = {
        "kind": "lucid_three_arm_training_comparison",
        "schema_version": 1,
        "experiment_id": f"training_{role}",
        "git_sha": git_sha,
        "git_status_short": [],
        "launcher_sha256": trainer_sha,
        "config": {
            "checkpoint": None,
            "num_envs": 1024,
            "iterations": A.EXPECTED_TRAINING_ITERATIONS,
            "warmup_iterations": A.EXPECTED_WARMUP_ITERATIONS,
            "seeds": [8600],
            "modes": [mode],
            "arm_order": [{"seed": 8600, "modes": [mode]}],
            "event_preset": "tracking/lucid_curriculum",
            "termination_thresholds": "default",
            "from_scratch": True,
            "arms": {mode: ["fixed", 0.0, None]},
            "consolidation_fraction": 0,
            "max_delay_steps": 12,
            "max_delay_ms": 60,
        },
        "commands": {branch: ["synthetic"]},
        "runtime": {branch: {"exit_code": 0}},
        "arms": {branch: arm},
        "verified": ["synthetic complete training mechanics"],
    }
    return write_json(directory / "training.json", receipt), checkpoint


def write_historical_training(tmp_path: Path):
    directory = tmp_path / "training" / "historical_fixed"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "final_checkpoint.pt"
    checkpoint.write_bytes(b"historical checkpoint")
    (directory / "config.yaml").write_text("model: historical\n")
    curriculum = write_curriculum(directory / "curriculum.jsonl", "historical_fixed")
    capsule = directory / "final.capsule.pt"
    capsule.write_bytes(b"historical capsule")
    branch = "historical_s8600_fixed"
    receipt = {
        "kind": "lucid_historical_training_cell_bridge",
        "schema_version": 1,
        "experiment_id": None,
        "config": {
            "checkpoint": None,
            "from_scratch": True,
            "num_envs": 1024,
            "iterations": A.EXPECTED_TRAINING_ITERATIONS,
            "warmup_iterations": A.EXPECTED_WARMUP_ITERATIONS,
            "seeds": [8600],
            "modes": ["fixed"],
            "termination_thresholds": "default",
            "consolidation_fraction": 0,
        },
        "arms": {
            branch: {
                "branch_id": branch,
                "seed": 8600,
                "mode": "fixed",
                "complete": True,
                "checkpoint_exported": True,
                "iterations_parsed": A.EXPECTED_TRAINING_ITERATIONS,
                "curriculum_rows": A.EXPECTED_TRAINING_ITERATIONS,
                "checkpoint": str(checkpoint.resolve()),
                "curriculum_path": str(curriculum.resolve()),
                "capsule": str(capsule.resolve()),
                "arm_spec": {
                    "curriculum_mode": "fixed",
                    "anchor_ratio": 0,
                    "spread_strata": 1,
                    "fixed_lambda": 1,
                    "allow_extrapolation": False,
                },
            }
        },
        "verified": ["historical source frozen"],
    }
    return write_json(directory / "training.json", receipt), checkpoint


def write_freeze(tmp_path, role, training_path, checkpoint, git_sha):
    training = json.loads(training_path.read_text())
    arm = next(iter(training["arms"].values()))
    config = checkpoint.parent / "config.yaml"
    curriculum, capsule = Path(arm["curriculum_path"]), Path(arm["capsule"])
    checkpoint.chmod(0o444)

    def section(path):
        return {
            "path": str(path.resolve()),
            "sha256": A.sha256(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = {
        "kind": "lucid_frozen_training_checkpoint",
        "schema_version": 1,
        "state": "frozen_for_evaluation",
        "evaluation_only": True,
        "seed": 8600,
        "mode": A.ROLE_MODE[role],
        "iterations": A.EXPECTED_TRAINING_ITERATIONS,
        "resume_forbidden": True,
        "code": {"git_sha": git_sha, "git_status_short": []},
        "checkpoint": {**section(checkpoint), "read_only": True},
        "config": section(config),
        "curriculum": {**section(curriculum), "rows": A.EXPECTED_TRAINING_ITERATIONS},
        "final_capsule": section(capsule),
        "training_receipt": section(training_path),
        "verified": ["synthetic frozen bundle"],
    }
    path = write_json(tmp_path / "freezes" / f"{role}.json", manifest)
    if role == "historical_fixed":
        path.chmod(0o444)
    return path


def write_panel(tmp_path: Path, keys: list[str]) -> Path:
    source = tmp_path / "source" / "m1_hob002.pkl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"motion")
    motion_file = tmp_path / "panel_aliases"
    motion_file.mkdir()
    for key in keys:
        (motion_file / f"{key}.pkl").symlink_to(source)
    return write_json(
        tmp_path / "panel.json",
        {
            "kind": "lucid_replicate_panel",
            "schema_version": 1,
            "motion_key": "m1_hob002",
            "source_clip": str(source.resolve()),
            "source_clip_sha256": A.sha256(source),
            "motion_file": str(motion_file.resolve()),
            "replicates": 512,
            "alias_keys_sha256": digest_keys(keys),
            "pool_sha256": "d" * 64,
            "split_sha256": "e" * 64,
            "partition": "adaptation",
            "verified": ["synthetic 512-alias panel"],
        },
    )


def write_h_r2(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "h_r2.json",
        {
            "kind": "lucid_ratchet_analysis",
            "instrument_audit": {"passed": True},
            "claim_scope": {
                "status": "three_seed_decision",
                "noninferiority_decision_eligible": True,
            },
            "preregistered_decision": {
                "status": "pass",
                "paired_training_seeds": ["8600", "8601", "8602"],
                "mechanism_pass": True,
                "capability_components_pass": True,
                "noninferiority_claim_authorized": True,
                "superiority_claim_authorized": False,
            },
            "mechanism": {"summary": {"all_available_seeds_pass": True}},
        },
    )


DEFAULT_PROFILES = {
    "historical_fixed": (0.95, 0.97, 0.70, 0.80, 0.85, 0.90, 0.50, 0.70),
    "fresh_fixed": (0.95, 0.97, 0.70, 0.80, 0.85, 0.90, 0.50, 0.70),
    "fixed_150": (0.945, 0.97, 0.735, 0.80, 0.86, 0.91, 0.56, 0.75),
    "fixed_u150": (0.95, 0.97, 0.765, 0.81, 0.86, 0.91, 0.56, 0.75),
}


def profile_dict(values):
    keys = (
        "in_success",
        "in_progress",
        "frontier_success",
        "frontier_progress",
        "lat50_success",
        "lat50_progress",
        "lat60_success",
        "lat60_progress",
    )
    return dict(zip(keys, values, strict=True))


def rates_for(profile, preset):
    overrides = profile.get("overrides", {}).get(preset, {})
    if preset in dict(A.IN_ENVELOPE_GRID):
        pair = profile["in_success"], profile["in_progress"]
    elif preset in dict(A.FRONTIER_GRID):
        pair = profile["frontier_success"], profile["frontier_progress"]
    elif preset == "lat_50ms":
        pair = profile["lat50_success"], profile["lat50_progress"]
    elif preset == "lat_60ms":
        pair = profile["lat60_success"], profile["lat60_progress"]
    else:
        pair = 0.90, 0.95
    return overrides.get("success_rate", pair[0]), overrides.get("progress_rate", pair[1])


def raw_ranges(preset):
    if preset not in A.PHYSICS_LEVELS:
        result = raw_ranges("phys_000")
        steps = float(A.LATENCY_STEPS[preset])
        result["randomize_action_delay"] = {"delay_range": [steps, steps]}
        return result
    scale, result = A.PHYSICS_LEVELS[preset], {}
    for term in A.EXPECTED_NON_LATENCY_TERMS:
        value = DS.scaled_term_params(baseline_ranges()[term], scale, allow_extrapolation=True)
        value, _ = DS.clamp_params_physical(value)
        result[term] = value
    result["randomize_action_delay"] = {"delay_range": [0.0, 0.0]}
    return result


def episode_arrays(success_target, progress_target, keys):
    successes = round(success_target * len(keys))
    failures = len(keys) - successes
    failed_progress = (progress_target * len(keys) - successes) / failures
    assert 0 <= failed_progress < 1
    terminated = [True] * failures + [False] * successes
    progress = [failed_progress] * failures + [1.0] * successes
    return terminated, progress, successes / len(keys), sum(progress) / len(keys)


def write_evaluation(
    tmp_path,
    role,
    panel,
    profile,
    training,
    freeze,
    keys,
    git_sha,
    evaluator_sha,
):
    mode = A.ROLE_MODE[role]
    directory = tmp_path / "evaluation" / role
    training_value = json.loads(training.read_text())
    freeze_value = json.loads(freeze.read_text())
    checkpoint = Path(freeze_value["checkpoint"]["path"])
    checkpoint_hash = freeze_value["checkpoint"]["sha256"]
    config_source = Path(freeze_value["config"]["path"])
    config_hash = freeze_value["config"]["sha256"]
    panel_value = json.loads(panel.read_text())
    runs, aggregate = {}, {}
    for preset in A.EXPECTED_PRESETS:
        terminated, progress, success, progress_rate = episode_arrays(
            *rates_for(profile, preset), keys
        )
        run_id = f"{role}_{preset}"
        steps = 0 if preset in A.PHYSICS_LEVELS else A.LATENCY_STEPS[preset]
        delay = {
            "action_delay_actuator_groups": 5,
            "action_delay_num_lags": 2560,
            "action_delay_min_steps": steps,
            "action_delay_max_steps": steps,
            "action_delay_mean_steps": float(steps),
            "action_delay_nonzero_fraction": 0.0 if steps == 0 else 1.0,
            "action_delay_histogram": [0] * steps + [2560],
        }
        ranges = raw_ranges(preset)
        failed = [index for index, value in enumerate(terminated) if value]
        metrics = {
            "eval/protocol/preset_id": preset,
            "eval/protocol/branch_id": run_id,
            "eval/protocol/non_latency_dr_scale": A.PHYSICS_LEVELS.get(preset),
            "eval/protocol/dr_scale_report": (
                {
                    "lambda_value": A.PHYSICS_LEVELS[preset],
                    "scaled_terms": sorted(A.EXPECTED_NON_LATENCY_TERMS),
                    "skipped_startup_terms": [],
                    "skipped_unknown_params": [],
                    "num_scaled": 5,
                }
                if preset in A.PHYSICS_LEVELS
                else None
            ),
            "eval/protocol/fixed_latency_steps": steps,
            "eval/protocol/fixed_latency_report": {
                "requested_steps": float(steps),
                "pinned_terms": ["randomize_action_delay"],
            },
            "eval/protocol/active_dr_terms": sorted(A.EXPECTED_SCALABLE_TERMS),
            "eval/protocol/dr_ranges": ranges,
            "eval/all_metrics_dict": {
                "motion_keys": keys,
                "terminated": terminated,
                "progress": progress,
            },
            "failed_idxes": failed,
            "failed_keys": [keys[index] for index in failed],
            "eval/success/success_rate": success,
            "eval/success/progress_rate": progress_rate,
            **{f"eval/delay/{key}": value for key, value in delay.items()},
        }
        metrics_path = write_json(directory / preset / "metrics_eval.json", metrics)
        summary = {
            "success_rate": success,
            "progress_rate": progress_rate,
            "motion_count": 512,
            "failed_count": len(failed),
            "active_dr_terms": sorted(A.EXPECTED_SCALABLE_TERMS),
            "dr_ranges": ranges,
            "delay": delay,
        }
        runs[run_id] = {
            "checkpoint_seed": 8600,
            "evaluation_seed": 8700,
            "mode": mode,
            "preset": preset,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "metrics_path": str(metrics_path.resolve()),
            "runtime": {"exit_code": 0},
            "summary": summary,
            "complete": True,
        }
        aggregate[preset] = {
            mode: {
                "num_runs": 1,
                "metrics": {
                    metric: {
                        "per_checkpoint_seed": {"8600": summary[metric]},
                        "mean": summary[metric],
                        "sample_std": None,
                    }
                    for metric in A.METRICS
                },
            }
        }
    receipt = {
        "kind": "lucid_frozen_checkpoint_robustness_evaluation",
        "schema_version": 1,
        "experiment_id": f"evaluation_{role}",
        "git_sha": git_sha,
        "git_status_short": [],
        "launcher_sha256": evaluator_sha,
        "training_receipt": str(training.resolve()),
        "training_experiment_id": training_value.get("experiment_id"),
        "protocol": {
            "num_envs": 512,
            "checkpoint_seeds": [8600],
            "evaluation_seed_by_checkpoint_seed": {"8600": 8700},
            "modes": [mode],
            "presets": A._expected_preset_metadata(),
            "max_delay_capacity_steps": 12,
            "physics_step_ms": 5,
            "suite": {
                "motion_count": 512,
                "motion_keys_sha256": panel_value["alias_keys_sha256"],
                "pool_sha256": panel_value["pool_sha256"],
                "split_sha256": panel_value["split_sha256"],
                "split_linkage": "replicate-panel",
                "partition": panel_value["partition"],
                "replicate_panel": {
                    "receipt": str(panel.resolve()),
                    "motion_key": panel_value["motion_key"],
                    "source_clip_sha256": panel_value["source_clip_sha256"],
                    "replicates": 512,
                    "alias_keys_sha256": panel_value["alias_keys_sha256"],
                },
            },
            "resolved_training_config": {
                "source": str(config_source.resolve()),
                "sha256": config_hash,
                "installed": [str((checkpoint.parent / "config.yaml").resolve())],
            },
            "no_learning": True,
        },
        "runs": runs,
        "mode_summary": aggregate,
        "checkpoint_sha256_before": {str(checkpoint.resolve()): checkpoint_hash},
        "checkpoint_sha256_after": {str(checkpoint.resolve()): checkpoint_hash},
        "verified": ["synthetic complete frozen evaluation"],
    }
    return write_json(directory / "evaluation.json", receipt)


def write_prereg(tmp_path, panel, h_r2, historical_training, historical_freeze, checkpoint):
    motion = Path(json.loads(panel.read_text())["source_clip"])
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"encoder")
    config = checkpoint.parent / "config.yaml"
    code_hashes = {
        relative: A.sha256(A.REPO / relative)
        for relative in (
            A.TRAINER_RELATIVE_PATH,
            A.EVALUATOR_RELATIVE_PATH,
            A.ANALYZER_RELATIVE_PATH,
        )
    }
    frozen = {
        "panel_receipt": panel,
        "h_r2_analysis": h_r2,
        "motion": motion,
        "encoder": encoder,
        "historical_fixed_training": historical_training,
        "historical_fixed_freeze_manifest": historical_freeze,
        "historical_fixed_checkpoint": checkpoint,
        "historical_fixed_config": config,
        "environment_bootstrap": A.EXPECTED_ENVIRONMENT_BOOTSTRAP,
    }
    prereg = {
        "kind": "lucid_tier2_support_screen_preregistration",
        "schema_version": 1,
        "frozen": True,
        "written_before_gpu": True,
        "code_state": {
            "worktree": str(A.REPO.resolve()),
            "clean_detached_worktree_required": True,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=A.REPO, text=True
            ).strip(),
            "file_sha256": code_hashes,
        },
        "frozen_inputs": {
            key: {"path": str(path.resolve()), "sha256": A.sha256(path)}
            for key, path in frozen.items()
        },
        "design": {
            "training": {
                "from_scratch": True,
                "seed": 8600,
                "num_envs": 1024,
                "iterations": A.EXPECTED_TRAINING_ITERATIONS,
                "warmup_iterations": A.EXPECTED_WARMUP_ITERATIONS,
                "order": ["fresh_fixed", "fixed_150", "fixed_u150"],
                "role_to_mode": {
                    "fresh_fixed": "fixed",
                    "fixed_150": "fixed_150",
                    "fixed_u150": "fixed_u150",
                },
                "max_delay_capacity_steps": 12,
                "resume_allowed": False,
            },
            "evaluation": {
                "num_envs": 512,
                "checkpoint_seed": 8600,
                "evaluation_seed": 8700,
                "roles": list(A.EVALUATION_ROLES),
                "presets": list(A.EXPECTED_PRESETS),
                "total_cells": 60,
            },
        },
        "evaluation": {
            "panel_receipt": str(panel.resolve()),
            "panel_sha256": A.sha256(panel),
            "evaluator_sha256": code_hashes[A.EVALUATOR_RELATIVE_PATH],
        },
        "analysis": {
            "script": A.ANALYZER_RELATIVE_PATH,
            "screening_only": True,
            "directional_claim_authorized": False,
            "superiority_claim_authorized": False,
        },
    }
    return write_json(tmp_path / "prereg.json", prereg)


def build_evidence(tmp_path: Path, changes=None):
    keys = [f"m1_hob002__rep{index:04d}" for index in range(512)]
    panel, h_r2 = write_panel(tmp_path, keys), write_h_r2(tmp_path)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=A.REPO, text=True).strip()
    trainer_sha = A.sha256(A.REPO / A.TRAINER_RELATIVE_PATH)
    evaluator_sha = A.sha256(A.REPO / A.EVALUATOR_RELATIVE_PATH)
    historical_training, historical_checkpoint = write_historical_training(tmp_path)
    historical_freeze = write_freeze(
        tmp_path, "historical_fixed", historical_training, historical_checkpoint, git_sha
    )
    trainings, checkpoints = {}, {}
    freezes = {"historical_fixed": historical_freeze}
    for role in A.TRAINING_ROLES:
        trainings[role], checkpoints[role] = write_training(tmp_path, role, git_sha, trainer_sha)
        freezes[role] = write_freeze(tmp_path, role, trainings[role], checkpoints[role], git_sha)
    prereg = write_prereg(
        tmp_path, panel, h_r2, historical_training, historical_freeze, historical_checkpoint
    )
    profiles = {role: profile_dict(values) for role, values in DEFAULT_PROFILES.items()}
    for role, updates in (changes or {}).items():
        profiles[role].update(updates)
    all_trainings = {"historical_fixed": historical_training, **trainings}
    evaluations = {
        role: write_evaluation(
            tmp_path,
            role,
            panel,
            profiles[role],
            all_trainings[role],
            freezes[role],
            keys,
            git_sha,
            evaluator_sha,
        )
        for role in A.EVALUATION_ROLES
    }
    kwargs = {
        **{role: evaluations[role] for role in A.EVALUATION_ROLES},
        **{f"{role}_training": trainings[role] for role in A.TRAINING_ROLES},
        "preregistration": prereg,
        "expected_preregistration_sha": A.sha256(prereg),
        **{f"{role}_freeze_manifest": freezes[role] for role in A.EVALUATION_ROLES},
    }
    return kwargs, {
        "panel": panel,
        "h_r2": h_r2,
        "prereg": prereg,
        "trainings": trainings,
        "freezes": freezes,
        "evaluations": evaluations,
    }


def repin_prereg(kwargs, evidence, key, path):
    prereg = json.loads(evidence["prereg"].read_text())
    prereg["frozen_inputs"][key] = {"path": str(path.resolve()), "sha256": A.sha256(path)}
    if key == "panel_receipt":
        prereg["evaluation"].update(panel_receipt=str(path.resolve()), panel_sha256=A.sha256(path))
    write_json(evidence["prereg"], prereg)
    kwargs["expected_preregistration_sha"] = A.sha256(evidence["prereg"])


def run_for_preset(receipt, preset):
    return next(run for run in receipt["runs"].values() if run["preset"] == preset)


def passing_profiles(fixed150_nominal, fixedu_nominal, frontier_delta=0.0):
    profiles = {
        "fixed_150": {"success_rate": {"nominal_phys_000": fixed150_nominal, "frontier_auc": 0.75}},
        "fixed_u150": {
            "success_rate": {
                "nominal_phys_000": fixedu_nominal,
                "frontier_auc": 0.75 + frontier_delta,
            }
        },
    }
    candidates = {"fixed_150": {"passed": True}, "fixed_u150": {"passed": True}}
    return profiles, candidates


def test_complete_screen_is_raw_evidence_bound_and_screening_only(tmp_path):
    kwargs, _ = build_evidence(tmp_path)
    receipt = A.analyze(**kwargs)
    assert receipt["instrument_audit"]["passed"] is True
    assert receipt["instrument_audit"]["unique_cells"] == 60
    assert receipt["decision"]["status"] == "screen_pass"
    assert receipt["decision"]["selected"] == "fixed_u150"
    assert receipt["decision"]["screening_only"] is True
    assert receipt["decision"]["superiority_claim_authorized"] is False
    assert receipt["inputs"]["training_receipts"]["fixed_u150"]["curriculum"]["tace_rows"] == 10
    assert receipt["instrument_audit"]["cross_role_live_dr"]["passed"] is True
    assert receipt["instrument_audit"]["cross_role_live_dr"]["roles"] == list(A.EVALUATION_ROLES)


def test_cross_role_canonical_semantics_normalize_only_object_key_order():
    assert A._canonical_json({"b": [1, 2], "a": {"y": 3, "x": 4}}) == A._canonical_json(
        {"a": {"x": 4, "y": 3}, "b": [1, 2]}
    )
    assert A._canonical_json({"value": 1}) != A._canonical_json({"value": 1.0})
    assert A._canonical_json({"value": [1, 2]}) != A._canonical_json({"value": [2, 1]})


def test_cross_role_live_dr_rejects_internally_coherent_role_drift(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    receipt_path = evidence["evaluations"]["fixed_u150"]
    receipt = json.loads(receipt_path.read_text())
    term = "add_joint_default_pos"
    changed_baseline = {"pos_distribution_params": [-0.02, 0.02]}
    for run in receipt["runs"].values():
        preset = run["preset"]
        scale = A.PHYSICS_LEVELS.get(preset, 0.0)
        changed = DS.scaled_term_params(changed_baseline, scale, allow_extrapolation=True)
        changed, _ = DS.clamp_params_physical(changed)
        metrics_path = Path(run["metrics_path"])
        metrics = json.loads(metrics_path.read_text())
        metrics["eval/protocol/dr_ranges"][term] = changed
        write_json(metrics_path, metrics)
        run["summary"]["dr_ranges"][term] = changed
    write_json(receipt_path, receipt)

    # The mutation is a valid self-consistent ladder for this role: phys_100
    # defines the changed baseline and every other rung is its exact scaling.
    # It must still fail because the instrument differed between roles.
    with pytest.raises(ValueError, match="differs from historical_fixed under canonical JSON"):
        A.analyze(**kwargs)


def test_preference_exact_nominal_boundary_does_not_select_fixedu():
    profiles, candidates = passing_profiles(0.92, 0.93)
    assert A.select_candidate(profiles, candidates)["selected"] == "fixed_150"


def test_preference_rejects_tiny_recovery_even_if_fixed150_lost_nominal():
    profiles, candidates = passing_profiles(0.939, 0.940)
    assert A.select_candidate(profiles, candidates)["selected"] == "fixed_150"


def test_preference_accepts_direct_phys000_gain_over_one_point():
    profiles, candidates = passing_profiles(0.920, 0.935)
    decision = A.select_candidate(profiles, candidates)
    assert decision["selected"] == "fixed_u150"
    assert decision["reason"] == "fixed_u150_phys_000_gain_gt_0.01_with_frontier_within_0.02"


def test_preference_frontier_within_exact_boundary_passes():
    profiles, candidates = passing_profiles(0.92, 0.935, frontier_delta=-0.02)
    assert A.select_candidate(profiles, candidates)["selected"] == "fixed_u150"


def test_failed_historical_bridge_blocks_selection(tmp_path):
    kwargs, _ = build_evidence(tmp_path, {"historical_fixed": {"frontier_success": 0.60}})
    receipt = A.analyze(**kwargs)
    assert receipt["historical_bridge"]["passed"] is False
    assert receipt["decision"]["status"] == "invalid_bridge"


def test_verified_must_be_a_nonempty_list(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    mutate_json(evidence["evaluations"]["fixed_150"], lambda value: value.update(verified=True))
    with pytest.raises(ValueError, match="nonempty list"):
        A.analyze(**kwargs)


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("missing", "missing one or more required frozen inputs"),
        ("path", "environment_bootstrap.path"),
        ("hash", "environment_bootstrap.sha256"),
    ],
)
def test_environment_bootstrap_is_path_and_hash_bound(tmp_path, attack, message):
    kwargs, evidence = build_evidence(tmp_path)
    prereg = json.loads(evidence["prereg"].read_text())
    entry = prereg["frozen_inputs"]["environment_bootstrap"]
    if attack == "missing":
        del prereg["frozen_inputs"]["environment_bootstrap"]
    elif attack == "path":
        foreign = tmp_path / "foreign_lucid_env.sh"
        foreign.write_bytes(A.EXPECTED_ENVIRONMENT_BOOTSTRAP.read_bytes())
        entry.update(path=str(foreign.resolve()), sha256=A.sha256(foreign))
    else:
        entry["sha256"] = "0" * 64
    write_json(evidence["prereg"], prereg)
    kwargs["expected_preregistration_sha"] = A.sha256(evidence["prereg"])

    with pytest.raises(ValueError, match=message):
        A.analyze(**kwargs)


def test_forged_h_r2_rejected_even_when_re_pinned(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    mutate_json(
        evidence["h_r2"],
        lambda value: value["preregistered_decision"].update(status="fail"),
    )
    repin_prereg(kwargs, evidence, "h_r2_analysis", evidence["h_r2"])
    with pytest.raises(ValueError, match="H_R2 decision.status"):
        A.analyze(**kwargs)


def test_forged_panel_rejected_even_when_re_pinned(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    mutate_json(evidence["panel"], lambda value: value.update(replicates=511))
    repin_prereg(kwargs, evidence, "panel_receipt", evidence["panel"])
    with pytest.raises(ValueError, match="panel.replicates"):
        A.analyze(**kwargs)


def test_mutated_alias_tree_cannot_hide_behind_frozen_panel_receipt(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    panel = json.loads(evidence["panel"].read_text())
    alias = next(Path(panel["motion_file"]).glob("*.pkl"))
    foreign = tmp_path / "foreign.pkl"
    foreign.write_bytes(Path(panel["source_clip"]).read_bytes())
    alias.unlink()
    alias.symlink_to(foreign)

    with pytest.raises(ValueError, match="canonical target"):
        A.analyze(**kwargs)


def test_historical_freeze_must_bind_preregistered_checkpoint(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    manifest = evidence["freezes"]["historical_fixed"]
    os.chmod(manifest, 0o644)
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"fake")
    fake.chmod(0o444)

    def replace_checkpoint(value):
        value["checkpoint"].update(
            path=str(fake.resolve()), sha256=A.sha256(fake), size_bytes=fake.stat().st_size
        )

    mutate_json(manifest, replace_checkpoint)
    manifest.chmod(0o444)
    repin_prereg(kwargs, evidence, "historical_fixed_freeze_manifest", manifest)
    with pytest.raises(ValueError, match="training checkpoint path|prereg binding"):
        A.analyze(**kwargs)


def test_copied_evaluation_receipt_content_is_fatal(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    evidence["evaluations"]["fixed_u150"].write_bytes(
        evidence["evaluations"]["fixed_150"].read_bytes()
    )
    with pytest.raises(ValueError, match="mode|receipt content hashes"):
        A.analyze(**kwargs)


def test_non_latency_drift_in_latency_cell_is_fatal(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    receipt_path = evidence["evaluations"]["fixed_150"]
    evaluation = json.loads(receipt_path.read_text())
    run = run_for_preset(evaluation, "lat_10ms")
    metrics_path = Path(run["metrics_path"])

    def drift(metrics):
        metrics["eval/protocol/dr_ranges"]["add_joint_default_pos"]["pos_distribution_params"] = [
            -0.2,
            0.2,
        ]

    mutate_json(metrics_path, drift)
    run["summary"]["dr_ranges"] = json.loads(metrics_path.read_text())["eval/protocol/dr_ranges"]
    write_json(receipt_path, evaluation)
    with pytest.raises(ValueError, match="clean DR"):
        A.analyze(**kwargs)


@pytest.mark.parametrize("attack", ["short", "missing_tace", "wrong_step", "wrong_lambda"])
def test_full_curriculum_history_is_binding(tmp_path, attack):
    kwargs, evidence = build_evidence(tmp_path)
    training = json.loads(evidence["trainings"]["fixed_u150"].read_text())
    path = Path(next(iter(training["arms"].values()))["curriculum_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if attack == "short":
        rows.pop()
    elif attack == "missing_tace":
        rows[-1].pop("tace")
    elif attack == "wrong_step":
        rows[-1]["global_step"] = 99
    else:
        rows[-1]["lambda"] = 1.49
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="rows|lacks TACE|global_step|lambda"):
        A.analyze(**kwargs)


@pytest.mark.parametrize("role", ["fresh_fixed", "fixed_150"])
def test_point_support_arms_are_audited_on_every_row(tmp_path, role):
    kwargs, evidence = build_evidence(tmp_path)
    training = json.loads(evidence["trainings"][role].read_text())
    path = Path(next(iter(training["arms"].values()))["curriculum_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["lambda"] = 0.5
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="lambda"):
        A.analyze(**kwargs)


def test_no_arm_may_hide_a_consolidation_row(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    training = json.loads(evidence["trainings"]["fixed_150"].read_text())
    path = Path(next(iter(training["arms"].values()))["curriculum_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["consolidation"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="consolidated"):
        A.analyze(**kwargs)


def test_dispatcher_params_are_recomputed_not_trusted_from_receipt(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    training = json.loads(evidence["trainings"]["fixed_u150"].read_text())
    path = Path(next(iter(training["arms"].values()))["curriculum_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["tace"]["dispatch"]["physics_material"]["stratum_params"][0][
        "static_friction_range"
    ] = ["garbage", "garbage"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((TypeError, ValueError)):
        A.analyze(**kwargs)


def test_early_tace_rows_may_precede_per_stratum_reset_counts(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    training = json.loads(evidence["trainings"]["fixed_u150"].read_text())
    curriculum = Path(next(iter(training["arms"].values()))["curriculum_path"])
    rows = [json.loads(line) for line in curriculum.read_text().splitlines()]
    first_postwarmup = rows[A.EXPECTED_WARMUP_ITERATIONS]
    for telemetry in first_postwarmup["tace"]["dispatch"].values():
        telemetry["env_counts"] = {"anchor": 0, "focus": 0}
    curriculum.write_text("".join(json.dumps(row) + "\n" for row in rows))

    freeze_path = evidence["freezes"]["fixed_u150"]
    freeze = json.loads(freeze_path.read_text())
    freeze["curriculum"].update(sha256=A.sha256(curriculum), size_bytes=curriculum.stat().st_size)
    write_json(freeze_path, freeze)

    receipt = A.analyze(**kwargs)
    assert receipt["instrument_audit"]["passed"] is True


def test_final_tace_requires_positive_evidence_for_every_stratum(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)

    def zero_final_count(receipt):
        arm = next(iter(receipt["arms"].values()))
        arm["tace_final"]["dispatch"]["physics_material"]["env_counts"]["focus_s0"] = 0

    mutate_json(evidence["trainings"]["fixed_u150"], zero_final_count)
    with pytest.raises(ValueError, match="focus_s0 is not positive"):
        A.analyze(**kwargs)


def test_bad_live_active_terms_is_fatal(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    receipt_path = evidence["evaluations"]["fresh_fixed"]
    evaluation = json.loads(receipt_path.read_text())
    run = run_for_preset(evaluation, "phys_150")
    metrics_path = Path(run["metrics_path"])
    terms = sorted(A.EXPECTED_SCALABLE_TERMS - {"push_robot"})
    mutate_json(
        metrics_path,
        lambda value: value.update({"eval/protocol/active_dr_terms": terms}),
    )
    run["summary"]["active_dr_terms"] = terms
    write_json(receipt_path, evaluation)
    with pytest.raises(ValueError, match="active_dr_terms"):
        A.analyze(**kwargs)


def test_bad_raw_arrays_are_fatal(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    evaluation = json.loads(evidence["evaluations"]["fixed_150"].read_text())
    metrics_path = Path(run_for_preset(evaluation, "phys_175")["metrics_path"])
    mutate_json(metrics_path, lambda value: value["eval/all_metrics_dict"]["progress"].pop())
    with pytest.raises(ValueError, match="progress length"):
        A.analyze(**kwargs)


def test_delay_histogram_is_binding(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    receipt_path = evidence["evaluations"]["fixed_u150"]
    evaluation = json.loads(receipt_path.read_text())
    run = run_for_preset(evaluation, "lat_50ms")
    metrics_path = Path(run["metrics_path"])
    mutate_json(
        metrics_path,
        lambda value: value.update({"eval/delay/action_delay_histogram": [2560]}),
    )
    run["summary"]["delay"]["action_delay_histogram"] = [2560]
    write_json(receipt_path, evaluation)
    with pytest.raises(ValueError, match="histogram"):
        A.analyze(**kwargs)


def test_raw_motion_panel_linkage_is_binding(tmp_path):
    kwargs, evidence = build_evidence(tmp_path)
    evaluation = json.loads(evidence["evaluations"]["fixed_150"].read_text())
    metrics_path = Path(run_for_preset(evaluation, "phys_125")["metrics_path"])

    def replace_key(value):
        value["eval/all_metrics_dict"]["motion_keys"][0] = "foreign_motion"
        if value["eval/all_metrics_dict"]["terminated"][0]:
            value["failed_keys"][0] = "foreign_motion"

    mutate_json(metrics_path, replace_key)
    with pytest.raises(ValueError, match="panel digest"):
        A.analyze(**kwargs)
