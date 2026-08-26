import json
from pathlib import Path

import pytest

from scripts.practice_utility import (
    run_curriculum_horizon_scaling as H,
    run_curriculum_robustness_eval as E,
)


def metric_arm(seed: int, mode: str, checkpoint: Path, capsule: Path) -> dict:
    return {
        "seed": seed,
        "mode": mode,
        "complete": True,
        "iterations_parsed": 32,
        "actuator_groups_swapped": 5,
        "checkpoint_exported": True,
        "mean_return_observed": mode == "lucid",
        "scalable_terms": sorted(H.EXPECTED_TERMS),
        "checkpoint": str(checkpoint),
        "capsule": str(capsule),
        "training": {metric: {"last4_mean": float(seed)} for metric in H.CC.TRAINING_METRICS},
        "observer_last4_mean": {},
        "final_lambda": {"lucid": 0.7, "fixed": 1.0, "off": 0.0}[mode],
    }


def make_historical_receipt(args, path: Path) -> dict:
    arms = {}
    runtime = {}
    commands = {}
    root = path.parent / "historical_artifacts"
    for seed_index, seed in enumerate(H.HISTORICAL_32_SEEDS):
        for mode in H.CC.arm_order(list(H.MODES), seed_index):
            branch_id = f"historical_s{seed}_{mode}"
            artifact = root / f"seed_{seed}" / mode
            artifact.mkdir(parents=True, exist_ok=True)
            checkpoint = artifact / "final_checkpoint.pt"
            capsule = artifact / "final.capsule.pt"
            checkpoint.write_bytes(f"checkpoint-{seed}-{mode}".encode())
            capsule.write_bytes(f"capsule-{seed}-{mode}".encode())
            arms[branch_id] = metric_arm(seed, mode, checkpoint, capsule)
            runtime[branch_id] = {"exit_code": 0}
            commands[branch_id] = H.CC.build_command(
                H.cc_args(args, 32), mode, seed, branch_id, artifact
            )
    payload = {
        "kind": "lucid_three_arm_training_comparison",
        "schema_version": 1,
        "experiment_id": "historical",
        "git_sha": "old",
        "git_status_short": [" M historical.py"],
        "launcher_sha256": "0" * 64,
        "config": {
            "checkpoint": str(H.resolved_file(args.checkpoint)),
            "num_envs": args.num_envs,
            "iterations": 32,
            "warmup_iterations": args.warmup_iterations,
            "seeds": list(H.HISTORICAL_32_SEEDS),
            "modes": list(H.MODES),
            "max_delay_steps": args.max_delay,
            "controller": {
                key: getattr(args, key)
                for key in (
                    "delta_target",
                    "kp",
                    "ki",
                    "alpha",
                    "integral_max",
                    "return_floor",
                )
            },
        },
        "commands": commands,
        "runtime": runtime,
        "arms": arms,
        "verified": ["fixture"],
    }
    path.write_text(json.dumps(payload))
    return payload


def campaign_args(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model.pt"
    checkpoint.write_bytes(b"model")
    (source / "config.yaml").write_text("model: fixture\n")
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"encoder")
    historical = tmp_path / "historical.json"
    args = H.parse_args(
        [
            "--campaign-id",
            "test_horizon_campaign",
            "--checkpoint",
            str(checkpoint),
            "--encoder",
            str(encoder),
            "--historical-32-receipt",
            str(historical),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--idle-samples",
            "1",
            "--idle-sample-seconds",
            "0",
        ]
    )
    make_historical_receipt(args, historical)
    return args


@pytest.fixture
def frozen_campaign(tmp_path, monkeypatch):
    args = campaign_args(tmp_path)
    monkeypatch.setattr(H, "git_identity", lambda: {"sha": "clean", "status_short": []})
    preregistration, status = H.create_campaign(args)
    return args, preregistration, status


def test_exact_matrix_is_deterministic_and_has_51_branches():
    left = H.build_specs("campaign")
    right = H.build_specs("campaign")
    assert left == right
    assert len(left) == 51
    assert len({spec.branch_id for spec in left}) == 51
    assert sum(spec.budget_iterations for spec in left) == 6912
    by_budget = {
        budget: sorted({spec.seed for spec in left if spec.budget_iterations == budget})
        for budget, _ in H.HORIZON_MATRIX
    }
    assert by_budget == {
        32: [8603, 8604],
        64: [8600, 8601, 8602, 8603, 8604],
        128: [8600, 8601, 8602, 8603, 8604],
        256: [8600, 8601, 8602, 8603, 8604],
    }


def test_execute_requires_a_prior_dry_run():
    with pytest.raises(SystemExit):
        H.parse_args(["--execute"])


def test_create_campaign_freezes_prereg_and_paths_exclusively(tmp_path, monkeypatch):
    args = campaign_args(tmp_path)
    monkeypatch.setattr(H, "git_identity", lambda: {"sha": "clean", "status_short": []})
    preregistration, status = H.create_campaign(args)
    H.verify_preregistration(preregistration)
    assert status["state"] == "preregistered"
    assert H.status_counts(status)["pending"] == 51
    assert Path(preregistration["paths"]["preregistration"]).is_file()
    assert Path(preregistration["paths"]["index"]).is_file()
    with pytest.raises(H.CampaignError, match="already exists"):
        H.create_campaign(args)


def test_preregistration_hash_rejects_mutation(frozen_campaign):
    _, preregistration, _ = frozen_campaign
    preregistration["training_config"]["kp"] = 9.0
    with pytest.raises(H.CampaignError, match="content hash"):
        H.verify_preregistration(preregistration)


def test_hash_bound_input_change_blocks_resume(frozen_campaign, monkeypatch):
    _, preregistration, _ = frozen_campaign
    monkeypatch.setattr(H, "git_identity", lambda: preregistration["git"])
    Path(preregistration["training_config"]["encoder"]).write_bytes(b"changed")
    with pytest.raises(H.CampaignError, match="encoder"):
        H.verify_current_inputs(preregistration)


def test_gpu_idle_gate_rejects_any_compute_process(monkeypatch, tmp_path):
    args = campaign_args(tmp_path)
    monkeypatch.setattr(
        H,
        "gpu_idle_sample",
        lambda: {
            "gpu": {"free_mib": 32000.0, "gpu_util_pct": 0.0},
            "compute_processes": [{"pid": 10, "process_name": "foreign", "used_memory_mib": 1.0}],
        },
    )
    with pytest.raises(H.GpuNotIdleError, match="active compute PIDs"):
        H.audit_gpu_idle(args)


def test_gpu_idle_gate_requires_every_sample(monkeypatch, tmp_path):
    args = campaign_args(tmp_path)
    samples = iter(
        [
            {"gpu": {"free_mib": 32000.0, "gpu_util_pct": 0.0}, "compute_processes": []},
            {"gpu": {"free_mib": 32000.0, "gpu_util_pct": 10.0}, "compute_processes": []},
        ]
    )
    args.idle_samples = 2
    monkeypatch.setattr(H, "gpu_idle_sample", lambda: next(samples))
    with pytest.raises(H.GpuNotIdleError, match="utilization"):
        H.audit_gpu_idle(args)


def test_gpu_block_is_receipted_before_any_training(frozen_campaign, monkeypatch):
    args, preregistration, status = frozen_campaign
    spec = H.prereg_specs(preregistration)[0]

    def blocked(_args):
        assert Path(preregistration["paths"]["preregistration"]).is_file()
        current = H.read_json(preregistration["paths"]["status"])
        assert current["branches"][spec.branch_id]["state"] == "running"
        raise H.GpuNotIdleError("foreign process")

    monkeypatch.setattr(H, "audit_gpu_idle", blocked)
    with pytest.raises(H.GpuNotIdleError):
        H.run_one_branch(args, preregistration, status, spec)
    persisted = H.read_json(preregistration["paths"]["status"])
    assert persisted["state"] == "blocked"
    assert persisted["branches"][spec.branch_id]["state"] == "blocked"
    assert persisted["branches"][spec.branch_id]["attempts"][0]["state"] == "blocked"


def test_resume_verification_rejects_tampered_completed_artifact(tmp_path, monkeypatch):
    spec = H.BranchSpec(32, 8603, "lucid", 0, "branch")
    artifacts = {}
    for label in ("training_log", "observer", "curriculum", "capsule", "checkpoint"):
        path = tmp_path / label
        path.write_bytes(label.encode())
        artifacts[label] = H.artifact_binding(path)
    record = {
        "spec_sha256": spec.spec_sha256,
        "runtime": {"exit_code": 0},
        "arm": {
            "complete": True,
            "iterations_parsed": 32,
            "actuator_groups_swapped": 5,
            "checkpoint_exported": True,
            "scalable_terms": sorted(H.EXPECTED_TERMS),
            "mean_return_observed": True,
            "capsule": artifacts["capsule"]["path"],
        },
        "artifact_hashes": artifacts,
    }
    monkeypatch.setattr(
        H.BC,
        "load_capsule",
        lambda *_args, **_kwargs: {"branch_id": "branch", "global_step": 32},
    )
    H.verify_completed_branch(spec, record)
    Path(artifacts["checkpoint"]["path"]).write_bytes(b"tampered")
    with pytest.raises(H.CampaignError, match="artifact changed"):
        H.verify_completed_branch(spec, record)


def test_stale_running_attempt_becomes_interrupted():
    status = {
        "branches": {
            "branch": {
                "state": "running",
                "attempts": [{"state": "running", "started_at": "before"}],
            }
        }
    }
    H.recover_stale_running(status)
    branch = status["branches"]["branch"]
    assert branch["state"] == "interrupted"
    assert branch["attempts"][0]["state"] == "interrupted"
    assert branch["attempts"][0]["error"]["type"] == "StaleRunningAttempt"


def test_combined_32_receipt_uses_evaluator_training_interface(frozen_campaign):
    _, preregistration, status = frozen_campaign
    specs = [spec for spec in H.prereg_specs(preregistration) if spec.budget_iterations == 32]
    root = Path(preregistration["paths"]["campaign_root"])
    for spec in specs:
        checkpoint = root / f"{spec.branch_id}.pt"
        capsule = root / f"{spec.branch_id}.capsule.pt"
        checkpoint.write_bytes(b"checkpoint")
        capsule.write_bytes(b"capsule")
        status["branches"][spec.branch_id].update(
            {
                "state": "complete",
                "completed": {
                    "arm": metric_arm(spec.seed, spec.mode, checkpoint, capsule),
                    "runtime": {"exit_code": 0},
                    "command": ["python", "train"],
                    "artifact_hashes": {},
                },
            }
        )
    historical = H.read_json(preregistration["historical_32"]["receipt_path"])
    combined = H.training_receipt(
        preregistration, status, H.prereg_specs(preregistration), 32, historical_payload=historical
    )
    assert combined["status"] == "complete"
    assert combined["config"]["seeds"] == [8600, 8601, 8602, 8603, 8604]
    assert [entry["git_sha"] for entry in combined["git_lineage"]] == ["clean", "old"]
    assert combined["launcher_lineage"][1] == {
        "stratum": "historical_32_seeds_8600_8602",
        "single_budget_launcher_sha256": "0" * 64,
        "same_as_current_single_budget_launcher": False,
    }
    assert len(combined["historical_32_artifact_bindings"]) == 9
    assert len(combined["historical_32_command_sha256_by_branch"]) == 9
    index = E.checkpoint_index(combined)
    assert len(index) == 15
    assert set(index) == {(seed, mode) for seed in range(8600, 8605) for mode in H.MODES}
