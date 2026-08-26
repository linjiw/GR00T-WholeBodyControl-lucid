"""Fail-closed contracts for the live passive-dose smoke driver."""

# Ruff's import combiner disagrees with the repository's isort profile.
# ruff: noqa: I001

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gear_sonic.research.practice_utility import branch_capsule as BC
from gear_sonic.research.practice_utility import dose_plan as DP
from gear_sonic.research.practice_utility.rng_capsule import RngState
from gear_sonic.research.practice_utility.schema import ContextKey
from scripts.practice_utility import run_passive_dose_smoke as S


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def fixture_args(tmp_path: Path, *, run_id: str = "smoke_test") -> list[str]:
    contexts = [
        ContextKey("motion_a", "a" * 64, 0, 0, 50),
        ContextKey("motion_b", "b" * 64, 1, 50, 100),
    ]
    manifest = {
        "kind": "practice_utility_probe_manifest",
        "schema_version": 1,
        "campaign_id": "probe_test",
        "stages": ["late"],
        "seeds": [9300],
        "epsilon": 0.1,
        "kernel_radius_bins": 1,
        "horizons": {"H_s": 4, "H_l": 32},
        "pool_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "contexts_per_stage": {
            "late": [
                {"context": context.to_dict(), "context_id": context.context_id}
                for context in contexts
            ]
        },
    }
    manifest["manifest_sha256"] = DP.probe_manifest_claim_sha256(manifest)
    manifest_path = write_json(tmp_path / "manifest.json", manifest)

    capsule = tmp_path / "origin.capsule.pt"
    provenance = BC.Provenance(
        resolved_config_sha256="1" * 64,
        motion_pool_manifest_sha256=manifest["pool_sha256"],
        dev_suite_sha256=manifest["split_sha256"],
        source_commit="2" * 40,
        checkpoint_sha256="3" * 64,
    )
    BC.save_capsule(
        capsule,
        branch_id="origin_s9300",
        pair_id="origin_s9300",
        role="control",
        global_step=56,
        model_state={
            "policy_state_dict": {"w": torch.tensor([1.0])},
            "value_state_dict": {"w": torch.tensor([2.0])},
            "combined_state_dict": {"w": torch.tensor([1.0])},
            "lr_scheduler_state_dict": {"last_epoch": 56},
        },
        optimizer_state={"state": {0: {"step": 56}}},
        trainer_state={"global_step": 56, "trainer_state_obj": {"global_step": 56}},
        env_state={"episode_count": 1},
        native_sampler_state={
            "adp_samp_num_episodes": torch.ones(2),
            "adp_samp_num_failures": torch.ones(2),
        },
        rng_state=RngState.capture("origin_s9300"),
        provenance=provenance,
    )
    checkpoint = tmp_path / "origin_checkpoint.pt"
    BC.export_sonic_checkpoint(capsule, checkpoint)
    snapshot = write_json(
        tmp_path / "origin_snapshot.json",
        {
            "kind": "practice_utility_sampler_snapshot",
            "schema_version": 1,
            "global_step": 56,
            "contexts": [
                {
                    **context.to_dict(),
                    "context_id": context.context_id,
                    "global_bin_id": index,
                }
                for index, context in enumerate(contexts)
            ],
        },
    )
    origin_row = {
        "origin_step": 56,
        "source_step": 24,
        "capsule": str(capsule),
        "capsule_sha256": S.file_sha256(capsule),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": S.file_sha256(checkpoint),
        "snapshot": str(snapshot),
        "snapshot_sha256": S.file_sha256(snapshot),
        "resident_context_ids": [context.context_id for context in contexts],
        "settled": True,
        "seed": 9300,
        "stage": "late",
        "num_envs": 256,
        "blockers": [],
    }
    robot_root = tmp_path / "robot"
    smpl_root = tmp_path / "smpl"
    robot_root.mkdir()
    smpl_root.mkdir()
    (robot_root / "motion_a.pkl").write_bytes(b"robot")
    (smpl_root / "motion_a.pkl").write_bytes(b"smpl")
    robot_binding = S.CPO.directory_tree_binding(robot_root, label="test robot")
    smpl_binding = S.CPO.directory_tree_binding(smpl_root, label="test smpl")
    origin_map = {
        "kind": "practice_utility_probe_origin_map",
        "schema_version": 1,
        "stage": "late",
        "origin_step": 56,
        "num_envs": 256,
        "motion_pool_manifest_sha256": manifest["pool_sha256"],
        "dev_suite_sha256": manifest["split_sha256"],
        "motion_lib_target_fps": 50.0,
        "motion_sources": {
            "robot": robot_binding,
            "smpl": smpl_binding,
        },
        "seeds": [9300],
        "origins": {"9300": origin_row},
        "common_resident_context_ids": [context.context_id for context in contexts],
        "usable_for_manifest_selection": True,
    }
    origin_map_path = write_json(tmp_path / "origin_map.json", origin_map)

    plan = DP.build_passive_dose_plan_payload(
        manifest,
        manifest_file_sha256=S.file_sha256(manifest_path),
        sigma_frames=50.0,
        created_at="2026-08-26T00:00:00+00:00",
        source_manifest_path=str(manifest_path),
        source_commit="f" * 40,
        git_status_short=[],
        launcher_path="/frozen/create_passive_dose_plan.py",
        launcher_sha256="e" * 64,
    )
    plan_path = write_json(tmp_path / "dose_plan.json", plan)
    return [
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        S.file_sha256(manifest_path),
        "--origin-map",
        str(origin_map_path),
        "--origin-map-sha256",
        S.file_sha256(origin_map_path),
        "--dose-plan",
        str(plan_path),
        "--dose-plan-sha256",
        S.file_sha256(plan_path),
        "--stage",
        "late",
        "--seed",
        "9300",
        "--run-id",
        run_id,
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--receipt-dir",
        str(tmp_path / "receipts"),
        "--gpu-lock",
        str(tmp_path / "gpu.lock"),
        "--idle-samples",
        "1",
        "--idle-sample-seconds",
        "0",
    ]


def test_command_wires_claim_control_resume_and_source_num_steps(tmp_path):
    args = S.parse_args(fixture_args(tmp_path))
    prepared = S.prepare(args)
    command = list(prepared.command)

    assert prepared.origin_step == 56
    assert prepared.short_horizon == 4
    assert prepared.target_step == 60
    assert prepared.num_steps_per_env == 24
    assert "+resume=true" in command
    assert "++algo.config.num_learning_iterations=60" in command
    assert f"checkpoint={prepared.runtime_checkpoint_path}" in command
    assert f"++callbacks.practice_resume.capsule_path={prepared.capsule_asset.path}" in command
    assert prepared.runtime_checkpoint_path.parent == prepared.artifact_dir
    assert "++callbacks.practice_context.role=control" in command
    assert "++callbacks.practice_context.epsilon=0.0" in command
    assert "++callbacks.practice_context.claim_mode=true" in command
    assert "++callbacks.practice_context.dose_report_horizons.H_s=60" in command
    assert "++callbacks.practice_context.dose_num_steps_per_iteration=24" in command
    assert not any(item.startswith("++algo.config.num_steps_per_env=") for item in command)


def test_input_hash_tamper_is_rejected(tmp_path):
    argv = fixture_args(tmp_path)
    manifest = Path(argv[argv.index("--manifest") + 1])
    manifest.write_text(manifest.read_text() + "\n")

    with pytest.raises(S.SmokeError, match="manifest file hash mismatch"):
        S.prepare(S.parse_args(argv))


def test_blocked_gpu_writes_fail_closed_status_after_preregistration(tmp_path, monkeypatch):
    args = S.parse_args(fixture_args(tmp_path) + ["--execute"])
    monkeypatch.setattr(S, "git_status", lambda: [])
    monkeypatch.setattr(S, "git_sha", lambda: "e" * 40)
    prepared = S.prepare(args)
    sample = {
        "gpu": {
            "total_mib": 32000.0,
            "used_mib": 2000.0,
            "free_mib": 30000.0,
            "gpu_util_pct": 0.0,
            "memory_util_pct": 0.0,
        },
        "compute_processes": [{"pid": 7, "process_name": "foreign", "used_memory_mib": 1.0}],
    }
    monkeypatch.setattr(
        S,
        "audit_gpu_idle",
        lambda _args: (_ for _ in ()).throw(S.GpuNotIdleError("busy", [sample])),
    )

    assert S.execute(prepared, args) == 2
    status = json.loads(prepared.status_path.read_text())
    assert status["state"] == "blocked"
    assert status["error"]["type"] == "GpuNotIdleError"
    assert prepared.preregistration_path.is_file()
    assert prepared.runtime_checkpoint_path.is_file()
    assert S.file_sha256(prepared.runtime_checkpoint_path) == prepared.checkpoint_asset.sha256
    assert not prepared.log_path.exists()


def test_missing_dose_receipt_fails_closed_and_writes_blocked_smoke(tmp_path, monkeypatch):
    args = S.parse_args(fixture_args(tmp_path) + ["--execute"])
    monkeypatch.setattr(S, "git_status", lambda: [])
    monkeypatch.setattr(S, "git_sha", lambda: "e" * 40)
    prepared = S.prepare(args)
    idle = {
        "gpu": {
            "total_mib": 32000.0,
            "used_mib": 1000.0,
            "free_mib": 31000.0,
            "gpu_util_pct": 0.0,
            "memory_util_pct": 0.0,
        },
        "compute_processes": [],
    }
    monkeypatch.setattr(S, "audit_gpu_idle", lambda _args: [idle])

    def fake_run(_command, log_path, _initial_gpu):
        log_path.write_text("fake successful simulator\n")
        return {"exit_code": 0, "wall_seconds": 1.0, "cuda_memory_growth_mib": 2000.0}

    monkeypatch.setattr(S, "run_command", fake_run)

    assert S.execute(prepared, args) == 1
    status = json.loads(prepared.status_path.read_text())
    smoke = json.loads(prepared.smoke_path.read_text())
    assert status["state"] == "failed"
    assert smoke["status"] == "blocked"
    assert smoke["checks"]["atomic_receipt_no_partial"] is False
    assert any("missing exact H_s dose receipt" in item for item in smoke["blockers"])


def test_complete_smoke_schema_limits_checks_to_live_observations(tmp_path):
    prepared = S.prepare(S.parse_args(fixture_args(tmp_path)))
    observed = {
        "valid": True,
        "checks": {
            "positive_exact_total": True,
            "exact_hook_calls": True,
            "exact_observations": True,
            "per_bin_total": True,
            "control": True,
            "registry_stable": True,
            "context_coverage": True,
            "atomic": True,
        },
        "expected_context_ids_sha256": "a" * 64,
        "actual_context_ids_sha256": "a" * 64,
        "cuda_execution_verified": True,
        "dose_receipt": {"path": "/frozen/dose.json", "sha256": "b" * 64},
        "blockers": [],
    }

    preregistration = write_json(tmp_path / "smoke.preregistration.json", {"frozen": True})
    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.log_path.write_text("live simulator log\n")
    smoke = S.smoke_payload(
        prepared,
        observed,
        S.Asset(preregistration, S.file_sha256(preregistration)),
        {"exit_code": 0, "wall_seconds": 1.0, "cuda_memory_growth_mib": 1024.0},
    )

    assert smoke["status"] == "complete"
    assert set(smoke["checks"]) == {
        "passive_completion_exact",
        "epsilon_zero_control",
        "dose_registry_stable",
        "exact_context_projection",
        "cuda_execution_verified",
        "atomic_receipt_no_partial",
        "immutable_preregistration_bound",
        "successful_runtime_and_log",
    }
    assert all(smoke["checks"].values())
    assert (
        "native distribution bitwise identity against a paired no-callback reference"
        in smoke["not_yet_verified"]
    )


def test_context_projection_postflight_is_recomputed_not_trusted(tmp_path):
    prepared = S.prepare(S.parse_args(fixture_args(tmp_path)))
    registry = "9" * 64
    completed = {"0": 7.0, "1": 11.0}
    rows = []
    for bin_id, context in enumerate(prepared.dose_plan.contexts_for(prepared.stage)):
        membership = {str(bin_id): 1.0}
        row = {
            "context": context.to_dict(),
            "context_id": context.context_id,
            "kernel_radius_bins": prepared.dose_plan.kernel_radius_bins,
            "sigma_frames": prepared.dose_plan.sigma_frames,
            "completed_kernel_steps": completed[str(bin_id)],
            "membership_by_global_bin": membership,
        }
        row["kernel_membership_sha256"] = S.sha256_of(
            {
                "context_id": context.context_id,
                "kernel_radius_bins": prepared.dose_plan.kernel_radius_bins,
                "sigma_frames": prepared.dose_plan.sigma_frames,
                "membership_by_global_bin": membership,
                "dose_registry_sha256": registry,
            }
        )
        rows.append(row)
    dose = {
        "per_bin_completed": completed,
        "dose_registry_sha256_at_report": registry,
        "context_doses": rows,
    }

    valid, actual = S._validate_context_projections(prepared, dose)
    assert valid
    assert set(actual) == {
        context.context_id for context in prepared.dose_plan.contexts_for("late")
    }

    rows[0]["completed_kernel_steps"] += 1.0
    valid, _ = S._validate_context_projections(prepared, dose)
    assert not valid


def test_execute_rechecks_assets_after_prepare_before_writing_receipts(tmp_path, monkeypatch):
    args = S.parse_args(fixture_args(tmp_path) + ["--execute"])
    monkeypatch.setattr(S, "git_status", lambda: [])
    monkeypatch.setattr(S, "git_sha", lambda: "e" * 40)
    prepared = S.prepare(args)
    prepared.snapshot_asset.path.write_text(prepared.snapshot_asset.path.read_text() + "\n")

    with pytest.raises(S.SmokeError, match="frozen snapshot changed after preparation"):
        S.execute(prepared, args)
    assert not prepared.preregistration_path.exists()


def test_execute_rechecks_git_status_after_prepare(tmp_path, monkeypatch):
    args = S.parse_args(fixture_args(tmp_path) + ["--execute"])
    statuses = iter([[], [" M gear_sonic/research/practice_utility/callbacks.py"]])
    monkeypatch.setattr(S, "git_status", lambda: next(statuses))
    monkeypatch.setattr(S, "git_sha", lambda: "e" * 40)
    prepared = S.prepare(args)

    with pytest.raises(S.SmokeError, match="source tree status changed after preparation"):
        S.execute(prepared, args)
    assert not prepared.preregistration_path.exists()


def test_execute_rehashes_motion_trees_before_reserving_outputs(tmp_path, monkeypatch):
    args = S.parse_args(fixture_args(tmp_path) + ["--execute"])
    monkeypatch.setattr(S, "git_status", lambda: [])
    monkeypatch.setattr(S, "git_sha", lambda: "e" * 40)
    prepared = S.prepare(args)
    robot_root = Path(prepared.motion_source_bindings["robot"]["resolved_path"])
    (robot_root / "motion_a.pkl").write_bytes(b"changed robot bytes")

    with pytest.raises(S.SmokeError, match="robot motion source differs"):
        S.execute(prepared, args)
    assert not prepared.preregistration_path.exists()
