# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for mixed import/from-import blocks.
# ruff: noqa: I001

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.practice_utility import run_branch as R


def args(role="intervention"):
    return SimpleNamespace(
        stage="late",
        seed=0,
        role=role,
        context_index=1 if role == "intervention" else None,
        checkpoint=Path("/tmp/settled.pt"),
        capsule=Path("/tmp/settled.capsule.pt"),
        encoder=Path("/tmp/frozen_encoder.pt"),
        exp="manager/universal_token/all_modes/sonic_release",
        num_envs=128,
        motion_file="robot_filtered",
        smpl_motion_file="smpl_filtered",
        artifact_dir=Path("/tmp/artifacts"),
        pool_manifest=None,
        epsilon_override=None,
    )


def manifest():
    return {
        "campaign_id": "screen",
        "manifest_sha256": "a" * 64,
        "seeds": [0],
        "epsilon": 0.1,
        "kernel_radius_bins": 1,
        "horizons": {"H_s": 4, "H_m": 12, "H_l": 32},
        "contexts_per_stage": {
            "late": [
                {"context": {"motion_key": "first", "bin_index": 0}},
                {"context": {"motion_key": "target", "bin_index": 7}},
            ]
        },
    }


def test_resume_command_uses_absolute_horizons_and_observer():
    branch_id, overrides = R.build_overrides(args(), manifest(), capsule_step=24)

    assert branch_id == "screen_late_s0_c1_intervention"
    for expected in (
        "checkpoint=/tmp/settled.pt",
        "+resume=true",
        "++algo.config.num_learning_iterations=56",
        f"++callbacks.practice_resume._target_={R.RESUME_CALLBACK}",
        "++callbacks.practice_resume.enabled=true",
        "++callbacks.practice_resume.capsule_path=/tmp/settled.capsule.pt",
        f"++callbacks.practice_observer._target_={R.OBSERVER_CALLBACK}",
        "++callbacks.practice_observer.enabled=true",
        "++callbacks.practice_observer.encoder_path=/tmp/frozen_encoder.pt",
        "++callbacks.practice_observer.output_dir=/tmp/artifacts/screen/"
        "screen_late_s0_c1_intervention",
        "++callbacks.practice_observer.branch_id=screen_late_s0_c1_intervention",
        "++callbacks.practice_capsule.horizons.H_s=28",
        "++callbacks.practice_capsule.horizons.H_m=36",
        "++callbacks.practice_capsule.horizons.H_l=56",
    ):
        assert expected in overrides

    resume_index = overrides.index(f"++callbacks.practice_resume._target_={R.RESUME_CALLBACK}")
    context_index = overrides.index(f"++callbacks.practice_context._target_={R.CALLBACK}")
    assert resume_index < context_index


def test_control_and_intervention_restart_from_the_same_origin():
    _, control = R.build_overrides(args("control"), manifest(), capsule_step=24)
    _, intervention = R.build_overrides(args("intervention"), manifest(), capsule_step=24)
    origin_overrides = {
        "checkpoint=/tmp/settled.pt",
        "+resume=true",
        "++algo.config.num_learning_iterations=56",
        "++callbacks.practice_resume.capsule_path=/tmp/settled.capsule.pt",
    }
    assert origin_overrides <= set(control)
    assert origin_overrides <= set(intervention)


def test_capsule_step_is_validated_against_its_trainer_state(monkeypatch):
    monkeypatch.setattr(
        R.BC,
        "load_capsule",
        lambda path, restore_rng: {
            "global_step": 24,
            "trainer_state": {"global_step": 23},
        },
    )
    with pytest.raises(ValueError, match="capsule global_step mismatch"):
        R.capsule_global_step(Path("capsule.pt"))


def test_resume_origin_rejects_checkpoint_capsule_step_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "origin.pt"
    capsule = tmp_path / "origin.capsule.pt"
    checkpoint.touch()
    capsule.touch()
    monkeypatch.setattr(R, "capsule_global_step", lambda path: 24)
    monkeypatch.setattr(R, "checkpoint_global_step", lambda path: 23)

    with pytest.raises(ValueError, match="checkpoint/capsule global_step mismatch"):
        R.validate_resume_origin(checkpoint, capsule)


def test_dry_run_prints_without_launching(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest()))
    monkeypatch.setattr(R, "validate_resume_origin", lambda checkpoint, capsule: 24)

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("dry run must not launch SONIC")

    monkeypatch.setattr(subprocess, "call", fail_if_launched)
    result = R.main(
        [
            "--manifest",
            str(manifest_path),
            "--stage",
            "late",
            "--seed",
            "0",
            "--role",
            "control",
            "--checkpoint",
            str(tmp_path / "settled.pt"),
            "--capsule",
            str(tmp_path / "settled.capsule.pt"),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "origin  step 24; continuation 32 iterations" in output
    assert "+resume=true" in output
    assert "dry run; low-level execution requires --execute --exploratory" in output


def test_execute_fails_closed_without_claim_launcher(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest()))
    monkeypatch.setattr(R, "validate_resume_origin", lambda checkpoint, capsule: 24)

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("blocked claim execution must not launch SONIC")

    monkeypatch.setattr(subprocess, "call", fail_if_launched)
    with pytest.raises(SystemExit, match="claim-grade execution is blocked"):
        R.main(
            [
                "--manifest",
                str(manifest_path),
                "--stage",
                "late",
                "--seed",
                "0",
                "--role",
                "control",
                "--checkpoint",
                str(tmp_path / "settled.pt"),
                "--capsule",
                str(tmp_path / "settled.capsule.pt"),
                "--execute",
            ]
        )
