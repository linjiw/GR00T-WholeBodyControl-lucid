import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.practice_utility import create_probe_origins as O


def args(tmp_path: Path):
    return SimpleNamespace(
        checkpoint=tmp_path / "settled.pt",
        stage="late",
        settle_iterations=12,
        num_envs=256,
        exp="manager/universal_token/all_modes/sonic_release",
        motion_file="robot_filtered",
        smpl_motion_file="smpl_filtered",
        snapshot_timeline_fps=50.0,
        pool_manifest=tmp_path / "pool.json",
        artifact_root=tmp_path / "artifacts",
    )


def provenance():
    return {
        "resolved_config_sha256": "a" * 64,
        "motion_pool_manifest_sha256": "b" * 64,
        "dev_suite_sha256": "c" * 64,
        "source_commit": "d" * 40,
        "checkpoint_sha256": "e" * 64,
    }


def test_origin_command_uses_absolute_step_and_captures_snapshot_capsule_together(tmp_path):
    command, paths = O.build_command(
        args(tmp_path),
        seed=9300,
        start_step=24,
        experiment_id="origins",
        provenance=provenance(),
    )
    assert "+resume=true" in command
    assert "++algo.config.num_learning_iterations=36" in command
    assert "++callbacks.practice_context.snapshot_at_step=36" in command
    assert "++callbacks.practice_capsule.horizons.origin=36" in command
    assert "seed=9300" in command
    assert f"++callbacks.practice_context.snapshot_path={paths['snapshot']}" in command
    assert f"++callbacks.practice_context.manifest_path={tmp_path / 'pool.json'}" in command
    assert "++manager_env.commands.motion.motion_lib_cfg.target_fps=50" in command
    assert f"++callbacks.practice_capsule.capsule_dir={paths['capsule'].parent}" in command
    assert any(item.startswith("++callbacks.practice_capsule.provenance=") for item in command)


def test_origin_command_does_not_collect_unattributed_global_observer_features(tmp_path):
    command, _ = O.build_command(
        args(tmp_path),
        seed=9301,
        start_step=24,
        experiment_id="origins",
        provenance=provenance(),
    )
    assert not any("practice_observer" in item for item in command)


def write_motion_tree(root: Path, keys: list[str], prefix: bytes) -> Path:
    root.mkdir()
    for index, key in enumerate(keys):
        (root / f"{key}.pkl").write_bytes(prefix + bytes([index]))
    return root


def test_motion_sources_resolve_robot_symlink_to_exact_pool_and_hash_smpl(tmp_path):
    keys = ["alpha__A001", "beta__A002"]
    robot = write_motion_tree(tmp_path / "robot", keys, b"robot")
    smpl = write_motion_tree(tmp_path / "smpl", keys, b"smpl")
    robot_link = tmp_path / "robot_link"
    robot_link.symlink_to(robot, target_is_directory=True)
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"source_root": str(robot), "pool_sha256": "b" * 64}))

    binding = O.validate_motion_sources(
        robot_link,
        smpl,
        pool,
        {key: str(index) * 64 for index, key in enumerate(keys, start=1)},
    )

    assert binding["robot"]["requested_path"] == str(robot_link)
    assert binding["robot"]["resolved_path"] == str(robot.resolve())
    assert binding["robot"]["pool_manifest_source_root_resolved"] == str(robot.resolve())
    assert binding["smpl"]["resolved_path"] == str(smpl.resolve())
    assert binding["smpl"]["file_count"] == 2
    assert len(binding["smpl"]["tree_sha256"]) == 64
    assert len(binding["smpl"]["paired_motion_keys_sha256"]) == 64


def test_motion_sources_reject_robot_tree_outside_pool_source_root(tmp_path):
    keys = ["alpha__A001"]
    robot = write_motion_tree(tmp_path / "robot", keys, b"robot")
    wrong_robot = write_motion_tree(tmp_path / "wrong_robot", keys, b"wrong")
    smpl = write_motion_tree(tmp_path / "smpl", keys, b"smpl")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"source_root": str(robot), "pool_sha256": "b" * 64}))

    with pytest.raises(ValueError, match="does not resolve to the pool manifest source_root"):
        O.validate_motion_sources(wrong_robot, smpl, pool, {keys[0]: "a" * 64})


def test_motion_sources_reject_smpl_tree_with_different_motion_support(tmp_path):
    robot = write_motion_tree(tmp_path / "robot", ["alpha__A001", "beta__A002"], b"robot")
    smpl = write_motion_tree(tmp_path / "smpl", ["alpha__A001"], b"smpl")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"source_root": str(robot), "pool_sha256": "b" * 64}))

    with pytest.raises(ValueError, match="does not exactly cover the frozen robot pool"):
        O.validate_motion_sources(
            robot,
            smpl,
            pool,
            {"alpha__A001": "a" * 64, "beta__A002": "b" * 64},
        )


def test_smpl_tree_hash_changes_with_source_bytes(tmp_path):
    smpl = write_motion_tree(tmp_path / "smpl", ["alpha__A001"], b"smpl")
    before = O.directory_tree_binding(smpl, label="smpl")["tree_sha256"]
    (smpl / "alpha__A001.pkl").write_bytes(b"changed")
    after = O.directory_tree_binding(smpl, label="smpl")["tree_sha256"]
    assert before != after


def run_with(metric: str, values: list[float]):
    return O.RL.RunLog(
        "fake.log",
        [O.RL.Iteration(index=i + 1, metrics={metric: value}) for i, value in enumerate(values)],
    )


def test_stability_compares_last_four_to_preceding_four():
    result = O.trailing_stability(run_with("Mean rewards", [10.0] * 8), "Mean rewards")
    assert result["passes"] is True
    assert result["relative_delta"] == pytest.approx(0.0)


def test_stability_fails_a_drifting_origin():
    result = O.trailing_stability(run_with("Mean rewards", [10.0] * 4 + [12.0] * 4), "Mean rewards")
    assert result["passes"] is False
    assert result["relative_delta"] == pytest.approx(0.2)


def test_stability_requires_two_complete_windows():
    result = O.trailing_stability(run_with("Mean length", [100.0] * 7), "Mean length")
    assert result["passes"] is False
    assert result["relative_delta"] is None


def test_snapshot_must_match_capsule_counters_and_frozen_pool():
    snapshot = {
        "num_active_bins": 1,
        "contexts": [
            {
                "context_id": "ctx",
                "motion_key": "motion",
                "motion_hash": "a" * 64,
                "global_bin_id": 1,
                "sampling_probability": 0.25,
                "num_episodes": 8.0,
                "num_failures": 3.0,
            }
        ],
    }
    capsule = {
        "native_sampler_state": {
            "adp_samp_num_episodes": torch.tensor([0.0, 8.0]),
            "adp_samp_num_failures": torch.tensor([0.0, 3.0]),
        }
    }
    rows, blockers = O.validate_snapshot_against_capsule_and_pool(
        snapshot, capsule, {"motion": "a" * 64}
    )
    assert blockers == []
    assert set(rows) == {"ctx"}

    snapshot["contexts"][0]["motion_hash"] = "b" * 64
    _, blockers = O.validate_snapshot_against_capsule_and_pool(
        snapshot, capsule, {"motion": "a" * 64}
    )
    assert any("motion hash differs" in blocker for blocker in blockers)


def test_snapshot_canonicalizes_identical_resident_copies_without_summing_counters():
    context = {
        "context_id": "ctx",
        "motion_key": "motion",
        "motion_hash": "a" * 64,
        "global_bin_id": 1,
        "sampling_probability": 0.25,
        "num_episodes": 8.0,
        "num_failures": 3.0,
    }
    snapshot = {"num_active_bins": 2, "contexts": [context, dict(context)]}
    capsule = {
        "native_sampler_state": {
            "adp_samp_num_episodes": torch.tensor([0.0, 8.0]),
            "adp_samp_num_failures": torch.tensor([0.0, 3.0]),
        }
    }

    rows, blockers = O.validate_snapshot_against_capsule_and_pool(
        snapshot, capsule, {"motion": "a" * 64}
    )

    assert blockers == []
    assert rows["ctx"]["sampling_probability"] == pytest.approx(0.5)
    assert rows["ctx"]["resident_multiplicity"] == 2
    assert rows["ctx"]["num_episodes"] == 8.0
    assert rows["ctx"]["num_failures"] == 3.0


def test_snapshot_rejects_conflicting_resident_copies():
    context = {
        "context_id": "ctx",
        "motion_key": "motion",
        "motion_hash": "a" * 64,
        "global_bin_id": 1,
        "sampling_probability": 0.25,
        "num_episodes": 8.0,
        "num_failures": 3.0,
    }
    snapshot = {
        "num_active_bins": 2,
        "contexts": [context, {**context, "num_failures": 4.0}],
    }
    capsule = {
        "native_sampler_state": {
            "adp_samp_num_episodes": torch.tensor([0.0, 8.0]),
            "adp_samp_num_failures": torch.tensor([0.0, 3.0]),
        }
    }

    _, blockers = O.validate_snapshot_against_capsule_and_pool(
        snapshot, capsule, {"motion": "a" * 64}
    )

    assert any("conflicting serialized rows" in blocker for blocker in blockers)


def test_exported_checkpoint_must_link_exact_capsule_and_step(tmp_path):
    checkpoint = tmp_path / "origin.pt"
    capsule = {"capsule_sha256": "c" * 64}
    torch.save(
        {
            "policy_state_dict": {"p": torch.tensor(1)},
            "value_state_dict": {"v": torch.tensor(2)},
            "optimizer_state_dict": {"o": torch.tensor(3)},
            "state": SimpleNamespace(global_step=36),
            "env_state_dict": {"motion_lib": {}},
            "practice_utility": {"capsule_sha256": "c" * 64, "global_step": 36},
        },
        checkpoint,
    )
    assert O.validate_exported_checkpoint(checkpoint, capsule=capsule, target_step=36) == []

    payload = torch.load(checkpoint, weights_only=False)
    payload["practice_utility"]["capsule_sha256"] = "d" * 64
    torch.save(payload, checkpoint)
    assert "logical capsule hash mismatch" in " ".join(
        O.validate_exported_checkpoint(checkpoint, capsule=capsule, target_step=36)
    )


def test_origin_validation_requires_exact_resumed_interval(tmp_path, monkeypatch):
    paths = {
        "capsule": tmp_path / "origin.capsule.pt",
        "snapshot": tmp_path / "snapshot.json",
        "checkpoint": tmp_path / "origin.pt",
    }
    for name in ("capsule", "checkpoint"):
        paths[name].touch()
    paths["snapshot"].write_text(
        json.dumps(
            {
                "global_step": 36,
                "snapshot_timeline_fps": 50.0,
                "num_active_bins": 2,
                "contexts": [
                    {
                        "context_id": "ctx",
                        "motion_key": "motion",
                        "motion_hash": "a" * 64,
                        "global_bin_id": 0,
                        "sampling_probability": 0.5,
                        "num_episodes": 8.0,
                        "num_failures": 3.0,
                    },
                    {
                        "context_id": "ctx",
                        "motion_key": "motion",
                        "motion_hash": "a" * 64,
                        "global_bin_id": 0,
                        "sampling_probability": 0.5,
                        "num_episodes": 8.0,
                        "num_failures": 3.0,
                    },
                ],
            }
        )
    )
    capsule = {
        "global_step": 36,
        "trainer_state": {"trainer_state_obj": object()},
        "optimizer_state": {"state": 1},
        "rng": {"counter_rng_enabled": False},
        "native_sampler_state": {
            "adp_samp_num_episodes": torch.tensor([8.0]),
            "adp_samp_num_failures": torch.tensor([3.0]),
        },
    }
    monkeypatch.setattr(O.BC, "load_capsule", lambda *args, **kwargs: capsule)
    monkeypatch.setattr(O, "validate_exported_checkpoint", lambda *args, **kwargs: [])

    metrics = {"Mean rewards": 10.0, "Mean length": 100.0}
    exact = O.RL.RunLog(
        "log",
        [O.RL.Iteration(index=index, metrics=metrics) for index in range(25, 37)],
    )
    monkeypatch.setattr(O.RL, "parse_run_log", lambda path: exact)
    origin, blockers = O.validate_origin(
        paths=paths,
        log_path=tmp_path / "run.log",
        start_step=24,
        target_step=36,
        settle_iterations=12,
        expected_provenance=provenance(),
        pool_hashes={"motion": "a" * 64},
        snapshot_timeline_fps=50.0,
    )
    assert blockers == []
    assert origin["resident_context_ids"] == ["ctx"]
    assert origin["num_resident_contexts"] == 1
    assert origin["num_active_context_rows"] == 2
    assert origin["num_duplicate_active_context_rows"] == 1

    shifted = O.RL.RunLog(
        "log",
        [O.RL.Iteration(index=index, metrics=metrics) for index in range(24, 36)],
    )
    monkeypatch.setattr(O.RL, "parse_run_log", lambda path: shifted)
    _, blockers = O.validate_origin(
        paths=paths,
        log_path=tmp_path / "run.log",
        start_step=24,
        target_step=36,
        settle_iterations=12,
        expected_provenance=provenance(),
        pool_hashes={"motion": "a" * 64},
        snapshot_timeline_fps=50.0,
    )
    assert any("interval differs" in blocker for blocker in blockers)


def test_execute_writes_preregistration_before_gpu_call(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source.pt"
    pool = tmp_path / "pool.json"
    split = tmp_path / "split.json"
    for path in (checkpoint, pool, split):
        path.touch()
    receipt_dir = tmp_path / "receipts"

    monkeypatch.setattr(
        O,
        "validate_claim_inputs",
        lambda *args: ("b" * 64, "c" * 64, {"motion": "a" * 64}),
    )
    monkeypatch.setattr(
        O,
        "validate_motion_sources",
        lambda *args: {
            "robot": {"resolved_path": "/frozen/robot", "tree_sha256": "1" * 64},
            "smpl": {"resolved_path": "/frozen/smpl", "tree_sha256": "2" * 64},
        },
    )
    monkeypatch.setattr(O, "checkpoint_global_step", lambda path: 24)
    monkeypatch.setattr(O.TP, "git_sha", lambda: "d" * 40)
    monkeypatch.setattr(O.TP, "git_status", lambda: [])

    observed = {}

    def fail_after_checking_receipt(command, log_path, min_free_mib):
        preregistrations = list(receipt_dir.glob("*_preregistration.json"))
        assert len(preregistrations) == 1
        assert json.loads(preregistrations[0].read_text())["frozen"] is True
        receipts = [
            path
            for path in receipt_dir.glob("probe_origins_ne*.json")
            if not path.name.endswith("_preregistration.json")
        ]
        assert len(receipts) == 1
        payload = json.loads(receipts[0].read_text())
        observed["status"] = payload["status"]
        observed["plan"] = payload["preregistered_before_run"]
        return {"exit_code": 7}

    monkeypatch.setattr(O.LA, "run_arm", fail_after_checking_receipt)
    result = O.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--pool-manifest",
            str(pool),
            "--dev-suite-manifest",
            str(split),
            "--seeds",
            "9300",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--receipt-dir",
            str(receipt_dir),
            "--execute",
        ]
    )
    assert result == 1
    assert observed["status"] == "running"
    assert observed["plan"]["required_iteration_indices"] == list(range(25, 37))
