import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.practice_utility import run_curriculum_robustness_eval as R


def args():
    return SimpleNamespace(
        max_delay=12,
        num_envs=128,
        smpl_motion_file="smpl",
    )


def test_ratchet_arm_is_selectable_for_the_matched_ladder():
    assert "lucid_ratchet_rg" in R.MODES


def test_expand_arms_are_selectable_for_the_matched_ladder():
    assert "fixed_u" in R.MODES
    assert "fixed_u150" in R.MODES


def test_protocol_metadata_names_only_the_actual_requested_presets():
    metadata = R.requested_preset_metadata(["phys_125", "lat_50ms"])
    assert set(metadata) == {"phys_125", "lat_50ms"}
    assert metadata["phys_125"] == {
        "event_preset": "tracking/lucid_curriculum",
        "non_latency_dr_scale": 1.25,
        "fixed_latency_steps": 0,
    }
    assert metadata["lat_50ms"] == {
        "event_preset": "tracking/lucid_eval_clean",
        "fixed_latency_steps": 10,
    }


def test_command_is_frozen_matched_evaluation():
    command = R.build_command(
        args(),
        Path("/tmp/model.pt"),
        "lucid",
        "dr_full",
        8700,
        "branch",
        Path("/tmp/out"),
        "/tmp/motions",
    )
    for expected in (
        "+manager_env/events=tracking/lucid_curriculum",
        "+num_envs=128",
        "+use_encoder=g1",
        "+eval_callbacks=[practice_eval]",
        "+run_eval_loop=false",
        "++manager_env.config.train_only_events=[]",
        "++callbacks.practice_eval.eval_only=true",
    ):
        assert expected in command
    assert not any("learning_iterations" in value for value in command)


def test_materialize_suite_selects_only_frozen_partition(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for key in ("a", "b"):
        (sources / f"{key}.pkl").write_bytes(key.encode())
    pool = {
        "pool_sha256": "pool",
        "motions": [{"motion_key": key, "path": str(sources / f"{key}.pkl")} for key in ("a", "b")],
    }
    split = {
        "pool_sha256": "pool",
        "split_sha256": "split",
        "linkage": "content",
        "assignment": {"a": "dev", "b": "test"},
    }
    pool_path, split_path = tmp_path / "pool.json", tmp_path / "split.json"
    pool_path.write_text(json.dumps(pool))
    split_path.write_text(json.dumps(split))
    suite = R.materialize_suite(pool_path, split_path, "dev", tmp_path / "suite")
    assert suite["motion_count"] == 1
    links = list((tmp_path / "suite" / "robot_filtered").iterdir())
    assert [link.name for link in links] == ["a.pkl"]
    assert links[0].is_symlink()


def test_delay_contract_distinguishes_presets():
    base = {"delay": {"action_delay_actuator_groups": 5}}
    assert R.delay_matches("id_clean", {"delay": {**base["delay"], "action_delay_max_steps": 0}})
    assert R.delay_matches(
        "dr_full",
        {
            "delay": {
                **base["delay"],
                "action_delay_min_steps": 0,
                "action_delay_max_steps": 8,
                "action_delay_nonzero_fraction": 0.8,
            }
        },
    )
    assert R.delay_matches(
        "latency_60ms",
        {
            "delay": {
                **base["delay"],
                "action_delay_min_steps": 12,
                "action_delay_max_steps": 12,
            }
        },
    )


def test_heldout_latency_presets_have_exact_step_values():
    assert R.PRESET_FIXED_LATENCY_STEPS["lat_80ms"] == 16
    assert R.PRESET_FIXED_LATENCY_STEPS["lat_100ms"] == 20
    assert R.PRESET_FIXED_LATENCY_STEPS["lat_120ms"] == 24
    for name in ("lat_80ms", "lat_100ms", "lat_120ms"):
        assert R.PRESETS[name] == "tracking/lucid_eval_clean"
        assert R.requested_preset_metadata([name])[name]["fixed_latency_steps"] == (
            R.PRESET_FIXED_LATENCY_STEPS[name]
        )


def test_existing_latency_ladder_values_are_unchanged():
    assert {k: v for k, v in R.PRESET_FIXED_LATENCY_STEPS.items() if v <= 12} == {
        "lat_10ms": 2,
        "lat_20ms": 4,
        "lat_30ms": 6,
        "lat_40ms": 8,
        "lat_50ms": 10,
        "lat_60ms": 12,
    }
    assert R.PRESET_PHYSICS_ONLY == {
        "phys_000": 0.0,
        "phys_025": 0.25,
        "phys_050": 0.5,
        "phys_075": 0.75,
        "phys_100": 1.0,
        "phys_125": 1.25,
        "phys_150": 1.5,
        "phys_175": 1.75,
        "phys_200": 2.0,
    }


def test_truncated_latency_cell_is_refused_at_default_capacity():
    with pytest.raises(ValueError) as excinfo:
        R.assert_latency_within_capacity(["lat_120ms"], 12)
    message = str(excinfo.value)
    assert "lat_120ms" in message
    assert "24 steps" in message
    assert "--max-delay is 12" in message
    assert "Raise --max-delay" in message


def test_every_truncated_cell_is_named_not_just_the_first():
    with pytest.raises(ValueError) as excinfo:
        R.assert_latency_within_capacity(["lat_60ms", "lat_80ms", "lat_100ms", "lat_120ms"], 12)
    message = str(excinfo.value)
    assert "lat_60ms" not in message
    for name in ("lat_80ms (16 steps)", "lat_100ms (20 steps)", "lat_120ms (24 steps)"):
        assert name in message
    assert "at least 24" in message


def test_latency_cell_within_capacity_is_accepted():
    R.assert_latency_within_capacity(["lat_120ms"], 24)
    R.assert_latency_within_capacity(["lat_80ms", "lat_100ms", "lat_120ms"], 24)
    R.assert_latency_within_capacity(["lat_60ms", "latency_60ms"], 12)


def test_physics_only_presets_ignore_latency_capacity():
    # Latency is pinned to zero in these cells, so no buffer depth can truncate them.
    R.assert_latency_within_capacity(list(R.PRESET_PHYSICS_ONLY), 0)
    R.assert_latency_within_capacity(["id_clean", "dr_full", "dr_125", "phys_125"], 12)
    command = R.build_command(
        args(), Path("/tmp/model.pt"), "lucid", "phys_125", 8700, "branch", Path("/tmp/out"), "/m"
    )
    assert "++callbacks.practice_eval.fixed_latency_steps=0" in command
    assert "++callbacks.practice_eval.non_latency_dr_scale=1.25" in command
    assert command[2:4] == ["--max-delay", "12"]


def test_build_command_refuses_a_truncating_latency_cell():
    with pytest.raises(ValueError, match="lat_120ms"):
        R.build_command(
            args(), Path("/tmp/model.pt"), "lucid", "lat_120ms", 8700, "b", Path("/tmp/out"), "/m"
        )
    wide = SimpleNamespace(max_delay=24, num_envs=128, smpl_motion_file="smpl")
    command = R.build_command(
        wide, Path("/tmp/model.pt"), "lucid", "lat_120ms", 8700, "b", Path("/tmp/out"), "/m"
    )
    assert "++callbacks.practice_eval.fixed_latency_steps=24" in command
    assert command[2:4] == ["--max-delay", "24"]


def test_main_fails_closed_before_touching_any_input(tmp_path):
    missing = tmp_path / "no_such_receipt.json"
    with pytest.raises(ValueError, match="lat_120ms"):
        R.main(["--presets", "lat_120ms", "--training-receipt", str(missing)])
    # The same launch with enough capacity gets past the guard and proceeds as before.
    with pytest.raises(FileNotFoundError):
        R.main(["--presets", "lat_120ms", "--max-delay", "24", "--training-receipt", str(missing)])
    # A physics-only run at the default capacity is not gated at all.
    with pytest.raises(FileNotFoundError):
        R.main(["--presets", "phys_125", "--training-receipt", str(missing)])
