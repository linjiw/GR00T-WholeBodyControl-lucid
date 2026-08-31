import json
from pathlib import Path
from types import SimpleNamespace

from scripts.practice_utility import run_curriculum_robustness_eval as R


def args():
    return SimpleNamespace(
        max_delay=12,
        num_envs=128,
        smpl_motion_file="smpl",
    )


def test_ratchet_arm_is_selectable_for_the_matched_ladder():
    assert "lucid_ratchet_rg" in R.MODES


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
