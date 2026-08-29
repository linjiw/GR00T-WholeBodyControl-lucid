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


def test_bounded_subset_is_deterministic_and_bounded():
    from scripts.practice_utility import run_curriculum_robustness_eval as E
    keys = {f"m{i}" for i in range(50)}
    a = E.bounded_subset(keys, 10, "salt")
    b = E.bounded_subset(keys, 10, "salt")
    assert a == b and len(a) == 10 and a <= keys
    assert E.bounded_subset(keys, 10, "other") != a
    assert E.bounded_subset(keys, None, "salt") == keys
    assert E.bounded_subset(keys, 99, "salt") == keys


def test_u60_common_presets_override_the_delay_process(tmp_path):
    from scripts.practice_utility import run_curriculum_robustness_eval as E
    args = E.parse_args(["--training-receipt", "/x.json"])
    cmd = E.build_command(args, tmp_path / "c.pt", "fixed", "lat_u60_common", 8700, "b", tmp_path, "/m")
    assert "++manager_env.events.randomize_action_delay.params.delay_range=[0,12]" in cmd
    assert "++manager_env.events.randomize_action_delay.params.coupling=common" in cmd
    assert "++callbacks.practice_eval.non_latency_dr_scale=0.0" in cmd
    cmd2 = E.build_command(args, tmp_path / "c.pt", "fixed", "dr_full_lat_u60_common", 8700, "b", tmp_path, "/m")
    assert not any("non_latency_dr_scale" in c for c in cmd2)
    good = {"delay": {"action_delay_actuator_groups": 5, "action_delay_min_steps": 0, "action_delay_max_steps": 12,
                      "action_delay_nonzero_fraction": 0.9, "action_delay_cross_group_equal_fraction": 1.0}}
    assert E.delay_matches("lat_u60_common", good)
    bad = {"delay": {**good["delay"], "action_delay_cross_group_equal_fraction": 0.2}}
    assert not E.delay_matches("lat_u60_common", bad)
