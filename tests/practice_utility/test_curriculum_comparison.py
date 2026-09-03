from pathlib import Path
from types import SimpleNamespace

from scripts.practice_utility import run_curriculum_comparison as R


def args():
    return SimpleNamespace(
        checkpoint="/tmp/model.pt",
        num_envs=128,
        iterations=32,
        warmup_iterations=10,
        max_delay=8,
        delta_target=0.778,
        kp=1.0,
        ki=0.02,
        alpha=0.05,
        integral_max=1.0,
        return_floor=8.0,
        exp="manager/universal_token/all_modes/sonic_release",
        encoder="/tmp/encoder.pt",
        motion_file="data/motion_lib_bones_seed/robot_filtered",
        smpl_motion_file="data/motion_lib_bones_seed/smpl_filtered",
        consolidation_fraction=0.0,
        spread_strata=1,
        latency_cap=0.5,
        return_guard="absolute",
        return_relative_drop=0.25,
        return_window=8,
    )


def test_command_wires_corrected_curriculum_and_final_checkpoint():
    command = R.build_command(args(), "lucid", 8600, "branch", Path("/tmp/artifact"))
    for expected in (
        "--max-delay",
        "8",
        "manager_env/events=tracking/lucid_curriculum",
        "++callbacks.lucid_curriculum.mode=lucid",
        "++callbacks.lucid_curriculum.ki=0.02",
        "++callbacks.lucid_curriculum.integral_max=1.0",
        "++callbacks.lucid_curriculum.return_floor=8.0",
        "++callbacks.practice_capsule.horizons.final=32",
    ):
        assert expected in command


def test_arm_order_rotates_across_seeds():
    modes = ["lucid", "fixed", "off"]
    assert R.arm_order(modes, 0) == ["lucid", "fixed", "off"]
    assert R.arm_order(modes, 1) == ["fixed", "off", "lucid"]
    assert R.arm_order(modes, 2) == ["off", "lucid", "fixed"]


def test_comparison_applies_frozen_noise_floors():
    summary = {
        mode: {
            "metrics": {
                "Mean rewards": {"mean": reward},
                "Mean length": {"mean": length},
            }
        }
        for mode, reward, length in (("lucid", 10.4, 104.0), ("fixed", 10.0, 100.0))
    }
    compared = R.comparisons(summary)["lucid_vs_fixed"]
    assert compared["Mean rewards"]["outside_settled_noise_floor"]
    assert compared["Mean length"]["outside_settled_noise_floor"]


def test_tace_arms_build_anchor_and_yoked_overrides(tmp_path):
    args = R.parse_args(["--checkpoint", "/x.pt", "--iterations", "8", "--warmup-iterations", "2"])
    cmd = R.build_command(args, "ta_lucid_25", 8600, "b", tmp_path)
    assert "++callbacks.lucid_curriculum.mode=lucid" in cmd
    assert "++callbacks.lucid_curriculum.anchor_ratio=0.25" in cmd
    assert "++callbacks.lucid_curriculum.anchor_seed=8600" in cmd
    sched = tmp_path / "curriculum_src.jsonl"
    cmd = R.build_command(args, "ta_yoked_25", 8600, "b", tmp_path, sched)
    assert "++callbacks.lucid_curriculum.mode=yoked" in cmd
    assert f"++callbacks.lucid_curriculum.yoked_schedule_path={sched}" in cmd
    assert "++callbacks.lucid_curriculum.anchor_ratio=0.25" in cmd
    plain = R.build_command(args, "lucid", 8600, "b", tmp_path)
    assert not any("anchor_ratio" in c for c in plain)
    import pytest
    with pytest.raises(ValueError, match="schedule"):
        R.build_command(args, "ta_yoked_25", 8600, "b", tmp_path)


def test_cross_seed_yoked_uses_next_seed_schedule_from_source_receipt(tmp_path):
    receipt = {
        "arms": {
            f"b{s}": {"mode": "ta_lucid_25", "seed": s, "curriculum_path": f"/art/seed_{s}/c.jsonl"}
            for s in (8600, 8601, 8602)
        }
    }
    path = tmp_path / "src.json"
    path.write_text(__import__("json").dumps(receipt))
    args = R.parse_args(["--checkpoint", "/x.pt", "--iterations", "8", "--warmup-iterations", "2",
                         "--seeds", "8600", "8601", "8602", "--modes", "ta_yoked_25x",
                         "--yoked-source-receipt", str(path)])
    assert "ta_yoked_25x" in R.CROSS_SEED_ARMS
    # exercise the resolution logic through main's dry run
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert R.main(["--checkpoint", "/x.pt", "--iterations", "8", "--warmup-iterations", "2",
                       "--seeds", "8600", "8601", "8602", "--modes", "ta_yoked_25x",
                       "--yoked-source-receipt", str(path)]) == 0
    lines = [l for l in out.getvalue().splitlines() if l.startswith("[seed=") or "yoked_schedule_path" in l]
    pairs = [(lines[k], lines[k + 1]) for k in range(0, len(lines), 2)]
    resolved = {h.split("seed=")[1].split(" ")[0]: v.split("=")[-1] for h, v in pairs}
    assert resolved == {
        "8600": "/art/seed_8601/c.jsonl",
        "8601": "/art/seed_8602/c.jsonl",
        "8602": "/art/seed_8600/c.jsonl",
    }


import pytest


def test_fixed_150_extends_support_explicitly():
    a = args()
    a.max_delay = 12
    cmd = R.build_command(a, "fixed_150", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.fixed_lambda=1.5" in cmd
    assert "++callbacks.lucid_curriculum.allow_extrapolation=true" in cmd
    assert "++callbacks.lucid_curriculum.mode=fixed" in cmd


def test_fixed_150_refuses_an_undersized_delay_buffer():
    # 1.5x the 0-40 ms envelope needs 12 steps of buffer; the delayed-actuator
    # process clamps silently, so the launcher must fail loudly instead.
    with pytest.raises(SystemExit, match="max-delay 12"):
        R.build_command(args(), "fixed_150", 8600, "b", Path("/tmp/artifact"))


def test_ordinary_arms_keep_the_envelope_and_no_flag():
    cmd = R.build_command(args(), "fixed", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.fixed_lambda=1.0" in cmd
    assert not any("allow_extrapolation" in part for part in cmd)


def test_box_150_is_the_gate_with_a_vector_frontier_and_the_same_support():
    a = args()
    a.max_delay = 12
    a.gate_threshold = 0.8
    a.gate_window = 100
    a.gate_dwell = 50
    a.gate_min_episodes = 200
    a.gate_guard_action = "freeze"
    a.box_channel_budget = 300
    a.ramp_begin_iteration = 0
    a.ramp_end_iteration = 1500
    a.num_envs = 1024
    box = R.build_command(a, "box_150", 8600, "b", Path("/tmp/artifact"))
    gate = R.build_command(a, "gate_150", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.mode=box" in box
    assert "++callbacks.lucid_curriculum.box_channel_budget=300" in box
    assert "++callbacks.lucid_curriculum.allow_extrapolation=true" in box
    # Identical strata, probe geometry, ceiling, gate law and survival observer.
    shared = (
        "++callbacks.lucid_curriculum.spread_strata=8",
        "++callbacks.lucid_curriculum.stratum_sizes=[43,43,43,43,42,42,640,128]",
        "++callbacks.lucid_curriculum.gate_probe_offset=0.125",
        "++callbacks.lucid_curriculum.gate_probe_max=1.5",
        "++callbacks.lucid_curriculum.gate_lambda_max=1.5",
        "++callbacks.lucid_curriculum.gate_threshold=0.8",
        "++callbacks.lucid_curriculum.gate_window=100",
        "++callbacks.lucid_curriculum.gate_dwell=50",
        "++callbacks.lucid_curriculum.return_guard=relative",
        "++callbacks.survival_observer.enabled=true",
    )
    for part in shared:
        assert part in box and part in gate
    assert not any("box_channel_budget" in part for part in gate)
    assert "box_150" in R.MODES and "box_150" in R.EXPANSION_ARMS
    assert R.ARM_LAMBDA_CEILING["box_150"] == 1.5


def test_box_150_refuses_an_undersized_delay_buffer():
    a = args()
    a.gate_threshold, a.gate_window, a.gate_dwell, a.gate_min_episodes = 0.8, 200, 200, 200
    a.gate_guard_action, a.box_channel_budget = "freeze", 0
    with pytest.raises(SystemExit, match="max-delay 12"):
        R.build_command(a, "box_150", 8600, "b", Path("/tmp/artifact"))


def _asym_args():
    a = args()
    a.max_delay = 12
    a.gate_threshold, a.gate_window, a.gate_dwell, a.gate_min_episodes = 0.8, 100, 50, 200
    a.gate_guard_action, a.box_channel_budget = "freeze", 300
    a.ramp_begin_iteration, a.ramp_end_iteration = 0, 1500
    a.num_envs = 1024
    return a


def test_asymmetric_arms_widen_cheap_channels_and_hold_the_binding_ones():
    a = _asym_args()
    box = R.build_command(a, "box_asym", 8600, "b", Path("/tmp/artifact"))
    ramp = R.build_command(a, "ramp_asym", 8600, "b", Path("/tmp/artifact"))
    fixed = R.build_command(a, "fixed_asym", 8600, "b", Path("/tmp/artifact"))
    # Box: per-channel ceilings, scalar ceiling at the max.
    assert "++callbacks.lucid_curriculum.mode=box" in box
    assert "++callbacks.lucid_curriculum.gate_lambda_max=2.0" in box
    assert "++callbacks.lucid_curriculum.box_lambda_max.push_robot=1.5" in box
    assert "++callbacks.lucid_curriculum.box_lambda_max.randomize_rigid_body_mass=2.0" in box
    assert "++callbacks.lucid_curriculum.box_lambda_max.randomize_action_delay=1.5" in box
    # Ramp and fixed: scalar to 2.0, binding channels capped at 1.5.
    assert "++callbacks.lucid_curriculum.ramp_end_lambda=2.0" in ramp
    assert "++callbacks.lucid_curriculum.fixed_lambda=2.0" in fixed
    for cmd in (ramp, fixed):
        assert "++callbacks.lucid_curriculum.term_lambda_caps.push_robot=1.5" in cmd
        assert "++callbacks.lucid_curriculum.term_lambda_caps.physics_material=1.5" in cmd
        assert "++callbacks.lucid_curriculum.term_lambda_caps.randomize_action_delay=1.5" in cmd
        assert not any("term_lambda_caps.randomize_rigid_body_mass" in part for part in cmd)
        assert "++callbacks.lucid_curriculum.allow_extrapolation=true" in cmd
    # Same strata and probe geometry as the 1.5 expansion arms.
    for cmd in (box, ramp):
        assert "++callbacks.lucid_curriculum.spread_strata=8" in cmd
        assert "++callbacks.lucid_curriculum.stratum_sizes=[43,43,43,43,42,42,640,128]" in cmd
        assert "++callbacks.survival_observer.enabled=true" in cmd
    assert R.ARM_LAMBDA_CEILING["box_asym"] == 2.0 and R.ARM_DELAY_CEILING["box_asym"] == 1.5


def test_asymmetric_arms_size_the_delay_buffer_from_the_latency_ceiling():
    # Latency is held at 1.5 (12 steps) even though mass reaches 2.0, so
    # --max-delay 12 is enough and 8 is not.
    a = _asym_args()
    for mode in ("box_asym", "ramp_asym", "fixed_asym"):
        R.build_command(a, mode, 8600, "b", Path("/tmp/artifact"))
    a.max_delay = 8
    with pytest.raises(SystemExit, match="max-delay 12"):
        R.build_command(a, "ramp_asym", 8600, "b", Path("/tmp/artifact"))


def test_wide_arms_hold_latency_and_reach_three():
    a = _asym_args()
    gate = R.build_command(a, "gate_300", 8600, "b", Path("/tmp/artifact"))
    fixed = R.build_command(a, "fixed_300", 8600, "b", Path("/tmp/artifact"))
    box = R.build_command(a, "box_fast_300", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.gate_lambda_max=3.0" in gate
    assert "++callbacks.lucid_curriculum.term_lambda_caps.randomize_action_delay=1.5" in gate
    assert "++callbacks.lucid_curriculum.fixed_lambda=3.0" in fixed
    assert "++callbacks.lucid_curriculum.term_lambda_caps.randomize_action_delay=1.5" in fixed
    assert "++callbacks.lucid_curriculum.mode=box" in box
    assert "++callbacks.lucid_curriculum.box_lambda_max.push_robot=3.0" in box
    assert "++callbacks.lucid_curriculum.box_lambda_max.randomize_action_delay=1.5" in box
    assert "++callbacks.lucid_curriculum.gate_lambda_max=3.0" in box
    for cmd in (gate, fixed, box):
        assert "++callbacks.lucid_curriculum.allow_extrapolation=true" in cmd
    # Latency held at 1.5 -> the 12-step buffer is exact; 8 is refused.
    a.max_delay = 8
    with pytest.raises(SystemExit, match="max-delay 12"):
        R.build_command(a, "gate_300", 8600, "b", Path("/tmp/artifact"))
    assert R.ARM_DELAY_CEILING["gate_300"] == 1.5 and R.ARM_LAMBDA_CEILING["fixed_300"] == 3.0


def test_guard_free_gate_disables_the_relative_return_guard_only():
    a = _asym_args()
    ng = R.build_command(a, "gate_300_ng", 8600, "b", Path("/tmp/artifact"))
    base = R.build_command(a, "gate_300", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.return_relative_drop=0.99" in ng
    assert f"++callbacks.lucid_curriculum.return_relative_drop={a.return_relative_drop}" in base
    # Everything else about the arm is identical to gate_300.
    strip = lambda cmd: [c for c in cmd if "return_relative_drop" not in c and "branch_id" not in c
                         and "output_dir" not in c and "capsule_dir" not in c]
    assert strip(ng) == strip(base)
    assert "gate_300_ng" in R.MODES
