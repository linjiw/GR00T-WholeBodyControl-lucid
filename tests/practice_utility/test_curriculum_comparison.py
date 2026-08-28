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
