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
    modes = list(R.MODES)
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
