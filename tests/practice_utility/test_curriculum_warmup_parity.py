from pathlib import Path
from types import SimpleNamespace

import pytest

from gear_sonic.research.practice_utility import run_log as RL
from scripts.practice_utility import run_curriculum_warmup_parity as W


def args():
    return SimpleNamespace(
        max_delay=8,
        artifact_root=Path("/tmp/warmup"),
        exp="manager/universal_token/all_modes/sonic_release",
        checkpoint="/tmp/model.pt",
        num_envs=128,
        seed=7,
        iterations=4,
        warmup_iterations=2,
        encoder="/tmp/encoder.pt",
    )


@pytest.mark.parametrize("mode", W.MODES)
def test_command_wires_equal_zero_lambda_warmup(mode):
    command = W.build_command(args(), mode, mode)
    assert "manager_env/events=tracking/lucid_curriculum" in command
    assert f'++callbacks.lucid_curriculum.mode="{mode}"' in command
    assert "++callbacks.lucid_curriculum.warmup_iterations=2" in command
    assert "seed=7" in command


def test_command_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported mode"):
        W.build_command(args(), "fixed", "bad")


def test_prefix_comparison_excludes_later_treatment(tmp_path):
    def run(name, rewards):
        iterations = [
            RL.Iteration(
                index=index,
                metrics={
                    "Mean rewards": reward,
                    "Mean length": 10.0,
                    "Mean entropy": 1.0,
                    "Mean action noise std": 0.5,
                },
            )
            for index, reward in enumerate(rewards)
        ]
        return RL.RunLog(str(tmp_path / name), iterations)

    left = run("left", [1.0, 2.0, 3.0, 5.0])
    right = run("right", [1.0, 2.0, 3.0, 4.0])
    assert W.compare_prefix(left, right, 3)["passes"] is True
    assert W.compare_prefix(left, right, 4)["passes"] is False
