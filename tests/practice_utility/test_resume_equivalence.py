from pathlib import Path
from types import SimpleNamespace

from gear_sonic.research.practice_utility import run_log as RL
from scripts.practice_utility import run_resume_equivalence as R


def args():
    return SimpleNamespace(
        checkpoint="/tmp/model.pt",
        num_envs=128,
        seed=7,
        total_iterations=20,
        split_iteration=10,
        exp="manager/universal_token/all_modes/sonic_release",
    )


def test_full_command_saves_a_capsule_at_the_split():
    command = R.build_full_command(args(), Path("/tmp/capsules"), "full")
    assert "++algo.config.num_learning_iterations=20" in command
    assert "++callbacks.practice_capsule.horizons.split=10" in command


def test_resume_command_restores_full_state_and_rng():
    command = R.build_resume_command(args(), Path("/tmp/export.pt"), Path("/tmp/split.capsule.pt"))
    assert "+resume=true" in command
    assert "++algo.config.num_learning_iterations=20" in command
    assert "++callbacks.practice_resume.enabled=true" in command
    assert "++callbacks.practice_resume.capsule_path=/tmp/split.capsule.pt" in command


def test_relative_delta_handles_zero_and_direction():
    assert R.relative_delta(10.0, 9.0) == -0.1
    assert R.relative_delta(0.0, 1.0) is None


def test_trailing_mean_requires_the_preregistered_window():
    run = RL.RunLog("short", [RL.Iteration(1, {"Mean rewards": 1.0})])
    assert R.trailing_mean(run, "Mean rewards") is None
