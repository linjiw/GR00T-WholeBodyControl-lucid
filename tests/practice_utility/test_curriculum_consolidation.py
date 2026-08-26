from pathlib import Path
from types import SimpleNamespace

from scripts.practice_utility import run_curriculum_consolidation as R


def args():
    return SimpleNamespace(
        max_delay=8,
        num_envs=128,
        exp="manager/universal_token/all_modes/sonic_release",
        encoder="/tmp/encoder.pt",
    )


def test_command_resumes_to_absolute_target_under_full_dr():
    command = R.build_command(
        args(),
        checkpoint=Path("/tmp/model.pt"),
        capsule=Path("/tmp/model.capsule.pt"),
        seed=8600,
        branch_id="branch",
        artifact_dir=Path("/tmp/artifact"),
        target_step=48,
    )
    for expected in (
        "+resume=true",
        "++algo.config.num_learning_iterations=48",
        "++callbacks.practice_resume.capsule_path=/tmp/model.capsule.pt",
        "++callbacks.lucid_curriculum.mode=fixed",
        "++callbacks.lucid_curriculum.fixed_lambda=1.0",
        "++callbacks.lucid_curriculum.warmup_iterations=0",
        "++callbacks.practice_capsule.horizons.final=48",
    ):
        assert expected in command


def test_source_index_keeps_only_comparison_modes():
    receipt = {
        "arms": {
            "a": {"seed": 1, "mode": "lucid"},
            "b": {"seed": 1, "mode": "fixed"},
            "c": {"seed": 1, "mode": "off"},
        }
    }
    assert sorted(R.source_index(receipt)) == [(1, "fixed"), (1, "lucid")]


def test_arm_order_is_rotated_across_seeds():
    modes = list(R.MODES)
    assert R.rotated(modes, 0) == ["lucid", "fixed"]
    assert R.rotated(modes, 1) == ["fixed", "lucid"]
