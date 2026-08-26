from pathlib import Path
from types import SimpleNamespace

from scripts.practice_utility import run_restart_pair_equivalence as R


def args():
    return SimpleNamespace(
        checkpoint=Path("/tmp/export.pt"),
        capsule=Path("/tmp/split.capsule.pt"),
        num_envs=128,
        seed=7,
        total_iterations=20,
        exp="manager/universal_token/all_modes/sonic_release",
    )


def test_both_branches_use_the_same_resume_contract():
    command = R.build_command(args())
    assert "+resume=true" in command
    assert "checkpoint=/tmp/export.pt" in command
    assert "++callbacks.practice_resume.capsule_path=/tmp/split.capsule.pt" in command
    assert "++algo.config.num_learning_iterations=20" in command


def test_sha256_reads_artifact_contents(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"lucid")
    assert R.sha256(artifact) == "948278737ada1997420a2cba8adaaff837c48e027d319d47b2555240c2ac091d"
