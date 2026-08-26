from types import SimpleNamespace

from scripts.practice_utility import run_throughput_probe as P


def args(**overrides):
    values = dict(
        variant="native",
        exp="manager/universal_token/all_modes/sonic_release",
        checkpoint="/tmp/model.pt",
        num_envs=128,
        iterations=4,
        seed=1,
        motion_file="motions",
        smpl_motion_file="smpl",
        encoder="/tmp/encoder.pt",
        artifact_root=P.Path("/tmp/artifacts"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_command_is_bounded_and_has_no_observer():
    command = P.build_command(args(), "probe")
    joined = " ".join(command)
    assert "num_learning_iterations=4" in joined
    assert "num_envs=128" in joined
    assert "practice_observer" not in joined


def test_observer_command_uses_frozen_encoder_and_is_enabled():
    command = P.build_command(args(variant="observer"), "probe")
    joined = " ".join(command)
    assert "practice_observer.enabled=true" in joined
    assert "practice_observer.encoder_path=/tmp/encoder.pt" in joined
    assert "practice_observer.branch_id=probe" in joined


def test_gpu_summary_reports_capacity_extrema():
    summary = P.summarize_gpu(
        [
            {"free_mib": 30_000.0, "used_mib": 1000.0, "gpu_util_pct": 0.0},
            {"free_mib": 12_000.0, "used_mib": 20_000.0, "gpu_util_pct": 99.0},
        ]
    )
    assert summary == {
        "samples": 2,
        "start_free_mib": 30_000.0,
        "min_free_mib": 12_000.0,
        "max_used_mib": 20_000.0,
        "max_gpu_util_pct": 99.0,
    }


def test_launcher_hash_is_a_sha256():
    assert len(P.launcher_sha256()) == 64
    int(P.launcher_sha256(), 16)
