from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.practice_utility import run_latency_ab as L


def args():
    return SimpleNamespace(
        max_delay=8,
        artifact_root=Path("/tmp/latency"),
        exp="manager/universal_token/all_modes/sonic_release",
        checkpoint="/tmp/model.pt",
        num_envs=256,
        seed=7,
        iterations=12,
        encoder="/tmp/encoder.pt",
    )


def test_commands_hold_everything_but_delay_and_output_identity_fixed():
    low = L.build_command(args(), 0, "low")
    high = L.build_command(args(), 1, "high")
    assert "++manager_env.events.randomize_action_delay.params.delay_range=[0.0,0.0]" in low
    assert "++manager_env.events.randomize_action_delay.params.delay_range=[0.0,8.0]" in high
    for expected in ("--max-delay", "8", "seed=7", "num_envs=256"):
        assert expected in low and expected in high


def test_rejects_nonendpoint_lambda():
    with pytest.raises(ValueError, match="0 or 1"):
        L.build_command(args(), 0.5, "bad")


def test_preregistered_activation_uses_frozen_noise_floors():
    def arm(reward, length):
        training = {
            metric: {"last4_mean": value}
            for metric, value in (
                ("Mean rewards", reward),
                ("Mean length", length),
                ("Mean entropy", 1.0),
            )
        }
        observer = {metric: 1.0 for metric in L.OBSERVER_METRICS}
        return {"training": training, "observer_last4_mean": observer}

    below = L.compare_arms(arm(10.0, 100.0), arm(10.3, 103.0))
    above = L.compare_arms(arm(10.0, 100.0), arm(10.4, 103.2))
    assert not below["latency_channel_behaviorally_active"]
    assert above["latency_channel_behaviorally_active"]


def test_latency_only_preset_has_no_other_dr_terms():
    path = L.REPO / "gear_sonic/config/manager_env/events/tracking/lucid_latency_only.yaml"
    text = path.read_text()
    assert "terms/randomize_action_delay@_here_" in text
    for excluded in ("physics_material@", "base_com@", "push_robot@", "mass@"):
        assert excluded not in text
