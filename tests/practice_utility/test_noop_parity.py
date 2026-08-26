from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.practice_utility import run_noop_parity as N


def args():
    return SimpleNamespace(
        checkpoint="/tmp/model.pt",
        num_envs=128,
        seed=7,
        iterations=4,
        exp="manager/universal_token/all_modes/sonic_release",
        log_dir=Path("/tmp/logs"),
    )


def test_native_command_has_no_research_callbacks():
    command = N.build_command(args(), "native")
    assert not any("practice_observer" in value for value in command)
    assert not any("lucid_curriculum" in value for value in command)


def test_research_command_installs_both_callbacks_disabled():
    command = N.build_command(args(), "research_disabled")
    assert "++callbacks.practice_observer.enabled=false" in command
    assert "++callbacks.lucid_curriculum.enabled=false" in command
    assert "seed=7" in command


def test_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unsupported arm"):
        N.build_command(args(), "other")
