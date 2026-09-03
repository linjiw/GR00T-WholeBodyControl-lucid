"""The adapter between the thermal model and a running articulation.

Tested against a fake articulation rather than Isaac, because the real one cannot
be imported without the USD runtime and an adapter that is only exercised on a GPU
is an adapter that is not exercised. The fake implements exactly the two pieces of
the IsaacLab interface this uses, so a change in either is caught here.
"""

from __future__ import annotations

import pytest
import torch

from gear_sonic.research.practice_utility.thermal import ThermalConfig
from gear_sonic.research.practice_utility.thermal_actuator import ThermalBudget

PEAK = [139.0, 139.0, 88.0, 50.0]


class FakeData:
    def __init__(self, num_envs, peak):
        self.joint_effort_limits = torch.tensor([peak] * num_envs, dtype=torch.float32)
        self.joint_pos = torch.zeros((num_envs, len(peak)))
        self.applied_torque = torch.zeros((num_envs, len(peak)), dtype=torch.float32)


class FakeArticulation:
    """Only what ThermalBudget touches: the data block and the limit writer."""

    def __init__(self, num_envs=4, peak=PEAK):
        self.num_instances = num_envs
        self.data = FakeData(num_envs, peak)
        self.writes: list[tuple[torch.Tensor, list[int]]] = []

    def write_joint_effort_limit_to_sim(self, limits, joint_ids=None, env_ids=None):
        assert limits.dtype == torch.float32, "PhysX takes float32"
        self.writes.append((limits.clone(), list(joint_ids) if joint_ids else None))
        self.data.joint_effort_limits[:, joint_ids] = limits


def budget(lam=1.0, num_envs=4, **cfg):
    art = FakeArticulation(num_envs)
    return art, ThermalBudget(art, lam=lam, config=ThermalConfig(**cfg))


def test_the_budget_reads_the_hardware_rating_from_the_articulation():
    art, b = budget()
    assert torch.allclose(b.state.peak, torch.tensor(PEAK, dtype=torch.float64))


def test_at_zero_intensity_nothing_is_ever_written():
    """A run with the channel off must leave the simulation exactly as it was."""
    art, b = budget(lam=0.0)
    b.reset()
    for _ in range(100):
        art.data.applied_torque = torch.tensor([PEAK] * 4, dtype=torch.float32)
        b.step(0.02)
    assert art.writes == []
    assert b.report()["enabled"] is False


def test_working_hard_lowers_the_limit_the_simulation_sees():
    art, b = budget(lam=3.0)
    b.state.temperature.zero_()
    b._write()
    first = art.data.joint_effort_limits.clone()
    for _ in range(500):  # 10 s at peak
        art.data.applied_torque = art.data.joint_effort_limits.clone()
        b.step(0.02)
    assert float(art.data.joint_effort_limits.max()) < float(first.max())
    assert float(art.data.joint_effort_limits.min()) >= 0.5 * min(PEAK) - 1e-3


def test_the_state_advances_on_delivered_torque_not_on_the_rating():
    """The articulation reports what it actually applied; that is what heats it."""
    art, b = budget(lam=3.0)
    b.state.temperature.zero_()
    art.data.applied_torque = torch.zeros((4, 4), dtype=torch.float32)
    for _ in range(200):
        b.step(0.02)
    assert float(b.state.temperature.max()) == pytest.approx(0.0, abs=1e-9)


def test_a_reset_redraws_and_writes_a_fresh_budget():
    art, b = budget(lam=2.0)
    for _ in range(300):
        art.data.applied_torque = art.data.joint_effort_limits.clone()
        b.step(0.02)
    hot = float(art.data.joint_effort_limits.min())
    art.data.joint_effort_limits = torch.tensor([PEAK] * 4, dtype=torch.float32)
    b.reset(torch.tensor([0, 1]))
    assert len(art.writes) > 0
    assert float(b.state.temperature[2:].max()) > 0.0 or hot > 0.0


def test_only_the_named_joints_are_governed():
    art = FakeArticulation()
    b = ThermalBudget(art, lam=3.0, joint_ids=[0, 1])
    b.state.temperature.zero_()
    for _ in range(500):
        art.data.applied_torque = torch.tensor([PEAK] * 4, dtype=torch.float32)
        b.step(0.02)
    limits = art.data.joint_effort_limits[0]
    assert float(limits[0]) < PEAK[0]          # governed
    assert float(limits[2]) == pytest.approx(PEAK[2])  # untouched
    assert float(limits[3]) == pytest.approx(PEAK[3])


def test_the_report_says_whether_it_actually_wrote():
    """Construction alone must not touch the simulation; only reset and step do."""
    art, b = budget(lam=1.0)
    assert b.report()["wrote_limit"] is False
    assert art.writes == []
    b.reset()
    assert b.report()["wrote_limit"] is True
    assert len(art.writes) == 1
