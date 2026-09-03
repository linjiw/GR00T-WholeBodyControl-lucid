"""Randomizing the actuator: the axes SONIC leaves fixed.

Tested against a fake articulation implementing exactly the Isaac Lab writers
this uses, because the real one needs the USD runtime and an adapter exercised
only on a GPU is an adapter that is not exercised.

The properties pinned here are the ones that have bitten this project before on
other channels: a draw must re-derive from the nominal rather than stack on the
previous one, intensity zero must be an exact no-op, and the term must report
what it wrote rather than what it intended.
"""

from __future__ import annotations

import pytest
import torch

from gear_sonic.research.practice_utility import actuator_dr as A

PEAK = [139.0, 139.0, 88.0, 50.0]
NUM_ENVS = 6


class FakeData:
    """Uses the CURRENT Isaac Lab field names. joint_velocity_limits and
    default_joint_friction are deprecated aliases that log a warning on read."""

    def __init__(self, legacy=False):
        self.joint_pos = torch.zeros((NUM_ENVS, 4))
        self.joint_effort_limits = torch.tensor([PEAK] * NUM_ENVS)
        vel = torch.tensor([[20.0, 20.0, 32.0, 37.0]] * NUM_ENVS)
        if legacy:
            self.joint_velocity_limits = vel
        else:
            self.joint_vel_limits = vel
        self.default_joint_armature = torch.full((NUM_ENVS, 4), 0.01)
        self.default_joint_friction_coeff = torch.zeros((NUM_ENVS, 4))


class FakeArticulation:
    """Only the writers actuator_dr calls, recording every write."""

    def __init__(self, friction_writers=3, legacy=False):
        self.data = FakeData(legacy=legacy)
        self.writes: dict[str, list[torch.Tensor]] = {}
        self._friction_writers = friction_writers

    def _record(self, name):
        def writer(values, joint_ids=None, env_ids=None):
            self.writes.setdefault(name, []).append(values.clone())
        return writer

    def __getattr__(self, name):
        if not name.startswith("write_joint_"):
            raise AttributeError(name)
        if "dynamic_friction" in name and self._friction_writers < 2:
            raise AttributeError(name)
        if "viscous_friction" in name and self._friction_writers < 3:
            raise AttributeError(name)
        return self._record(name)


def gen(seed=8600):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------- off means off

@pytest.mark.parametrize("channel", sorted(A.CHANNELS))
def test_at_zero_intensity_every_channel_writes_the_nominal_back(channel):
    art = FakeArticulation()
    report = A.apply(art, channel, lam=0.0, generator=gen())
    assert report["written_mean"] == pytest.approx(report["nominal_mean"])
    written = art.writes[A.CHANNELS[channel].writer][0]
    nominal = A._nominal(art, A.CHANNELS[channel])
    if nominal.ndim == 1:
        nominal = nominal.unsqueeze(0).expand(NUM_ENVS, -1)
    assert torch.allclose(written, nominal.to(torch.float32))


# ------------------------------------------------ draws must not accumulate

@pytest.mark.parametrize("channel", sorted(A.CHANNELS))
def test_repeated_resets_re_derive_from_the_nominal(channel):
    """The bug events_reset_safe exists to prevent, checked on every new channel."""
    art = FakeArticulation()
    first = A.apply(art, channel, lam=1.0, generator=gen(1))
    for _ in range(20):
        A.apply(art, channel, lam=1.0, generator=gen(2))
    last = A.apply(art, channel, lam=1.0, generator=gen(1))
    assert last["written_mean"] == pytest.approx(first["written_mean"], rel=1e-6)
    assert last["nominal_mean"] == pytest.approx(first["nominal_mean"], rel=1e-9)


# ---------------------------------------------------- the channels do a thing

def test_derating_only_ever_removes_torque():
    art = FakeArticulation()
    report = A.apply(art, "effort_limit", lam=1.0, generator=gen())
    assert report["written_max"] <= max(PEAK) + 1e-6
    assert report["written_mean"] < report["nominal_mean"]
    assert report["written_min"] >= 0.5 * min(PEAK) - 1e-3


def test_friction_is_added_because_the_simulated_robot_has_none():
    """A scale operation on a nominal of zero would do nothing at all."""
    art = FakeArticulation()
    assert float(art.data.default_joint_friction_coeff.max()) == 0.0
    report = A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert report["combine"] == "add"
    assert report["written_mean"] > 0.0
    assert report["written_max"] <= 6.0 + 1e-6


def test_friction_reaches_all_three_physx_coefficients():
    art = FakeArticulation()
    report = A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert report["writers_called"] == 3
    static = art.writes["write_joint_friction_coefficient_to_sim"][0]
    dynamic = art.writes["write_joint_dynamic_friction_coefficient_to_sim"][0]
    viscous = art.writes["write_joint_viscous_friction_coefficient_to_sim"][0]
    assert torch.equal(static, dynamic) and torch.equal(static, viscous)


def test_an_older_isaac_sim_with_fewer_writers_is_reported_not_assumed():
    art = FakeArticulation(friction_writers=1)
    report = A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert report["writers_called"] == 1
    assert report["writers_available"] == 3


def test_higher_intensity_widens_the_draw():
    art = FakeArticulation()
    mild = A.apply(art, "armature", lam=0.5, generator=gen(7))
    harsh = A.apply(art, "armature", lam=2.0, generator=gen(7))
    assert abs(harsh["written_mean"] - harsh["nominal_mean"]) > abs(
        mild["written_mean"] - mild["nominal_mean"])


def test_no_channel_can_be_driven_negative():
    art = FakeArticulation()
    for channel in A.CHANNELS:
        report = A.apply(art, channel, lam=5.0, generator=gen())
        assert report["written_min"] >= 0.0


# ------------------------------------------------------------- housekeeping

def test_only_the_named_environments_are_written():
    art = FakeArticulation()
    report = A.apply(art, "effort_limit", lam=1.0, env_ids=torch.tensor([0, 3]), generator=gen())
    assert report["envs"] == 2
    assert art.writes["write_joint_effort_limit_to_sim"][0].shape[0] == 2


def test_only_the_named_joints_are_written():
    art = FakeArticulation()
    report = A.apply(art, "joint_friction", lam=1.0, joint_ids=[0, 1], generator=gen())
    assert report["joints"] == 2
    assert art.writes["write_joint_friction_coefficient_to_sim"][0].shape[1] == 2


def test_the_nominal_survives_an_isaac_lab_rename():
    """joint_velocity_limits is deprecated in favour of joint_vel_limits; both work."""
    modern = FakeArticulation()
    legacy = FakeArticulation(legacy=True)
    a = A.apply(modern, "velocity_limit", lam=0.0, generator=gen())
    b = A.apply(legacy, "velocity_limit", lam=0.0, generator=gen())
    assert a["nominal_mean"] == pytest.approx(b["nominal_mean"])
    assert a["nominal_source"] == "joint_vel_limits"
    assert b["nominal_source"] == "joint_velocity_limits"


def test_friction_with_no_field_at_all_falls_back_to_zero():
    """A robot whose USD declares no joint friction still gets the channel."""
    art = FakeArticulation()
    del art.data.default_joint_friction_coeff
    report = A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert report["nominal_mean"] == pytest.approx(0.0)
    assert report["nominal_source"] == "assumed zero"
    assert report["written_mean"] > 0.0


def test_an_unknown_nominal_is_an_error_not_a_silent_zero():
    art = FakeArticulation()
    del art.data.joint_vel_limits
    with pytest.raises(KeyError, match="no nominal source"):
        A.apply(art, "velocity_limit", lam=1.0, generator=gen())


def test_a_swapped_argument_is_refused_at_definition_time():
    """This mistake was made twice while writing the module and was silent both times."""
    with pytest.raises(TypeError, match="tuple of attribute names"):
        A.ActuatorChannel("armature", "default_joint_armature",
                          "write_joint_armature_to_sim", "scale", (-0.3, 0.6))
    with pytest.raises(ValueError, match="not an articulation joint writer"):
        A.ActuatorChannel("x", ("a",), "default_joint_armature", "scale", (0.0, 1.0))
    with pytest.raises(ValueError, match="inverted"):
        A.ActuatorChannel("x", ("a",), "write_joint_armature_to_sim", "scale", (1.0, 0.0))


def test_every_channel_key_matches_its_own_name():
    for key, channel in A.CHANNELS.items():
        assert key == channel.name


def test_a_negative_intensity_is_refused():
    with pytest.raises(ValueError, match="lam"):
        A.apply(FakeArticulation(), "effort_limit", lam=-0.5)


def test_every_channel_states_its_own_range_in_the_report():
    art = FakeArticulation()
    for name, channel in A.CHANNELS.items():
        report = A.apply(art, name, lam=1.0, generator=gen())
        assert report["deviation_at_lam_1"] == list(channel.deviation)
        assert report["channel"] == name
