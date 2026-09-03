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
    ceiling = A.CHANNELS["joint_friction"].deviation[1]
    assert report["written_max"] <= ceiling * max(PEAK) + 1e-6


def test_friction_is_a_fraction_of_each_joints_own_torque_rating():
    """A flat N.m figure locks a 5 N.m wrist while barely touching a 139 N.m knee.

    Expressing the channel relative to each joint's rating makes one range mean
    the same physical thing everywhere, which is what a gearbox friction actually
    does: it scales with the size of the reducer.
    """
    art = FakeArticulation()
    assert A.CHANNELS["joint_friction"].relative_to == "effort_limit"
    A.apply(art, "joint_friction", lam=1.0, generator=gen())
    written = art.writes["write_joint_friction_coefficient_to_sim"][0]
    ceiling = A.CHANNELS["joint_friction"].deviation[1]
    peak = torch.tensor(PEAK)
    # Every joint stays inside its own fraction, and none is driven past its rating.
    assert bool((written <= ceiling * peak.unsqueeze(0) + 1e-6).all())
    assert bool((written < peak.unsqueeze(0)).all())
    # The strong joints receive proportionally more friction than the weak ones.
    assert float(written[:, 0].mean()) > float(written[:, 3].mean())


def test_friction_writes_both_coulomb_coefficients_and_the_same_value_to_each():
    """Static and dynamic Coulomb friction share units and magnitude; one draw is right."""
    art = FakeArticulation()
    report = A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert report["writers_called"] == 2
    static = art.writes["write_joint_friction_coefficient_to_sim"][0]
    dynamic = art.writes["write_joint_dynamic_friction_coefficient_to_sim"][0]
    assert torch.equal(static, dynamic)


def test_friction_never_touches_the_viscous_coefficient():
    """Viscous friction is a DIFFERENT physical quantity, in N.m.s/rad.

    Writing a Coulomb magnitude into it would add a large velocity-proportional
    damping term while the channel claimed to be modelling stiction, so a run
    would be measuring something other than what its own docstring said.
    """
    art = FakeArticulation()
    A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert "write_joint_viscous_friction_coefficient_to_sim" not in art.writes
    channel = A.CHANNELS["joint_friction"]
    assert not any("viscous" in w for w in channel.also_write)


def test_an_older_isaac_sim_with_fewer_writers_is_reported_not_assumed():
    art = FakeArticulation(friction_writers=1)
    report = A.apply(art, "joint_friction", lam=1.0, generator=gen())
    assert report["writers_called"] == 1
    assert report["writers_available"] == 2


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


def test_the_draw_lands_on_the_same_device_as_the_nominal():
    """A CPU draw against a CUDA nominal raises on the first reset in Isaac."""
    art = FakeArticulation()
    A.apply(art, "effort_limit", lam=1.0, generator=gen())
    written = art.writes["write_joint_effort_limit_to_sim"][0]
    assert written.device == art.data.joint_effort_limits.device


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


# ---------------------------------------- proving the write reached the engine

class PhysxView:
    """The few getters actuator_dr reads back through, over a settable store."""

    def __init__(self, store, honest=True):
        self.store, self.honest = store, honest

    def get_dof_max_forces(self):
        return self.store if self.honest else torch.zeros_like(self.store)

    def get_dof_armatures(self):
        return self.store

    def get_dof_max_velocities(self):
        return self.store


def test_without_a_physx_view_the_readback_says_so_rather_than_claiming_success():
    """On a fake, or an older build, 'unavailable' is the honest answer."""
    art = FakeArticulation()
    report = A.apply(art, "effort_limit", lam=1.0, generator=gen())
    assert report["physx_readback"] == "unavailable"
    assert "root_physx_view" in report["physx_reason"]


def test_a_matching_readback_is_reported_as_matched():
    art = FakeArticulation()
    store = torch.tensor([PEAK] * NUM_ENVS)
    art.root_physx_view = PhysxView(store)
    original = A.CHANNELS["effort_limit"].writer

    def capture(values, joint_ids=None, env_ids=None):
        store[env_ids.reshape(-1)[:, None], torch.tensor(joint_ids)] = values
    setattr(art, "_captured", capture)
    art.writes.setdefault(original, [])
    # route the writer into the store so the engine and the write agree
    object.__setattr__(art, original, capture)
    report = A.apply(art, "effort_limit", lam=1.0, generator=gen())
    assert report["physx_readback"] == "matched"
    assert report["physx_max_abs_gap"] < 1e-3


def test_an_engine_that_ignored_the_write_is_reported_as_a_mismatch():
    """The articulation updates its own mirror unconditionally; only this catches it."""
    art = FakeArticulation()
    art.root_physx_view = PhysxView(torch.tensor([PEAK] * NUM_ENVS), honest=False)
    report = A.apply(art, "effort_limit", lam=1.0, generator=gen())
    assert report["physx_readback"] == "MISMATCH"
    assert report["physx_max_abs_gap"] > 1.0


def test_a_readback_that_raises_does_not_kill_the_run():
    class Exploding:
        def get_dof_max_forces(self):
            raise RuntimeError("view detached")

    art = FakeArticulation()
    art.root_physx_view = Exploding()
    report = A.apply(art, "effort_limit", lam=1.0, generator=gen())
    assert report["physx_readback"] == "error"
    assert "view detached" in report["physx_reason"]


def test_a_negative_intensity_is_refused():
    with pytest.raises(ValueError, match="lam"):
        A.apply(FakeArticulation(), "effort_limit", lam=-0.5)


def test_every_channel_states_its_own_range_in_the_report():
    art = FakeArticulation()
    for name, channel in A.CHANNELS.items():
        report = A.apply(art, name, lam=1.0, generator=gen())
        assert report["deviation_at_lam_1"] == list(channel.deviation)
        assert report["channel"] == name
