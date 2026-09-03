"""Actuator channels as ordinary event terms: are they visible to the curriculum?

The point of routing these through the event manager rather than a parallel path
is that everything else keys on it. The evaluator refuses a channel absent from
the baseline, the curriculum scales what the baseline holds, and the strata
dispatchers read the same structure. So the tests that matter are not "does the
write happen" (that is covered in test_actuator_dr) but "does the rest of the
system see this channel at all", and those can be answered on a CPU with a fake
event manager, which is what this file does.
"""

from __future__ import annotations

import pytest
import torch

from gear_sonic.research.practice_utility import actuator_dr as ADR
from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import events_actuator as EA

PEAK = [139.0, 139.0, 88.0, 50.0]
NUM_ENVS = 5


class FakeData:
    def __init__(self):
        self.joint_pos = torch.zeros((NUM_ENVS, 4))
        self.joint_effort_limits = torch.tensor([PEAK] * NUM_ENVS)
        self.joint_vel_limits = torch.tensor([[20.0, 20.0, 32.0, 37.0]] * NUM_ENVS)
        self.default_joint_armature = torch.full((NUM_ENVS, 4), 0.01)
        self.default_joint_friction_coeff = torch.zeros((NUM_ENVS, 4))


class FakeAsset:
    def __init__(self):
        self.data = FakeData()
        self.writes = {}

    def __getattr__(self, name):
        if not name.startswith("write_joint_"):
            raise AttributeError(name)
        def writer(values, joint_ids=None, env_ids=None):
            self.writes.setdefault(name, []).append(values.clone())
            live = {"write_joint_effort_limit_to_sim": "joint_effort_limits",
                    "write_joint_velocity_limit_to_sim": "joint_vel_limits",
                    "write_joint_armature_to_sim": "default_joint_armature"}.get(name)
            if live is not None:
                getattr(self.data, live)[env_ids][:, joint_ids] = values
        return writer


class FakeEnv:
    def __init__(self):
        self.asset = FakeAsset()
        self.scene = {"robot": self.asset}


class AssetCfg:
    name = "robot"
    joint_ids = slice(None)


class TermCfg:
    """The shape dr_scaling._iter_terms expects: a mode and a params dict."""

    def __init__(self, func, mode, params):
        self.func, self.mode, self.params = func, mode, params


class FakeEventManager:
    def __init__(self, terms):
        self.active_terms = list(terms)
        self._term_cfgs = [terms[n] for n in terms]


def actuator_terms():
    return {
        "randomize_joint_effort_limit": TermCfg(
            EA.randomize_joint_effort_limit, "reset",
            {"asset_cfg": AssetCfg(), "effort_limit_scale_range": [0.5, 1.0]}),
        "randomize_joint_friction": TermCfg(
            EA.randomize_joint_friction, "reset",
            {"asset_cfg": AssetCfg(), "joint_friction_range": [0.0, 6.0]}),
        "randomize_joint_armature": TermCfg(
            EA.randomize_joint_armature, "reset",
            {"asset_cfg": AssetCfg(), "armature_scale_range": [0.7, 1.6]}),
        "randomize_joint_velocity_limit": TermCfg(
            EA.randomize_joint_velocity_limit, "reset",
            {"asset_cfg": AssetCfg(), "velocity_limit_scale_range": [0.6, 1.0]}),
    }


# ------------------------------------- the machinery has to be able to see them

def test_every_actuator_channel_appears_in_the_baseline():
    """The evaluator FAILS CLOSED on a channel missing from the baseline."""
    manager = FakeEventManager(actuator_terms())
    baseline = DS.capture_baseline(manager)
    assert set(baseline) == set(actuator_terms())
    for term, captured in baseline.items():
        assert captured, f"{term} contributed no scalable range"


def test_every_actuator_channel_is_schedulable_at_runtime():
    manager = FakeEventManager(actuator_terms())
    assert set(DS.scalable_terms(manager)) == set(actuator_terms())


def test_a_startup_mode_actuator_term_is_correctly_reported_as_unschedulable():
    terms = actuator_terms()
    terms["randomize_joint_friction"].mode = "startup"
    manager = FakeEventManager(terms)
    assert "randomize_joint_friction" not in DS.scalable_terms(manager)


# ------------------------------------------ lambda zero must be a genuine no-op

@pytest.mark.parametrize("term_name", sorted(actuator_terms()))
def test_at_lambda_zero_the_range_collapses_to_the_nominal(term_name):
    manager = FakeEventManager(actuator_terms())
    baseline = DS.capture_baseline(manager)
    scaled = DS.scaled_term_params(baseline[term_name], 0.0)
    (key, value), = scaled.items()
    nominal = DS.RANGE_NOMINALS[key]
    assert value[0] == pytest.approx(nominal)
    assert value[1] == pytest.approx(nominal)


@pytest.mark.parametrize("term_name,channel", [
    ("randomize_joint_effort_limit", "effort_limit"),
    ("randomize_joint_friction", "joint_friction"),
    ("randomize_joint_armature", "armature"),
    ("randomize_joint_velocity_limit", "velocity_limit"),
])
def test_running_a_term_at_lambda_zero_writes_the_nominal_back(term_name, channel):
    env, terms = FakeEnv(), actuator_terms()
    baseline = DS.capture_baseline(FakeEventManager(terms))
    scaled = DS.scaled_term_params(baseline[term_name], 0.0)
    cfg = terms[term_name]
    cfg.func(env, torch.arange(NUM_ENVS), cfg.params["asset_cfg"], **scaled)
    report = EA.actuator_telemetry(env.asset)[channel]
    assert report["written_mean"] == pytest.approx(report["nominal_mean"], rel=1e-6)


# ------------------------------------------------- lambda actually moves them

def test_lambda_widens_the_applied_range_in_the_expected_direction():
    terms = actuator_terms()
    baseline = DS.capture_baseline(FakeEventManager(terms))
    mild = DS.scaled_term_params(baseline["randomize_joint_effort_limit"], 0.5,
                                 allow_extrapolation=True)["effort_limit_scale_range"]
    harsh = DS.scaled_term_params(baseline["randomize_joint_effort_limit"], 2.0,
                                  allow_extrapolation=True)["effort_limit_scale_range"]
    assert harsh[0] < mild[0] < 1.0        # derating deepens with lambda
    assert harsh[1] == pytest.approx(1.0)  # and never hands back more than peak

    friction = DS.scaled_term_params(baseline["randomize_joint_friction"], 2.0,
                                     allow_extrapolation=True)["joint_friction_range"]
    assert friction[0] == pytest.approx(0.0) and friction[1] > 6.0


# ---------------------------------------------- reset safety, per channel

@pytest.mark.parametrize("term_name,channel", [
    ("randomize_joint_effort_limit", "effort_limit"),
    ("randomize_joint_friction", "joint_friction"),
    ("randomize_joint_armature", "armature"),
    ("randomize_joint_velocity_limit", "velocity_limit"),
])
def test_repeated_resets_do_not_stack(term_name, channel):
    """The accumulation bug events_reset_safe exists to prevent."""
    env, terms = FakeEnv(), actuator_terms()
    cfg = terms[term_name]
    baseline = DS.capture_baseline(FakeEventManager(terms))
    scaled = DS.scaled_term_params(baseline[term_name], 1.0)
    nominals = []
    for _ in range(15):
        cfg.func(env, torch.arange(NUM_ENVS), cfg.params["asset_cfg"], **scaled)
        nominals.append(EA.actuator_telemetry(env.asset)[channel]["nominal_mean"])
    assert max(nominals) == pytest.approx(min(nominals), rel=1e-9)


def test_the_telemetry_records_what_each_channel_applied():
    env, terms = FakeEnv(), actuator_terms()
    for name, cfg in terms.items():
        cfg.func(env, torch.arange(NUM_ENVS), cfg.params["asset_cfg"],
                 **{k: v for k, v in cfg.params.items() if k != "asset_cfg"})
    telemetry = EA.actuator_telemetry(env.asset)
    assert set(telemetry) == set(ADR.CHANNELS)
    for channel, report in telemetry.items():
        assert report["writers_called"] >= 1
        assert report["applied_range"]
        assert report["nominal_source"]
