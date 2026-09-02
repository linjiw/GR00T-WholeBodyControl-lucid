from types import SimpleNamespace

import pytest
import torch

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility.eval_callback import (
    PracticeRobustnessEvalCallback,
)


class FakeBuffer:
    def __init__(self):
        self.time_lags = torch.tensor([12, 12])


def test_post_evaluation_adds_quality_and_live_delay(monkeypatch):
    callback = PracticeRobustnessEvalCallback(preset_id="latency_60ms", branch_id="b")
    callback.env = SimpleNamespace(event_manager=SimpleNamespace(active_terms=[], _term_cfgs=[]))
    robot = SimpleNamespace(
        actuators={"legs": SimpleNamespace(positions_delay_buffer=FakeBuffer())}
    )
    monkeypatch.setattr(
        "gear_sonic.research.practice_utility.eval_callback._scene_entity",
        lambda env, name: robot,
    )
    result = callback._post_evaluate_policy(
        {
            "metrics_success": {"success_rate": 0.5},
            "metrics_all": {"mpjpe_g": 42.0},
            "all_metrics_dict": {},
            "failed_metrics_dict": {},
            "failed_keys": [],
            "failed_idxes": [],
        }
    )
    assert result["eval/protocol/preset_id"] == "latency_60ms"
    assert result["eval/quality/steps"] == 0
    assert result["eval/protocol/active_dr_terms"] == []
    assert result["eval/delay/action_delay_min_steps"] == 12
    assert result["eval/delay/action_delay_max_steps"] == 12


def test_non_latency_scale_is_validated():
    # Evaluation may extrapolate past the training envelope; the ceiling is
    # DS.MAX_EXTRAPOLATION, not 1. Training is capped at 1 elsewhere.
    PracticeRobustnessEvalCallback(non_latency_dr_scale=1.25)
    try:
        PracticeRobustnessEvalCallback(non_latency_dr_scale=DS.MAX_EXTRAPOLATION + 0.1)
    except ValueError as error:
        assert "non_latency_dr_scale" in str(error)
    else:
        raise AssertionError("a scale past the extrapolation ceiling must fail")


def test_curriculum_scaling_is_still_hard_capped_at_one():
    # The evaluator's freedom must not leak into the curriculum: a training
    # distribution allowed past its own envelope cannot be falsified by the
    # evaluation that follows it.
    DS.scale_range([0.8, 1.2], 1.25, 1.0, allow_extrapolation=True)
    try:
        DS.scale_range([0.8, 1.2], 1.25, 1.0)
    except ValueError as error:
        assert "lambda must be in [0, 1.0]" in str(error)
    else:
        raise AssertionError("the default path must refuse lambda > 1")


def test_extrapolation_widens_about_the_nominal():
    low, high = DS.scale_range([0.8, 1.2], 1.5, 1.0, allow_extrapolation=True)
    assert low == pytest.approx(0.7)
    assert high == pytest.approx(1.3)


class Term:
    def __init__(self, params):
        self.params = params
        self.mode = "reset"


class Manager:
    def __init__(self, terms):
        self.active_terms = list(terms)
        self._term_cfgs = list(terms.values())


def test_pinning_latency_rewrites_every_delay_term_and_reports_which():
    manager = Manager({
        "randomize_action_delay": Term({"delay_range": [0.0, 8.0]}),
        "randomize_action_delay_interval": Term({"delay_range": [0.0, 8.0]}),
        "push_robot": Term({"velocity_range": {"x": [-1.0, 1.0]}}),
    })
    from gear_sonic.research.practice_utility.eval_callback import _pin_action_delay

    report = _pin_action_delay(manager, 6.0)
    assert report["pinned_terms"] == [
        "randomize_action_delay",
        "randomize_action_delay_interval",
    ]
    assert manager._term_cfgs[0].params["delay_range"] == [6.0, 6.0]
    assert manager._term_cfgs[1].params["delay_range"] == [6.0, 6.0]
    # A term with no delay range is untouched.
    assert manager._term_cfgs[2].params == {"velocity_range": {"x": [-1.0, 1.0]}}


def test_fixed_latency_steps_is_validated():
    PracticeRobustnessEvalCallback(fixed_latency_steps=0)
    PracticeRobustnessEvalCallback(fixed_latency_steps=12)
    with pytest.raises(ValueError):
        PracticeRobustnessEvalCallback(fixed_latency_steps=-1)


def test_the_ladder_rungs_cover_the_training_envelope_and_beyond():
    from scripts.practice_utility import run_curriculum_robustness_eval as E

    # 5 ms per physics step; training samples 0-40 ms, so the ladder must reach
    # inside it and past it, or it cannot show where the cliff is.
    ms = {name: steps * 5 for name, steps in E.PRESET_FIXED_LATENCY_STEPS.items()}
    assert min(ms.values()) < 40 and max(ms.values()) > 40
    for name, value in ms.items():
        assert name == f"lat_{value}ms"
    # Every rung runs on nominal physics, so latency is the only axis moving.
    for name in E.PRESET_FIXED_LATENCY_STEPS:
        assert E.PRESETS[name] == "tracking/lucid_eval_clean"
        assert name not in E.PRESET_DR_SCALE


class _MatTerm:
    def __init__(self):
        self.material_buckets = torch.tensor([[0.3, 0.3, 0.0], [1.6, 1.2, 0.5]])
    def __call__(self, *a, **k): return None


class _Cfg:
    def __init__(self, params, func=None):
        self.params = params; self.mode = "reset"; self.func = func


class _Mgr:
    def __init__(self, terms):
        self.active_terms = list(terms); self._term_cfgs = list(terms.values())


def test_extrapolated_friction_is_clamped_to_physical_validity():
    # 1.5x the [0.3, 1.6] envelope about 0.95 is [-0.025, 1.925]; PhysX does
    # not accept negative friction. Every extrapolated cell before this clamp
    # carried that tail.
    term = _MatTerm()
    mgr = _Mgr({
        "physics_material": _Cfg({"static_friction_range": [-0.025, 1.925],
                                  "dynamic_friction_range": [0.075, 1.425],
                                  "restitution_range": [0.0, 0.75], "num_buckets": 2}, term),
        "randomize_rigid_body_mass": _Cfg({"mass_distribution_params": [-0.2, 2.5]}),
        "push_robot": _Cfg({"velocity_range": {"x": [-9.0, 9.0]}}),
    })
    report = DS.clamp_physical(mgr)
    assert mgr._term_cfgs[0].params["static_friction_range"] == [0.05, 1.925]
    assert mgr._term_cfgs[1].params["mass_distribution_params"] == [0.1, 2.5]
    assert mgr._term_cfgs[2].params["velocity_range"]["x"] == [-9.0, 9.0]  # no limit declared
    assert report["clamped"]["physics_material"]["static_friction_range"]["from"] == [-0.025, 1.925]
    assert report["clamped"]["physics_material"]["material_buckets_redrawn_consistent"] is True
    # redrawn buckets respect the floor and dynamic <= static
    b = term.material_buckets
    assert float(b[:, 0].min()) >= 0.05 and bool((b[:, 1] <= b[:, 0]).all())


def test_an_in_envelope_range_is_untouched():
    mgr = _Mgr({"physics_material": _Cfg({"static_friction_range": [0.3, 1.6]})})
    report = DS.clamp_physical(mgr)
    assert report["clamped"] == {}
    assert mgr._term_cfgs[0].params["static_friction_range"] == [0.3, 1.6]


class _ChannelCfg:
    def __init__(self, params, mode="reset"):
        self.params = params
        self.mode = mode
        self.func = None


def _manager(**terms):
    return SimpleNamespace(
        active_terms=list(terms), _term_cfgs=[_ChannelCfg(params) for params in terms.values()]
    )


def test_channel_scales_widen_only_the_named_term():
    # The scalar scale puts every channel at its envelope; the channel scale
    # then widens ONE term from its baseline. Every other range must still be
    # exactly the training envelope, or the cell is not a marginal.
    manager = _manager(
        randomize_rigid_body_mass={"mass_distribution_params": [0.8, 1.5]},
        push_robot={"velocity_range": {"x": [-0.5, 0.5]}},
    )
    callback = PracticeRobustnessEvalCallback(
        non_latency_dr_scale=1.0, channel_dr_scales={"randomize_rigid_body_mass": 2.0}
    )
    callback.env = SimpleNamespace(event_manager=manager)
    callback.quality.reset = lambda: None
    PracticeRobustnessEvalCallback.__mro__[1]._pre_evaluate_policy = lambda self, reset_env=True: None
    callback._pre_evaluate_policy()
    mass, push = manager._term_cfgs
    assert mass.params["mass_distribution_params"] == pytest.approx([0.6, 2.0])
    assert push.params["velocity_range"]["x"] == pytest.approx([-0.5, 0.5])
    assert callback._dr_scale_report["channels"]["randomize_rigid_body_mass"]["scaled_terms"] == [
        "randomize_rigid_body_mass"
    ]
    assert "physical_clamp_channels" in callback._dr_scale_report


def test_channel_scales_refuse_an_unknown_term():
    manager = _manager(push_robot={"velocity_range": {"x": [-0.5, 0.5]}})
    callback = PracticeRobustnessEvalCallback(channel_dr_scales={"randomize_rigid_body_mass": 2.0})
    callback.env = SimpleNamespace(event_manager=manager)
    callback.quality.reset = lambda: None
    with pytest.raises(ValueError, match="no scalable range"):
        callback._pre_evaluate_policy()


def test_channel_scales_are_validated_at_construction():
    with pytest.raises(ValueError, match="channel_dr_scales"):
        PracticeRobustnessEvalCallback(channel_dr_scales={"push_robot": DS.MAX_EXTRAPOLATION + 0.5})
    assert PracticeRobustnessEvalCallback(channel_dr_scales=None).channel_dr_scales is None
    assert PracticeRobustnessEvalCallback(channel_dr_scales={}).channel_dr_scales is None
