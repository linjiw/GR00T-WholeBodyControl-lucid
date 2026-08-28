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
