from types import SimpleNamespace

import torch

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
    try:
        PracticeRobustnessEvalCallback(non_latency_dr_scale=1.1)
    except ValueError as error:
        assert "non_latency_dr_scale" in str(error)
    else:
        raise AssertionError("invalid non-latency scale must fail")
