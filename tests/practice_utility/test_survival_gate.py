"""The gate's safety properties, stated as tests.

The whole point of this module is that one failure mode cannot occur, so most
of these tests are attempts to make it occur.
"""

from __future__ import annotations

import pytest

from gear_sonic.research.practice_utility.survival_gate import (
    GateStep,
    SurvivalGateConfig,
    SurvivalGateController,
    linear_ramp_lambda,
)


def gate(**overrides) -> SurvivalGateController:
    config = {
        "threshold": 0.8,
        "window": 5,
        "step_size": 0.125,
        "probe_offset": 0.125,
        "dwell": 0,
        "min_episodes": 5,
        "lambda_max": 1.5,
    }
    config.update(overrides)
    return SurvivalGateController(SurvivalGateConfig(**config), initial_lambda=1.0)


class TestMonotonicity:
    def test_sustained_failure_never_lowers_the_frontier(self):
        # The evacuation this module exists to delete: a long run of terrible
        # probe survival must not move the frontier down by a float epsilon.
        controller = gate()
        for _ in range(500):
            step = controller.update(probe_survival=0.0, probe_episodes=20, mean_return=10.0)
            assert step.frontier_after == 1.0
            assert step.applied_decrease is False
        assert controller.frontier == 1.0
        assert controller.incidents == []

    def test_frontier_never_decreases_across_any_input_sequence(self):
        controller = gate()
        rates = [0.0, 1.0, 0.5, 0.95, 0.1, 1.0, 0.0, 0.99, 0.3, 1.0] * 30
        previous = controller.frontier
        for index, rate in enumerate(rates):
            step = controller.update(
                probe_survival=rate,
                probe_episodes=10,
                mean_return=10.0 + (index % 3),
            )
            assert step.frontier_after >= previous - 1e-12
            previous = step.frontier_after

    def test_guard_freeze_does_not_contract_support(self):
        # Return collapse must halt expansion, not discard support already paid
        # for: contracting is the cost the campaign measured at 7.97 points.
        controller = gate(return_window=4, guard_action="freeze", guard_freeze_iterations=50)
        for _ in range(8):
            controller.update(probe_survival=0.9, probe_episodes=10, mean_return=100.0)
        frontier = controller.frontier
        step = controller.update(probe_survival=0.9, probe_episodes=10, mean_return=1.0)
        for _ in range(6):
            step = controller.update(probe_survival=0.9, probe_episodes=10, mean_return=1.0)
        assert step.guard_tripped is True
        assert step.guard_action == "freeze"
        assert controller.frontier >= frontier
        assert controller.incidents == []

    def test_guard_decay_is_recorded_as_an_incident(self):
        # The opt-in contracting brake still exists, and using it is reportable.
        controller = gate(return_window=4, guard_action="decay", guard_decay=0.9)
        for _ in range(8):
            controller.update(probe_survival=0.9, probe_episodes=10, mean_return=100.0)
        for _ in range(6):
            step = controller.update(probe_survival=0.9, probe_episodes=10, mean_return=1.0)
        assert step.guard_tripped is True
        assert controller.incidents
        assert controller.incidents[0]["cause"] == "return_guard_decay"
        assert any(s.applied_decrease for s in controller.history)


class TestExpansion:
    def test_expands_after_a_full_window_above_threshold(self):
        controller = gate()
        fired = [controller.update(probe_survival=0.9, probe_episodes=10).fired for _ in range(5)]
        assert fired == [False, False, False, False, True]
        assert controller.frontier == pytest.approx(1.125)

    def test_probe_runs_one_step_above_the_frontier(self):
        controller = gate()
        assert controller.probe_lambda == pytest.approx(1.125)
        for _ in range(5):
            controller.update(probe_survival=0.9, probe_episodes=10)
        assert controller.frontier == pytest.approx(1.125)
        assert controller.probe_lambda == pytest.approx(1.25)

    def test_one_good_window_yields_exactly_one_step(self):
        # Hysteresis: without clearing the window a single good stretch would
        # ratchet several steps in succession on the same evidence.
        controller = gate()
        for _ in range(5):
            controller.update(probe_survival=0.9, probe_episodes=10)
        assert controller.expansions == 1
        step = controller.update(probe_survival=0.9, probe_episodes=10)
        assert step.fired is False
        assert step.withheld == "window_not_full"

    def test_below_threshold_never_expands(self):
        controller = gate()
        for _ in range(200):
            step = controller.update(probe_survival=0.79, probe_episodes=10)
        assert controller.expansions == 0
        assert step.withheld == "below_threshold"

    def test_stops_at_the_ceiling(self):
        controller = gate()
        for _ in range(500):
            controller.update(probe_survival=1.0, probe_episodes=10)
        assert controller.frontier == pytest.approx(1.5)
        assert controller.history[-1].withheld == "at_ceiling"

    def test_dwell_delays_the_next_expansion(self):
        controller = gate(dwell=20)
        for _ in range(5):
            controller.update(probe_survival=1.0, probe_episodes=10)
        assert controller.expansions == 1
        withheld = set()
        for _ in range(10):
            withheld.add(controller.update(probe_survival=1.0, probe_episodes=10).withheld)
        assert controller.expansions == 1
        assert "dwell" in withheld or "window_not_full" in withheld


class TestEvidence:
    def test_absent_evidence_is_a_hold_not_a_pass(self):
        controller = gate()
        for _ in range(100):
            step = controller.update(probe_survival=None, probe_episodes=0)
        assert controller.expansions == 0
        assert step.withheld == "window_not_full"
        assert step.window_episodes == 0

    def test_absent_evidence_is_not_a_failure_either(self):
        # Missing iterations must not poison the window: they are skipped, so a
        # sparse but uniformly good probe still expands.
        controller = gate()
        for index in range(10):
            controller.update(
                probe_survival=0.9 if index % 2 == 0 else None,
                probe_episodes=10 if index % 2 == 0 else 0,
            )
        assert controller.expansions == 1

    def test_coverage_floor_blocks_expansion_on_too_few_episodes(self):
        controller = gate(min_episodes=1000)
        for _ in range(50):
            step = controller.update(probe_survival=1.0, probe_episodes=10)
        assert controller.expansions == 0
        assert step.withheld == "insufficient_episodes"


class TestPersistence:
    def test_state_round_trips(self):
        controller = gate()
        for _ in range(7):
            controller.update(probe_survival=0.9, probe_episodes=10, mean_return=10.0)
        state = controller.state_dict()
        restored = gate()
        restored.load_state_dict(state)
        assert restored.frontier == controller.frontier
        assert restored.expansions == controller.expansions
        assert restored.iteration == controller.iteration

    def test_resume_never_rolls_the_frontier_back(self):
        controller = gate()
        for _ in range(5):
            controller.update(probe_survival=1.0, probe_episodes=10)
        advanced = controller.frontier
        controller.load_state_dict({"frontier": advanced})
        assert controller.frontier == advanced


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 0.0},
            {"threshold": 1.5},
            {"window": 1},
            {"step_size": 0.0},
            {"probe_offset": 0.0},
            {"lambda_max": 0.0},
            {"probe_max": 1.0},
            {"guard_action": "lower"},
            {"return_relative_drop": 1.0},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            SurvivalGateConfig(**kwargs)

    def test_initial_lambda_must_be_within_the_ceiling(self):
        with pytest.raises(ValueError):
            SurvivalGateController(SurvivalGateConfig(lambda_max=1.5), initial_lambda=2.0)


class TestRamp:
    def test_ramp_is_flat_then_linear_then_flat(self):
        kwargs = {
            "start_lambda": 1.0,
            "end_lambda": 1.5,
            "begin_iteration": 1000,
            "end_iteration": 5000,
        }
        assert linear_ramp_lambda(0, **kwargs) == pytest.approx(1.0)
        assert linear_ramp_lambda(1000, **kwargs) == pytest.approx(1.0)
        assert linear_ramp_lambda(3000, **kwargs) == pytest.approx(1.25)
        assert linear_ramp_lambda(5000, **kwargs) == pytest.approx(1.5)
        assert linear_ramp_lambda(8000, **kwargs) == pytest.approx(1.5)

    def test_ramp_is_monotone_nondecreasing(self):
        kwargs = {
            "start_lambda": 1.0,
            "end_lambda": 1.5,
            "begin_iteration": 1000,
            "end_iteration": 5000,
        }
        values = [linear_ramp_lambda(i, **kwargs) for i in range(0, 8000, 37)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_ramp_rejects_an_empty_span(self):
        with pytest.raises(ValueError):
            linear_ramp_lambda(
                10, start_lambda=1.0, end_lambda=1.5, begin_iteration=100, end_iteration=100
            )


def test_gate_step_is_serialisable():
    step = GateStep(iteration=1, frontier_before=1.0, frontier_after=1.0, probe_lambda=1.125)
    assert step.to_dict()["probe_lambda"] == pytest.approx(1.125)
    assert step.to_dict()["applied_decrease"] is False
