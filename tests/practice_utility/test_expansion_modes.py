"""The gate and ramp arms, wired to a live event manager.

These two modes exist as a matched pair. The gate reads probe-stratum survival
and decides when to widen support; the ramp widens support on a fixed schedule
and reads nothing. Everything else about them -- stratum count, stratum sizes,
where the probe sits, which channels move -- is identical, so that a difference
between the arms can only be attributed to the feedback.

The ratchet result is why the pair is built this way: a constrained controller
that reaches lambda = 1 and stays there is *distributionally identical* to
fixed randomization, which made its noninferiority test near-tautological. An
arm that widens support must be compared against a schedule that widens it the
same amount.
"""

from __future__ import annotations

import pytest
import torch

from gear_sonic.research.practice_utility import (
    dr_scaling as DS,
    observer as OBS,
    survival_observer as SO,
)
from gear_sonic.research.practice_utility.dr_curriculum import (
    LucidCurriculumCallback,
    clear_curricula,
)
from tests.practice_utility.test_tace import FakeEnv, State, manager


@pytest.fixture(autouse=True)
def _clean():
    OBS.clear_observers()
    SO.clear_survival_observers()
    clear_curricula()
    yield
    OBS.clear_observers()
    SO.clear_survival_observers()
    clear_curricula()


class StubSurvival:
    """Stands in for the observer, feeding the gate a chosen probe rate."""

    def __init__(self, survival, episodes=50, branch_id="exp"):
        self.branch_id = branch_id
        self._survival = survival
        self._episodes = episodes
        self.strata = None
        self.probe_index = None
        self.flushes = []

    def set_strata(self, masks, probe_index=None):
        self.strata = tuple(masks)
        self.probe_index = probe_index

    def ensure_flushed(self, step):
        self.flushes.append(step)

    def current_probe(self):
        return self._survival, self._episodes

    def current_population(self):
        return 0.95


def build(mode, num_envs=32, strata=8, **kwargs):
    env = FakeEnv(manager(num_envs=num_envs), num_envs=num_envs)
    settings = {
        "gate_lambda_max": 1.5,
        "ramp_start_lambda": 1.0,
        "ramp_end_lambda": 1.5,
        "ramp_begin_iteration": 10,
        "ramp_end_iteration": 50,
        "gate_window": 3,
        "gate_dwell": 0,
        "gate_min_episodes": 10,
    }
    settings.update(kwargs)
    callback = LucidCurriculumCallback(
        enabled=True,
        mode=mode,
        branch_id="exp",
        survival_branch_id="exp",
        spread_strata=strata,
        allow_extrapolation=True,
        initial_lambda=1.0,
        **settings,
    )
    return env, callback


class TestStratumLayout:
    def test_probe_stratum_sits_above_the_frontier(self):
        # The signal is read where we are considering moving to, not where we
        # already are. That is what stops the gate from having a fixed point at
        # "make it easier and the signal improves".
        env, callback = build("gate")
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(1.0)
        lambdas = callback._stratum_lambdas_absolute
        assert lambdas is not None
        assert lambdas[-1] == pytest.approx(1.125)
        assert lambdas[-2] == pytest.approx(1.0)
        assert all(a < b for a, b in zip(lambdas, lambdas[1:]))

    def test_tail_strata_retain_the_intensities_already_trained(self):
        env, callback = build("gate", strata=4)
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(1.0)
        lambdas = callback._stratum_lambdas_absolute
        # Two tail strata evenly spaced strictly inside (0, frontier).
        assert lambdas[0] == pytest.approx(1.0 / 3.0)
        assert lambdas[1] == pytest.approx(2.0 / 3.0)
        assert min(lambdas) > 0.0

    def test_gate_and_ramp_place_strata_identically_at_equal_frontier(self):
        # The matched-control property, asserted rather than assumed.
        env_a, gate = build("gate")
        env_b, ramp = build("ramp")
        gate.on_train_begin(None, None, None, env=env_a)
        ramp.on_train_begin(None, None, None, env=env_b)
        gate._apply(1.25)
        ramp._apply(1.25)
        assert gate._stratum_lambdas_absolute == pytest.approx(ramp._stratum_lambdas_absolute)
        assert [len(s) for s in gate.assignment.focus_strata] == [
            len(s) for s in ramp.assignment.focus_strata
        ]

    def test_probe_stratum_params_reach_the_event_dispatcher(self):
        env, callback = build("gate")
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(1.0)
        recorded = callback.dispatchers["randomize_rigid_body_mass"].telemetry()["stratum_params"]
        # baseline [0.8, 1.2] about nominal 1.0 -> half-width 0.2 * lambda.
        low, high = recorded[-1]["mass_distribution_params"]
        assert low == pytest.approx(1.0 - 0.2 * 1.125)
        assert high == pytest.approx(1.0 + 0.2 * 1.125)

    def test_frontier_stratum_is_explicit_not_the_manager_default(self):
        # With a stratum above the frontier the top-stratum "None" shortcut
        # would hand the probe the frontier's params, silently deleting the
        # probe. Every stratum must therefore be explicit here.
        env, callback = build("gate")
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(1.0)
        recorded = callback.dispatchers["randomize_rigid_body_mass"].telemetry()["stratum_params"]
        assert all(entry is not None for entry in recorded)

    def test_unstratified_arms_keep_the_top_stratum_shortcut(self):
        # Identity guard: an arm that places nothing above the frontier must
        # still reproduce the pre-expansion behaviour exactly.
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = LucidCurriculumCallback(
            enabled=True,
            mode="lucid",
            branch_id="legacy",
            spread_strata=4,
            initial_lambda=0.8,
        )
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(0.8)
        assert callback._stratum_lambdas_absolute is None
        recorded = callback.dispatchers["randomize_rigid_body_mass"].telemetry()["stratum_params"]
        assert recorded[-1] is None


class TestGateBehaviour:
    def _run(self, survival, iterations=12, **kwargs):
        env, callback = build("gate", **kwargs)
        SO._ACTIVE["exp"] = StubSurvival(survival)
        callback.on_train_begin(None, None, None, env=env)
        for step in range(1, iterations + 1):
            callback.on_step_end(None, State(step, mean_reward=10.0), None, env=env)
        return callback

    def test_healthy_probe_expands_support(self):
        callback = self._run(0.95)
        assert callback._frontier_lambda > 1.0
        assert callback.gate.expansions >= 1
        assert callback.history[-1]["frontier_lambda"] == callback._frontier_lambda

    def test_failing_probe_never_lowers_support(self):
        callback = self._run(0.0, iterations=60)
        assert callback._frontier_lambda == pytest.approx(1.0)
        assert callback.gate.expansions == 0
        assert callback.gate.incidents == []
        assert not any(step.applied_decrease for step in callback.gate.history)

    def test_expansion_stops_at_the_configured_ceiling(self):
        callback = self._run(1.0, iterations=200)
        assert callback._frontier_lambda == pytest.approx(1.5)

    def test_missing_observer_holds_rather_than_expanding(self):
        env, callback = build("gate")
        callback.on_train_begin(None, None, None, env=env)
        for step in range(1, 40):
            callback.on_step_end(None, State(step, mean_reward=10.0), None, env=env)
        assert callback._frontier_lambda == pytest.approx(1.0)
        assert callback.history[-1]["survival_observer_present"] is False

    def test_record_carries_the_audit_fields(self):
        callback = self._run(0.95)
        record = callback.history[-1]
        for key in (
            "frontier_lambda",
            "probe_lambda",
            "probe_survival",
            "probe_episodes",
            "window_mean",
            "fired",
            "applied_decrease",
            "guard_tripped",
            "population_survival",
        ):
            assert key in record, key
        assert record["signal"] == "survival"

    def test_observer_is_told_which_stratum_is_the_probe(self):
        env, callback = build("gate")
        stub = StubSurvival(0.9)
        SO._ACTIVE["exp"] = stub
        callback.on_train_begin(None, None, None, env=env)
        assert stub.probe_index == callback.spread_strata - 1
        assert stub.strata is not None
        assert len(stub.strata) == callback.spread_strata


class TestRampBehaviour:
    def test_ramp_follows_the_schedule_and_ignores_the_probe(self):
        env, callback = build("ramp")
        SO._ACTIVE["exp"] = StubSurvival(0.0)  # a probe in total failure
        callback.on_train_begin(None, None, None, env=env)
        for step in range(1, 61):
            callback.on_step_end(None, State(step, mean_reward=10.0), None, env=env)
        # Schedule reaches its end regardless of what the probe reported.
        assert callback._frontier_lambda == pytest.approx(1.5)
        assert callback.history[-1]["signal"] == "none"

    def test_ramp_is_monotone_over_the_whole_run(self):
        env, callback = build("ramp")
        callback.on_train_begin(None, None, None, env=env)
        seen = []
        for step in range(1, 61):
            callback.on_step_end(None, State(step, mean_reward=10.0), None, env=env)
            seen.append(callback.history[-1]["lambda"])
        assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))

    def test_ramp_records_probe_telemetry_without_gating_on_it(self):
        env, callback = build("ramp")
        SO._ACTIVE["exp"] = StubSurvival(0.42)
        callback.on_train_begin(None, None, None, env=env)
        callback.on_step_end(None, State(1, mean_reward=10.0), None, env=env)
        record = callback.history[-1]
        assert record["probe_survival"] == pytest.approx(0.42)
        assert "fired" not in record


class TestPersistence:
    def test_gate_state_rides_along_with_the_checkpoint(self):
        callback = TestGateBehaviour()._run(1.0, iterations=20)
        state = callback.state_dict()
        assert state["gate"] is not None
        assert state["frontier_lambda"] == pytest.approx(callback._frontier_lambda)

    def test_resume_restores_the_frontier(self):
        original = TestGateBehaviour()._run(1.0, iterations=20)
        env, restored = build("gate")
        restored.load_state_dict(original.state_dict())
        assert restored._frontier_lambda == pytest.approx(original._frontier_lambda)
        assert restored.gate.expansions == original.gate.expansions


class TestSurvivalObserver:
    def _result(self, dones, time_outs):
        return (None, None, torch.tensor(dones), {"time_outs": torch.tensor(time_outs)})

    def test_per_stratum_survival_splits_the_population(self):
        observer = SO.SurvivalObserverCallback(enabled=True, branch_id="s")
        low = torch.zeros(8, dtype=torch.bool)
        low[:4] = True
        high = ~low
        observer.set_strata([low, high], probe_index=1)
        # Stratum 0 all time out, stratum 1 all terminate early.
        observer._after(
            self._result([True] * 8, [True, True, True, True, False, False, False, False])
        )
        observer.ensure_flushed(1)
        record = observer.history[-1]
        assert record["per_stratum"][0]["survival"] == pytest.approx(1.0)
        assert record["per_stratum"][1]["survival"] == pytest.approx(0.0)
        assert observer.current_probe() == (pytest.approx(0.0), 4)

    def test_no_ended_episodes_reports_no_evidence(self):
        observer = SO.SurvivalObserverCallback(enabled=True, branch_id="s")
        observer.set_strata([torch.ones(4, dtype=torch.bool)], probe_index=0)
        observer.ensure_flushed(1)
        record = observer.history[-1]
        assert record["episodes_ended"] == 0
        assert observer.current_probe() == (None, 0)

    def test_overlapping_strata_are_rejected(self):
        observer = SO.SurvivalObserverCallback(enabled=True, branch_id="s")
        both = torch.ones(4, dtype=torch.bool)
        with pytest.raises(ValueError, match="share an environment"):
            observer.set_strata([both, both])

    def test_probe_index_must_exist(self):
        observer = SO.SurvivalObserverCallback(enabled=True, branch_id="s")
        with pytest.raises(ValueError, match="probe_index"):
            observer.set_strata([torch.ones(4, dtype=torch.bool)], probe_index=3)

    def test_missing_time_outs_is_recorded_as_an_error_not_a_survival(self):
        observer = SO.SurvivalObserverCallback(enabled=True, branch_id="s")
        observer.set_strata([torch.ones(4, dtype=torch.bool)], probe_index=0)
        observer._after((None, None, torch.tensor([True] * 4), {}))
        observer.ensure_flushed(1)
        record = observer.history[-1]
        assert "no_time_outs" in record["errors"]
        assert record["survival"] == pytest.approx(0.0)


def test_extrapolated_probe_params_are_physically_clamped():
    # Above lambda ~1.385 the static-friction floor clamps; the probe must not
    # be handed a negative friction range.
    env, callback = build("gate", gate_lambda_max=2.0, gate_probe_max=2.5)
    callback.on_train_begin(None, None, None, env=env)
    callback._apply(1.9)
    lambdas = callback._stratum_lambdas_absolute
    assert lambdas[-1] == pytest.approx(2.025)
    # The probe intensity is what the friction channel would be handed. Taking
    # the same scaling path the dispatcher takes, the floor must hold.
    params = DS.scaled_term_params(
        {"static_friction_range": [0.3, 1.6], "dynamic_friction_range": [0.3, 1.2]},
        lambdas[-1],
        True,
    )
    clamped, report = DS.clamp_params_physical(params)
    low, high = DS._as_pair(clamped["static_friction_range"])
    assert low >= 0.05
    assert high > low
    assert report


def test_consolidation_is_refused_for_expansion_modes():
    # Consolidation pins every cohort at lambda = 1.0 for the last stretch of
    # training. For an arm whose frontier is above 1.0 that is a silent support
    # contraction the gate never asked for and would not log as an incident.
    for mode in ("gate", "ramp"):
        with pytest.raises(ValueError, match="contract the support"):
            LucidCurriculumCallback(
                enabled=True,
                mode=mode,
                spread_strata=8,
                allow_extrapolation=True,
                gate_lambda_max=1.5,
                ramp_end_lambda=1.5,
                consolidation_fraction=0.3,
            )


def test_consolidation_still_works_for_legacy_modes():
    callback = LucidCurriculumCallback(
        enabled=True, mode="lucid", spread_strata=4, consolidation_fraction=0.3
    )
    assert callback.consolidation_fraction == pytest.approx(0.3)
