"""LUCID-S: intensity strata, and the scale-free return guard.

Two changes are under test, and they are separable on purpose.

**Strata.** ``spread_strata = K`` splits the focus cohort into K intensity
strata so the training mixture spans ``(0, lambda]`` instead of sitting at the
single point ``lambda``. The invariant that matters most is that ``K = 1``
changes nothing: every existing receipt was produced by the un-stratified path
and must stay reproducible.

**Relative guard.** The absolute return floor was calibrated in the
32-iteration regime and fired continuously at 128 iterations, where the reward
scale had halved for every arm including the no-DR control. The replacement
compares a return against the best of a trailing window of its own history, so
it is invariant to the scale of the reward and to slow drift in it.
"""

import pytest
import torch

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import observer as OBS
from gear_sonic.research.practice_utility import tace as TACE
from gear_sonic.research.practice_utility.dr_controller import LucidDRController, PIConfig
from gear_sonic.research.practice_utility.dr_curriculum import (
    LucidCurriculumCallback,
    clear_curricula,
)
from tests.practice_utility.test_tace import (
    FakeEnv,
    MaterialTerm,
    Recorder,
    State,
    StubObserver,
    Term,
    manager,
)


@pytest.fixture(autouse=True)
def _clean():
    OBS.clear_observers()
    clear_curricula()
    yield
    OBS.clear_observers()
    clear_curricula()


# ------------------------------------------------------------- assignment --


class TestStratifiedAssignment:
    def test_single_stratum_holds_every_focus_env(self):
        plan = TACE.assign_cohorts(16, 0.25, seed=7, num_strata=1)
        assert plan.num_strata == 1
        assert len(plan.focus_strata) == 1
        assert set(plan.focus_strata[0]) == set(range(16)) - set(plan.anchor_ids)

    def test_strata_partition_the_focus_cohort_exactly(self):
        plan = TACE.assign_cohorts(64, 0.25, seed=11, num_strata=4)
        flat = [env for stratum in plan.focus_strata for env in stratum]
        assert len(flat) == len(set(flat)) == plan.num_focus
        assert set(flat).isdisjoint(plan.anchor_ids)
        assert set(flat) | set(plan.anchor_ids) == set(range(64))

    def test_strata_sizes_are_near_equal(self):
        plan = TACE.assign_cohorts(128, 0.0, seed=3, num_strata=4)
        sizes = [len(stratum) for stratum in plan.focus_strata]
        assert max(sizes) - min(sizes) <= 1

    def test_reserved_env_lands_in_the_top_stratum(self):
        # The controller reads its gap from the reserved env. It must observe
        # the frontier it is deciding whether to expand, not easier company.
        plan = TACE.assign_cohorts(32, 0.25, seed=5, reserved_focus_ids=(0,), num_strata=4)
        assert 0 in plan.focus_strata[-1]
        assert all(0 not in stratum for stratum in plan.focus_strata[:-1])

    def test_assignment_is_seed_deterministic(self):
        a = TACE.assign_cohorts(64, 0.5, seed=99, num_strata=3)
        b = TACE.assign_cohorts(64, 0.5, seed=99, num_strata=3)
        assert a.focus_strata == b.focus_strata
        assert TACE.assign_cohorts(64, 0.5, seed=100, num_strata=3).focus_strata != a.focus_strata

    def test_weights_put_the_top_stratum_at_the_frontier(self):
        assert TACE.stratum_weights(4) == (0.25, 0.5, 0.75, 1.0)
        assert TACE.stratum_weights(1) == (1.0,)
        with pytest.raises(ValueError):
            TACE.stratum_weights(0)

    def test_masks_match_the_stratum_membership(self):
        plan = TACE.assign_cohorts(16, 0.25, seed=1, num_strata=2)
        masks = plan.stratum_masks()
        assert len(masks) == 2
        for ids, mask in zip(plan.focus_strata, masks, strict=True):
            assert mask.nonzero().flatten().tolist() == list(ids)


# --------------------------------------------------------------- dispatch --


class TestStratifiedDispatch:
    def test_one_stratum_makes_a_single_focus_call(self):
        # K = 1 must be the pre-strata path: one focus call, no overrides.
        recorder = Recorder()
        plan = TACE.assign_cohorts(8, 0.25, seed=2, num_strata=1)
        dispatch = TACE.CohortDispatch(
            recorder, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks()
        )
        dispatch(None, torch.arange(8), velocity_range=[-0.5, 0.5])
        focus_calls = [c for c in recorder.calls if c[1]["velocity_range"] == [-0.5, 0.5]]
        assert len(focus_calls) == 1

    def test_each_stratum_gets_its_own_call_and_params(self):
        recorder = Recorder()
        plan = TACE.assign_cohorts(8, 0.0, seed=2, num_strata=4)
        dispatch = TACE.CohortDispatch(
            recorder, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks()
        )
        for index in range(3):
            dispatch.set_stratum(index, {"velocity_range": [-0.25 * (index + 1), 0.25 * (index + 1)]})
        dispatch.set_stratum(3, None)
        dispatch(None, torch.arange(8), velocity_range=[-1.0, 1.0])
        seen = [call[1]["velocity_range"] for call in recorder.calls]
        assert seen == [[-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-1.0, 1.0]]

    def test_every_env_is_sampled_exactly_once(self):
        recorder = Recorder()
        plan = TACE.assign_cohorts(16, 0.25, seed=4, num_strata=4)
        dispatch = TACE.CohortDispatch(
            recorder, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks()
        )
        for index in range(3):
            dispatch.set_stratum(index, {"velocity_range": [-0.1, 0.1]})
        dispatch(None, torch.arange(16), velocity_range=[-1.0, 1.0])
        sampled = torch.cat([call[0] for call in recorder.calls]).tolist()
        assert sorted(sampled) == list(range(16))

    def test_partial_reset_only_touches_the_reset_envs(self):
        recorder = Recorder()
        plan = TACE.assign_cohorts(16, 0.25, seed=6, num_strata=4)
        dispatch = TACE.CohortDispatch(
            recorder, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks()
        )
        for index in range(3):
            dispatch.set_stratum(index, {"velocity_range": [-0.1, 0.1]})
        subset = torch.tensor([1, 3, 5, 7, 9])
        dispatch(None, subset, velocity_range=[-1.0, 1.0])
        sampled = torch.cat([call[0] for call in recorder.calls]).tolist()
        assert sorted(sampled) == subset.tolist()

    def test_consolidation_still_overrides_every_stratum(self):
        recorder = Recorder()
        plan = TACE.assign_cohorts(8, 0.25, seed=2, num_strata=4)
        dispatch = TACE.CohortDispatch(
            recorder, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks()
        )
        dispatch.all_envs_mode = True
        dispatch(None, torch.arange(8), velocity_range=[-0.1, 0.1])
        assert len(recorder.calls) == 1
        assert recorder.calls[0][1]["velocity_range"] == [-1.0, 1.0]

    def test_stratum_buckets_are_swapped_in_and_restored(self):
        term = MaterialTerm(n=8)
        plan = TACE.assign_cohorts(8, 0.0, seed=2, num_strata=2)
        dispatch = TACE.CohortDispatch(
            term, "physics_material", {"static_friction_range": [0.3, 1.6]}, plan.mask(),
            plan.stratum_masks(),
        )
        live = term.material_buckets.clone()
        narrow = torch.zeros_like(live)
        dispatch.set_stratum(0, {"static_friction_range": [0.9, 1.0]}, narrow)
        dispatch.set_stratum(1, None)
        dispatch(None, torch.arange(8), static_friction_range=[0.3, 1.6])
        assert torch.equal(term.seen_buckets[0], narrow)
        assert torch.equal(term.seen_buckets[1], live)
        assert torch.equal(term.material_buckets, live)

    def test_telemetry_counts_every_stratum(self):
        recorder = Recorder()
        plan = TACE.assign_cohorts(16, 0.25, seed=8, num_strata=4)
        dispatch = TACE.CohortDispatch(
            recorder, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks()
        )
        for index in range(3):
            dispatch.set_stratum(index, {"velocity_range": [-0.1, 0.1]})
        dispatch(None, torch.arange(16), velocity_range=[-1.0, 1.0])
        counts = dispatch.telemetry()["env_counts"]
        assert counts["anchor"] == plan.num_anchor
        assert sum(counts[f"focus_s{i}"] for i in range(4)) == plan.num_focus

    def test_set_stratum_rejects_an_out_of_range_index(self):
        plan = TACE.assign_cohorts(8, 0.0, seed=2, num_strata=2)
        dispatch = TACE.CohortDispatch(
            Recorder(), "t", {}, plan.mask(), plan.stratum_masks()
        )
        with pytest.raises(IndexError):
            dispatch.set_stratum(5, None)


# ------------------------------------------------------- curriculum wiring --


class TestCurriculumStrata:
    def _callback(self, num_envs=16, **kwargs):
        return LucidCurriculumCallback(
            enabled=True,
            mode="lucid",
            branch_id="spread",
            observer_branch_id="obs",
            initial_lambda=0.8,
            **kwargs,
        )

    def test_strata_alone_install_dispatchers_without_an_anchor(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(anchor_ratio=0.0, spread_strata=4)
        callback.on_train_begin(None, None, None, env=env)
        assert callback.assignment is not None
        assert callback.assignment.num_anchor == 0
        assert callback.assignment.num_strata == 4
        assert callback.dispatchers

    def test_one_stratum_installs_nothing_new(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(anchor_ratio=0.0, spread_strata=1)
        callback.on_train_begin(None, None, None, env=env)
        assert callback.assignment is None
        assert callback.dispatchers == {}

    def test_stratum_params_are_lambda_times_the_weight(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(anchor_ratio=0.0, spread_strata=4)
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(0.8)
        dispatch = callback.dispatchers["randomize_rigid_body_mass"]
        recorded = dispatch.telemetry()["stratum_params"]
        # baseline [0.8, 1.2] about nominal 1.0, so half-width 0.2 * lambda * w.
        for index, weight in enumerate((0.25, 0.5, 0.75)):
            low, high = recorded[index]["mass_distribution_params"]
            assert low == pytest.approx(1.0 - 0.2 * 0.8 * weight)
            assert high == pytest.approx(1.0 + 0.2 * 0.8 * weight)
        assert recorded[3] is None

    def test_top_stratum_matches_the_scalar_curriculum_exactly(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(anchor_ratio=0.0, spread_strata=4)
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(0.6)
        term = env.event_manager._terms["randomize_rigid_body_mass"]
        assert term.params["mass_distribution_params"] == DS.scale_range([0.8, 1.2], 0.6, 1.0)

    def test_anchor_and_strata_compose(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(anchor_ratio=0.5, anchor_seed=3, spread_strata=2)
        callback.on_train_begin(None, None, None, env=env)
        plan = callback.assignment
        assert plan.num_anchor == 8
        assert sum(len(s) for s in plan.focus_strata) == 8

    def test_telemetry_reports_realized_stratum_intensities(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(anchor_ratio=0.0, spread_strata=4)
        OBS._ACTIVE_OBSERVERS["obs"] = StubObserver([0.5] * 8, branch_id="obs")
        callback.on_train_begin(None, None, None, env=env)
        callback.on_step_end(None, State(1, mean_reward=10.0), None, env=env)
        record = callback.history[-1]
        assert record["tace"]["num_strata"] == 4
        assert len(record["tace"]["stratum_lambdas"]) == 4
        assert record["tace"]["stratum_lambdas"][-1] == pytest.approx(record["lambda"])

    def test_spread_strata_is_persisted_in_state(self):
        callback = self._callback(anchor_ratio=0.0, spread_strata=3)
        assert callback.state_dict()["spread_strata"] == 3

    def test_rejects_a_zero_stratum_count(self):
        with pytest.raises(ValueError):
            self._callback(spread_strata=0)


# ------------------------------------------------------------ return guard --


class TestRelativeReturnGuard:
    def _controller(self, **kwargs):
        config = PIConfig(
            delta_target=0.1,
            return_guard="relative",
            return_relative_drop=0.25,
            return_window=8,
            **kwargs,
        )
        return LucidDRController(config, initial_lambda=1.0)

    def test_steady_returns_never_trip(self):
        controller = self._controller()
        for _ in range(20):
            step = controller.update(gaps=[0.05], mean_return=10.0)
            assert not step.guard_tripped
        assert controller.lambda_value == 1.0

    def test_a_halved_return_trips_after_the_patience_window(self):
        controller = self._controller()
        for _ in range(4):
            controller.update(gaps=[0.05], mean_return=20.0)
        first = controller.update(gaps=[0.05], mean_return=9.0)
        assert not first.guard_tripped
        second = controller.update(gaps=[0.05], mean_return=9.0)
        assert second.guard_tripped
        assert controller.lambda_value == pytest.approx(0.5)

    def test_the_guard_is_invariant_to_reward_scale(self):
        # The 32 -> 128 iteration collapse halved every arm's reward. An
        # absolute floor cannot survive that; a relative one must.
        small, large = self._controller(), self._controller()
        for value in (2.0, 2.0, 2.0, 2.0, 0.9, 0.9):
            small_step = small.update(gaps=[0.05], mean_return=value)
        for value in (2000.0, 2000.0, 2000.0, 2000.0, 900.0, 900.0):
            large_step = large.update(gaps=[0.05], mean_return=value)
        assert small_step.guard_tripped and large_step.guard_tripped
        assert small.lambda_value == large.lambda_value

    def test_slow_drift_inside_the_window_does_not_trip(self):
        controller = self._controller()
        value = 20.0
        for _ in range(30):
            step = controller.update(gaps=[0.05], mean_return=value)
            assert not step.guard_tripped
            value *= 0.97  # 3% per epoch, far below the 25% deadband

    def test_a_low_absolute_return_alone_does_not_trip(self):
        # The very thing the absolute floor got wrong: a hard environment
        # returns less without the policy having failed.
        controller = self._controller()
        for _ in range(12):
            step = controller.update(gaps=[0.05], mean_return=0.4)
            assert not step.guard_tripped

    def test_the_window_resets_after_a_trip(self):
        controller = self._controller()
        for _ in range(4):
            controller.update(gaps=[0.05], mean_return=20.0)
        controller.update(gaps=[0.05], mean_return=9.0)
        tripped = controller.update(gaps=[0.05], mean_return=9.0)
        assert tripped.guard_tripped
        # Without the reset the old peak of 20 would re-trip immediately.
        follow = controller.update(gaps=[0.05], mean_return=9.0)
        assert not follow.guard_tripped

    def test_the_reference_is_read_before_the_new_sample_joins(self):
        controller = self._controller()
        for _ in range(4):
            controller.update(gaps=[0.05], mean_return=20.0)
        step = controller.update(gaps=[0.05], mean_return=1.0)
        assert step.return_reference == pytest.approx(20.0)

    def test_a_missing_return_is_not_evidence(self):
        controller = self._controller()
        for _ in range(4):
            controller.update(gaps=[0.05], mean_return=20.0)
        for _ in range(3):
            step = controller.update(gaps=[0.05], mean_return=None)
            assert not step.guard_tripped

    def test_absolute_guard_is_untouched_by_default(self):
        controller = LucidDRController(
            PIConfig(delta_target=0.1, return_floor=8.0), initial_lambda=1.0
        )
        assert controller.config.return_guard == "absolute"
        controller.update(gaps=[0.05], mean_return=1.0)
        step = controller.update(gaps=[0.05], mean_return=1.0)
        assert step.guard_tripped

    def test_window_survives_a_resume(self):
        controller = self._controller()
        for value in (20.0, 19.0, 18.0):
            controller.update(gaps=[0.05], mean_return=value)
        restored = self._controller()
        restored.load_state_dict(controller.state_dict())
        assert list(restored.return_window) == list(controller.return_window)
        assert restored.return_reference == controller.return_reference

    def test_config_rejects_nonsense(self):
        with pytest.raises(ValueError):
            PIConfig(return_guard="sometimes")
        with pytest.raises(ValueError):
            PIConfig(return_guard="relative", return_relative_drop=0.0)
        with pytest.raises(ValueError):
            PIConfig(return_guard="relative", return_window=1)


class TestRealizedStratumDose:
    """The simulator must show the mixture, not just the config."""

    class _Buffer:
        def __init__(self, lags):
            self.time_lags = lags

    class _Actuator:
        def __init__(self, lags):
            self.positions_delay_buffer = TestRealizedStratumDose._Buffer(lags)

    class _Asset:
        def __init__(self, lags):
            self.actuators = {"legs": TestRealizedStratumDose._Actuator(lags)}

    def test_per_stratum_means_are_reported_and_ordered(self):
        plan = TACE.assign_cohorts(16, 0.25, seed=4, num_strata=4)
        # Give each env a lag proportional to its stratum's weight, and the
        # anchor cohort the full lag: what a working mixture would produce.
        lags = torch.zeros(16, dtype=torch.long)
        for index, stratum in enumerate(plan.focus_strata):
            lags[list(stratum)] = index + 1
        lags[list(plan.anchor_ids)] = 8
        stats = TACE.cohort_delay_stats(
            self._Asset(lags.unsqueeze(0)), plan.mask(), plan.stratum_masks()
        )
        means = [stats[f"focus_s{i}_delay_mean_steps"] for i in range(4)]
        assert means == sorted(means)
        assert stats["anchor_delay_mean_steps"] == 8.0
        assert stats["focus_delay_mean_steps"] < stats["anchor_delay_mean_steps"]

    def test_unstratified_call_reports_only_the_two_cohorts(self):
        plan = TACE.assign_cohorts(16, 0.25, seed=4)
        stats = TACE.cohort_delay_stats(
            self._Asset(torch.zeros(16, dtype=torch.long).unsqueeze(0)), plan.mask()
        )
        assert not any(key.startswith("focus_s") for key in stats)

    def test_curriculum_telemetry_carries_the_realized_dose(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        asset = self._Asset(torch.arange(16, dtype=torch.long).unsqueeze(0))
        # The scene is reached by subscript, as IsaacLab's is.
        env.scene.__class__.__getitem__ = lambda _self, key: (
            asset if key == "robot" else (_ for _ in ()).throw(KeyError(key))
        )
        callback = LucidCurriculumCallback(
            enabled=True, mode="lucid", branch_id="dose", observer_branch_id="obs",
            initial_lambda=0.8, anchor_ratio=0.25, anchor_seed=4, spread_strata=4,
        )
        OBS._ACTIVE_OBSERVERS["obs"] = StubObserver([0.5] * 8, branch_id="obs")
        callback.on_train_begin(None, None, None, env=env)
        callback.on_step_end(None, State(1, mean_reward=10.0), None, env=env)
        tace_block = callback.history[-1]["tace"]
        assert any(k.startswith("focus_s0") for k in tace_block)


class TestPerChannelCaps:
    """A cap schedules a channel to its own ceiling; an override pins it."""

    def _callback(self, **kwargs):
        return LucidCurriculumCallback(
            enabled=True, mode="lucid", branch_id="cap", observer_branch_id="obs",
            initial_lambda=0.0, **kwargs,
        )

    def test_a_capped_channel_follows_lambda_until_its_ceiling(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(term_lambda_caps={"randomize_rigid_body_mass": 0.5})
        callback.on_train_begin(None, None, None, env=env)
        term = env.event_manager._terms["randomize_rigid_body_mass"]
        other = env.event_manager._terms["push_robot"]

        callback._apply(0.3)
        assert term.params["mass_distribution_params"] == DS.scale_range([0.8, 1.2], 0.3, 1.0)
        callback._apply(0.9)
        assert term.params["mass_distribution_params"] == DS.scale_range([0.8, 1.2], 0.5, 1.0)
        # The uncapped channel is untouched by another channel's ceiling.
        assert other.params["velocity_range"]["x"] == DS.scale_range([-0.5, 0.5], 0.9, 0.0)

    def test_an_override_ignores_lambda_entirely(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(term_lambda_overrides={"randomize_rigid_body_mass": 0.0})
        callback.on_train_begin(None, None, None, env=env)
        term = env.event_manager._terms["randomize_rigid_body_mass"]
        for value in (0.2, 0.9):
            callback._apply(value)
            assert term.params["mass_distribution_params"] == [1.0, 1.0]

    def test_caps_compose_with_strata(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(
            spread_strata=4, anchor_seed=1, term_lambda_caps={"randomize_rigid_body_mass": 0.5}
        )
        callback.on_train_begin(None, None, None, env=env)
        callback._apply(1.0)
        recorded = callback.dispatchers["randomize_rigid_body_mass"].telemetry()["stratum_params"]
        # Capped at 0.5, then scaled by each stratum weight.
        for index, weight in enumerate((0.25, 0.5, 0.75)):
            assert recorded[index]["mass_distribution_params"] == pytest.approx(
                DS.scale_range([0.8, 1.2], 0.5 * weight, 1.0)
            )

    def test_the_realized_channel_intensity_is_logged(self):
        env = FakeEnv(manager(num_envs=16), num_envs=16)
        callback = self._callback(term_lambda_caps={"randomize_rigid_body_mass": 0.4})
        OBS._ACTIVE_OBSERVERS["obs"] = StubObserver([0.5] * 8, branch_id="obs")
        callback.on_train_begin(None, None, None, env=env)
        callback.controller.lambda_value = 0.9
        callback.on_step_end(None, State(1, mean_reward=10.0), None, env=env)
        record = callback.history[-1]
        assert record["term_lambda_caps"] == {"randomize_rigid_body_mass": 0.4}
        assert record["realized_channel_lambdas"]["randomize_rigid_body_mass"] <= 0.4

    def test_a_channel_cannot_be_both_pinned_and_capped(self):
        with pytest.raises(ValueError, match="pick one"):
            self._callback(
                term_lambda_overrides={"push_robot": 0.0},
                term_lambda_caps={"push_robot": 0.5},
            )

    def test_a_cap_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError):
            self._callback(term_lambda_caps={"push_robot": 1.5})
