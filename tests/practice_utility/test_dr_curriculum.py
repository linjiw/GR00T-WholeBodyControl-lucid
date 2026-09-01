"""Tests for LUCID's PI controller and the lambda -> DR-range scaling."""

import json

import pytest

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility.dr_controller import (
    LucidDRController,
    PIConfig,
    calibrate_target,
)


def controller(**overrides):
    params = dict(delta_target=0.10, kp=1.0, ki=0.1, alpha=0.05)
    params.update(overrides)
    return LucidDRController(PIConfig(**params))


class TestPIDirection:
    def test_good_tracking_raises_difficulty(self):
        """Gap below target means the robot is coping: make it harder."""
        c = controller()
        step = c.update(gaps=[0.02] * 20, mean_return=10.0)
        assert step.error > 0 and step.lambda_after > step.lambda_before

    def test_poor_tracking_lowers_difficulty(self):
        c = controller(delta_target=0.10)
        c.lambda_value = 0.5
        step = c.update(gaps=[0.40] * 20, mean_return=10.0)
        assert step.error < 0 and step.lambda_after < step.lambda_before

    def test_gap_at_target_barely_moves_lambda(self):
        c = controller(ki=0.0)
        c.lambda_value = 0.5
        step = c.update(gaps=[0.10] * 20, mean_return=10.0)
        assert step.lambda_after == pytest.approx(0.5)

    def test_uses_a_high_quantile_not_the_mean(self):
        """A calm epoch with a few near-failures must register as difficult."""
        calm = controller().update(gaps=[0.02] * 20, mean_return=10.0)
        spiky = controller().update(gaps=[0.02] * 18 + [0.9, 0.9], mean_return=10.0)
        assert spiky.gap_quantile > calm.gap_quantile
        assert spiky.lambda_after < calm.lambda_after

    def test_lambda_climbs_monotonically_under_sustained_good_tracking(self):
        c = controller()
        values = [c.update(gaps=[0.01] * 10, mean_return=10.0).lambda_after for _ in range(10)]
        assert values == sorted(values)
        assert values[-1] > values[0]

    def test_lambda_is_bounded(self):
        c = controller(alpha=0.5)
        for _ in range(50):
            c.update(gaps=[0.0] * 10, mean_return=10.0)
        assert c.lambda_value <= 1.0
        for _ in range(50):
            c.update(gaps=[2.0] * 10, mean_return=10.0)
        assert c.lambda_value >= 0.0


class TestGuards:
    def test_integral_is_clamped(self):
        c = controller(integral_max=0.5)
        for _ in range(50):
            c.update(gaps=[0.0] * 10, mean_return=10.0)
        assert abs(c.integral) <= 0.5

    def test_alpha_bounds_a_single_epoch_move(self):
        c = controller(alpha=0.01)
        step = c.update(gaps=[0.0] * 10, mean_return=10.0)
        assert abs(step.lambda_after - step.lambda_before) <= 0.01 + 1e-9

    def test_return_guard_needs_two_consecutive_epochs(self):
        c = controller(return_floor=5.0)
        c.lambda_value = 0.8
        first = c.update(gaps=[0.01] * 10, mean_return=1.0)
        assert first.guard_tripped is False
        second = c.update(gaps=[0.01] * 10, mean_return=1.0)
        assert second.guard_tripped is True
        assert second.lambda_after < 0.8

    def test_a_recovering_return_resets_the_streak(self):
        c = controller(return_floor=5.0)
        c.update(gaps=[0.01] * 10, mean_return=1.0)
        c.update(gaps=[0.01] * 10, mean_return=9.0)  # recovered
        step = c.update(gaps=[0.01] * 10, mean_return=1.0)
        assert step.guard_tripped is False

    def test_guard_clears_the_integral(self):
        """Otherwise accumulated integral immediately pushes difficulty back up."""
        c = controller(return_floor=5.0)
        for _ in range(10):
            c.update(gaps=[0.0] * 10, mean_return=10.0)
        assert c.integral > 0
        c.update(gaps=[0.0] * 10, mean_return=1.0)
        c.update(gaps=[0.0] * 10, mean_return=1.0)
        assert c.integral == 0.0

    def test_guard_catches_what_the_gap_misses(self):
        """A healthy-looking gap while the policy is actually failing."""
        c = controller(return_floor=5.0)
        c.lambda_value = 0.9
        c.update(gaps=[0.001] * 10, mean_return=0.5)
        step = c.update(gaps=[0.001] * 10, mean_return=0.5)
        assert step.guard_tripped and step.lambda_after < 0.9

    def test_no_guard_without_a_floor(self):
        c = controller(return_floor=None)
        for _ in range(5):
            step = c.update(gaps=[0.01] * 10, mean_return=-100.0)
        assert step.guard_tripped is False


class TestEmptyEvidence:
    def test_no_samples_holds_lambda_still(self):
        """Moving difficulty on no evidence is the failure this exists to avoid."""
        c = controller()
        c.lambda_value = 0.4
        step = c.update(gaps=[], mean_return=10.0)
        assert step.lambda_after == pytest.approx(0.4)
        assert step.num_gap_samples == 0

    def test_precomputed_quantile_is_honoured(self):
        c = controller()
        step = c.update(gap_quantile=0.02, mean_return=10.0)
        assert step.gap_quantile == pytest.approx(0.02)
        assert step.lambda_after > 0


class TestPersistence:
    def test_state_round_trips(self):
        c = controller()
        for _ in range(5):
            c.update(gaps=[0.01] * 10, mean_return=10.0)
        restored = controller()
        restored.load_state_dict(c.state_dict())
        assert restored.lambda_value == c.lambda_value
        assert restored.integral == c.integral
        assert restored.epoch == c.epoch

    def test_history_records_every_epoch(self):
        c = controller()
        for _ in range(4):
            c.update(gaps=[0.01] * 5, mean_return=1.0)
        assert len(c.history) == 4 and c.history[-1].epoch == 4


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"quantile": 0.0},
            {"quantile": 1.0},
            {"alpha": 0.0},
            {"integral_max": 0.0},
            {"return_decay": 0.0},
            {"low_return_patience": 0},
            {"lambda_min": 0.5, "lambda_max": 0.5},
        ],
    )
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            PIConfig(**kwargs)

    def test_rejects_out_of_range_initial_lambda(self):
        with pytest.raises(ValueError, match="outside"):
            LucidDRController(PIConfig(), initial_lambda=1.5)


class TestCalibration:
    def test_target_exceeds_the_nominal_mean(self):
        target = calibrate_target([0.05, 0.06, 0.04, 0.05], num_sigma=3.0)
        assert target > 0.05

    def test_more_sigma_gives_a_looser_target(self):
        gaps = [0.05, 0.06, 0.04, 0.05]
        assert calibrate_target(gaps, 3.0) > calibrate_target(gaps, 1.0)

    def test_needs_enough_samples(self):
        with pytest.raises(ValueError, match="at least two"):
            calibrate_target([0.05])


class TestRangeScaling:
    def test_lambda_zero_collapses_to_nominal(self):
        assert DS.scale_range([0.8, 1.2], 0.0, 1.0) == [1.0, 1.0]

    def test_lambda_one_restores_the_baseline_exactly(self):
        assert DS.scale_range([0.8, 1.2], 1.0, 1.0) == pytest.approx([0.8, 1.2])

    def test_half_lambda_halves_the_deviation(self):
        assert DS.scale_range([0.8, 1.2], 0.5, 1.0) == pytest.approx([0.9, 1.1])

    def test_asymmetric_range_scales_each_side_about_the_nominal(self):
        assert DS.scale_range([0.5, 2.0], 0.5, 1.0) == pytest.approx([0.75, 1.5])

    def test_additive_channel_uses_zero_nominal(self):
        assert DS.scale_range([-0.5, 0.5], 0.5, 0.0) == pytest.approx([-0.25, 0.25])

    def test_midpoint_nominal_when_unspecified(self):
        assert DS.scale_range([0.3, 1.6], 0.0, None) == pytest.approx([0.95, 0.95])

    @pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0]])
    def test_rejects_malformed_range(self, bad):
        with pytest.raises(ValueError, match="lo, hi"):
            DS.scale_range(bad, 0.5, 0.0)

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="inverted"):
            DS.scale_range([1.0, 0.0], 0.5, 0.0)

    @pytest.mark.parametrize("lam", [-0.1, 1.1])
    def test_rejects_lambda_out_of_range(self, lam):
        with pytest.raises(ValueError, match="lambda"):
            DS.scale_range([0.0, 1.0], lam, 0.0)

    def test_nested_dict_ranges_scale(self):
        scaled = DS.scale_params({"x": [-0.02, 0.02], "y": [-0.05, 0.05]}, 0.5, 0.0)
        assert scaled["x"] == pytest.approx([-0.01, 0.01])
        assert scaled["y"] == pytest.approx([-0.025, 0.025])

    def test_non_range_values_pass_through(self):
        assert DS.scale_params("scale", 0.5, 0.0) == "scale"
        assert DS.scale_params(64, 0.5, 0.0) == 64


class Term:
    def __init__(self, mode, params):
        self.mode = mode
        self.params = params


class Manager:
    def __init__(self, terms):
        self._terms = terms
        self.active_terms = list(terms)
        self._term_cfgs = list(terms.values())


def manager():
    return Manager(
        {
            "randomize_rigid_body_mass": Term(
                "reset", {"mass_distribution_params": [0.8, 1.2], "operation": "scale"}
            ),
            "push_robot": Term("interval", {"velocity_range": {"x": [-0.5, 0.5]}}),
            "physics_material": Term(
                "startup", {"static_friction_range": [0.3, 1.6], "num_buckets": 64}
            ),
        }
    )


class MaterialTerm:
    """Mimics IsaacLab's class-based material term: buckets fixed at __init__.

    IsaacLab stores the constructed instance back on ``cfg.func``
    (manager_base.py:418), which is where the live buckets live.
    """

    def __init__(self, n=32):
        import torch

        self.material_buckets = torch.stack(
            [
                torch.empty(n).uniform_(0.3, 1.6),
                torch.empty(n).uniform_(0.3, 1.2),
                torch.empty(n).uniform_(0.0, 0.5),
            ],
            dim=1,
        )


class MaterialCfg(Term):
    def __init__(self, mode="reset"):
        super().__init__(
            mode,
            {
                "static_friction_range": [0.3, 1.6],
                "dynamic_friction_range": [0.3, 1.2],
                "restitution_range": [0.0, 0.5],
                "num_buckets": 32,
            },
        )
        self.func = MaterialTerm()


class TestMaterialBucketsFollowLambda:
    """Scaling the range parameters alone changes nothing -- buckets must move."""

    def manager_with_material(self):
        return Manager({"physics_material": MaterialCfg(mode="reset")})

    def test_buckets_are_resampled_when_scaled(self):
        m = self.manager_with_material()
        term = m._terms["physics_material"].func
        before = term.material_buckets.clone()
        report = DS.apply_lambda(m, DS.capture_baseline(m), 0.2)
        assert "physics_material" in report.material_terms_resampled
        import torch

        assert not torch.allclose(before, term.material_buckets)

    def test_resampled_buckets_respect_the_scaled_range(self):
        m = self.manager_with_material()
        DS.apply_lambda(m, DS.capture_baseline(m), 0.0)
        term = m._terms["physics_material"].func
        static = term.material_buckets[:, 0]
        # lambda = 0 collapses friction to its nominal (the midpoint, 0.95)
        assert float(static.min()) == pytest.approx(0.95, abs=1e-5)
        assert float(static.max()) == pytest.approx(0.95, abs=1e-5)

    def test_full_lambda_restores_the_configured_span(self):
        m = self.manager_with_material()
        baseline = DS.capture_baseline(m)
        DS.apply_lambda(m, baseline, 0.0)
        DS.apply_lambda(m, baseline, 1.0)
        static = m._terms["physics_material"].func.material_buckets[:, 0]
        assert float(static.min()) >= 0.3 - 1e-6
        assert float(static.max()) <= 1.6 + 1e-6
        assert float(static.max()) - float(static.min()) > 0.5

    def test_bucket_count_never_grows(self):
        """PhysX caps unique materials in the scene at 64000."""
        m = self.manager_with_material()
        baseline = DS.capture_baseline(m)
        for lam in (0.1, 0.5, 0.9):
            DS.apply_lambda(m, baseline, lam)
        assert m._terms["physics_material"].func.material_buckets.shape == (32, 3)

    def test_startup_material_is_not_resampled(self):
        m = Manager({"physics_material": MaterialCfg(mode="startup")})
        report = DS.apply_lambda(m, DS.capture_baseline(m), 0.5)
        assert report.material_terms_resampled == []


class TestApplyLambda:
    def test_reports_only_runtime_scalable_terms(self):
        assert DS.scalable_terms(manager()) == ["push_robot", "randomize_rigid_body_mass"]

    def test_startup_terms_are_skipped_and_named(self):
        m = manager()
        report = DS.apply_lambda(m, DS.capture_baseline(m), 0.5)
        assert "physics_material" in report.skipped_startup_terms
        assert m._terms["physics_material"].params["static_friction_range"] == [0.3, 1.6]

    def test_reset_and_interval_terms_are_scaled(self):
        m = manager()
        DS.apply_lambda(m, DS.capture_baseline(m), 0.5)
        assert m._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.9, 1.1])
        assert m._terms["push_robot"].params["velocity_range"]["x"] == pytest.approx([-0.25, 0.25])

    def test_scaling_is_computed_from_the_baseline_not_the_current_value(self):
        """Compounding would drive every range to zero after a few epochs."""
        m = manager()
        baseline = DS.capture_baseline(m)
        for _ in range(5):
            DS.apply_lambda(m, baseline, 0.5)
        assert m._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.9, 1.1])

    def test_lambda_one_restores_the_configured_maximum(self):
        m = manager()
        baseline = DS.capture_baseline(m)
        DS.apply_lambda(m, baseline, 0.0)
        DS.apply_lambda(m, baseline, 1.0)
        assert m._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.8, 1.2])

    def test_lambda_zero_disables_randomization(self):
        m = manager()
        DS.apply_lambda(m, DS.capture_baseline(m), 0.0)
        assert m._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([1.0, 1.0])
        assert m._terms["push_robot"].params["velocity_range"]["x"] == pytest.approx([0.0, 0.0])

    def test_non_range_params_are_untouched(self):
        m = manager()
        DS.apply_lambda(m, DS.capture_baseline(m), 0.3)
        assert m._terms["randomize_rigid_body_mass"].params["operation"] == "scale"

    def test_evaluation_can_scale_physics_without_scaling_latency(self):
        m = manager()
        m._terms["randomize_action_delay"] = Term("reset", {"delay_range": [0.0, 12.0]})
        m.active_terms.append("randomize_action_delay")
        m._term_cfgs.append(m._terms["randomize_action_delay"])
        baseline = DS.capture_baseline(m)
        DS.apply_lambda(m, baseline, 0.5, exclude_terms=("randomize_action_delay",))
        assert m._terms["randomize_action_delay"].params["delay_range"] == [0.0, 12.0]
        assert m._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.9, 1.1])

    def test_baseline_capture_is_a_deep_copy(self):
        m = manager()
        baseline = DS.capture_baseline(m)
        DS.apply_lambda(m, baseline, 0.1)
        assert baseline["randomize_rigid_body_mass"]["mass_distribution_params"] == [0.8, 1.2]

    def test_report_serializes(self):
        m = manager()
        payload = DS.apply_lambda(m, DS.capture_baseline(m), 0.5).to_dict()
        assert payload["num_scaled"] == 2 and payload["lambda_value"] == 0.5

    def test_missing_manager_is_safe(self):
        assert DS.scalable_terms(None) == []
        assert DS.capture_baseline(None) == {}


# ---------------------------------------------------------------------------
# The callback that ties controller, scaling, and the live gap together.
# ---------------------------------------------------------------------------

from gear_sonic.research.practice_utility import observer as OBS  # noqa: E402
from gear_sonic.research.practice_utility.dr_curriculum import (  # noqa: E402
    LucidCurriculumCallback,
    _event_manager_of,
)


class FakeEnvWithEvents:
    def __init__(self):
        self.event_manager = manager()


class NestedEnvWithEvents:
    def __init__(self):
        self.env = FakeEnvWithEvents()


class FakeStateWithHistory:
    def __init__(self, step, mean_reward=None):
        self.global_step = step
        self.log_history = [{"mean_reward": mean_reward}] if mean_reward is not None else []


class StubObserver:
    def __init__(self, gaps, branch_id="b0"):
        self._gaps = list(gaps)
        self.branch_id = branch_id

    def drain_gaps(self):
        return list(self._gaps)


@pytest.fixture(autouse=True)
def _clean_registry():
    OBS.clear_observers()
    yield
    OBS.clear_observers()


def curriculum(**overrides):
    params = dict(enabled=True, mode="lucid", delta_target=0.10, alpha=0.05, branch_id="b0")
    params.update(overrides)
    return LucidCurriculumCallback(**params)


class TestCurriculumWiring:
    def test_disabled_touches_nothing(self):
        env = FakeEnvWithEvents()
        before = list(
            env.event_manager._terms["randomize_rigid_body_mass"].params["mass_distribution_params"]
        )
        callback = curriculum(enabled=False)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1), None, env=env)
        assert (
            env.event_manager._terms["randomize_rigid_body_mass"].params["mass_distribution_params"]
            == before
        )
        assert callback.history == []

    def test_binds_and_reports_scalable_terms(self):
        env = FakeEnvWithEvents()
        callback = curriculum()
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        assert callback.scalable == ["push_robot", "randomize_rigid_body_mass"]

    def test_finds_the_event_manager_through_a_wrapper(self):
        assert _event_manager_of(NestedEnvWithEvents()) is not None

    def test_missing_event_manager_is_survivable(self):
        callback = curriculum()
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=object())
        callback.on_step_end(None, FakeStateWithHistory(1), None, env=object())
        assert callback.history  # still records, just cannot scale

    def test_initial_lambda_is_applied_before_the_first_rollout(self):
        env = FakeEnvWithEvents()
        curriculum(initial_lambda=0.0).on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        assert env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([1.0, 1.0])

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown curriculum mode"):
            LucidCurriculumCallback(enabled=True, mode="adaptive")


class TestCurriculumModes:
    @pytest.mark.parametrize(
        ("mode", "fixed_lambda", "expected_range", "expected_lambda"),
        [
            ("lucid", 1.0, [1.0, 1.0], 0.0),
            ("fixed", 0.5, [0.9, 1.1], 0.5),
            ("off", 1.0, [1.0, 1.0], 0.0),
        ],
    )
    def test_first_rollout_uses_the_mode_intensity(
        self, mode, fixed_lambda, expected_range, expected_lambda
    ):
        env = FakeEnvWithEvents()
        callback = curriculum(mode=mode, fixed_lambda=fixed_lambda, warmup_iterations=3)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        assert callback.controller.lambda_value == pytest.approx(expected_lambda)
        assert env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx(expected_range)

    def test_lucid_mode_raises_lambda_on_good_tracking(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.01] * 20))
        callback = curriculum()
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        assert callback.history[-1]["lambda"] > 0.0

    def test_lucid_mode_actually_widens_the_dr_range(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.01] * 20))
        callback = curriculum(alpha=0.5)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        low, high = env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ]
        assert low < 1.0 < high

    def test_fixed_mode_pins_lambda(self):
        env = FakeEnvWithEvents()
        callback = curriculum(mode="fixed", fixed_lambda=1.0)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        for step in (1, 2, 3):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=1.0), None, env=env)
        assert {r["lambda"] for r in callback.history} == {1.0}
        assert env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.8, 1.2])

    def test_off_mode_disables_randomization(self):
        env = FakeEnvWithEvents()
        callback = curriculum(mode="off")
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1), None, env=env)
        assert env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([1.0, 1.0])

    def test_lucid_without_an_observer_holds_lambda_still(self):
        env = FakeEnvWithEvents()
        callback = curriculum()
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        assert callback.history[-1]["lambda"] == pytest.approx(0.0)
        assert callback.history[-1]["num_gap_samples"] == 0

    def test_update_every_throttles(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.01] * 20))
        callback = curriculum(update_every=3)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        for step in range(1, 7):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=10.0), None, env=env)
        assert len(callback.history) == 2  # steps 3 and 6


class TestCurriculumReporting:
    def test_reads_sonics_native_reward_log_key(self):
        state = FakeStateWithHistory(1)
        state.log_history = [{"objective/rewards": 7.25}]
        assert LucidCurriculumCallback._mean_return(state) == pytest.approx(7.25)

    def test_records_the_gap_that_drove_the_decision(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.05] * 20))
        callback = curriculum()
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        record = callback.history[-1]
        assert record["gap_quantile"] == pytest.approx(0.05)
        assert record["num_gap_samples"] == 20

    def test_writes_jsonl(self, tmp_path):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.01] * 10))
        callback = curriculum(output_dir=str(tmp_path))
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        import json as _json

        record = _json.loads((tmp_path / "curriculum_b0.jsonl").read_text().strip())
        assert record["mode"] == "lucid" and "lambda" in record

    def test_return_guard_is_reachable_end_to_end(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.001] * 20))
        callback = curriculum(alpha=0.5, return_floor=5.0)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        for step in (1, 2, 3):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=10.0), None, env=env)
        high = callback.history[-1]["lambda"]
        for step in (4, 5):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=0.1), None, env=env)
        assert callback.history[-1]["guard_tripped"] is True
        assert callback.history[-1]["lambda"] < high


class TestObserverRegistry:
    def test_single_observer_is_found_without_a_name(self):
        OBS.register_observer(StubObserver([0.1], branch_id="only"))
        assert OBS.get_active_observer() is not None

    def test_ambiguous_registry_requires_a_name(self):
        OBS.register_observer(StubObserver([0.1], branch_id="a"))
        OBS.register_observer(StubObserver([0.1], branch_id="b"))
        assert OBS.get_active_observer() is None
        assert OBS.get_active_observer("a") is not None

    def test_unknown_name_returns_none(self):
        assert OBS.get_active_observer("nope") is None


class TestCurriculumSurvivesResume:
    """A resumed run must not silently restart the curriculum at lambda = 0."""

    def trained(self, tmp_path, steps=6):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.01] * 20))
        callback = curriculum(alpha=0.2, output_dir=str(tmp_path))
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        for step in range(1, steps + 1):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=10.0), None, env=env)
        return callback, env

    def test_lambda_is_restored(self, tmp_path):
        first, _ = self.trained(tmp_path)
        assert first.controller.lambda_value > 0.1

        resumed = curriculum(
            resume_state_path=str(tmp_path / f"curriculum_state_{first.branch_id}.json")
        )
        resumed.on_train_begin(None, FakeStateWithHistory(0), None, env=FakeEnvWithEvents())
        assert resumed.controller.lambda_value == pytest.approx(first.controller.lambda_value)

    def test_integral_is_restored(self, tmp_path):
        first, _ = self.trained(tmp_path)
        resumed = curriculum(
            resume_state_path=str(tmp_path / f"curriculum_state_{first.branch_id}.json")
        )
        resumed.on_train_begin(None, FakeStateWithHistory(0), None, env=FakeEnvWithEvents())
        assert resumed.controller.integral == pytest.approx(first.controller.integral)

    def test_without_resume_the_curriculum_restarts_at_zero(self, tmp_path):
        """The defect this guards: nothing errors, the curve just looks worse."""
        first, _ = self.trained(tmp_path)
        naive = curriculum()
        naive.on_train_begin(None, FakeStateWithHistory(0), None, env=FakeEnvWithEvents())
        assert naive.controller.lambda_value == 0.0
        assert first.controller.lambda_value > naive.controller.lambda_value

    def test_restored_intensity_is_applied_before_the_first_rollout(self, tmp_path):
        first, _ = self.trained(tmp_path)
        env = FakeEnvWithEvents()
        resumed = curriculum(
            resume_state_path=str(tmp_path / f"curriculum_state_{first.branch_id}.json")
        )
        resumed.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        low, high = env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ]
        assert low < 1.0 < high  # DR already widened, not reset to nominal

    def test_resume_records_where_it_came_from(self, tmp_path):
        first, _ = self.trained(tmp_path)
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.01] * 20))
        resumed = curriculum(
            resume_state_path=str(tmp_path / f"curriculum_state_{first.branch_id}.json")
        )
        resumed.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        resumed.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        assert "resumed_from" in resumed.history[-1]

    def test_a_missing_state_file_is_survivable(self, tmp_path):
        callback = curriculum(resume_state_path=str(tmp_path / "absent.json"))
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=FakeEnvWithEvents())
        assert callback.controller.lambda_value == 0.0

    def test_resume_jump_can_be_rate_limited(self, tmp_path):
        first, _ = self.trained(tmp_path, steps=10)
        resumed = curriculum(
            resume_state_path=str(tmp_path / f"curriculum_state_{first.branch_id}.json"),
            max_lambda_step_on_resume=0.02,
        )
        resumed.on_train_begin(None, FakeStateWithHistory(0), None, env=FakeEnvWithEvents())
        assert resumed.controller.lambda_value <= 0.02 + 1e-9
        assert resumed.controller.lambda_value < first.controller.lambda_value

    def test_state_file_is_written_every_update(self, tmp_path):
        self.trained(tmp_path, steps=3)
        assert (tmp_path / "curriculum_state_b0.json").exists()


class TestWarmupHold:
    def test_lambda_is_held_during_warmup(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.001] * 20))
        callback = curriculum(alpha=0.5, warmup_iterations=3)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        for step in (1, 2, 3):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=10.0), None, env=env)
        assert all(r.get("warmup_hold") for r in callback.history)
        assert callback.controller.lambda_value == 0.0

    def test_warmup_records_are_written(self, tmp_path):
        env = FakeEnvWithEvents()
        callback = curriculum(output_dir=str(tmp_path), warmup_iterations=2)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1), None, env=env)
        record = json.loads((tmp_path / "curriculum_b0.jsonl").read_text())
        assert record["warmup_hold"] is True
        assert record["lambda"] == 0.0
        assert (tmp_path / "curriculum_state_b0.json").exists()

    def test_updates_resume_after_warmup(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.001] * 20))
        callback = curriculum(alpha=0.5, warmup_iterations=2)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        for step in range(1, 6):
            callback.on_step_end(None, FakeStateWithHistory(step, mean_reward=10.0), None, env=env)
        assert callback.controller.lambda_value > 0.0

    def test_warmup_still_applies_the_restored_intensity(self):
        env = FakeEnvWithEvents()
        callback = curriculum(warmup_iterations=5, initial_lambda=0.6)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        low, high = env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ]
        assert low == pytest.approx(1.0 - 0.6 * 0.2)

    def test_zero_warmup_updates_immediately(self):
        env = FakeEnvWithEvents()
        OBS.register_observer(StubObserver([0.001] * 20))
        callback = curriculum(alpha=0.5, warmup_iterations=0)
        callback.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        callback.on_step_end(None, FakeStateWithHistory(1, mean_reward=10.0), None, env=env)
        assert callback.controller.lambda_value > 0.0


class TestSupportExtension:
    """fixed_150: training past lambda = 1 is explicit, fixed-mode-only, and clamped."""

    def test_scaled_term_params_refuses_extrapolation_without_the_flag(self):
        with pytest.raises(ValueError, match="lambda must be in"):
            DS.scaled_term_params({"mass_distribution_params": [0.8, 1.2]}, 1.5)

    def test_scaled_term_params_extends_about_the_nominal_with_the_flag(self):
        params = DS.scaled_term_params(
            {"mass_distribution_params": [0.8, 1.2]}, 1.5, allow_extrapolation=True
        )
        assert params["mass_distribution_params"] == pytest.approx([0.7, 1.3])

    def test_controller_modes_may_not_extrapolate(self):
        # The bidirectional gap-driven controller stays hard-capped at 1: a
        # scheduler that can both lower difficulty and leave the envelope turns
        # a support experiment into an uncontrolled one.
        with pytest.raises(ValueError, match="hard-capped"):
            LucidCurriculumCallback(enabled=True, mode="lucid", allow_extrapolation=True)

    def test_monotone_expansion_modes_may_extrapolate(self):
        # gate and ramp are admitted because neither can lower applied support
        # by its own decision rule.
        for mode in ("gate", "ramp"):
            cur = LucidCurriculumCallback(
                enabled=True,
                mode=mode,
                allow_extrapolation=True,
                spread_strata=8,
                gate_lambda_max=1.5,
                ramp_end_lambda=1.5,
            )
            assert cur.allow_extrapolation is True

    def test_expansion_modes_past_one_require_the_flag(self):
        with pytest.raises(ValueError, match="allow_extrapolation"):
            LucidCurriculumCallback(enabled=True, mode="gate", spread_strata=8, gate_lambda_max=1.5)
        with pytest.raises(ValueError, match="allow_extrapolation"):
            LucidCurriculumCallback(enabled=True, mode="ramp", spread_strata=8, ramp_end_lambda=1.5)

    def test_fixed_lambda_past_one_requires_the_flag(self):
        with pytest.raises(ValueError, match="allow_extrapolation"):
            LucidCurriculumCallback(enabled=True, mode="fixed", fixed_lambda=1.5)

    def test_fixed_lambda_is_bounded_by_the_extrapolation_ceiling(self):
        with pytest.raises(ValueError, match="must be in"):
            LucidCurriculumCallback(
                enabled=True, mode="fixed", fixed_lambda=5.0, allow_extrapolation=True
            )

    def test_apply_at_150_extends_and_physically_clamps_the_live_config(self):
        cur = LucidCurriculumCallback(
            enabled=True, mode="fixed", fixed_lambda=1.5, allow_extrapolation=True
        )
        m = Manager(
            {
                "physics_material": MaterialCfg(mode="reset"),
                "randomize_rigid_body_mass": Term(
                    "reset", {"mass_distribution_params": [0.8, 1.2], "operation": "scale"}
                ),
            }
        )
        cur._event_manager = m
        cur.baseline = DS.capture_baseline(m)
        cur._apply(1.5)
        mass = m._terms["randomize_rigid_body_mass"].params["mass_distribution_params"]
        assert mass == pytest.approx([0.7, 1.3])
        # friction [0.3, 1.6] about its midpoint 0.95 extends to [-0.025, 1.925];
        # the clamp must lift the low edge to physical validity.
        static = m._terms["physics_material"].params["static_friction_range"]
        assert static[0] == pytest.approx(0.05) and static[1] == pytest.approx(1.925)
        assert cur._clamp_report is not None
        assert "physics_material" in cur._clamp_report["clamped"]
        buckets = m._terms["physics_material"].func.material_buckets
        assert float(buckets[:, 0].min()) >= 0.049

    def test_fixed_150_is_applied_on_the_first_rollout_and_through_warmup(self):
        env = FakeEnvWithEvents()
        cur = LucidCurriculumCallback(
            enabled=True,
            mode="fixed",
            fixed_lambda=1.5,
            allow_extrapolation=True,
            warmup_iterations=3,
        )
        cur.on_train_begin(None, FakeStateWithHistory(0), None, env=env)
        mass = env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ]
        assert mass == pytest.approx([0.7, 1.3])

        cur.on_step_end(None, FakeStateWithHistory(1), None, env=env)
        assert cur.history[-1]["warmup_hold"] is True
        assert cur.history[-1]["lambda"] == pytest.approx(1.5)
        assert env.event_manager._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.7, 1.3])

    def test_apply_at_one_is_unchanged_with_the_flag_off(self):
        cur = LucidCurriculumCallback(enabled=True, mode="fixed", fixed_lambda=1.0)
        m = manager()
        cur._event_manager = m
        cur.baseline = DS.capture_baseline(m)
        cur._apply(1.0)
        assert m._terms["randomize_rigid_body_mass"].params[
            "mass_distribution_params"
        ] == pytest.approx([0.8, 1.2])
        assert cur._clamp_report is None


class TestClampParamsPhysical:
    def test_friction_floor_and_report(self):
        out, report = DS.clamp_params_physical({"static_friction_range": [-0.025, 1.925]})
        assert out["static_friction_range"] == pytest.approx([0.05, 1.925])
        assert report["static_friction_range"]["from"] == pytest.approx([-0.025, 1.925])

    def test_dynamic_kept_at_or_below_static(self):
        out, report = DS.clamp_params_physical(
            {"static_friction_range": [0.1, 1.0], "dynamic_friction_range": [0.2, 1.4]}
        )
        assert out["dynamic_friction_range"] == pytest.approx([0.2, 1.0])
        assert "dynamic_friction_range" in report

    def test_untouched_params_pass_through(self):
        out, report = DS.clamp_params_physical(
            {"velocity_range": {"x": [-0.75, 0.75]}, "mass_distribution_params": [0.7, 1.3]}
        )
        assert report == {}
        assert out["mass_distribution_params"] == pytest.approx([0.7, 1.3])
