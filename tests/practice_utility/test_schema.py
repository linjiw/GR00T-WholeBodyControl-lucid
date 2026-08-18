"""Contract tests for practice-utility data schemas.

These guard the invariants that make a utility label trustworthy: stable
context identity, honest dose accounting, and the refusal to emit a label when
the intervention did not actually happen.
"""

import dataclasses

import pytest

from gear_sonic.research.practice_utility import schema as S


def make_context(**overrides):
    base = dict(
        motion_key="walk_forward_001",
        motion_hash=S.motion_hash("walk_forward_001", 300, 50.0),
        bin_index=3,
        bin_start_frame=150,
        bin_end_frame=200,
    )
    base.update(overrides)
    return S.ContextKey(**base)


def make_dose(role, **overrides):
    base = dict(
        branch_id=f"pair0_{role}",
        context_id="ctx0",
        role=role,
        completed_env_steps=10_000.0,
        completed_kernel_steps=1_000.0,
    )
    base.update(overrides)
    return S.DoseReport(**base)


class TestContextKey:
    def test_identity_is_stable_across_instances(self):
        assert make_context().context_id == make_context().context_id

    def test_identity_depends_on_motion_hash_not_just_key(self):
        other = make_context(motion_hash=S.motion_hash("walk_forward_001", 301, 50.0))
        assert make_context().context_id != other.context_id

    def test_identity_separates_perturbation_and_severity(self):
        native = make_context()
        perturbed = make_context(perturbation_group="latency", severity_level=2)
        assert native.context_id != perturbed.context_id

    def test_roundtrip_through_dict(self):
        ctx = make_context(perturbation_group="material", severity_level=1)
        assert S.ContextKey.from_dict(ctx.to_dict()) == ctx

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_context().bin_index = 7

    @pytest.mark.parametrize(
        "overrides",
        [
            {"bin_start_frame": 200, "bin_end_frame": 200},
            {"bin_start_frame": 200, "bin_end_frame": 150},
            {"bin_index": -1},
            {"severity_level": -1},
        ],
    )
    def test_rejects_malformed(self, overrides):
        with pytest.raises(ValueError):
            make_context(**overrides)


class TestMotionHash:
    def test_distinguishes_resampled_clip(self):
        assert S.motion_hash("m", 300, 50.0) != S.motion_hash("m", 300, 30.0)

    def test_distinguishes_renamed_clip(self):
        assert S.motion_hash("a", 300, 50.0) != S.motion_hash("b", 300, 50.0)


class TestMotionPoolManifest:
    def test_rejects_motion_without_hash(self):
        with pytest.raises(ValueError, match="lack hashes"):
            S.MotionPoolManifest(
                manifest_id="p", motion_keys=["a", "b"],
                motion_hashes={"a": "h"}, source_root="/tmp",
            )

    def test_rejects_duplicate_keys(self):
        with pytest.raises(ValueError, match="duplicates"):
            S.MotionPoolManifest(
                manifest_id="p", motion_keys=["a", "a"],
                motion_hashes={"a": "h"}, source_root="/tmp",
            )

    def test_sha_is_order_independent(self):
        hashes = {"a": "h1", "b": "h2"}
        first = S.MotionPoolManifest("p", ["a", "b"], hashes, "/tmp")
        second = S.MotionPoolManifest("p", ["b", "a"], hashes, "/tmp")
        assert first.manifest_sha256 == second.manifest_sha256


class TestSamplingSnapshot:
    def make(self, prob):
        return S.SamplingSnapshot(
            global_step=10, num_bins=len(prob), active_bin_ids=list(range(len(prob))),
            active_prob=prob, failure_rate_raw=[0.1] * len(prob),
            num_episodes=[1.0] * len(prob), num_failures=[0.1] * len(prob),
            uniform_sampling_rate=0.1, failure_rate_max_over_mean=200.0, manifest_id="p",
        )

    def test_rejects_unnormalized(self):
        with pytest.raises(ValueError, match="sums to"):
            self.make([0.5, 0.4])

    def test_rejects_negative_probability(self):
        with pytest.raises(ValueError, match="negative"):
            self.make([1.2, -0.2])

    def test_effective_num_bins_is_uniform_count_for_uniform(self):
        assert self.make([0.25] * 4).effective_num_bins == pytest.approx(4.0)

    def test_effective_num_bins_drops_under_concentration(self):
        assert self.make([0.97, 0.01, 0.01, 0.01]).effective_num_bins < 1.2


class TestDoseReport:
    def test_kernel_steps_cannot_exceed_total_steps(self):
        with pytest.raises(ValueError, match="exceeds"):
            make_dose("control", completed_env_steps=100.0, completed_kernel_steps=101.0)

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            make_dose("control", drawn_episodes=-1.0)

    def test_kernel_fraction(self):
        assert make_dose("control").kernel_step_fraction == pytest.approx(0.1)

    def test_kernel_fraction_is_zero_when_nothing_ran(self):
        dose = make_dose("control", completed_env_steps=0.0, completed_kernel_steps=0.0)
        assert dose.kernel_step_fraction == 0.0


class TestHarmVector:
    def make(self, **overrides):
        base = dict(
            clean_delta=0.0, action_rate_delta=0.0, slip_delta=0.0,
            contact_impulse_delta=0.0, torque_saturation_delta=0.0,
        )
        base.update(overrides)
        return S.HarmVector(**base)

    GATES = {
        "action_rate": 0.05, "slip": 0.05, "contact_impulse": 0.05,
        "torque_saturation": 0.05, "clean_noninferiority": 0.02,
    }

    def test_clean_passes_when_no_harm(self):
        assert self.make().exceeds(self.GATES) == []

    def test_flags_each_channel(self):
        assert self.make(slip_delta=0.5).exceeds(self.GATES) == ["slip"]

    def test_improvement_is_not_harm(self):
        assert self.make(slip_delta=-0.5).exceeds(self.GATES) == []

    def test_clean_regression_flagged(self):
        assert "clean_noninferiority" in self.make(clean_delta=-0.5).exceeds(self.GATES)


class TestUtilityRecord:
    def make(self, control_kernel=1_000.0, intervention_kernel=1_500.0, **overrides):
        base = dict(
            branch_pair_id="pair0",
            context=make_context(),
            policy_stage="middle",
            seed=0,
            horizons={"H_s": 8, "H_m": 32, "H_l": 128},
            base_distribution_sha256="a" * 64,
            intervention_distribution_sha256="b" * 64,
            epsilon=0.10,
            kernel_radius_bins=1,
            control_dose=make_dose("control", completed_kernel_steps=control_kernel),
            intervention_dose=make_dose("intervention", completed_kernel_steps=intervention_kernel),
            efficacy_delta={"H_s": 0.02, "H_m": 0.05, "H_l": -0.01},
        )
        base.update(overrides)
        return S.UtilityRecord(**base)

    def test_roles_must_match_slots(self):
        with pytest.raises(ValueError, match="role='control'"):
            self.make(control_dose=make_dose("intervention"))
        with pytest.raises(ValueError, match="role='intervention'"):
            self.make(intervention_dose=make_dose("control"))

    def test_epsilon_range_enforced(self):
        with pytest.raises(ValueError, match="epsilon"):
            self.make(epsilon=1.5)

    def test_realized_extra_dose_is_a_difference_not_the_nominal_plan(self):
        assert self.make().realized_extra_dose == pytest.approx(500.0)

    def test_utility_normalizes_by_realized_dose(self):
        assert self.make().utility_at("H_m") == pytest.approx(0.05 / (500.0 + 1e-6))

    def test_utility_sign_is_preserved(self):
        assert self.make().utility_at("H_l") < 0

    def test_refuses_label_when_dose_was_not_delivered(self):
        record = self.make(intervention_kernel=1_000.0)
        with pytest.raises(ValueError, match="non-positive realized extra dose"):
            record.utility_at("H_m")

    def test_refuses_label_when_intervention_got_less_than_control(self):
        record = self.make(intervention_kernel=900.0)
        with pytest.raises(ValueError, match="non-positive realized extra dose"):
            record.utility_at("H_m")

    def test_unknown_horizon_raises(self):
        with pytest.raises(KeyError):
            self.make().utility_at("H_xl")


class TestHashing:
    def test_canonical_json_is_key_order_independent(self):
        assert S.sha256_of({"a": 1, "b": 2}) == S.sha256_of({"b": 2, "a": 1})

    def test_hash_is_value_sensitive(self):
        assert S.sha256_of({"a": 1}) != S.sha256_of({"a": 2})
