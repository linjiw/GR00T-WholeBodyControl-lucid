"""Tests for utility-label construction and the Gate A identifiability check."""

import pytest

from gear_sonic.research.practice_utility import utility_label as U
from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    DoseReport,
    HarmVector,
    UtilityRecord,
    motion_hash,
)

HORIZONS = {"H_s": 8, "H_m": 32, "H_l": 128}


def context(index=0):
    key = f"walk_{index:03d}__A{index:03d}"
    return ContextKey(
        motion_key=key, motion_hash=motion_hash(key, 300, 50.0),
        bin_index=1, bin_start_frame=50, bin_end_frame=100,
    )


def dose(role, kernel_steps):
    return DoseReport(
        branch_id=f"b_{role}", context_id="ctx", role=role,
        completed_env_steps=10_000.0, completed_kernel_steps=kernel_steps,
    )


def evaluation(role, horizon, j_eff, **overrides):
    base = dict(
        branch_id=f"b_{role}", role=role, horizon_label=horizon, j_eff=j_eff,
        clean_j_eff=0.80, action_rate=0.010, foot_slip=0.010,
        contact_impulse=50.0, torque_saturation=0.010,
    )
    base.update(overrides)
    return U.BranchEvaluation(**base)


def record(
    control_j=0.50, treated_j=0.55, control_kernel=1_000.0, treated_kernel=1_500.0,
    seed=0, index=0, treated_overrides=None, horizons=None,
):
    horizons = horizons or HORIZONS
    treated_overrides = treated_overrides or {}
    return U.build_utility_record(
        branch_pair_id=f"pair_{index}_{seed}",
        context=context(index),
        policy_stage="middle",
        seed=seed,
        horizons=horizons,
        control_dose=dose("control", control_kernel),
        intervention_dose=dose("intervention", treated_kernel),
        control_evaluations=[evaluation("control", h, control_j) for h in horizons],
        intervention_evaluations=[
            evaluation("intervention", h, treated_j, **treated_overrides) for h in horizons
        ],
        epsilon=0.10,
        kernel_radius_bins=1,
        base_distribution_sha256="a" * 64,
        intervention_distribution_sha256="b" * 64,
    )


class TestHarmVector:
    def test_deltas_are_intervention_minus_control(self):
        harm = U.build_harm_vector(
            evaluation("control", "H_m", 0.5, foot_slip=0.01),
            evaluation("intervention", "H_m", 0.5, foot_slip=0.05),
        )
        assert harm.slip_delta == pytest.approx(0.04)

    def test_improvement_is_negative_harm(self):
        harm = U.build_harm_vector(
            evaluation("control", "H_m", 0.5, foot_slip=0.05),
            evaluation("intervention", "H_m", 0.5, foot_slip=0.01),
        )
        assert harm.slip_delta < 0

    def test_clean_delta_follows_efficacy_convention(self):
        harm = U.build_harm_vector(
            evaluation("control", "H_m", 0.5, clean_j_eff=0.80),
            evaluation("intervention", "H_m", 0.5, clean_j_eff=0.85),
        )
        assert harm.clean_delta > 0

    def test_rejects_horizon_mismatch(self):
        with pytest.raises(ValueError, match="horizon mismatch"):
            U.build_harm_vector(
                evaluation("control", "H_s", 0.5), evaluation("intervention", "H_l", 0.5)
            )


class TestClassify:
    CLEAN = HarmVector(0.0, 0.0, 0.0, 0.0, 0.0)

    def test_positive_and_clean_is_safe_positive(self):
        assert U.classify_context(0.05, self.CLEAN)[0] == "safe_positive"

    def test_negative_efficacy_is_harmful(self):
        assert U.classify_context(-0.05, self.CLEAN)[0] == "harmful"

    def test_no_effect_is_neutral(self):
        assert U.classify_context(0.0, self.CLEAN)[0] == "neutral"

    def test_a_breached_gate_overrides_positive_efficacy(self):
        """Gates cannot be traded against success."""
        label, reasons = U.classify_context(0.50, HarmVector(0.0, 0.0, 0.5, 0.0, 0.0))
        assert label == "harmful" and reasons == ["gate:slip"]

    def test_clean_regression_is_harmful_even_with_efficacy_gain(self):
        label, reasons = U.classify_context(0.50, HarmVector(-0.5, 0.0, 0.0, 0.0, 0.0))
        assert label == "harmful" and "gate:clean_noninferiority" in reasons

    def test_all_breached_gates_are_reported(self):
        _, reasons = U.classify_context(0.1, HarmVector(0.0, 0.5, 0.5, 0.0, 0.0))
        assert set(reasons) == {"gate:action_rate", "gate:slip"}

    def test_deadband_suppresses_noise(self):
        assert U.classify_context(1e-9, self.CLEAN)[0] == "neutral"

    def test_gates_are_configurable(self):
        harm = HarmVector(0.0, 0.0, 0.5, 0.0, 0.0)
        assert U.classify_context(0.1, harm, gates={"slip": 10.0})[0] == "safe_positive"


class TestBuildUtilityRecord:
    def test_efficacy_delta_per_horizon(self):
        result = record(control_j=0.50, treated_j=0.55)
        assert all(v == pytest.approx(0.05) for v in result.efficacy_delta.values())

    def test_utility_is_normalized_by_realized_dose(self):
        result = record(control_j=0.50, treated_j=0.55,
                        control_kernel=1_000.0, treated_kernel=1_500.0)
        assert result.utility["H_m"] == pytest.approx(0.05 / (500.0 + 1e-6))

    def test_dose_denominator_changes_the_label(self):
        """Same efficacy, less delivered dose, higher per-unit utility."""
        generous = record(treated_kernel=2_000.0).utility["H_m"]
        stingy = record(treated_kernel=1_100.0).utility["H_m"]
        assert stingy > generous

    def test_undelivered_dose_yields_no_label(self):
        result = record(treated_kernel=1_000.0)   # intervention got no extra practice
        assert result.utility == {}
        assert U.is_usable(result) is False

    def test_negative_dose_yields_no_label(self):
        assert U.is_usable(record(treated_kernel=800.0)) is False

    def test_safety_label_recorded_per_horizon(self):
        result = record()
        assert set(result.safety_label) == set(HORIZONS)
        assert result.safety_label["H_l"] == "safe_positive"

    def test_harm_breach_labels_every_horizon(self):
        result = record(treated_overrides={"foot_slip": 0.9})
        assert all(v == "harmful" for v in result.safety_label.values())

    def test_missing_control_horizon_raises(self):
        with pytest.raises(ValueError, match="control branch is missing"):
            U.build_utility_record(
                branch_pair_id="p", context=context(), policy_stage="mid", seed=0,
                horizons=HORIZONS,
                control_dose=dose("control", 1000.0),
                intervention_dose=dose("intervention", 1500.0),
                control_evaluations=[evaluation("control", "H_s", 0.5)],
                intervention_evaluations=[evaluation("intervention", h, 0.5) for h in HORIZONS],
                epsilon=0.1, kernel_radius_bins=1,
                base_distribution_sha256="a" * 64, intervention_distribution_sha256="b" * 64,
            )

    def test_missing_intervention_horizon_raises(self):
        with pytest.raises(ValueError, match="intervention branch is missing"):
            U.build_utility_record(
                branch_pair_id="p", context=context(), policy_stage="mid", seed=0,
                horizons=HORIZONS,
                control_dose=dose("control", 1000.0),
                intervention_dose=dose("intervention", 1500.0),
                control_evaluations=[evaluation("control", h, 0.5) for h in HORIZONS],
                intervention_evaluations=[evaluation("intervention", "H_s", 0.5)],
                epsilon=0.1, kernel_radius_bins=1,
                base_distribution_sha256="a" * 64, intervention_distribution_sha256="b" * 64,
            )


class TestHorizonReversals:
    def build(self, short_u, long_u):
        result = record()
        result.utility = {"H_s": short_u, "H_l": long_u}
        return result

    def test_reversal_harmful(self):
        """The curriculum false positive the programme exists to find."""
        assert U.horizon_reversals(self.build(0.5, -0.5), "H_s", "H_l") == "reversal_harmful"

    def test_delayed_useful(self):
        assert U.horizon_reversals(self.build(-0.5, 0.5), "H_s", "H_l") == "delayed_useful"

    def test_immediate_only(self):
        assert U.horizon_reversals(self.build(0.5, 0.0), "H_s", "H_l") == "immediate_only"

    def test_consistent_positive_is_not_flagged(self):
        assert U.horizon_reversals(self.build(0.5, 0.5), "H_s", "H_l") is None

    def test_missing_horizon_returns_none(self):
        assert U.horizon_reversals(record(), "H_s", "H_xl") is None


class TestIdentifiability:
    def replicated(self, spread, noise, num_contexts=8, replicates=3):
        """Contexts separated by ``spread``, each measured ``replicates`` times."""
        records = []
        for index in range(num_contexts):
            for r in range(replicates):
                result = record(index=index, seed=r)
                jitter = noise * (r - (replicates - 1) / 2)
                result.utility = {"H_l": index * spread + jitter}
                records.append(result)
        return records

    def test_passes_when_contexts_separate_cleanly(self):
        report = U.assess_identifiability(self.replicated(spread=1.0, noise=0.01), "H_l")
        assert report.passes is True
        assert report.variance_ratio > 2.0

    def test_fails_when_noise_swamps_the_signal(self):
        report = U.assess_identifiability(self.replicated(spread=0.01, noise=5.0), "H_l")
        assert report.passes is False
        assert any("between/within" in reason for reason in report.reasons)

    def test_reports_both_variance_components(self):
        report = U.assess_identifiability(self.replicated(spread=1.0, noise=0.1), "H_l")
        assert report.between_context_sd > report.within_context_sd > 0

    def test_icc_is_between_zero_and_one(self):
        report = U.assess_identifiability(self.replicated(spread=1.0, noise=0.5), "H_l")
        assert 0.0 <= report.intraclass_correlation <= 1.0

    def test_epsilon_zero_pairs_supply_the_noise_floor(self):
        records = self.replicated(spread=1.0, noise=0.01)
        report = U.assess_identifiability(records, "H_l", noise_floor=[0.0, 0.02, -0.02, 0.01])
        assert report.noise_floor_sd is not None
        assert "noise floor taken from epsilon=0 pairs" in report.reasons

    def test_a_large_noise_floor_can_fail_an_otherwise_clean_set(self):
        """Replicates can look tight and still be uninformative."""
        records = self.replicated(spread=0.1, noise=0.001)
        optimistic = U.assess_identifiability(records, "H_l")
        pessimistic = U.assess_identifiability(
            records, "H_l", noise_floor=[-2.0, 2.0, -1.5, 1.5, 0.0]
        )
        assert optimistic.passes and not pessimistic.passes

    def test_single_context_cannot_pass(self):
        report = U.assess_identifiability([record()], "H_l")
        assert report.passes is False
        assert "fewer than two contexts" in report.reasons[0]

    def test_no_replicates_and_no_noise_floor_cannot_pass(self):
        records = [record(index=i) for i in range(6)]
        for i, r in enumerate(records):
            r.utility = {"H_l": float(i)}
        report = U.assess_identifiability(records, "H_l")
        assert report.passes is False

    def test_thresholds_are_configurable(self):
        records = self.replicated(spread=0.2, noise=0.1)
        strict = U.assess_identifiability(records, "H_l", min_variance_ratio=1e6)
        assert strict.passes is False

    def test_records_without_the_horizon_are_ignored(self):
        records = self.replicated(spread=1.0, noise=0.01)
        records.append(record())          # carries H_s/H_m/H_l from build, different scale
        report = U.assess_identifiability(records, "H_zz")
        assert report.num_contexts == 0

    def test_report_serializes(self):
        report = U.assess_identifiability(self.replicated(1.0, 0.01), "H_l")
        assert set(report.to_dict()) >= {"passes", "variance_ratio", "reasons"}


class TestSummarizeLabels:
    def test_counts_usable_and_dropped(self):
        records = [record(index=0), record(index=1, treated_kernel=1_000.0)]
        summary = U.summarize_labels(records, "H_l")
        assert summary["num_records"] == 2
        assert summary["num_usable"] == 1
        assert summary["num_dropped_no_dose"] == 1

    def test_reports_distribution(self):
        records = [record(index=i, treated_j=0.50 + 0.01 * i) for i in range(5)]
        summary = U.summarize_labels(records, "H_l")
        assert summary["utility_min"] <= summary["utility_median"] <= summary["utility_max"]

    def test_counts_safety_labels(self):
        records = [record(index=0), record(index=1, treated_overrides={"foot_slip": 0.9})]
        summary = U.summarize_labels(records, "H_l")
        assert summary["safety_labels"]["harmful"] == 1

    def test_empty_is_safe(self):
        assert U.summarize_labels([], "H_l")["num_usable"] == 0
