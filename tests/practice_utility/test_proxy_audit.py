"""Tests for the difficulty-proxy audit and the Gate B decision."""

import math

import pytest

from gear_sonic.research.practice_utility import proxy_audit as P
from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    DoseReport,
    UtilityRecord,
    motion_hash,
)


def make_record(index, stage, proxy_values, utilities):
    key = f"walk_{index:03d}__A{index:03d}"
    record = UtilityRecord(
        branch_pair_id=f"pair_{index}",
        context=ContextKey(key, motion_hash(key, 300, 50.0), 1, 50, 100),
        policy_stage=stage,
        seed=0,
        horizons={"H_s": 8, "H_l": 128},
        base_distribution_sha256="a" * 64,
        intervention_distribution_sha256="b" * 64,
        epsilon=0.1,
        kernel_radius_bins=1,
        control_dose=DoseReport("c", "ctx", "control", completed_kernel_steps=1000.0,
                                completed_env_steps=10000.0),
        intervention_dose=DoseReport("i", "ctx", "intervention",
                                     completed_kernel_steps=1500.0,
                                     completed_env_steps=10000.0),
        proxy_features=dict(proxy_values),
    )
    record.utility = dict(utilities)
    return record


def perfect_set(n=12, stage="middle", proxy="native_failure_rate"):
    """A proxy that predicts utility exactly."""
    return [
        make_record(i, stage, {proxy: float(i)}, {"H_l": float(i), "H_s": float(i)})
        for i in range(n)
    ]


def useless_set(n=12, stage="middle", proxy="native_failure_rate"):
    """A proxy uncorrelated with utility."""
    scrambled = [(7 * i) % n for i in range(n)]
    return [
        make_record(i, stage, {proxy: float(scrambled[i])},
                    {"H_l": float(i), "H_s": float(i)})
        for i in range(n)
    ]


class TestRankStatistics:
    def test_ties_get_average_rank(self):
        assert P.rank_data([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]

    def test_all_tied(self):
        assert P.rank_data([5, 5, 5]) == [2.0, 2.0, 2.0]

    def test_spearman_monotone(self):
        assert P.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_spearman_inverse(self):
        assert P.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_spearman_is_rank_based_not_linear(self):
        """A monotone but wildly nonlinear relation still scores 1."""
        assert P.spearman([1, 2, 3, 4], [1, 10, 1000, 1e6]) == pytest.approx(1.0)

    def test_spearman_zero_variance_is_zero(self):
        assert P.spearman([1, 1, 1], [1, 2, 3]) == 0.0

    def test_spearman_too_short(self):
        assert P.spearman([1], [1]) == 0.0

    def test_pearson_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            P.pearson([1, 2], [1])


class TestSignAccuracy:
    def test_perfect(self):
        assert P.sign_accuracy([1, -1, 1], [2, -2, 3]) == 1.0

    def test_all_wrong(self):
        assert P.sign_accuracy([1, 1], [-1, -1]) == 0.0

    def test_partial(self):
        assert P.sign_accuracy([1, -1, 1], [1, -1, -1]) == pytest.approx(2 / 3)

    def test_deadband_excludes_ambiguous_truth(self):
        """Contexts with no real effect must not be scored."""
        assert P.sign_accuracy([1, 1], [0.0001, 5.0], deadband=0.01) == 1.0

    def test_everything_in_deadband_returns_zero(self):
        assert P.sign_accuracy([1, 1], [0.0, 0.0], deadband=0.01) == 0.0


class TestPairwiseAccuracy:
    def test_perfect_ordering(self):
        assert P.pairwise_ranking_accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_reversed_ordering(self):
        assert P.pairwise_ranking_accuracy([1, 2, 3], [3, 2, 1]) == 0.0

    def test_ties_in_truth_are_skipped(self):
        assert P.pairwise_ranking_accuracy([1, 2, 3], [5, 5, 5]) == 0.0

    def test_partial(self):
        assert 0.0 < P.pairwise_ranking_accuracy([1, 3, 2], [1, 2, 3]) < 1.0

    def test_rejects_mismatch(self):
        with pytest.raises(ValueError, match="align"):
            P.pairwise_ranking_accuracy([1, 2], [1])


class TestCalibration:
    def test_perfect_relationship_is_well_calibrated(self):
        values = [float(i) for i in range(20)]
        assert P.calibration_error(values, values) == pytest.approx(0.0, abs=1e-9)

    def test_scale_invariant(self):
        values = [float(i) for i in range(20)]
        scaled = [1000.0 * v + 7.0 for v in values]
        assert P.calibration_error(scaled, values) == pytest.approx(0.0, abs=1e-9)

    def test_inverted_relationship_is_badly_calibrated(self):
        values = [float(i) for i in range(20)]
        assert P.calibration_error(values, list(reversed(values))) > 0.5

    def test_too_few_samples_is_nan(self):
        assert math.isnan(P.calibration_error([1.0, 2.0], [1.0, 2.0]))


class TestAuditProxy:
    def test_perfect_proxy_scores_high(self):
        result = P.audit_proxy(perfect_set(), "native_failure_rate", "H_l")
        assert result.spearman == pytest.approx(1.0)
        assert result.pairwise_accuracy == pytest.approx(1.0)
        assert result.is_sufficient is True

    def test_useless_proxy_scores_low(self):
        result = P.audit_proxy(useless_set(), "native_failure_rate", "H_l")
        assert abs(result.spearman) < 0.5
        assert result.is_sufficient is False

    def test_grouping_defeats_a_between_group_confound(self):
        """The reason correlations are computed within group and then averaged.

        Here the proxy carries no within-stage information at all, but the two
        stages sit at different levels of both proxy and utility. Pooled, that
        looks like strong prediction; grouped, it correctly reads as nothing.
        """
        # Within each stage the proxy is constant, so it carries no information
        # about which context in that stage is worth practising. Between stages
        # both proxy and utility jump together, purely because training
        # progressed.
        records = []
        for i in range(10):
            records.append(make_record(i, "early", {"p": 1.0}, {"H_l": 1.0 + 0.1 * i}))
        for i in range(10):
            records.append(make_record(100 + i, "late", {"p": 9.0}, {"H_l": 9.0 + 0.1 * i}))

        pooled = P.spearman([r.proxy_features["p"] for r in records],
                            [r.utility["H_l"] for r in records])
        grouped = P.audit_proxy(records, "p", "H_l").spearman

        assert pooled > 0.8          # pooling says the proxy is excellent
        assert grouped == 0.0        # grouping says it knows nothing at all

    def test_grouped_audit_rejects_the_confounded_proxy(self):
        records = []
        for i in range(10):
            records.append(make_record(i, "early", {"p": 1.0}, {"H_l": 1.0 + 0.1 * i}))
        for i in range(10):
            records.append(make_record(100 + i, "late", {"p": 9.0}, {"H_l": 9.0 + 0.1 * i}))
        assert P.audit_proxy(records, "p", "H_l").is_sufficient is False

    def test_detects_direction_flip_between_groups(self):
        records = []
        for i in range(8):
            records.append(make_record(i, "early", {"p": float(i)}, {"H_l": float(i)}))
        for i in range(8):
            records.append(make_record(100 + i, "late", {"p": float(i)}, {"H_l": float(-i)}))
        result = P.audit_proxy(records, "p", "H_l")
        assert result.sign_flips_across_groups is True
        assert result.is_sufficient is False

    def test_consistent_direction_is_not_flagged(self):
        records = perfect_set(8, "early") + perfect_set(8, "late")
        assert P.audit_proxy(records, "native_failure_rate", "H_l").sign_flips_across_groups is False

    def test_counts_groups_and_samples(self):
        records = perfect_set(6, "early") + perfect_set(6, "late")
        result = P.audit_proxy(records, "native_failure_rate", "H_l")
        assert result.num_samples == 12 and result.num_groups == 2

    def test_records_without_the_proxy_are_skipped(self):
        records = perfect_set(8)
        records.append(make_record(99, "middle", {}, {"H_l": 1.0}))
        assert P.audit_proxy(records, "native_failure_rate", "H_l").num_samples == 8

    def test_records_without_the_horizon_are_skipped(self):
        records = perfect_set(8)
        records.append(make_record(99, "middle", {"native_failure_rate": 1.0}, {}))
        assert P.audit_proxy(records, "native_failure_rate", "H_l").num_samples == 8

    def test_too_few_samples_is_safe(self):
        result = P.audit_proxy(perfect_set(1), "native_failure_rate", "H_l")
        assert result.num_samples <= 1 and result.spearman == 0.0

    def test_custom_grouping(self):
        records = perfect_set(8, "early") + perfect_set(8, "late")
        result = P.audit_proxy(records, "native_failure_rate", "H_l",
                               group_by=lambda r: "all")
        assert result.num_groups == 1

    def test_serializes(self):
        payload = P.audit_proxy(perfect_set(), "native_failure_rate", "H_l").to_dict()
        assert set(payload) >= {"spearman", "is_sufficient", "per_group_spearman"}


class TestAuditAllProxies:
    def test_audits_only_present_proxies(self):
        records = [
            make_record(i, "middle",
                        {"native_failure_rate": float(i), "latent_gap_p90": float(-i)},
                        {"H_l": float(i)})
            for i in range(10)
        ]
        results = P.audit_all_proxies(records, "H_l")
        assert set(results) == {"native_failure_rate", "latent_gap_p90"}

    def test_reports_opposite_directions(self):
        records = [
            make_record(i, "middle",
                        {"native_failure_rate": float(i), "latent_gap_p90": float(-i)},
                        {"H_l": float(i)})
            for i in range(10)
        ]
        results = P.audit_all_proxies(records, "H_l")
        assert results["native_failure_rate"].spearman > 0
        assert results["latent_gap_p90"].spearman < 0


class TestReversals:
    def test_counts_each_pattern(self):
        records = [
            make_record(0, "m", {"p": 1.0}, {"H_s": 1.0, "H_l": -1.0}),   # reversal
            make_record(1, "m", {"p": 1.0}, {"H_s": -1.0, "H_l": 1.0}),   # delayed
            make_record(2, "m", {"p": 1.0}, {"H_s": 1.0, "H_l": 0.0}),    # immediate only
            make_record(3, "m", {"p": 1.0}, {"H_s": 1.0, "H_l": 1.0}),    # consistent
        ]
        counts = P.count_reversals(records, "H_s", "H_l")
        assert counts == {"delayed_useful": 1, "immediate_only": 1, "reversal_harmful": 1}


class TestGateB:
    def test_a_sufficient_proxy_blocks_the_estimator(self):
        """Finding that a simple signal suffices is a result, not a failure."""
        report = P.assess_sufficiency(perfect_set(), "H_l")
        assert report.authorizes_estimator is False
        assert "native_failure_rate" in report.sufficient_proxies
        assert any("not warranted" in reason for reason in report.reasons)

    def test_no_sufficient_proxy_authorizes_the_estimator(self):
        report = P.assess_sufficiency(useless_set(), "H_l")
        assert report.authorizes_estimator is True
        assert report.sufficient_proxies == []
        assert any("no proxy reached sufficiency" in reason for reason in report.reasons)

    def test_reports_the_best_proxy_even_when_insufficient(self):
        report = P.assess_sufficiency(useless_set(), "H_l")
        assert report.best_proxy == "native_failure_rate"

    def test_counts_horizon_reversals(self):
        records = [
            make_record(i, "middle", {"native_failure_rate": float(i)},
                        {"H_s": 1.0, "H_l": -1.0 if i % 2 else 1.0})
            for i in range(10)
        ]
        report = P.assess_sufficiency(records, "H_l", short_horizon="H_s")
        assert report.num_reversals == 5
        assert report.reversal_fraction == pytest.approx(0.5)

    def test_flags_unstable_proxies(self):
        records = []
        for i in range(8):
            records.append(make_record(i, "early", {"native_failure_rate": float(i)},
                                       {"H_l": float(i)}))
        for i in range(8):
            records.append(make_record(100 + i, "late", {"native_failure_rate": float(i)},
                                       {"H_l": float(-i)}))
        report = P.assess_sufficiency(records, "H_l")
        assert "native_failure_rate" in report.unstable_proxies
        assert report.authorizes_estimator is True

    def test_no_proxies_recorded_does_not_authorize(self):
        records = [make_record(i, "m", {}, {"H_l": float(i)}) for i in range(5)]
        report = P.assess_sufficiency(records, "H_l")
        assert report.authorizes_estimator is False
        assert "cannot audit" in report.reasons[0]

    def test_serializes_with_per_proxy_detail(self):
        payload = P.assess_sufficiency(perfect_set(), "H_l").to_dict()
        assert "proxy_results" in payload
        assert payload["proxy_results"]["native_failure_rate"]["is_sufficient"] is True
