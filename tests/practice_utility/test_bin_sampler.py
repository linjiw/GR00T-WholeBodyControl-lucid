"""Expanding-support samplers: the safeguards, not just the arithmetic.

Every test here corresponds to a way the sampler could silently stop being what
its receipt says it is -- easy bins sampled away, weights coupled to the batch
they are drawing, an active bin missing from an update, a resume that restarts
the draw sequence. Those are the failures that produce a plausible number rather
than an error, which is the only kind worth writing tests against.
"""

import copy
import math

import pytest
import torch

from gear_sonic.research.practice_utility import bin_sampler as BS

# ------------------------------------------------------------------- bins --


class TestDifficultyBins:
    def test_uniform_spans_the_closed_unit_interval(self):
        bins = BS.DifficultyBins.uniform(5)
        assert bins.centres == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))
        assert len(bins) == 5

    def test_centres_must_increase_and_stay_in_range(self):
        with pytest.raises(ValueError):
            BS.DifficultyBins((0.0, 0.5, 0.5))
        with pytest.raises(ValueError):
            BS.DifficultyBins((0.0, 1.5))
        with pytest.raises(ValueError):
            BS.DifficultyBins((0.5,))

    def test_active_grows_with_d_max_and_never_reaches_zero(self):
        bins = BS.DifficultyBins.uniform(5)
        assert bins.active(0.0) == 1
        assert bins.active(0.5) == 3
        assert bins.active(1.0) == 5
        # A support that excluded the nominal distribution would be a moving
        # point, not an expanding support.
        assert bins.active(0.0) >= 1

    def test_easy_indices_are_defined_over_the_whole_bin_set(self):
        bins = BS.DifficultyBins.uniform(5)
        # 40% of 5 bins = 2, regardless of how far the support has expanded.
        assert bins.easy_indices(0.4) == (0, 1)
        assert bins.easy_indices(1.0) == (0, 1, 2, 3, 4)
        with pytest.raises(ValueError):
            bins.easy_indices(0.0)


# ------------------------------------------------------------- easy floor --


class TestEasyFloor:
    def test_a_satisfied_floor_changes_nothing(self):
        probabilities = [0.3, 0.3, 0.2, 0.2]
        out = BS.apply_easy_floor(probabilities, (0, 1), 0.15)
        assert out == pytest.approx(probabilities)

    def test_a_starved_easy_end_is_lifted_to_exactly_the_floor(self):
        # The failure this exists to prevent: error weighting drives the easy
        # bins to ~0 and the curriculum silently becomes a moving point.
        out = BS.apply_easy_floor([0.01, 0.01, 0.49, 0.49], (0, 1), 0.15)
        assert sum(out[:2]) == pytest.approx(0.15)
        assert sum(out) == pytest.approx(1.0)

    def test_the_hard_end_keeps_its_relative_ordering(self):
        out = BS.apply_easy_floor([0.01, 0.01, 0.20, 0.78], (0, 1), 0.15)
        assert out[3] > out[2]
        assert out[3] / out[2] == pytest.approx(0.78 / 0.20)

    def test_the_easy_end_keeps_its_relative_ordering(self):
        out = BS.apply_easy_floor([0.02, 0.01, 0.97], (0, 1), 0.15)
        assert out[0] / out[1] == pytest.approx(2.0)

    def test_all_easy_bins_means_nothing_to_do(self):
        out = BS.apply_easy_floor([0.5, 0.5], (0, 1), 0.15)
        assert out == pytest.approx([0.5, 0.5])

    def test_a_dead_hard_end_stays_dead_when_the_floor_is_already_satisfied(self):
        out = BS.apply_easy_floor([0.5, 0.5, 0.0], (0, 1), 0.15)
        assert sum(out) == pytest.approx(1.0)
        assert out == pytest.approx([0.5, 0.5, 0.0])

    def test_an_out_of_range_floor_is_rejected(self):
        with pytest.raises(ValueError):
            BS.apply_easy_floor([0.5, 0.5], (0,), 1.0)


# --------------------------------------------------------------- coverage --


class TestEffectiveBinCount:
    def test_balanced_bins_give_back_the_number_of_bins(self):
        assert BS.effective_bin_count([25, 25, 25, 25]) == pytest.approx(4.0)

    def test_one_dominant_bin_collapses_the_effective_count(self):
        # 97 of 100 samples in one bin: nominally four bins, effectively one.
        assert BS.effective_bin_count([97, 1, 1, 1]) < 1.1
        assert BS.effective_bin_count([97, 1, 1, 1]) == pytest.approx(
            100 * 100 / (97 * 97 + 3), rel=1e-9
        )

    def test_empty_is_zero_not_an_error(self):
        assert BS.effective_bin_count([0, 0]) == 0.0


# ---------------------------------------------------------------- support --


class TestExpandingSupport:
    def _sampler(self, **kwargs):
        return BS.UniformExpandingSampler(BS.DifficultyBins.uniform(5), seed=7, **kwargs)

    def test_probabilities_are_uniform_over_the_active_prefix(self):
        sampler = self._sampler(d_max=0.5)
        assert sampler.probabilities() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])

    def test_the_easy_bin_never_leaves_the_mixture(self):
        for d_max in (0.0, 0.25, 0.5, 0.75, 1.0):
            sampler = self._sampler(d_max=d_max)
            assert sampler.probabilities()[0] > 0.0

    def test_support_may_expand_but_not_shrink(self):
        sampler = self._sampler(d_max=0.25)
        sampler.d_max = 0.75
        assert sampler.num_active == 4
        with pytest.raises(ValueError, match="only expand"):
            sampler.d_max = 0.25

    def test_draws_stay_inside_the_active_support(self):
        sampler = self._sampler(d_max=0.5)
        drawn = sampler.draw(500)
        assert drawn.numel() == 500
        assert int(drawn.max()) <= 2

    def test_draws_are_seed_deterministic(self):
        a = self._sampler(d_max=1.0).draw(64)
        b = self._sampler(d_max=1.0).draw(64)
        assert torch.equal(a, b)

    def test_difficulties_map_through_the_bin_centres(self):
        sampler = self._sampler(d_max=1.0)
        assignment = torch.tensor([0, 2, 4])
        assert sampler.difficulties(assignment).tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_an_empty_draw_is_empty_not_an_error(self):
        assert self._sampler().draw(0).numel() == 0


class TestCoverage:
    def _sampler(self, minimum=2):
        return BS.UniformExpandingSampler(
            BS.DifficultyBins.uniform(4),
            seed=3,
            d_max=1.0,
            min_samples_per_active_bin=minimum,
        )

    def test_a_covered_batch_passes_and_is_counted(self):
        sampler = self._sampler(minimum=1)
        counts = sampler.check_coverage(torch.tensor([0, 1, 2, 3]))
        assert counts == [1, 1, 1, 1]
        assert sampler.telemetry.updates == 1

    def test_a_missing_active_bin_raises(self):
        # Fail closed: a batch without an active bin trains on a different
        # distribution than the receipt claims, and nothing else would show it.
        sampler = self._sampler(minimum=1)
        with pytest.raises(BS.CoverageError, match=r"active bins \[3\]"):
            sampler.check_coverage(torch.tensor([0, 1, 2, 2]))
        assert sampler.telemetry.coverage_failures == 1

    def test_an_under_represented_bin_raises(self):
        sampler = self._sampler(minimum=2)
        with pytest.raises(BS.CoverageError):
            sampler.check_coverage(torch.tensor([0, 0, 1, 1, 2, 2, 3]))

    def test_inactive_bins_are_not_required(self):
        sampler = BS.UniformExpandingSampler(
            BS.DifficultyBins.uniform(4),
            seed=3,
            d_max=0.34,
            min_samples_per_active_bin=1,
        )
        assert sampler.num_active == 2
        sampler.check_coverage(torch.tensor([0, 1]))

    def test_effective_bin_count_is_recorded_and_falls_with_imbalance(self):
        sampler = self._sampler(minimum=1)
        sampler.check_coverage(torch.tensor([0, 1, 2, 3]))
        balanced = sampler.telemetry.min_effective_bins
        sampler.check_coverage(torch.tensor([0] * 97 + [1, 2, 3]))
        assert sampler.telemetry.min_effective_bins < balanced
        assert sampler.receipt()["telemetry"]["min_effective_bin_count"] == pytest.approx(
            sampler.telemetry.min_effective_bins
        )


# ---------------------------------------------------------- error weights --


class TestErrorWeighted:
    def _sampler(self, **kwargs):
        options = dict(
            seed=11,
            d_max=1.0,
            easy_fraction=0.4,
            easy_floor=0.15,
            lag=2,
            update_every=1,
            smoothing=1.0,
        )
        options.update(kwargs)
        return BS.ErrorWeightedSampler(BS.DifficultyBins.uniform(5), **options)

    def test_the_floor_band_is_enforced_at_construction(self):
        # 10-20% is the preregistered band; anything else is a different
        # experiment and must not be reachable by a config typo.
        for bad in (0.0, 0.05, 0.25, 0.9):
            with pytest.raises(ValueError, match="preregistered band"):
                self._sampler(easy_floor=bad)
        self._sampler(easy_floor=0.10)
        self._sampler(easy_floor=0.20)

    def test_weighting_is_inert_until_the_lag_has_elapsed(self):
        sampler = self._sampler(lag=2)
        uniform = [0.2] * 5
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        assert sampler.probabilities() == pytest.approx(uniform)
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        assert sampler.probabilities() == pytest.approx(uniform)
        # The third observation releases the first, so weighting begins.
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        assert sampler.probabilities() != pytest.approx(uniform)

    def test_weights_follow_the_lagged_statistics_not_the_latest(self):
        sampler = self._sampler(lag=2, smoothing=1.0)
        sampler.observe_failure_rates([1.0, 0.0, 0.0, 0.0, 0.0])  # released 3rd
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        # The live weights come from the FIRST observation, not the last.
        probabilities = sampler.probabilities()
        assert probabilities[0] == max(probabilities)

    def test_the_easy_floor_survives_an_all_hard_signal(self):
        sampler = self._sampler(lag=1, smoothing=1.0)
        for _ in range(3):
            sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        probabilities = sampler.probabilities()
        assert sum(probabilities[:2]) == pytest.approx(0.15)
        assert probabilities[4] == max(probabilities)
        assert sum(probabilities) == pytest.approx(1.0)

    def test_inactive_bins_get_no_probability_even_if_they_look_hard(self):
        sampler = self._sampler(lag=1, d_max=0.5, smoothing=1.0)
        for _ in range(3):
            sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        probabilities = sampler.probabilities()
        assert probabilities[3] == 0.0 and probabilities[4] == 0.0
        assert sum(probabilities) == pytest.approx(1.0)

    def test_an_all_zero_signal_falls_back_to_uniform(self):
        sampler = self._sampler(lag=1, smoothing=1.0)
        for _ in range(3):
            sampler.observe_failure_rates([0.0] * 5)
        assert sampler.probabilities() == pytest.approx([0.2] * 5)

    def test_update_every_holds_the_weights_between_refreshes(self):
        sampler = self._sampler(lag=1, update_every=3, smoothing=1.0)
        sampler.observe_failure_rates([1.0, 0.0, 0.0, 0.0, 0.0])
        sampler.observe_failure_rates([1.0, 0.0, 0.0, 0.0, 0.0])
        first = sampler.probabilities()
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        assert sampler.probabilities() == pytest.approx(first)
        # The fourth observation releases the third statistic: that is the
        # third *released* value, so this is the first legal refresh.
        sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        assert sampler.probabilities() != pytest.approx(first)
        assert sampler.probabilities()[4] == max(sampler.probabilities())

    def test_malformed_statistics_are_rejected(self):
        sampler = self._sampler()
        with pytest.raises(ValueError):
            sampler.observe_failure_rates([0.1, 0.2])
        with pytest.raises(ValueError):
            sampler.observe_failure_rates([0.1, 0.2, 0.3, 0.4, -1.0])
        with pytest.raises(ValueError):
            sampler.observe_failure_rates([0.1, 0.2, 0.3, 0.4, math.nan])

    def test_the_receipt_records_the_realised_easy_mass(self):
        sampler = self._sampler(lag=1, smoothing=1.0)
        for _ in range(3):
            sampler.observe_failure_rates([0.0, 0.0, 0.0, 0.0, 1.0])
        receipt = sampler.receipt()
        assert receipt["easy_floor"] == 0.15
        assert receipt["realised_easy_mass"] == pytest.approx(0.15)
        assert receipt["weighting_active"] is True
        assert receipt["easy_bin_indices"] == [0, 1]


# ----------------------------------------------------------- direct mixed --


class TestFixedMixture:
    def test_the_full_mixture_ignores_d_max(self):
        sampler = BS.FixedMixtureSampler(BS.DifficultyBins.uniform(4), seed=1, d_max=0.0)
        assert sampler.probabilities() == pytest.approx([0.25] * 4)
        assert sampler.num_active == 4

    def test_it_is_the_same_distribution_as_a_fully_expanded_curriculum(self):
        # The direct-mixed baseline and the curriculum's finish must be the
        # same distribution, not two implementations meant to agree.
        mixed = BS.FixedMixtureSampler(BS.DifficultyBins.uniform(5), seed=2)
        expanded = BS.UniformExpandingSampler(BS.DifficultyBins.uniform(5), seed=2, d_max=1.0)
        assert mixed.probabilities() == pytest.approx(expanded.probabilities())


# ----------------------------------------------------------------- resume --


class TestResume:
    def test_a_resumed_sampler_continues_the_same_draw_sequence(self):
        original = BS.UniformExpandingSampler(BS.DifficultyBins.uniform(5), seed=5, d_max=1.0)
        original.draw(32)
        state = original.state_dict()
        expected = original.draw(32)

        resumed = BS.UniformExpandingSampler(BS.DifficultyBins.uniform(5), seed=5)
        resumed.load_state_dict(state)
        assert torch.equal(resumed.draw(32), expected)
        assert resumed.d_max == 1.0

    def test_resuming_into_different_bins_is_refused(self):
        state = BS.UniformExpandingSampler(BS.DifficultyBins.uniform(5), seed=5).state_dict()
        other = BS.UniformExpandingSampler(BS.DifficultyBins.uniform(4), seed=5)
        with pytest.raises(ValueError, match="different bin definition"):
            other.load_state_dict(state)

    def test_error_weighted_state_round_trips_including_the_lag_queue(self):
        original = BS.ErrorWeightedSampler(
            BS.DifficultyBins.uniform(5), seed=4, d_max=1.0, lag=2, smoothing=1.0
        )
        for rates in ([1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0]):
            original.observe_failure_rates(rates)
        before = original.probabilities()

        resumed = BS.ErrorWeightedSampler(
            BS.DifficultyBins.uniform(5), seed=4, lag=2, smoothing=1.0
        )
        resumed.load_state_dict(original.state_dict())
        assert resumed.probabilities() == pytest.approx(before)
        assert resumed._released_statistics == original._released_statistics
        # And the lag queue really came back, so the next release matches.
        original.observe_failure_rates([0, 0, 0, 1.0, 0])
        resumed.observe_failure_rates([0, 0, 0, 1.0, 0])
        assert resumed.probabilities() == pytest.approx(original.probabilities())

    def test_telemetry_survives_resume_including_failures(self):
        bins = BS.DifficultyBins.uniform(4)
        original = BS.UniformExpandingSampler(bins, seed=5, d_max=1.0)
        original.check_coverage(torch.tensor([0, 1, 2, 3]))
        with pytest.raises(BS.CoverageError):
            original.check_coverage(torch.tensor([0, 1, 2, 2]))

        resumed = BS.UniformExpandingSampler(bins, seed=5)
        resumed.load_state_dict(original.state_dict())
        assert resumed.telemetry.to_dict() == original.telemetry.to_dict()
        assert resumed.receipt()["telemetry"] == original.receipt()["telemetry"]

    def test_released_cadence_survives_resume_between_refreshes(self):
        options = dict(
            bins=BS.DifficultyBins.uniform(5),
            seed=4,
            d_max=1.0,
            lag=1,
            update_every=3,
            smoothing=1.0,
        )
        original = BS.ErrorWeightedSampler(**options)
        original.observe_failure_rates([1.0, 0, 0, 0, 0])
        original.observe_failure_rates([1.0, 0, 0, 0, 0])
        original.observe_failure_rates([0, 0, 0, 0, 1.0])
        assert original._released_statistics == 2

        resumed = BS.ErrorWeightedSampler(**options)
        resumed.load_state_dict(original.state_dict())
        update = [0, 0, 0, 0, 1.0]
        original.observe_failure_rates(update)
        resumed.observe_failure_rates(update)
        assert resumed._released_statistics == 3
        assert resumed.probabilities() == pytest.approx(original.probabilities())

    def test_sampler_kind_and_frozen_configuration_must_match(self):
        bins = BS.DifficultyBins.uniform(5)
        state = BS.UniformExpandingSampler(bins, seed=5, d_max=1.0).state_dict()
        with pytest.raises(ValueError, match="UniformExpandingSampler.*FixedMixtureSampler"):
            BS.FixedMixtureSampler(bins, seed=5).load_state_dict(state)
        with pytest.raises(ValueError, match="different sampler configuration"):
            BS.UniformExpandingSampler(bins, seed=6).load_state_dict(state)
        with pytest.raises(ValueError, match="different sampler configuration"):
            BS.UniformExpandingSampler(bins, seed=5, min_samples_per_active_bin=2).load_state_dict(
                state
            )

        weighted = BS.ErrorWeightedSampler(
            bins, seed=5, lag=2, update_every=2, smoothing=0.5
        ).state_dict()
        with pytest.raises(ValueError, match="different sampler configuration"):
            BS.ErrorWeightedSampler(
                bins, seed=5, lag=1, update_every=2, smoothing=0.5
            ).load_state_dict(weighted)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("schema_version", 999, "unsupported sampler state schema"),
            ("d_max", math.nan, "d_max must be in"),
            ("generator_state", [1, 2, 3], "invalid RNG state"),
        ),
    )
    def test_corrupt_base_state_is_rejected_without_mutating_the_sampler(
        self, field, value, message
    ):
        bins = BS.DifficultyBins.uniform(5)
        source = BS.UniformExpandingSampler(bins, seed=5, d_max=1.0)
        state = source.state_dict()
        state[field] = value
        target = BS.UniformExpandingSampler(bins, seed=5)
        before = copy.deepcopy(target.state_dict())
        with pytest.raises(ValueError, match=message):
            target.load_state_dict(state)
        assert target.state_dict() == before

    def test_corrupt_telemetry_and_cadence_state_are_rejected(self):
        bins = BS.DifficultyBins.uniform(5)
        source = BS.UniformExpandingSampler(bins, seed=5, d_max=1.0)
        source.check_coverage(torch.tensor([0, 1, 2, 3, 4]))
        bad_telemetry = source.state_dict()
        bad_telemetry["telemetry"]["cumulative_counts"] = [5]
        with pytest.raises(ValueError, match="bin counts"):
            BS.UniformExpandingSampler(bins, seed=5).load_state_dict(bad_telemetry)

        weighted = BS.ErrorWeightedSampler(
            bins, seed=5, d_max=1.0, lag=1, update_every=3, smoothing=1.0
        )
        weighted.observe_failure_rates([1.0, 0, 0, 0, 0])
        weighted.observe_failure_rates([1.0, 0, 0, 0, 0])
        bad_cadence = weighted.state_dict()
        bad_cadence["released_statistics"] = 99
        with pytest.raises(ValueError, match="lag queue/counters are inconsistent"):
            BS.ErrorWeightedSampler(
                bins, seed=5, d_max=1.0, lag=1, update_every=3, smoothing=1.0
            ).load_state_dict(bad_cadence)
