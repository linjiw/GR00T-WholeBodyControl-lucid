"""Tests for intervention kernels and identity-preserving distribution mixing.

The identity properties tested here are load-bearing for the science, not
cosmetic: if ``epsilon = 0`` did not reproduce the base distribution exactly,
the epsilon=0 branch could not be used to measure the branch noise floor, and
Gate A would be unfalsifiable.
"""

import math

import pytest
import torch

from gear_sonic.research.practice_utility import intervention as I


def uniform(n):
    return torch.full((n,), 1.0 / n, dtype=torch.float64)


def normalize(values):
    t = torch.tensor(values, dtype=torch.float64)
    return t / t.sum()


class TestKernelSpec:
    @pytest.mark.parametrize("kwargs", [{"radius_bins": -1}, {"sigma_frames": 0.0},
                                        {"sigma_frames": -5.0}])
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            I.KernelSpec(**kwargs)


class TestBuildLocalKernel:
    """Two motions of 4 bins each, laid out end to end."""

    def setup_method(self):
        self.bin_positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        self.bin_motion_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        self.bin_centres = torch.tensor([25, 75, 125, 175, 25, 75, 125, 175])

    def build(self, target_position, target_motion_id, spec=I.KernelSpec()):
        return I.build_local_kernel(
            target_position=target_position,
            bin_positions=self.bin_positions,
            bin_motion_ids=self.bin_motion_ids,
            target_motion_id=target_motion_id,
            bin_centre_frames=self.bin_centres,
            spec=spec,
        )

    def test_normalized(self):
        assert float(self.build(1, 0).sum()) == pytest.approx(1.0)

    def test_never_crosses_motion_boundary(self):
        kernel = self.build(3, 0)          # last bin of motion 0
        assert float(kernel[4:].sum()) == 0.0  # motion 1 untouched

    def test_respects_radius(self):
        kernel = self.build(1, 0)
        assert float(kernel[3]) == 0.0     # two bins away
        assert float(kernel[0]) > 0.0
        assert float(kernel[2]) > 0.0

    def test_target_has_the_largest_weight(self):
        kernel = self.build(1, 0)
        assert int(kernel.argmax()) == 1

    def test_radius_zero_is_a_point_mass(self):
        kernel = self.build(1, 0, I.KernelSpec(radius_bins=0))
        assert float(kernel[1]) == pytest.approx(1.0)

    def test_wider_radius_covers_more_bins(self):
        narrow = (self.build(1, 0, I.KernelSpec(radius_bins=1)) > 0).sum()
        wide = (self.build(1, 0, I.KernelSpec(radius_bins=3)) > 0).sum()
        assert wide > narrow

    def test_smaller_sigma_concentrates_mass(self):
        tight = self.build(1, 0, I.KernelSpec(sigma_frames=5.0))
        loose = self.build(1, 0, I.KernelSpec(sigma_frames=500.0))
        assert float(tight[1]) > float(loose[1])

    def test_boundary_bin_kernel_is_still_normalized(self):
        for position in (0, 3):
            assert float(self.build(position, 0).sum()) == pytest.approx(1.0)

    def test_rejects_out_of_range_target(self):
        with pytest.raises(IndexError):
            self.build(99, 0)

    def test_rejects_target_motion_mismatch(self):
        with pytest.raises(ValueError, match="empty"):
            self.build(1, 7)


class TestMixIntervention:
    def test_epsilon_zero_is_exact_identity(self):
        base, kernel = normalize([4.0, 3.0, 2.0, 1.0]), normalize([0.0, 1.0, 1.0, 0.0])
        assert torch.equal(I.mix_intervention(base, kernel, 0.0), base)

    def test_epsilon_one_returns_the_kernel(self):
        base, kernel = uniform(4), normalize([0.0, 1.0, 1.0, 0.0])
        assert torch.allclose(I.mix_intervention(base, kernel, 1.0), kernel)

    def test_stays_a_distribution(self):
        base, kernel = normalize([4.0, 3.0, 2.0, 1.0]), normalize([0.0, 1.0, 1.0, 0.0])
        mixed = I.mix_intervention(base, kernel, 0.1)
        assert float(mixed.sum()) == pytest.approx(1.0)
        assert bool((mixed >= 0).all())

    def test_increases_mass_on_kernel_support(self):
        base, kernel = uniform(4), normalize([0.0, 1.0, 0.0, 0.0])
        mixed = I.mix_intervention(base, kernel, 0.1)
        assert float(mixed[1]) > float(base[1])

    def test_decreases_mass_off_kernel_support(self):
        base, kernel = uniform(4), normalize([0.0, 1.0, 0.0, 0.0])
        mixed = I.mix_intervention(base, kernel, 0.1)
        assert float(mixed[0]) < float(base[0])

    def test_added_mass_equals_epsilon_times_kernel(self):
        base, kernel = uniform(4), normalize([0.0, 1.0, 0.0, 0.0])
        eps = 0.1
        mixed = I.mix_intervention(base, kernel, eps)
        assert float(mixed[1] - base[1]) == pytest.approx(eps * (1.0 - float(base[1])))

    def test_is_monotone_in_epsilon(self):
        base, kernel = uniform(8), normalize([0.0, 1.0] + [0.0] * 6)
        masses = [float(I.mix_intervention(base, kernel, e)[1]) for e in (0.0, 0.05, 0.1, 0.2)]
        assert masses == sorted(masses)

    @pytest.mark.parametrize("epsilon", [-0.01, 1.01])
    def test_rejects_epsilon_out_of_range(self, epsilon):
        with pytest.raises(ValueError, match="epsilon"):
            I.mix_intervention(uniform(4), uniform(4), epsilon)

    def test_rejects_unnormalized_base(self):
        with pytest.raises(ValueError, match="sums to"):
            I.mix_intervention(torch.tensor([0.5, 0.4]), uniform(2), 0.1)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            I.mix_intervention(uniform(4), uniform(5), 0.1)


class TestResidualDistribution:
    def test_alpha_zero_is_exact_identity(self):
        base = normalize([4.0, 3.0, 2.0, 1.0])
        out = I.residual_distribution(base, torch.tensor([0.0, 1.0, 2.0, 3.0]), 0.0, 1.0)
        assert torch.equal(out, base)

    def test_constant_scores_is_exact_identity(self):
        base = normalize([4.0, 3.0, 2.0, 1.0])
        out = I.residual_distribution(base, torch.full((4,), 2.5), 0.5, 1.0)
        assert torch.equal(out, base)

    def test_stays_a_distribution(self):
        base = normalize([4.0, 3.0, 2.0, 1.0])
        out = I.residual_distribution(base, torch.tensor([0.0, 1.0, 2.0, 3.0]), 0.25, 1.0)
        assert float(out.sum()) == pytest.approx(1.0)
        assert bool((out >= 0).all())

    def test_upweights_high_scores(self):
        base = uniform(4)
        out = I.residual_distribution(base, torch.tensor([0.0, 0.0, 0.0, 3.0]), 0.5, 1.0)
        assert float(out[3]) > float(base[3])
        assert float(out[0]) < float(base[0])

    def test_never_leaves_the_base_support(self):
        base = normalize([0.5, 0.5, 0.0])
        out = I.residual_distribution(base, torch.tensor([0.0, 0.0, 10.0]), 0.9, 1.0)
        assert float(out[2]) == 0.0

    def test_kl_radius_is_respected(self):
        base = uniform(16)
        scores = torch.arange(16, dtype=torch.float64)
        for max_kl in (0.002, 0.02, 0.05):
            out = I.residual_distribution(base, scores, 0.5, 0.5, max_kl=max_kl)
            assert I.kl_divergence(out, base) <= max_kl + 1e-9

    def test_tighter_kl_moves_less(self):
        base = uniform(16)
        scores = torch.arange(16, dtype=torch.float64)
        tight = I.residual_distribution(base, scores, 0.5, 0.5, max_kl=0.002)
        loose = I.residual_distribution(base, scores, 0.5, 0.5, max_kl=0.05)
        assert I.kl_divergence(tight, base) < I.kl_divergence(loose, base)

    def test_larger_alpha_moves_further(self):
        base = uniform(16)
        scores = torch.arange(16, dtype=torch.float64)
        small = I.residual_distribution(base, scores, 0.10, 1.0)
        large = I.residual_distribution(base, scores, 0.25, 1.0)
        assert I.kl_divergence(large, base) > I.kl_divergence(small, base)

    def test_higher_temperature_flattens_the_tilt(self):
        base = uniform(16)
        scores = torch.arange(16, dtype=torch.float64)
        hot = I.residual_distribution(base, scores, 0.5, 100.0)
        cold = I.residual_distribution(base, scores, 0.5, 0.5)
        assert I.kl_divergence(hot, base) < I.kl_divergence(cold, base)

    def test_coverage_floor_is_enforced(self):
        base = uniform(10)
        scores = torch.tensor([10.0] + [0.0] * 9)
        out = I.residual_distribution(base, scores, 0.9, 0.1, coverage_floor=0.02)
        assert float(out.min()) >= 0.02 - 1e-9
        assert float(out.sum()) == pytest.approx(1.0)

    def test_infeasible_coverage_floor_rejected(self):
        with pytest.raises(ValueError, match="infeasible"):
            I.residual_distribution(uniform(10), torch.arange(10.0), 0.5, 1.0, coverage_floor=0.5)

    def test_max_prob_ratio_caps_concentration(self):
        base = uniform(8)
        scores = torch.tensor([50.0] + [0.0] * 7)
        capped = I.residual_distribution(base, scores, 1.0, 1.0, max_prob_ratio=3.0)
        uncapped = I.residual_distribution(base, scores, 1.0, 1.0)
        assert float(capped.max()) < float(uncapped.max())

    @pytest.mark.parametrize("max_ratio", [1.5, 3.0, 10.0])
    def test_max_prob_ratio_bound_actually_holds(self, max_ratio):
        base = normalize([5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0])
        out = I.residual_distribution(base, scores, 1.0, 1.0, max_prob_ratio=max_ratio)
        assert float((out / base).max()) <= max_ratio + 1e-6
        assert float(out.sum()) == pytest.approx(1.0)

    def test_ratio_bound_survives_blending(self):
        base = normalize([5.0, 4.0, 3.0, 2.0, 1.0])
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 30.0])
        out = I.residual_distribution(base, scores, 0.3, 1.0, max_prob_ratio=2.0)
        assert float((out / base).max()) <= 2.0 + 1e-6

    def test_globally_binding_cap_degrades_gracefully(self):
        # max_ratio == 1 forces q == base for a uniform base; must stay valid.
        base = uniform(6)
        out = I.residual_distribution(base, torch.arange(6.0), 1.0, 1.0, max_prob_ratio=1.0)
        assert float(out.sum()) == pytest.approx(1.0)
        assert float((out / base).max()) <= 1.0 + 1e-6

    def test_rejects_nonpositive_max_prob_ratio(self):
        with pytest.raises(ValueError, match="max_prob_ratio"):
            I.residual_distribution(uniform(4), torch.arange(4.0), 0.5, 1.0, max_prob_ratio=0.0)

    @pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"alpha": 1.1}, {"temperature": 0.0},
                                        {"coverage_floor": -0.1}])
    def test_rejects_invalid_parameters(self, kwargs):
        params = dict(alpha=0.5, temperature=1.0)
        params.update(kwargs)
        with pytest.raises(ValueError):
            I.residual_distribution(uniform(4), torch.arange(4.0), **params)


class TestKlDivergence:
    def test_zero_for_identical(self):
        assert I.kl_divergence(uniform(8), uniform(8)) == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_different(self):
        assert I.kl_divergence(normalize([0.7, 0.3]), uniform(2)) > 0

    def test_infinite_when_q_leaves_support_of_p(self):
        q, p = normalize([0.5, 0.5]), torch.tensor([1.0, 0.0], dtype=torch.float64)
        assert math.isinf(I.kl_divergence(q, p))

    def test_asymmetric(self):
        a, b = normalize([0.9, 0.1]), normalize([0.5, 0.5])
        assert I.kl_divergence(a, b) != pytest.approx(I.kl_divergence(b, a))
