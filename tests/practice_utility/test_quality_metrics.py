"""Tests for physical-quality metrics and quality-qualified success.

Each metric is checked against a case whose answer is known analytically, not
merely against itself: a stationary foot must slip exactly zero, a foot sliding
at 1 m/s for 0.2 s must slip exactly 0.2 m, a joint held at its limit must read
saturation 1.0, and a pure 20 Hz action must put all its energy above a 10 Hz
cutoff.
"""

import math

import pytest
import torch

from gear_sonic.research.practice_utility import quality_metrics as Q


def sine(freq_hz, steps=200, control_hz=50.0, dims=1, amplitude=1.0):
    t = torch.arange(steps, dtype=torch.float64) / control_hz
    wave = amplitude * torch.sin(2 * math.pi * freq_hz * t)
    return wave.unsqueeze(1).repeat(1, dims)


def episode(**overrides):
    base = dict(
        completed=True, completion_fraction=1.0, mpjpe=0.10,
        action_rate=0.01, action_acceleration=0.001, hf_action_ratio=0.05,
        foot_slip=0.01, contact_impulse=50.0, undesired_contact_rate=0.0,
        torque_saturation=0.0, joint_limit_proximity=0.3, energy_proxy=10.0,
        episode_length=200, family="walk",
    )
    base.update(overrides)
    return Q.EpisodeQuality(**base)


class TestActionSmoothness:
    def test_constant_action_has_zero_rate(self):
        assert Q.action_rate(torch.ones(50, 6)) == 0.0

    def test_linear_ramp_has_constant_rate_but_zero_acceleration(self):
        ramp = torch.arange(50, dtype=torch.float64).unsqueeze(1)
        assert Q.action_rate(ramp) == pytest.approx(1.0)
        assert Q.action_acceleration(ramp) == pytest.approx(0.0, abs=1e-12)

    def test_alternating_action_is_penalized(self):
        chatter = torch.tensor([[1.0], [-1.0]] * 25, dtype=torch.float64)
        assert Q.action_rate(chatter) == pytest.approx(4.0)

    def test_rate_sums_over_joints(self):
        ramp = torch.arange(50, dtype=torch.float64).unsqueeze(1).repeat(1, 3)
        assert Q.action_rate(ramp) == pytest.approx(3.0)

    def test_short_sequences_degrade_gracefully(self):
        assert Q.action_rate(torch.ones(1, 4)) == 0.0
        assert Q.action_acceleration(torch.ones(2, 4)) == 0.0

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\(T, D\)"):
            Q.action_rate(torch.ones(5))


class TestHighFrequencyRatio:
    def test_pure_low_frequency_has_no_high_energy(self):
        assert Q.high_frequency_action_ratio(sine(1.0)) == pytest.approx(0.0, abs=1e-9)

    def test_pure_high_frequency_is_all_high_energy(self):
        assert Q.high_frequency_action_ratio(sine(20.0)) == pytest.approx(1.0, abs=1e-9)

    def test_ratio_is_between_zero_and_one_for_a_mixture(self):
        mixed = sine(1.0) + sine(20.0)
        assert 0.2 < Q.high_frequency_action_ratio(mixed) < 0.8

    def test_amplitude_invariant(self):
        """A vigorous smooth motion must not read as jitter."""
        small = Q.high_frequency_action_ratio(sine(1.0, amplitude=0.1) + sine(20.0, amplitude=0.1))
        large = Q.high_frequency_action_ratio(sine(1.0, amplitude=10.0) + sine(20.0, amplitude=10.0))
        assert small == pytest.approx(large, abs=1e-9)

    def test_constant_offset_is_not_oscillation(self):
        assert Q.high_frequency_action_ratio(torch.ones(200, 2)) == 0.0

    def test_offset_does_not_change_the_ratio(self):
        wave = sine(20.0)
        assert Q.high_frequency_action_ratio(wave + 5.0) == pytest.approx(
            Q.high_frequency_action_ratio(wave), abs=1e-9
        )

    def test_cutoff_moves_the_boundary(self):
        wave = sine(5.0)
        assert Q.high_frequency_action_ratio(wave, cutoff_hz=2.0) > 0.9
        assert Q.high_frequency_action_ratio(wave, cutoff_hz=10.0) < 0.1

    @pytest.mark.parametrize("cutoff", [0.0, -1.0, 25.0, 30.0])
    def test_rejects_cutoff_outside_the_band(self, cutoff):
        with pytest.raises(ValueError, match="cutoff_hz"):
            Q.high_frequency_action_ratio(sine(1.0), cutoff_hz=cutoff)

    def test_rejects_nonpositive_rate(self):
        with pytest.raises(ValueError, match="control_hz"):
            Q.high_frequency_action_ratio(sine(1.0), control_hz=0.0)

    def test_short_sequence_returns_zero(self):
        assert Q.high_frequency_action_ratio(torch.ones(3, 2)) == 0.0


class TestFootSlip:
    def test_stationary_foot_does_not_slip(self):
        assert Q.foot_slip(torch.zeros(10, 2, 2), torch.ones(10, 2), 0.02) == 0.0

    def test_known_slide_gives_known_distance(self):
        velocity = torch.zeros(10, 2, 2)
        velocity[:, 0, 0] = 1.0                       # 1 m/s, 10 steps of 0.02 s
        assert Q.foot_slip(velocity, torch.ones(10, 2), 0.02) == pytest.approx(0.2)

    def test_swing_phase_motion_is_not_slip(self):
        velocity = torch.ones(10, 2, 2)
        assert Q.foot_slip(velocity, torch.zeros(10, 2), 0.02) == 0.0

    def test_only_contact_steps_count(self):
        velocity = torch.zeros(10, 1, 2)
        velocity[:, 0, 0] = 1.0
        mask = torch.zeros(10, 1)
        mask[:5] = 1.0
        assert Q.foot_slip(velocity, mask, 0.02) == pytest.approx(0.1)

    def test_uses_speed_not_signed_velocity(self):
        """Sliding back and forth is still slipping."""
        velocity = torch.zeros(4, 1, 2)
        velocity[:, 0, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert Q.foot_slip(velocity, torch.ones(4, 1), 0.02) == pytest.approx(0.08)

    def test_diagonal_slide_uses_euclidean_norm(self):
        velocity = torch.zeros(1, 1, 2)
        velocity[0, 0] = torch.tensor([3.0, 4.0])
        assert Q.foot_slip(velocity, torch.ones(1, 1), 1.0) == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "vel,mask", [(torch.zeros(10, 2, 3), torch.ones(10, 2)),
                     (torch.zeros(10, 2, 2), torch.ones(10, 3))]
    )
    def test_rejects_mismatched_shapes(self, vel, mask):
        with pytest.raises(ValueError):
            Q.foot_slip(vel, mask, 0.02)

    def test_rejects_nonpositive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            Q.foot_slip(torch.zeros(4, 1, 2), torch.ones(4, 1), 0.0)


class TestContact:
    def test_impulse_integrates_force_over_time(self):
        forces = torch.full((10, 1), 100.0)
        peak, impulse = Q.contact_impulse(forces, 0.02)
        assert peak == pytest.approx(100.0)
        assert impulse == pytest.approx(20.0)

    def test_peak_captures_a_single_spike(self):
        forces = torch.zeros(10, 1)
        forces[5] = 900.0
        assert Q.contact_impulse(forces, 0.02)[0] == pytest.approx(900.0)

    def test_vector_forces_use_magnitude(self):
        forces = torch.zeros(1, 1, 3)
        forces[0, 0] = torch.tensor([3.0, 4.0, 0.0])
        assert Q.contact_impulse(forces, 1.0)[0] == pytest.approx(5.0)

    def test_empty_is_zero(self):
        assert Q.contact_impulse(torch.zeros(0, 1), 0.02) == (0.0, 0.0)

    def test_undesired_contact_rate(self):
        contacts = torch.zeros(10, 3, dtype=torch.bool)
        contacts[:2, 0] = True
        assert Q.undesired_contact_rate(contacts) == pytest.approx(0.2)

    def test_simultaneous_contacts_count_once_per_step(self):
        contacts = torch.ones(10, 4, dtype=torch.bool)
        assert Q.undesired_contact_rate(contacts) == pytest.approx(1.0)


class TestActuator:
    def test_torque_at_limit_saturates_fully(self):
        assert Q.torque_saturation(torch.full((10, 4), 100.0), torch.full((4,), 100.0)) == 1.0

    def test_zero_torque_never_saturates(self):
        assert Q.torque_saturation(torch.zeros(10, 4), torch.full((4,), 100.0)) == 0.0

    def test_sign_does_not_matter(self):
        assert Q.torque_saturation(torch.full((10, 4), -100.0), torch.full((4,), 100.0)) == 1.0

    def test_partial_saturation_is_a_fraction(self):
        torques = torch.zeros(10, 2)
        torques[:, 0] = 100.0
        assert Q.torque_saturation(torques, torch.full((2,), 100.0)) == pytest.approx(0.5)

    def test_respects_per_joint_limits(self):
        torques = torch.full((10, 2), 50.0)
        assert Q.torque_saturation(torques, torch.tensor([50.0, 500.0])) == pytest.approx(0.5)

    def test_rejects_mismatched_limits(self):
        with pytest.raises(ValueError, match="entries"):
            Q.torque_saturation(torch.zeros(10, 4), torch.full((3,), 1.0))

    def test_rejects_nonpositive_limits(self):
        with pytest.raises(ValueError, match="positive"):
            Q.torque_saturation(torch.zeros(10, 2), torch.tensor([1.0, 0.0]))

    def test_joint_limit_proximity_is_zero_at_mid_range(self):
        lower, upper = torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0])
        assert Q.joint_limit_proximity(torch.zeros(5, 2), lower, upper) == pytest.approx(0.0)

    def test_joint_limit_proximity_is_one_on_the_stop(self):
        lower, upper = torch.tensor([-1.0]), torch.tensor([1.0])
        assert Q.joint_limit_proximity(torch.ones(5, 1), lower, upper) == pytest.approx(1.0)

    def test_joint_limit_proximity_is_clamped_beyond_the_stop(self):
        lower, upper = torch.tensor([-1.0]), torch.tensor([1.0])
        assert Q.joint_limit_proximity(torch.full((5, 1), 9.0), lower, upper) == pytest.approx(1.0)

    def test_rejects_inverted_limits(self):
        with pytest.raises(ValueError, match="exceed"):
            Q.joint_limit_proximity(torch.zeros(5, 1), torch.tensor([1.0]), torch.tensor([-1.0]))

    def test_energy_proxy_is_absolute_power(self):
        torques = torch.full((10, 2), 2.0)
        velocities = torch.full((10, 2), -3.0)
        assert Q.energy_proxy(torques, velocities) == pytest.approx(12.0)

    def test_energy_proxy_rejects_mismatch(self):
        with pytest.raises(ValueError, match="align"):
            Q.energy_proxy(torch.zeros(10, 2), torch.zeros(10, 3))


class TestGates:
    THRESHOLDS = Q.QualityThresholds()

    def test_clean_episode_passes(self):
        assert Q.evaluate_gates(episode(), self.THRESHOLDS) == []

    @pytest.mark.parametrize(
        "field,value,gate",
        [
            ("mpjpe", 0.9, "mpjpe"),
            ("foot_slip", 0.9, "foot_slip"),
            ("hf_action_ratio", 0.9, "hf_action"),
            ("contact_impulse", 9999.0, "contact_impulse"),
            ("torque_saturation", 0.9, "torque_saturation"),
            ("completion_fraction", 0.1, "completion"),
        ],
    )
    def test_each_gate_fires(self, field, value, gate):
        assert Q.evaluate_gates(episode(**{field: value}), self.THRESHOLDS) == [gate]

    def test_multiple_failures_are_all_reported(self):
        bad = episode(mpjpe=0.9, foot_slip=0.9)
        assert set(Q.evaluate_gates(bad, self.THRESHOLDS)) == {"mpjpe", "foot_slip"}

    def test_upright_but_jittery_is_not_a_success(self):
        """The whole point: staying upright is not enough."""
        jittery = Q.apply_gates([episode(hf_action_ratio=0.9)], self.THRESHOLDS)[0]
        assert jittery.completed is True
        assert jittery.quality_success is False

    def test_fall_is_never_a_success(self):
        fallen = Q.apply_gates([episode(completed=False, completion_fraction=0.4)],
                               self.THRESHOLDS)[0]
        assert fallen.quality_success is False


class TestAggregation:
    THRESHOLDS = Q.QualityThresholds()

    def test_macro_mean_weights_families_equally(self):
        """A large good family must not mask a small bad one."""
        episodes = [episode(family="walk") for _ in range(99)]
        episodes += [episode(family="crawl", mpjpe=0.9)]
        Q.apply_gates(episodes, self.THRESHOLDS)
        assert Q.macro_mean_quality_success(episodes) == pytest.approx(0.5)

    def test_micro_average_would_have_hidden_it(self):
        episodes = [episode(family="walk") for _ in range(99)]
        episodes += [episode(family="crawl", mpjpe=0.9)]
        Q.apply_gates(episodes, self.THRESHOLDS)
        micro = sum(e.quality_success for e in episodes) / len(episodes)
        assert micro > 0.98 and Q.macro_mean_quality_success(episodes) == pytest.approx(0.5)

    def test_empty_is_zero(self):
        assert Q.macro_mean_quality_success([]) == 0.0

    def test_family_rates(self):
        episodes = Q.apply_gates(
            [episode(family="walk"), episode(family="crawl", mpjpe=0.9)], self.THRESHOLDS
        )
        assert Q.family_success_rates(episodes) == {"crawl": 0.0, "walk": 1.0}

    def test_gate_failure_counts_expose_a_dominant_gate(self):
        episodes = Q.apply_gates(
            [episode(foot_slip=0.9) for _ in range(5)] + [episode(mpjpe=0.9)], self.THRESHOLDS
        )
        assert Q.gate_failure_counts(episodes) == {"foot_slip": 5, "mpjpe": 1}

    def test_summary_separates_raw_from_quality_success(self):
        episodes = [episode(hf_action_ratio=0.9) for _ in range(4)]
        report = Q.summarize(episodes, self.THRESHOLDS)
        assert report["raw_completion_rate"] == 1.0
        assert report["quality_success_rate_micro"] == 0.0
        assert report["gate_failure_counts"] == {"hf_action": 4}

    def test_summary_reports_the_worst_family(self):
        episodes = [episode(family="walk"), episode(family="crawl", mpjpe=0.9)]
        report = Q.summarize(episodes, self.THRESHOLDS)
        assert report["worst_family"] == "crawl" and report["worst_family_rate"] == 0.0

    def test_summary_records_the_thresholds_used(self):
        report = Q.summarize([episode()], Q.QualityThresholds(max_mpjpe=0.5))
        assert report["thresholds"]["max_mpjpe"] == 0.5
