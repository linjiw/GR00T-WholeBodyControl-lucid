"""Tests for offline motion-structure features.

Each feature is checked against a synthetic motion whose answer is known in
closed form, so a plausible-but-wrong implementation cannot pass.
"""

import math

import numpy as np
import pytest

from gear_sonic.research.practice_utility import proxy_features as PF

FPS = 50.0


def still(frames=100, joints=6):
    return np.zeros((frames, joints))


def ramp(frames=100, joints=6, rate=1.0):
    """Joints advancing at exactly ``rate`` rad/s."""
    t = np.arange(frames) / FPS
    return np.repeat((rate * t)[:, None], joints, axis=1)


def sine(freq_hz, frames=200, joints=6, amplitude=1.0):
    t = np.arange(frames) / FPS
    return np.repeat((amplitude * np.sin(2 * math.pi * freq_hz * t))[:, None], joints, axis=1)


class TestJointKinematics:
    def test_static_pose_has_no_motion(self):
        f = PF.compute_motion_features(still(), fps=FPS)
        assert f.joint_speed_rms == 0.0
        assert f.joint_acceleration_rms == 0.0
        assert f.joint_range_max == 0.0

    def test_constant_rate_recovers_the_rate(self):
        f = PF.compute_motion_features(ramp(rate=2.0), fps=FPS)
        assert f.joint_speed_rms == pytest.approx(2.0)

    def test_constant_rate_has_no_acceleration(self):
        f = PF.compute_motion_features(ramp(rate=2.0), fps=FPS)
        assert f.joint_acceleration_rms == pytest.approx(0.0, abs=1e-9)

    def test_range_is_peak_to_peak(self):
        f = PF.compute_motion_features(sine(1.0, amplitude=3.0), fps=FPS)
        assert f.joint_range_max == pytest.approx(6.0, rel=0.02)

    def test_faster_motion_has_higher_speed(self):
        slow = PF.compute_motion_features(sine(1.0), fps=FPS).joint_speed_rms
        fast = PF.compute_motion_features(sine(5.0), fps=FPS).joint_speed_rms
        assert fast > slow

    def test_jerk_separates_smooth_from_chattering(self):
        smooth = PF.compute_motion_features(sine(1.0), fps=FPS).joint_jerk_rms
        chatter = np.tile(np.array([[0.0] * 6, [1.0] * 6]), (50, 1))
        assert PF.compute_motion_features(chatter, fps=FPS).joint_jerk_rms > smooth

    def test_duration_from_fps(self):
        f = PF.compute_motion_features(still(frames=100), fps=FPS)
        assert f.duration_seconds == pytest.approx(2.0)
        assert f.num_frames == 100

    @pytest.mark.parametrize("bad", [np.zeros((5,)), np.zeros((2, 3, 4))])
    def test_rejects_wrong_rank(self, bad):
        with pytest.raises(ValueError, match=r"\(T, J\)"):
            PF.compute_motion_features(bad, fps=FPS)

    def test_rejects_nonpositive_fps(self):
        with pytest.raises(ValueError, match="fps"):
            PF.compute_motion_features(still(), fps=0.0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="no frames"):
            PF.compute_motion_features(np.zeros((0, 6)), fps=FPS)


class TestRootKinematics:
    def test_stationary_root(self):
        f = PF.compute_motion_features(still(), root_trans=np.zeros((100, 3)), fps=FPS)
        assert f.root_speed_mean == 0.0 and f.root_vertical_range == 0.0

    def test_constant_velocity_recovers_speed(self):
        t = np.arange(100) / FPS
        trans = np.stack([1.5 * t, np.zeros(100), np.zeros(100)], axis=1)
        f = PF.compute_motion_features(still(), root_trans=trans, fps=FPS)
        assert f.root_speed_mean == pytest.approx(1.5)

    def test_diagonal_motion_uses_euclidean_speed(self):
        t = np.arange(100) / FPS
        trans = np.stack([3.0 * t, 4.0 * t, np.zeros(100)], axis=1)
        f = PF.compute_motion_features(still(), root_trans=trans, fps=FPS)
        assert f.root_speed_mean == pytest.approx(5.0)

    def test_vertical_range(self):
        trans = np.zeros((100, 3))
        trans[:, 2] = np.linspace(0.8, 1.3, 100)
        f = PF.compute_motion_features(still(), root_trans=trans, fps=FPS)
        assert f.root_vertical_range == pytest.approx(0.5)

    def test_missing_root_data_is_zero_not_an_error(self):
        f = PF.compute_motion_features(sine(1.0), fps=FPS)
        assert f.root_speed_mean == 0.0 and f.joint_speed_rms > 0

    def test_rejects_mismatched_root_length(self):
        with pytest.raises(ValueError, match="matching dof"):
            PF.compute_motion_features(still(100), root_trans=np.zeros((50, 3)), fps=FPS)


class TestQuaternionAngularSpeed:
    def test_constant_orientation_has_no_rotation(self):
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (50, 1))
        assert PF.quaternion_angular_speed(quats, FPS).max() == pytest.approx(0.0, abs=1e-9)

    def test_known_rotation_rate(self):
        """Rotating at 1 rad/s about z."""
        t = np.arange(50) / FPS
        half = 0.5 * 1.0 * t
        quats = np.stack([np.cos(half), np.zeros(50), np.zeros(50), np.sin(half)], axis=1)
        speeds = PF.quaternion_angular_speed(quats, FPS)
        assert np.median(speeds) == pytest.approx(1.0, rel=1e-3)

    def test_double_cover_sign_flip_is_not_a_jump(self):
        """q and -q are the same rotation; a naive dot would see 180 degrees."""
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (10, 1))
        quats[5:] *= -1.0
        assert PF.quaternion_angular_speed(quats, FPS).max() == pytest.approx(0.0, abs=1e-9)

    def test_unnormalized_input_is_handled(self):
        quats = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (10, 1))
        assert np.isfinite(PF.quaternion_angular_speed(quats, FPS)).all()

    def test_too_short_is_zero(self):
        assert PF.quaternion_angular_speed(np.array([[1.0, 0, 0, 0]]), FPS).max() == 0.0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(T, 4\)"):
            PF.compute_motion_features(still(10), root_rot=np.zeros((10, 3)), fps=FPS)


class TestBallisticFraction:
    def test_free_fall_is_fully_ballistic(self):
        t = np.arange(60) / FPS
        height = 2.0 - 0.5 * PF.GRAVITY * t**2
        assert PF.ballistic_fraction(height, FPS) > 0.95

    def test_standing_still_is_not_ballistic(self):
        assert PF.ballistic_fraction(np.full(60, 0.8), FPS) == 0.0

    def test_constant_velocity_is_not_ballistic(self):
        assert PF.ballistic_fraction(np.linspace(0.8, 1.2, 60), FPS) == 0.0

    def test_partial_flight(self):
        t = np.arange(30) / FPS
        flight = 1.0 - 0.5 * PF.GRAVITY * t**2
        height = np.concatenate([np.full(30, 0.8), flight])
        assert 0.2 < PF.ballistic_fraction(height, FPS) < 0.8

    def test_too_short_is_zero(self):
        assert PF.ballistic_fraction(np.array([1.0, 1.0]), FPS) == 0.0


class TestSpectral:
    def test_single_frequency_is_simple(self):
        assert PF.spectral_complexity(sine(2.0)) < 0.2

    def test_noise_is_complex(self):
        rng = np.random.default_rng(0)
        assert PF.spectral_complexity(rng.standard_normal((200, 6))) > 0.8

    def test_static_pose_is_not_complex(self):
        assert PF.spectral_complexity(still(200)) == 0.0

    def test_low_frequency_has_little_high_band_energy(self):
        assert PF.high_frequency_ratio(sine(1.0)) < 0.05

    def test_high_frequency_has_most_high_band_energy(self):
        assert PF.high_frequency_ratio(sine(20.0)) > 0.9

    def test_rejects_invalid_band(self):
        with pytest.raises(ValueError, match="band_start"):
            PF.high_frequency_ratio(sine(1.0), band_start=1.5)


class TestBinSlicing:
    def clip(self, frames=200):
        return {
            "dof": sine(1.0, frames=frames),
            "root_trans_offset": np.zeros((frames, 3)),
            "root_rot": np.tile(np.array([1.0, 0, 0, 0]), (frames, 1)),
            "fps": FPS,
        }

    def test_uses_only_the_bin(self):
        clip = self.clip()
        clip["dof"][:100] = 0.0            # calm first half, active second half
        calm = PF.features_for_bin(clip, 0, 100)
        active = PF.features_for_bin(clip, 100, 200)
        assert calm.joint_speed_rms < active.joint_speed_rms

    def test_bin_features_differ_from_clip_average(self):
        """Why bin-level features exist: an average blurs the hard moment."""
        clip = self.clip()
        clip["dof"][:150] = 0.0
        whole = PF.compute_motion_features(clip["dof"], fps=FPS)
        hard = PF.features_for_bin(clip, 150, 200)
        assert hard.joint_speed_rms > whole.joint_speed_rms

    def test_frame_count_matches_the_bin(self):
        assert PF.features_for_bin(self.clip(), 50, 100).num_frames == 50

    @pytest.mark.parametrize("bounds", [(-1, 50), (0, 500), (100, 50), (50, 50)])
    def test_rejects_out_of_range_bins(self, bounds):
        with pytest.raises(ValueError, match="out of range"):
            PF.features_for_bin(self.clip(), *bounds)


class TestRegimeProxy:
    def make(self, **overrides):
        base = dict(
            num_frames=100, fps=FPS, duration_seconds=2.0,
            root_speed_mean=0.1, root_speed_max=0.2, root_vertical_range=0.05,
            root_vertical_speed_rms=0.1, root_angular_speed_mean=0.1,
            root_angular_speed_max=0.2, joint_speed_rms=0.5, joint_speed_q90=0.8,
            joint_acceleration_rms=1.0, joint_jerk_rms=2.0, joint_range_mean=0.5,
            joint_range_max=1.0, spectral_complexity=0.3, high_frequency_ratio=0.05,
            ballistic_fraction=0.0,
        )
        base.update(overrides)
        return PF.MotionFeatures(**base)

    def test_flight_reads_aerial(self):
        assert PF.contact_regime_proxy(self.make(ballistic_fraction=0.4)) == "aerial"

    def test_fast_locomotion_reads_dynamic(self):
        assert PF.contact_regime_proxy(self.make(root_speed_mean=1.5)) == "dynamic"

    def test_fast_joints_read_dynamic(self):
        assert PF.contact_regime_proxy(self.make(joint_speed_rms=3.0)) == "dynamic"

    def test_quiet_motion_reads_steady(self):
        assert PF.contact_regime_proxy(self.make()) == "steady"

    def test_flight_outranks_speed(self):
        assert PF.contact_regime_proxy(
            self.make(ballistic_fraction=0.4, root_speed_mean=1.5)
        ) == "aerial"


class TestSerialization:
    def test_proxy_features_are_namespaced(self):
        features = PF.compute_motion_features(sine(1.0), fps=FPS).as_proxy_features()
        assert all(k.startswith("motion_") for k in features)
        assert "motion_joint_speed_rms" in features

    def test_all_values_are_floats(self):
        for value in PF.compute_motion_features(sine(1.0), fps=FPS).to_dict().values():
            assert isinstance(value, float)
