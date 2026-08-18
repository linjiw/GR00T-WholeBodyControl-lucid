"""Tests for the trainer callbacks that wire measurement into a SONIC run.

The guarantee under test is the same one the whole layer rests on: **disabled
means untouched**. A disabled callback must not patch, wrap, or observe
anything, so that a "research code off" baseline really is the native run.
"""

import json

import pytest
import torch

from gear_sonic.research.practice_utility import callbacks as C
from gear_sonic.research.practice_utility.schema import ContextKey, motion_hash

BIN_SIZE = 50
BINS_PER_MOTION = 4
NUM_MOTIONS = 3


class FakeMotionLib:
    """Mirrors the MotionLibBase surface the callbacks touch."""

    def __init__(self):
        self.adp_samp_bin_size = BIN_SIZE
        self.use_adaptive_sampling = True
        rows, self._keys = [], []
        for motion in range(NUM_MOTIONS):
            self._keys.append(f"motion_{motion:02d}")
            for b in range(BINS_PER_MOTION):
                rows.append([motion, b * BIN_SIZE, (b + 1) * BIN_SIZE])
        self.adp_samp_bins = torch.tensor(rows, dtype=torch.long)
        self.adp_samp_num_bins = len(rows)
        self.adp_samp_num_frames = torch.full((NUM_MOTIONS,), BINS_PER_MOTION * BIN_SIZE)
        self.adp_samp_active_motion_bins = torch.arange(len(rows))
        self.adp_sampling_active_prob = torch.full((len(rows),), 1.0 / len(rows),
                                                   dtype=torch.float64)
        self.adp_samp_failure_rate_raw = torch.linspace(0.0, 1.0, len(rows), dtype=torch.float64)
        self.adp_samp_num_episodes = torch.full((len(rows),), 10.0)
        self.adp_samp_num_failures = torch.full((len(rows),), 2.0)
        self.uniform_sampling_rate = 0.1
        self.adp_samp_failure_rate_max_over_mean = 200.0
        self._motion_data_keys = self._keys
        self._motion_fps = torch.full((NUM_MOTIONS,), 30.0)   # per-motion tensor upstream
        self._sim_fps = 50.0
        self._device = torch.device("cpu")

        frames_per_motion = BINS_PER_MOTION * BIN_SIZE
        self.adp_samp_length_starts = torch.arange(NUM_MOTIONS) * frames_per_motion
        self.adp_samp_frame_to_bin = torch.arange(
            NUM_MOTIONS * frames_per_motion
        ) // BIN_SIZE
        self.update_calls = 0
        self.sample_calls = 0

    def update_adaptive_sampling_probabilities(self):
        self.update_calls += 1
        n = self.adp_samp_active_motion_bins.numel()
        self.adp_sampling_active_prob = torch.full((n,), 1.0 / n, dtype=torch.float64)

    def sample_motion_ids_and_time_steps(self, n):
        self.sample_calls += 1
        motion_ids = torch.zeros(n, dtype=torch.long)
        time_steps = torch.full((n,), 60, dtype=torch.long)   # bin 1 of motion 0
        return motion_ids, time_steps

    def get_motion_ids_in_dataset(self, motion_ids):
        return motion_ids

    def get_env_state_dict(self):
        return {"motion_lib": {"stub": True}}


class FakeEnv:
    def __init__(self):
        self._motion_lib = FakeMotionLib()

    def get_env_state_dict(self):
        return {"motion_lib": {"stub": True}}


class FakeState:
    def __init__(self, global_step=0):
        self.global_step = global_step


def context_for(bin_index=1):
    key = "motion_00"
    return ContextKey(
        motion_key=key,
        motion_hash=motion_hash(key, BINS_PER_MOTION * BIN_SIZE, 50.0),
        bin_index=bin_index,
        bin_start_frame=bin_index * BIN_SIZE,
        bin_end_frame=(bin_index + 1) * BIN_SIZE,
    )


@pytest.fixture
def env():
    return FakeEnv()


class TestDisabledIsUntouched:
    def test_no_adapter_installed(self, env):
        callback = C.PracticeContextCallback(enabled=False)
        callback.on_train_begin(None, FakeState(), None, env=env)
        assert callback.adapter is None

    def test_native_methods_are_not_wrapped(self, env):
        callback = C.PracticeContextCallback(enabled=False)
        callback.on_train_begin(None, FakeState(), None, env=env)
        callback.on_step_end(None, FakeState(1), None, env=env)
        assert C.PracticeContextCallback.is_patched(env._motion_lib) is False

    def test_sampling_is_not_observed(self, env):
        callback = C.PracticeContextCallback(enabled=False)
        callback.on_step_end(None, FakeState(1), None, env=env)
        env._motion_lib.sample_motion_ids_and_time_steps(8)
        assert callback.adapter is None


class TestInstallation:
    def test_control_branch_installs_without_changing_the_distribution(self, env):
        before = env._motion_lib.adp_sampling_active_prob.clone()
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        assert torch.allclose(env._motion_lib.adp_sampling_active_prob, before)

    def test_native_computation_still_runs(self, env):
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        assert env._motion_lib.update_calls == 1   # not bypassed, only post-processed

    def test_intervention_changes_the_distribution(self, env):
        callback = C.PracticeContextCallback(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for().to_dict(), epsilon=0.20,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        probs = env._motion_lib.adp_sampling_active_prob
        assert float(probs[1]) > 1.0 / 12
        assert float(probs.sum()) == pytest.approx(1.0)

    def test_epsilon_zero_arms_but_does_not_change(self, env):
        callback = C.PracticeContextCallback(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for().to_dict(), epsilon=0.0,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        assert callback._armed is True
        assert torch.allclose(
            env._motion_lib.adp_sampling_active_prob,
            torch.full((12,), 1.0 / 12, dtype=torch.float64),
        )

    def test_uninstall_removes_the_patch_entirely(self, env):
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        callback.on_train_begin(None, FakeState(), None, env=env)
        assert C.PracticeContextCallback.is_patched(env._motion_lib) is True
        callback.uninstall()
        assert C.PracticeContextCallback.is_patched(env._motion_lib) is False
        assert "sample_motion_ids_and_time_steps" not in vars(env._motion_lib)

    def test_native_behaviour_returns_after_uninstall(self, env):
        callback = C.PracticeContextCallback(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for().to_dict(), epsilon=0.5,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        callback.uninstall()
        env._motion_lib.update_adaptive_sampling_probabilities()
        assert torch.allclose(
            env._motion_lib.adp_sampling_active_prob,
            torch.full((12,), 1.0 / 12, dtype=torch.float64),
        )

    def test_reinstall_does_not_double_wrap(self, env):
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        callback.on_train_begin(None, FakeState(), None, env=env)
        callback.uninstall()
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        assert env._motion_lib.update_calls == 1

    def test_train_end_uninstalls(self, env):
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        callback.on_train_begin(None, FakeState(), None, env=env)
        callback.on_train_end(None, FakeState(10), None, env=env)
        assert C.PracticeContextCallback.is_patched(env._motion_lib) is False

    def test_missing_environment_raises(self):
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        with pytest.raises(RuntimeError, match="no motion library"):
            callback.on_train_begin(None, FakeState(), None, env=None)

    def test_adaptive_sampling_off_raises(self, env):
        env._motion_lib.use_adaptive_sampling = False
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        with pytest.raises(RuntimeError, match="adaptive sampling is disabled"):
            callback.on_train_begin(None, FakeState(), None, env=env)

    def test_intervention_without_context_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="requires a context"):
            C.PracticeContextCallback(enabled=True, role="intervention")


class TestDoseRecording:
    def install(self, env, **overrides):
        params = dict(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for().to_dict(), epsilon=0.10,
        )
        params.update(overrides)
        callback = C.PracticeContextCallback(**params)
        callback.on_train_begin(None, FakeState(), None, env=env)
        return callback

    def test_sampling_records_draws(self, env):
        callback = self.install(env)
        env._motion_lib.sample_motion_ids_and_time_steps(16)
        assert callback.adapter.get_exact_dose_report().drawn_episodes == 16.0

    def test_draws_land_on_the_expected_bin(self, env):
        callback = self.install(env)
        env._motion_lib.sample_motion_ids_and_time_steps(8)
        report = callback.adapter.get_exact_dose_report()
        assert report.per_bin_drawn == {1: 8.0}      # frame 60 -> bin 1

    def test_kernel_mass_accumulates_for_the_target(self, env):
        callback = self.install(env, kernel_radius_bins=0)
        env._motion_lib.sample_motion_ids_and_time_steps(8)
        assert callback.adapter.get_exact_dose_report().drawn_kernel_mass == pytest.approx(8.0)

    def test_native_sampling_still_returns_its_values(self, env):
        self.install(env)
        motion_ids, time_steps = env._motion_lib.sample_motion_ids_and_time_steps(4)
        assert motion_ids.shape == (4,) and int(time_steps[0]) == 60
        assert env._motion_lib.sample_calls == 1

    def test_writes_a_dose_report(self, env, tmp_path):
        callback = self.install(env, dose_report_dir=str(tmp_path))
        env._motion_lib.sample_motion_ids_and_time_steps(8)
        path = callback.write_dose_report(global_step=100)
        payload = json.loads(open(path).read())
        assert payload["role"] == "intervention"
        assert payload["drawn_episodes"] == 8.0
        assert payload["epsilon"] == 0.10
        assert payload["armed"] is True

    def test_periodic_reports_are_written(self, env, tmp_path):
        callback = self.install(env, dose_report_dir=str(tmp_path), dose_report_frequency=5)
        env._motion_lib.sample_motion_ids_and_time_steps(4)
        callback.on_step_end(None, FakeState(5), None, env=env)
        assert list(tmp_path.glob("dose_*.json"))

    def test_no_report_without_a_directory(self, env):
        callback = self.install(env)
        assert callback.write_dose_report(10) is None


class TestStaleKernelHandling:
    def test_rearms_after_a_motion_resample(self, env):
        callback = C.PracticeContextCallback(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for().to_dict(), epsilon=0.10,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        # The resident batch shrinks, as after load_motions_for_training.
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(8)
        callback.on_step_end(None, FakeState(1), None, env=env)
        assert callback.adapter._kernel_weights.numel() == 8
        assert callback._armed is True

    def test_disarms_when_the_context_leaves_the_batch(self, env):
        callback = C.PracticeContextCallback(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for(bin_index=1).to_dict(), epsilon=0.10,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        # Only motion 2's bins remain resident; the context is gone.
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(8, 12)
        callback.on_step_end(None, FakeState(1), None, env=env)
        assert callback._armed is False
        assert callback.adapter.override_active is False

    def test_falls_back_to_the_native_distribution_when_disarmed(self, env):
        callback = C.PracticeContextCallback(
            enabled=True, role="intervention", pair_id="p0",
            context=context_for(bin_index=1).to_dict(), epsilon=0.10,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(8, 12)
        callback.on_step_end(None, FakeState(1), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        probs = env._motion_lib.adp_sampling_active_prob
        assert torch.allclose(probs, torch.full((4,), 0.25, dtype=torch.float64))


class TestCapsuleCallback:
    def make(self, tmp_path, **overrides):
        params = dict(
            enabled=True, capsule_dir=str(tmp_path),
            horizons={"H_s": 8, "H_m": 32}, pair_id="p0", role="control",
        )
        params.update(overrides)
        return C.PracticeCapsuleCallback(**params)

    def test_saves_at_the_horizon(self, env, tmp_path):
        callback = self.make(tmp_path)
        callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        assert "H_s" in callback.saved
        assert (tmp_path / "p0_control_H_s.capsule.pt").exists()

    def test_does_not_save_off_horizon(self, env, tmp_path):
        callback = self.make(tmp_path)
        callback.on_step_end(None, FakeState(7), None, env=env, model=None, optimizer=None)
        assert callback.saved == {}

    def test_saves_each_horizon_once(self, env, tmp_path):
        callback = self.make(tmp_path)
        for _ in range(3):
            callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        assert len(list(tmp_path.glob("*.capsule.pt"))) == 1

    def test_saves_multiple_horizons(self, env, tmp_path):
        callback = self.make(tmp_path)
        callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        callback.on_step_end(None, FakeState(32), None, env=env, model=None, optimizer=None)
        assert set(callback.saved) == {"H_s", "H_m"}

    def test_capsule_carries_the_sampler_counters(self, env, tmp_path):
        callback = self.make(tmp_path)
        callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        payload = torch.load(tmp_path / "p0_control_H_s.capsule.pt", weights_only=False)
        assert "adp_samp_num_episodes" in payload["native_sampler_state"]

    def test_capsule_carries_rng_state(self, env, tmp_path):
        callback = self.make(tmp_path)
        callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        payload = torch.load(tmp_path / "p0_control_H_s.capsule.pt", weights_only=False)
        assert "torch_cpu_state" in payload["rng"]

    def test_disabled_saves_nothing(self, env, tmp_path):
        callback = self.make(tmp_path, enabled=False)
        callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        assert callback.saved == {} and not list(tmp_path.glob("*.pt"))


class TestHelpers:
    def test_finds_motion_lib_through_private_attribute(self, env):
        assert C._motion_lib_of(env) is env._motion_lib

    def test_finds_motion_lib_through_command(self):
        class Command:
            motion_lib = "lib"

        class Env:
            motion_command = Command()

        assert C._motion_lib_of(Env()) == "lib"

    def test_returns_none_without_an_env(self):
        assert C._motion_lib_of(None) is None

    def test_maps_time_steps_to_global_bins(self, env):
        library = env._motion_lib
        bins = C._global_bins_for(
            library, torch.tensor([0, 1]), torch.tensor([60, 10])
        )
        assert bins.tolist() == [1, 4]   # motion 0 bin 1; motion 1 bin 0
