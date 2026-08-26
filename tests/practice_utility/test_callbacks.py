"""Tests for the trainer callbacks that wire measurement into a SONIC run.

The guarantee under test is the same one the whole layer rests on: **disabled
means untouched**. A disabled callback must not patch, wrap, or observe
anything, so that a "research code off" baseline really is the native run.
"""

# Ruff's force-sort setting conflicts with the repository's isort profile.
# ruff: noqa: I001

import json
from pathlib import Path

import pytest
import torch

from gear_sonic.research.practice_utility import callbacks as C
from gear_sonic.research.practice_utility import dose_plan as DP
from gear_sonic.research.practice_utility.schema import ContextKey, motion_hash
from scripts.practice_utility import create_passive_dose_plan as DOSE_CLI

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
        self.adp_sampling_active_prob = torch.full(
            (len(rows),), 1.0 / len(rows), dtype=torch.float64
        )
        self.adp_samp_failure_rate_raw = torch.linspace(0.0, 1.0, len(rows), dtype=torch.float64)
        self.adp_samp_num_episodes = torch.full((len(rows),), 10.0)
        self.adp_samp_num_failures = torch.full((len(rows),), 2.0)
        self.uniform_sampling_rate = 0.1
        self.adp_samp_failure_rate_max_over_mean = 200.0
        self._motion_data_keys = self._keys
        self._motion_fps = torch.full((NUM_MOTIONS,), 30.0)  # per-motion tensor upstream
        self._sim_fps = 50.0
        self.target_fps = 50.0
        self._device = torch.device("cpu")

        frames_per_motion = BINS_PER_MOTION * BIN_SIZE
        self.adp_samp_length_starts = torch.arange(NUM_MOTIONS) * frames_per_motion
        self.adp_samp_frame_to_bin = torch.arange(NUM_MOTIONS * frames_per_motion) // BIN_SIZE
        self.update_calls = 0
        self.adaptive_update_calls = 0
        self.sample_calls = 0

    def update_adaptive_sampling_probabilities(self):
        self.update_calls += 1
        n = self.adp_samp_active_motion_bins.numel()
        self.adp_sampling_active_prob = torch.full((n,), 1.0 / n, dtype=torch.float64)

    def update_adaptive_sampling(self, failure, motion_ids, motion_time_steps):
        self.adaptive_update_calls += 1
        self.last_adaptive_update = (
            failure.clone(),
            motion_ids.clone(),
            motion_time_steps.clone(),
        )
        return "native-update-result"

    def sample_motion_ids_and_time_steps(self, n):
        self.sample_calls += 1
        motion_ids = torch.zeros(n, dtype=torch.long)
        time_steps = torch.full((n,), 60, dtype=torch.long)  # bin 1 of motion 0
        return motion_ids, time_steps

    def get_motion_ids_in_dataset(self, motion_ids):
        return motion_ids

    def get_env_state_dict(self):
        return {"motion_lib": {"stub": True}}


class FakeEnv:
    def __init__(self):
        self._motion_lib = FakeMotionLib()
        self.num_envs = 2
        self.motion_command = type("FakeMotionCommand", (), {})()
        self.motion_command.motion_ids = torch.tensor([0, 0])
        self.motion_command.motion_start_time_steps = torch.tensor([0, 0])
        self.motion_command.time_steps = torch.tensor([10, 60])
        self.return_dones = torch.tensor([False, True])
        self.return_time_outs = torch.tensor([False, False])
        self.next_motion_ids = None
        self.next_motion_start_time_steps = None
        self.next_time_steps = None
        self.step_calls = 0
        self.last_step_result = None

    def step(self, actions=None):
        self.step_calls += 1
        if self.next_motion_ids is not None:
            self.motion_command.motion_ids = self.next_motion_ids.clone()
        if self.next_motion_start_time_steps is not None:
            self.motion_command.motion_start_time_steps = self.next_motion_start_time_steps.clone()
        if self.next_time_steps is not None:
            self.motion_command.time_steps = self.next_time_steps.clone()
        self.last_step_result = (
            {"actions": actions},
            torch.zeros(self.num_envs),
            self.return_dones.clone(),
            {"time_outs": self.return_time_outs.clone()},
        )
        return self.last_step_result

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


def write_dose_plan(tmp_path, *contexts):
    launcher = Path(DOSE_CLI.__file__).resolve()
    rows = [
        {"context_id": context.context_id, "context": context.to_dict()} for context in contexts
    ]
    payload = {
        "kind": DP.PASSIVE_DOSE_PLAN_KIND,
        "schema_version": DP.PASSIVE_DOSE_PLAN_SCHEMA_VERSION,
        "campaign_id": "screen_test",
        "manifest_sha256": "a" * 64,
        "manifest_file_sha256": "b" * 64,
        "provenance": {
            "source_manifest": {
                "path": str(tmp_path / "manifest.json"),
                "logical_sha256": "a" * 64,
                "file_sha256": "b" * 64,
            },
            "git": {"sha": "c" * 40, "status_short": []},
            "launcher": {
                "path": str(launcher),
                "sha256": DP.file_sha256(launcher),
            },
        },
        "control_strategy": "shared_per_stage_seed",
        "measurement_hook": DP.PASSIVE_DOSE_HOOK,
        "kernel": {
            "radius_bins": 0,
            "reference_bin_size_frames": 50,
            "sigma_frames": 50.0,
            "membership_normalization": "peak_equals_one",
        },
        "contexts_per_stage": {"late": rows},
    }
    payload["dose_plan_sha256"] = DP.logical_sha256(payload)
    path = tmp_path / "dose_plan.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def claim_callback(tmp_path, plan_path, **overrides):
    params = dict(
        enabled=True,
        role="control",
        pair_id="screen_test_late_s11",
        branch_id="screen_test_late_s11_control",
        kernel_radius_bins=0,
        dose_report_dir=str(tmp_path / "dose"),
        claim_mode=True,
        dose_plan_path=str(plan_path),
        dose_plan_sha256=DP.file_sha256(plan_path),
        dose_plan_stage="late",
        dose_report_horizons={"H_s": 1},
        dose_origin_global_step=0,
        dose_num_steps_per_iteration=1,
        dose_num_envs=2,
        dose_lineage={
            "campaign_id": "screen_test",
            "manifest_sha256": "a" * 64,
            "manifest_file_sha256": "b" * 64,
            "source_commit": "c" * 40,
        },
    )
    params.update(overrides)
    return C.PracticeContextCallback(**params)


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
        assert env._motion_lib.update_calls == 1  # not bypassed, only post-processed

    def test_intervention_changes_the_distribution(self, env):
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for().to_dict(),
            epsilon=0.20,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        probs = env._motion_lib.adp_sampling_active_prob
        assert float(probs[1]) > 1.0 / 12
        assert float(probs.sum()) == pytest.approx(1.0)

    def test_epsilon_zero_arms_but_does_not_change(self, env):
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for().to_dict(),
            epsilon=0.0,
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
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for().to_dict(),
            epsilon=0.5,
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
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for().to_dict(),
            epsilon=0.10,
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
        assert report.per_bin_drawn == {1: 8.0}  # frame 60 -> bin 1

    def test_kernel_mass_accumulates_for_the_target(self, env):
        callback = self.install(env, kernel_radius_bins=0)
        env._motion_lib.sample_motion_ids_and_time_steps(8)
        assert callback.adapter.get_exact_dose_report().drawn_kernel_mass == pytest.approx(8.0)

    def test_native_sampling_still_returns_its_values(self, env):
        self.install(env)
        motion_ids, time_steps = env._motion_lib.sample_motion_ids_and_time_steps(4)
        assert motion_ids.shape == (4,) and int(time_steps[0]) == 60
        assert env._motion_lib.sample_calls == 1

    def test_completed_step_wrapper_preserves_native_call_and_return(self, env):
        callback = self.install(env)
        result = env.step({"actions": "unchanged"})
        report = callback.adapter.get_exact_dose_report()
        assert result is env.last_step_result
        assert env.step_calls == 1
        assert report.per_bin_completed == {0: 1.0, 1: 1.0}
        assert report.completed_env_steps == 2.0
        assert report.completion_hook_calls == 1
        assert report.termination_observations == 2
        assert report.early_terminations == 1

    def test_completed_step_observation_does_not_change_distribution(self, env):
        callback = self.install(env, epsilon=0.0)
        before = env._motion_lib.adp_sampling_active_prob.clone()
        env.return_dones = torch.tensor([False, False])
        env.step(None)
        assert torch.equal(env._motion_lib.adp_sampling_active_prob, before)
        assert callback.adapter.get_exact_dose_report().completed_env_steps == 2.0

    def test_reset_to_different_bin_credits_the_pre_transition_context(self, env):
        callback = self.install(env)
        env.next_time_steps = torch.tensor([110, 160])
        env.return_dones = torch.tensor([True, True])
        env.return_time_outs = torch.tensor([False, True])

        env.step(None)

        report = callback.adapter.get_exact_dose_report()
        assert report.per_bin_completed == {0: 1.0, 1: 1.0}
        assert report.early_terminations == 1
        assert torch.equal(env.motion_command.time_steps, torch.tensor([110, 160]))

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

    def test_snapshot_records_its_timeline_rate(self, env, tmp_path):
        path = tmp_path / "snapshot.json"
        callback = self.install(
            env,
            snapshot_path=str(path),
            snapshot_timeline_fps=50.0,
        )
        callback.write_snapshot(global_step=12)
        assert json.loads(path.read_text())["snapshot_timeline_fps"] == 50.0

    def test_snapshot_refuses_self_report_that_differs_from_live_target_fps(self, env, tmp_path):
        env._motion_lib.target_fps = 60.0
        with pytest.raises(RuntimeError, match="live motion library target_fps differs"):
            self.install(
                env,
                snapshot_path=str(tmp_path / "snapshot.json"),
                snapshot_timeline_fps=50.0,
            )


class TestPassiveClaimReceipts:
    def test_shared_control_emits_every_planned_context_at_exact_horizon(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(0), context_for(1))
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        env.step(None)
        callback.on_step_end(None, FakeState(1), None, env=env)

        reports = list((tmp_path / "dose").glob("dose_*_H_s_step000001.json"))
        assert len(reports) == 1
        payload = json.loads(reports[0].read_text())
        assert payload["schema_version"] == 2
        assert payload["status"] == "complete"
        assert payload["valid_for_claim"] is True
        assert payload["completed_env_steps"] == payload["expected_env_steps"] == 2.0
        assert payload["completion_hook_calls"] == payload["expected_completion_hook_calls"] == 1
        assert payload["termination_observations"] == 2
        assert len(payload["context_doses"]) == 2
        assert {row["context_id"] for row in payload["context_doses"]} == {
            context_for(0).context_id,
            context_for(1).context_id,
        }
        assert payload["passive_dose_plan"]["file_sha256"] == DP.file_sha256(plan)
        assert not list((tmp_path / "dose").glob("*.partial"))

    def test_claim_mode_fails_at_receipt_on_stale_global_registry(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        env._motion_lib.adp_samp_bins[0, 1] += 1
        env.return_dones = torch.tensor([False, False])
        env.step(None)
        with pytest.raises(RuntimeError, match="registry changed"):
            callback.on_step_end(None, FakeState(1), None, env=env)
        assert env.step_calls == 1
        assert callback.adapter.get_exact_dose_report().dropped_completion_batches == 0
        blocked = next((tmp_path / "dose").glob("dose_*_H_s_step000001.json"))
        assert json.loads(blocked.read_text())["status"] == "blocked"
        callback.uninstall()

    def test_one_registry_hash_serves_24_receipt_projections(self, env, tmp_path, monkeypatch):
        base = context_for(1).to_dict()
        contexts = [
            ContextKey.from_dict({**base, "encoder_mode": f"g1_projection_{index:02d}"})
            for index in range(24)
        ]
        plan = write_dose_plan(tmp_path, *contexts)
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        assert callback.adapter is not None
        calls = 0
        original = callback.adapter.dose_registry_sha256

        def counted():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(callback.adapter, "dose_registry_sha256", counted)
        env.return_dones = torch.tensor([False, False])
        env.step(None)
        callback.on_step_end(None, FakeState(1), None, env=env)
        report = next((tmp_path / "dose").glob("dose_*_H_s_step000001.json"))
        assert len(json.loads(report.read_text())["context_doses"]) == 24
        assert calls == 1

    def test_claim_mode_fails_after_native_step_on_dropped_batch(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        env.return_dones = torch.tensor([False])
        env.return_time_outs = torch.tensor([False])
        with pytest.raises(RuntimeError, match="captured transition contexts"):
            env.step(None)
        assert env.step_calls == 1
        assert callback.adapter.get_exact_dose_report().dropped_completion_batches == 1
        callback.uninstall()

    def test_claim_receipt_is_exclusive_and_preserves_first_evidence(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        env.step(None)
        path = Path(callback.write_dose_report(1, horizon_label="H_s"))
        original = path.read_bytes()

        with pytest.raises(FileExistsError):
            callback.write_dose_report(1, horizon_label="H_s")

        assert path.read_bytes() == original
        assert not list(path.parent.glob(".*.partial"))

    def test_claim_install_rejects_live_bin_size_different_from_plan(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        env._motion_lib.adp_samp_bin_size = 25
        callback = claim_callback(tmp_path, plan)
        with pytest.raises(ValueError, match="live adp_samp_bin_size"):
            callback.on_train_begin(None, FakeState(0), None, env=env)

    def test_nonclaim_plan_receipt_is_never_marked_valid_for_claim(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        callback = C.PracticeContextCallback(
            enabled=True,
            role="control",
            pair_id="p0",
            dose_report_dir=str(tmp_path / "dose"),
            dose_plan_path=str(plan),
            dose_plan_sha256=DP.file_sha256(plan),
            dose_plan_stage="late",
            kernel_radius_bins=0,
        )
        callback.on_train_begin(None, FakeState(0), None, env=env)
        env.step(None)
        payload = json.loads(Path(callback.write_dose_report(1)).read_text())
        assert payload["valid_for_claim"] is False

    def test_claim_receipt_without_exact_horizon_is_blocked_not_valid(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        env.step(None)
        with pytest.raises(RuntimeError, match="not tied to an exact horizon"):
            callback.write_dose_report(1)
        path = next((tmp_path / "dose").glob("dose_*_step000001.json"))
        payload = json.loads(path.read_text())
        assert payload["status"] == "blocked"
        assert payload["valid_for_claim"] is False

    def test_train_end_fails_if_exact_horizon_was_not_seen_and_restores_hooks(self, env, tmp_path):
        plan = write_dose_plan(tmp_path, context_for(1))
        callback = claim_callback(tmp_path, plan)
        callback.on_train_begin(None, FakeState(0), None, env=env)
        with pytest.raises(RuntimeError, match="without exact horizon receipts"):
            callback.on_train_end(None, FakeState(0), None, env=env)
        assert "step" not in vars(env)

    def test_uninstall_restores_a_preexisting_instance_hook_exactly(self, env):
        def instance_hook(actions):
            return actions

        env.step = instance_hook
        callback = C.PracticeContextCallback(enabled=True, role="control", pair_id="p0")
        callback.on_train_begin(None, FakeState(), None, env=env)
        callback.uninstall()
        assert env.step is instance_hook


class TestStaleKernelHandling:
    def test_rearms_after_a_motion_resample(self, env):
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for().to_dict(),
            epsilon=0.10,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        # The resident batch shrinks, as after load_motions_for_training.
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(8)
        callback.on_step_end(None, FakeState(1), None, env=env)
        assert callback.adapter._kernel_weights.numel() == 8
        assert callback._armed is True

    def test_disarms_when_the_context_leaves_the_batch(self, env):
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for(bin_index=1).to_dict(),
            epsilon=0.10,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        # Only motion 2's bins remain resident; the context is gone.
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(8, 12)
        callback.on_step_end(None, FakeState(1), None, env=env)
        assert callback._armed is False
        assert callback.adapter.override_active is False

    def test_falls_back_to_the_native_distribution_when_disarmed(self, env):
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for(bin_index=1).to_dict(),
            epsilon=0.10,
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(8, 12)
        callback.on_step_end(None, FakeState(1), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        probs = env._motion_lib.adp_sampling_active_prob
        assert torch.allclose(probs, torch.full((4,), 0.25, dtype=torch.float64))


class TestFailSoftArming:
    """A context absent from the resident batch must not kill the branch.

    SONIC keeps only part of the pool loaded and rotates it -- 195 of 512
    motions in a measured run -- so a randomly chosen context is often absent at
    install. Raising there killed a real noise-floor branch and would have killed
    most of a campaign.
    """

    def absent_context(self):
        """A context on motion 2, which the shrunken batch will not contain."""
        key = "motion_02"
        return ContextKey(
            motion_key=key,
            motion_hash=motion_hash(key, BINS_PER_MOTION * BIN_SIZE, 50.0),
            bin_index=1,
            bin_start_frame=BIN_SIZE,
            bin_end_frame=2 * BIN_SIZE,
        )

    def callback_with(self, context):
        return C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context.to_dict(),
            epsilon=0.25,
        )

    def test_install_survives_an_absent_context(self, env):
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(4)  # motion 0 only
        n = 4
        env._motion_lib.adp_sampling_active_prob = torch.full((n,), 1.0 / n, dtype=torch.float64)
        callback = self.callback_with(self.absent_context())
        callback.on_train_begin(None, FakeState(), None, env=env)  # must not raise
        assert callback._armed is False
        assert callback.adapter is not None

    def test_absent_context_leaves_the_distribution_native(self, env):
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(4)
        env._motion_lib.adp_sampling_active_prob = torch.full((4,), 0.25, dtype=torch.float64)
        callback = self.callback_with(self.absent_context())
        callback.on_train_begin(None, FakeState(), None, env=env)
        env._motion_lib.update_adaptive_sampling_probabilities()
        assert torch.allclose(
            env._motion_lib.adp_sampling_active_prob,
            torch.full((4,), 0.25, dtype=torch.float64),
        )

    def test_arms_once_the_context_becomes_resident(self, env):
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(4)
        env._motion_lib.adp_sampling_active_prob = torch.full((4,), 0.25, dtype=torch.float64)
        callback = self.callback_with(self.absent_context())
        callback.on_train_begin(None, FakeState(), None, env=env)
        assert callback._armed is False

        # A resample brings the whole pool back, including motion 2.
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(12)
        env._motion_lib.adp_sampling_active_prob = torch.full((12,), 1 / 12, dtype=torch.float64)
        callback.on_step_end(None, FakeState(3), None, env=env)
        assert callback._armed is True
        assert callback._first_armed_step == 3

    def test_dose_report_records_never_armed(self, env, tmp_path):
        env._motion_lib.adp_samp_active_motion_bins = torch.arange(4)
        env._motion_lib.adp_sampling_active_prob = torch.full((4,), 0.25, dtype=torch.float64)
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=self.absent_context().to_dict(),
            epsilon=0.25,
            dose_report_dir=str(tmp_path),
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        payload = json.loads(open(callback.write_dose_report(5)).read())
        assert payload["never_armed"] is True
        assert payload["armed"] is False
        assert payload["arm_attempts"] >= 1

    def test_dose_report_counts_armed_steps(self, env, tmp_path):
        callback = C.PracticeContextCallback(
            enabled=True,
            role="intervention",
            pair_id="p0",
            context=context_for().to_dict(),
            epsilon=0.1,
            dose_report_dir=str(tmp_path),
        )
        callback.on_train_begin(None, FakeState(), None, env=env)
        for step in range(1, 4):
            callback.on_step_end(None, FakeState(step), None, env=env)
        payload = json.loads(open(callback.write_dose_report(3)).read())
        assert payload["armed_steps"] == 3
        assert payload["never_armed"] is False
        assert payload["first_armed_step"] == 0


class TestCapsuleCallback:
    def make(self, tmp_path, **overrides):
        params = dict(
            enabled=True,
            capsule_dir=str(tmp_path),
            horizons={"H_s": 8, "H_m": 32},
            pair_id="p0",
            role="control",
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

    def test_capsule_carries_the_positional_trainer_state(self, env, tmp_path):
        callback = self.make(tmp_path)
        state = FakeState(8)
        callback.on_step_end(None, state, None, env=env, model=None, optimizer=None)
        payload = torch.load(tmp_path / "p0_control_H_s.capsule.pt", weights_only=False)
        assert payload["trainer_state"]["trainer_state_obj"].global_step == 8

    def test_disabled_saves_nothing(self, env, tmp_path):
        callback = self.make(tmp_path, enabled=False)
        callback.on_step_end(None, FakeState(8), None, env=env, model=None, optimizer=None)
        assert callback.saved == {} and not list(tmp_path.glob("*.pt"))


class TestCapsuleResumeCallback:
    def test_restores_rng_at_matching_step(self, monkeypatch):
        calls = []

        def load(path, restore_rng):
            calls.append((path, restore_rng))
            return {
                "global_step": 10,
                "capsule_sha256": "abc",
                "pair_id": "p0",
            }

        monkeypatch.setattr(C, "load_capsule", load)
        callback = C.PracticeCapsuleResumeCallback(enabled=True, capsule_path="split.capsule.pt")
        callback.on_train_begin(None, FakeState(10), None)
        assert calls == [("split.capsule.pt", True)]
        assert callback.restored["global_step"] == 10

    def test_rejects_step_mismatch(self, monkeypatch):
        monkeypatch.setattr(
            C,
            "load_capsule",
            lambda *args, **kwargs: {"global_step": 10},
        )
        callback = C.PracticeCapsuleResumeCallback(enabled=True, capsule_path="split.capsule.pt")
        with pytest.raises(RuntimeError, match="does not match"):
            callback.on_train_begin(None, FakeState(9), None)

    def test_disabled_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            C,
            "load_capsule",
            lambda *args, **kwargs: pytest.fail("must not load"),
        )
        callback = C.PracticeCapsuleResumeCallback(enabled=False)
        callback.on_train_begin(None, FakeState(0), None)
        assert callback.restored is None


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
        bins = C._global_bins_for(library, torch.tensor([0, 1]), torch.tensor([60, 10]))
        assert bins.tolist() == [1, 4]  # motion 0 bin 1; motion 1 bin 0
