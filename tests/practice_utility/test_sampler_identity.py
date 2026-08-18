"""Identity and dose-accounting tests for :mod:`sampler_adapter`.

The central guarantee: **with no override armed, the adapter is a
pass-through.** If that ever breaks, every "research code disabled" baseline in
the program is silently contaminated, so it is tested from several angles.

The fake motion library mirrors the attribute contract of
``gear_sonic.utils.motion_lib.motion_lib_base.MotionLibBase``.
``TestUpstreamContract`` guards against drift between the fake and upstream.
"""

import pytest
import torch

from gear_sonic.research.practice_utility import intervention as I
from gear_sonic.research.practice_utility.sampler_adapter import PracticeSamplerAdapter
from gear_sonic.research.practice_utility.schema import MotionPoolManifest, motion_hash

BIN_SIZE = 50


class FakeMotionLib:
    """Three motions of four 50-frame bins each; all twelve bins resident."""

    def __init__(self, num_motions=3, bins_per_motion=4, active_bin_ids=None, probs=None):
        self.adp_samp_bin_size = BIN_SIZE
        rows, self._keys = [], []
        for motion in range(num_motions):
            self._keys.append(f"motion_{motion:02d}")
            for b in range(bins_per_motion):
                rows.append([motion, b * BIN_SIZE, (b + 1) * BIN_SIZE])
        self.adp_samp_bins = torch.tensor(rows, dtype=torch.long)
        self.adp_samp_num_bins = len(rows)
        self.adp_samp_num_frames = torch.full(
            (num_motions,), bins_per_motion * BIN_SIZE, dtype=torch.long
        )

        active = torch.arange(len(rows)) if active_bin_ids is None else torch.tensor(active_bin_ids)
        self.adp_samp_active_motion_bins = active
        n = active.numel()
        p = torch.full((n,), 1.0 / n, dtype=torch.float64) if probs is None else torch.tensor(
            probs, dtype=torch.float64
        )
        self.adp_sampling_active_prob = p / p.sum()

        self.adp_samp_failure_rate_raw = torch.linspace(0.0, 1.0, len(rows), dtype=torch.float64)
        self.adp_samp_num_episodes = torch.full((len(rows),), 10.0, dtype=torch.float64)
        self.adp_samp_num_failures = torch.full((len(rows),), 2.0, dtype=torch.float64)
        self.uniform_sampling_rate = 0.1
        self.adp_samp_failure_rate_max_over_mean = 200.0
        self._motion_data_keys = self._keys
        # Upstream stores a per-motion tensor of SOURCE clip rates here, indexed
        # by batch-local motion id -- not a scalar. The fake mirrors that.
        self._motion_fps = torch.full((num_motions,), 30.0)
        self._sim_fps = 50.0
        self._device = torch.device("cpu")


@pytest.fixture
def lib():
    return FakeMotionLib()


@pytest.fixture
def adapter(lib):
    return PracticeSamplerAdapter(lib, branch_id="pair0_control", role="control")


class TestPassThroughIdentity:
    def test_no_override_by_default(self, adapter):
        assert adapter.override_active is False

    def test_apply_returns_the_same_object(self, adapter, lib):
        out = adapter.apply(lib.adp_sampling_active_prob)
        assert out is lib.adp_sampling_active_prob

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_apply_is_bitwise_identity(self, adapter, dtype):
        prob = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=dtype)
        assert torch.equal(adapter.apply(prob), prob)

    def test_identity_holds_for_a_skewed_distribution(self, adapter):
        prob = torch.tensor([0.97, 0.01, 0.01, 0.01], dtype=torch.float64)
        assert torch.equal(adapter.apply(prob), prob)

    def test_clear_override_restores_identity(self, adapter, lib):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.5)
        assert adapter.override_active is True
        adapter.clear_override()
        assert adapter.override_active is False
        assert adapter.apply(lib.adp_sampling_active_prob) is lib.adp_sampling_active_prob


class TestEpsilonZero:
    """epsilon=0 must traverse the override path yet change nothing."""

    def test_arms_the_override(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.0)
        assert adapter.override_active is True

    def test_leaves_the_distribution_unchanged(self, adapter, lib):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.0)
        out = adapter.apply(lib.adp_sampling_active_prob)
        assert torch.allclose(out, lib.adp_sampling_active_prob, atol=1e-12)

    def test_preserves_dtype_and_device(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.0)
        prob = torch.full((12,), 1.0 / 12, dtype=torch.float32)
        out = adapter.apply(prob)
        assert out.dtype == prob.dtype and out.device == prob.device


class TestInterventionShape:
    def test_adds_mass_to_the_target_bin(self, adapter, lib):
        context = adapter.context_for_bin(1)
        before = lib.adp_sampling_active_prob.clone()
        adapter.set_intervention(context, epsilon=0.10)
        after = adapter.apply(before)
        assert float(after[1]) > float(before[1])

    def test_remains_a_distribution(self, adapter, lib):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.10)
        after = adapter.apply(lib.adp_sampling_active_prob)
        assert float(after.sum()) == pytest.approx(1.0)
        assert bool((after >= 0).all())

    def test_never_leaks_into_another_motion(self, adapter, lib):
        # Bin 3 is the last bin of motion 0; bin 4 is the first of motion 1.
        adapter.set_intervention(adapter.context_for_bin(3), epsilon=0.5)
        before = lib.adp_sampling_active_prob.clone()
        after = adapter.apply(before)
        assert float(after[4]) < float(before[4])  # only diluted, never boosted

    def test_spreads_over_neighbours_within_the_clip(self, adapter, lib):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.5, kernel_radius=1)
        before = lib.adp_sampling_active_prob.clone()
        after = adapter.apply(before)
        assert float(after[0]) > float(before[0])
        assert float(after[2]) > float(before[2])
        assert float(after[3]) < float(before[3])  # outside radius: diluted

    def test_radius_zero_targets_a_single_bin(self, adapter, lib):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.5, kernel_radius=0)
        before = lib.adp_sampling_active_prob.clone()
        after = adapter.apply(before)
        assert float(after[0]) < float(before[0])
        assert float(after[1]) > float(before[1])

    def test_added_mass_totals_epsilon(self, adapter, lib):
        eps = 0.10
        before = lib.adp_sampling_active_prob.clone()
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=eps)
        after = adapter.apply(before)
        gained = float((after - before).clamp(min=0).sum())
        assert gained == pytest.approx(eps * (1.0 - float(before[0:3].sum())), abs=0.02)

    def test_rejects_epsilon_out_of_range(self, adapter):
        with pytest.raises(ValueError, match="epsilon"):
            adapter.set_intervention(adapter.context_for_bin(1), epsilon=1.5)

    def test_rejects_context_absent_from_the_resident_batch(self, lib):
        partial = FakeMotionLib(active_bin_ids=[0, 1, 2, 3])   # motion 0 only
        full = PracticeSamplerAdapter(FakeMotionLib())
        adapter = PracticeSamplerAdapter(partial)
        with pytest.raises(ValueError, match="not resident"):
            adapter.set_intervention(full.context_for_bin(8), epsilon=0.1)  # motion 2


class TestResidualDistribution:
    def test_installs_and_applies(self, adapter, lib):
        target = torch.full((12,), 1.0 / 12, dtype=torch.float64)
        target[0] += 0.05
        target[1] -= 0.05
        adapter.set_residual_distribution(target, manifest_id="resid_v1")
        assert torch.allclose(adapter.apply(lib.adp_sampling_active_prob), target)

    def test_rejects_wrong_length(self, adapter):
        with pytest.raises(ValueError, match="entries"):
            adapter.set_residual_distribution(torch.full((5,), 0.2), manifest_id="bad")

    def test_rejects_unnormalized(self, adapter):
        with pytest.raises(ValueError, match="sums to"):
            adapter.set_residual_distribution(torch.full((12,), 0.5), manifest_id="bad")

    def test_mutually_exclusive_with_intervention(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1)
        with pytest.raises(RuntimeError, match="intervention is armed"):
            adapter.set_residual_distribution(torch.full((12,), 1 / 12), manifest_id="r")

    def test_intervention_rejected_while_residual_set(self, adapter):
        adapter.set_residual_distribution(torch.full((12,), 1 / 12), manifest_id="r")
        with pytest.raises(RuntimeError, match="residual distribution is set"):
            adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1)


class TestContextKeys:
    def test_fields_match_the_bin_table(self, adapter):
        context = adapter.context_for_bin(5)   # motion 1, bin 1
        assert context.motion_key == "motion_01"
        assert (context.bin_start_frame, context.bin_end_frame) == (50, 100)
        assert context.bin_index == 1

    def test_ids_are_unique_across_bins(self, adapter):
        ids = {adapter.context_for_bin(b).context_id for b in range(12)}
        assert len(ids) == 12

    def test_manifest_hash_wins_over_derived_hash(self, lib):
        manifest = MotionPoolManifest(
            manifest_id="pool",
            motion_keys=["motion_00", "motion_01", "motion_02"],
            motion_hashes={k: f"pinned_{k}" for k in ["motion_00", "motion_01", "motion_02"]},
            source_root="/tmp",
        )
        adapter = PracticeSamplerAdapter(lib, manifest=manifest)
        assert adapter.context_for_bin(0).motion_hash == "pinned_motion_00"

    def test_derived_hash_used_without_manifest(self, adapter):
        assert adapter.context_for_bin(0).motion_hash == motion_hash("motion_00", 200, 50.0)

    def test_derived_hash_uses_the_sim_timeline_not_source_fps(self, lib):
        """Bins live on the resampled sim timeline; source clip fps is a
        per-motion tensor on a different indexing scheme."""
        adapter = PracticeSamplerAdapter(lib)
        assert adapter._timeline_fps() == 50.0

    def test_per_motion_fps_tensor_is_not_mistaken_for_a_scalar(self, lib):
        """The bug the first live run found: float(tensor_of_N) raises."""
        del lib._sim_fps
        adapter = PracticeSamplerAdapter(lib)
        assert adapter._timeline_fps() == 50.0          # falls back, does not raise
        assert adapter.context_for_bin(0).motion_hash

    def test_scalar_fps_tensor_is_accepted(self, lib):
        del lib._sim_fps
        lib._motion_fps = torch.tensor([25.0])
        assert PracticeSamplerAdapter(lib)._timeline_fps() == 25.0


class TestSnapshot:
    def test_captures_a_normalized_distribution(self, adapter):
        snapshot = adapter.snapshot_native_distribution(global_step=42)
        assert snapshot.global_step == 42
        assert sum(snapshot.active_prob) == pytest.approx(1.0)
        assert len(snapshot.active_bin_ids) == 12

    def test_is_stable_for_the_same_state(self, adapter):
        first = adapter.snapshot_native_distribution()
        second = adapter.snapshot_native_distribution()
        assert first.distribution_sha256 == second.distribution_sha256

    def test_changes_when_the_distribution_changes(self, adapter, lib):
        before = adapter.snapshot_native_distribution().distribution_sha256
        lib.adp_sampling_active_prob = torch.tensor([0.5] + [0.5 / 11] * 11, dtype=torch.float64)
        assert adapter.snapshot_native_distribution().distribution_sha256 != before

    def test_snapshot_is_not_affected_by_an_armed_override(self, adapter):
        before = adapter.snapshot_native_distribution().distribution_sha256
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.5)
        assert adapter.snapshot_native_distribution().distribution_sha256 == before

    def test_uniform_snapshot_reports_full_diversity(self, adapter):
        assert adapter.snapshot_native_distribution().effective_num_bins == pytest.approx(12.0)


class TestDoseAccounting:
    def test_counts_drawn_episodes(self, adapter):
        adapter.record_draw(torch.tensor([1, 1, 2, 5]))
        report = adapter.get_exact_dose_report()
        assert report.drawn_episodes == 4.0
        assert report.per_bin_drawn[1] == 2.0

    def test_aggregates_draws_per_motion(self, adapter):
        adapter.record_draw(torch.tensor([0, 1, 2, 3, 4]))
        report = adapter.get_exact_dose_report()
        assert report.per_motion_drawn["motion_00"] == 4.0
        assert report.per_motion_drawn["motion_01"] == 1.0

    def test_kernel_mass_only_counted_when_armed(self, adapter):
        adapter.record_draw(torch.tensor([1, 1, 1]))
        assert adapter.get_exact_dose_report().drawn_kernel_mass == 0.0

    def test_kernel_mass_counted_for_target_draws(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1, kernel_radius=0)
        adapter.record_draw(torch.tensor([1, 1, 1]))
        assert adapter.get_exact_dose_report().drawn_kernel_mass == pytest.approx(3.0)

    def test_draws_outside_the_kernel_contribute_nothing(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1, kernel_radius=0)
        adapter.record_draw(torch.tensor([7, 8, 9]))
        assert adapter.get_exact_dose_report().drawn_kernel_mass == pytest.approx(0.0)

    def test_completed_steps_are_the_dose_denominator(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1, kernel_radius=0)
        adapter.record_completion(torch.tensor([1, 1]), torch.tensor([100.0, 40.0]))
        report = adapter.get_exact_dose_report()
        assert report.completed_kernel_steps == pytest.approx(140.0)
        assert report.completed_env_steps == pytest.approx(140.0)

    def test_early_termination_reduces_realized_dose(self, adapter):
        """The point of counting steps rather than episodes."""
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1, kernel_radius=0)
        adapter.record_draw(torch.tensor([1, 1]))
        adapter.record_completion(
            torch.tensor([1, 1]), torch.tensor([200.0, 3.0]), torch.tensor([False, True])
        )
        report = adapter.get_exact_dose_report()
        assert report.drawn_episodes == 2.0          # both episodes were started
        assert report.completed_kernel_steps == 203.0  # but one delivered almost nothing
        assert report.early_terminations == 1

    def test_completion_rejects_mismatched_lengths(self, adapter):
        with pytest.raises(ValueError, match="align"):
            adapter.record_completion(torch.tensor([1, 2]), torch.tensor([10.0]))

    def test_reset_dose_clears_counters_but_keeps_override(self, adapter):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1)
        adapter.record_draw(torch.tensor([1, 1]))
        adapter.reset_dose()
        assert adapter.get_exact_dose_report().drawn_episodes == 0.0
        assert adapter.override_active is True

    def test_report_carries_branch_identity(self, adapter):
        report = adapter.get_exact_dose_report()
        assert report.branch_id == "pair0_control" and report.role == "control"

    def test_stale_kernel_is_detected_after_a_motion_resample(self, adapter, lib):
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.1)
        lib.adp_samp_active_motion_bins = torch.arange(8)   # library reloaded
        with pytest.raises(RuntimeError, match="stale"):
            adapter.record_draw(torch.tensor([1]))


class TestDuplicateResidentBins:
    """SONIC loads resident motions with replacement.

    ``update_adaptive_sampling_motion_frames`` appends a motion's bins once per
    resident copy, so the same global bin id can occupy several positions in
    ``adp_samp_active_motion_bins``. Observed live: 18 of 535 active entries
    were duplicates. Both the kernel and the dose accounting must stay correct
    when that happens.
    """

    def duplicated_lib(self):
        # Motion 0's four bins appear twice, as if it were resident twice.
        lib = FakeMotionLib()
        lib.adp_samp_active_motion_bins = torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]
        )
        n = lib.adp_samp_active_motion_bins.numel()
        lib.adp_sampling_active_prob = torch.full((n,), 1.0 / n, dtype=torch.float64)
        return lib

    def test_kernel_covers_every_copy_of_the_context(self):
        lib = self.duplicated_lib()
        adapter = PracticeSamplerAdapter(lib)
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.2, kernel_radius=0)
        kernel = adapter._kernel_weights
        # Positions 1 and 5 are both bin 1; each must carry mass.
        assert float(kernel[1]) > 0 and float(kernel[5]) > 0

    def test_distribution_stays_normalized_with_duplicates(self):
        lib = self.duplicated_lib()
        adapter = PracticeSamplerAdapter(lib)
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.2)
        out = adapter.apply(lib.adp_sampling_active_prob)
        assert float(out.sum()) == pytest.approx(1.0)
        assert bool((out >= 0).all())

    def test_boost_lands_on_the_duplicated_context(self):
        lib = self.duplicated_lib()
        adapter = PracticeSamplerAdapter(lib)
        before = lib.adp_sampling_active_prob.clone()
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.2, kernel_radius=0)
        after = adapter.apply(before)
        assert float(after[1]) > float(before[1])
        assert float(after[5]) > float(before[5])

    def test_dose_is_attributed_once_per_draw_not_per_copy(self):
        lib = self.duplicated_lib()
        adapter = PracticeSamplerAdapter(lib)
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.2, kernel_radius=0)
        adapter.record_draw(torch.tensor([1, 1, 1]))
        report = adapter.get_exact_dose_report()
        assert report.drawn_episodes == 3.0
        assert report.drawn_kernel_mass == pytest.approx(3.0)

    def test_context_identity_is_shared_by_duplicates(self):
        lib = self.duplicated_lib()
        adapter = PracticeSamplerAdapter(lib)
        assert adapter.context_for_bin(1).context_id == adapter.context_for_bin(1).context_id

    def test_completion_dose_handles_duplicates(self):
        lib = self.duplicated_lib()
        adapter = PracticeSamplerAdapter(lib)
        adapter.set_intervention(adapter.context_for_bin(1), epsilon=0.2, kernel_radius=0)
        adapter.record_completion(torch.tensor([1, 1]), torch.tensor([100.0, 50.0]))
        assert adapter.get_exact_dose_report().completed_kernel_steps == pytest.approx(150.0)


class TestPairedBranchesShareABase:
    """Control and intervention must differ only through the intervention."""

    def test_same_native_snapshot(self):
        control = PracticeSamplerAdapter(FakeMotionLib(), role="control")
        treated = PracticeSamplerAdapter(FakeMotionLib(), role="intervention")
        assert (
            control.snapshot_native_distribution().distribution_sha256
            == treated.snapshot_native_distribution().distribution_sha256
        )

    def test_control_distribution_is_the_native_one(self):
        lib = FakeMotionLib()
        control = PracticeSamplerAdapter(lib, role="control")
        assert torch.equal(
            control.apply(lib.adp_sampling_active_prob), lib.adp_sampling_active_prob
        )

    def test_branches_diverge_only_by_epsilon(self):
        lib_c, lib_t = FakeMotionLib(), FakeMotionLib()
        control = PracticeSamplerAdapter(lib_c, role="control")
        treated = PracticeSamplerAdapter(lib_t, role="intervention")
        treated.set_intervention(treated.context_for_bin(1), epsilon=0.10)

        out_c = control.apply(lib_c.adp_sampling_active_prob)
        out_t = treated.apply(lib_t.adp_sampling_active_prob)
        expected = I.mix_intervention(
            lib_c.adp_sampling_active_prob, treated._kernel_weights, 0.10
        )
        assert torch.allclose(out_t, expected)
        assert not torch.allclose(out_t, out_c)


class TestUpstreamContract:
    """Guard against drift between the fake and the real ``MotionLibBase``."""

    def test_real_motion_lib_defines_every_attribute_the_adapter_uses(self):
        import inspect

        from gear_sonic.utils.motion_lib import motion_lib_base

        source = inspect.getsource(motion_lib_base.MotionLibBase)
        required = [
            "adp_samp_bins",
            "adp_samp_active_motion_bins",
            "adp_sampling_active_prob",
            "adp_samp_bin_size",
            "adp_samp_num_bins",
            "adp_samp_num_frames",
            "adp_samp_failure_rate_raw",
            "adp_samp_num_episodes",
            "adp_samp_num_failures",
            "uniform_sampling_rate",
            "adp_samp_failure_rate_max_over_mean",
            "_motion_data_keys",
            "_sim_fps",
        ]
        missing = [name for name in required if name not in source]
        assert not missing, f"adapter depends on attributes absent upstream: {missing}"

    def test_motion_fps_really_is_a_tensor_upstream(self):
        """Pins the reality that broke the first live run.

        The fake must keep mirroring this: a scalar stand-in made a live-only
        failure invisible to the whole CPU suite.
        """
        import inspect

        from gear_sonic.utils.motion_lib import motion_lib_base

        source = inspect.getsource(motion_lib_base.MotionLibBase)
        assert "self._motion_fps = torch.tensor(" in source

    def test_sim_fps_really_is_a_scalar_upstream(self):
        import inspect

        from gear_sonic.utils.motion_lib import motion_lib_base

        source = inspect.getsource(motion_lib_base.MotionLibBase)
        assert "self._sim_fps = 1 / self.m_cfg.get(" in source

    def test_sampling_entry_point_still_exists(self):
        from gear_sonic.utils.motion_lib import motion_lib_base

        assert hasattr(motion_lib_base.MotionLibBase, "sample_motion_ids_and_time_steps")
        assert hasattr(motion_lib_base.MotionLibBase, "update_adaptive_sampling_probabilities")

    def test_bin_table_layout_is_motion_start_end(self):
        """``adp_samp_bins`` rows are (orig_motion_id, bin_start, bin_end)."""
        import inspect

        from gear_sonic.utils.motion_lib import motion_lib_base

        source = inspect.getsource(motion_lib_base.MotionLibBase.sample_motion_ids_and_time_steps)
        assert "bins[:, 0], bins[:, 1], bins[:, 2]" in source.replace(" ", "").replace("\n", "") \
            or "orig_motion_ids, bin_start, bin_end" in source
