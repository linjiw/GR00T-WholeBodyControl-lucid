"""Tests for random-state capture and counter-based streams.

These verify the property the paired design depends on: two branches with the
same ``pair_id`` draw identical values on every *matched* channel, while the
context-selection channel is free to differ.
"""

import random

import numpy as np
import pytest
import torch

from gear_sonic.research.practice_utility import rng_capsule as R


class TestDeriveSeed:
    def test_deterministic(self):
        assert R.derive_seed("p", 1, 2, "friction") == R.derive_seed("p", 1, 2, "friction")

    def test_independent_of_consumption_order(self):
        first = R.derive_seed("p", 1, 2, "friction")
        random.random(), np.random.rand(), torch.rand(10)
        assert R.derive_seed("p", 1, 2, "friction") == first

    @pytest.mark.parametrize(
        "a,b",
        [
            (("p", 1, 2, "friction"), ("q", 1, 2, "friction")),
            (("p", 1, 2, "friction"), ("p", 9, 2, "friction")),
            (("p", 1, 2, "friction"), ("p", 1, 9, "friction")),
            (("p", 1, 2, "friction"), ("p", 1, 2, "push_time")),
        ],
    )
    def test_distinct_keys_give_distinct_seeds(self, a, b):
        assert R.derive_seed(*a) != R.derive_seed(*b)

    def test_fits_in_uint32(self):
        for env in range(50):
            seed = R.derive_seed("p", env, 0, "action_noise")
            assert 0 <= seed < 2**32

    def test_rejects_undeclared_channel(self):
        with pytest.raises(ValueError, match="unknown random channel"):
            R.derive_seed("p", 1, 2, "some_new_noise")

    def test_treatment_channel_is_allowed(self):
        assert isinstance(R.derive_seed("p", 1, 2, R.TREATMENT_CHANNEL), int)


class TestPairedBranchesShareMatchedChannels:
    """The core common-random-numbers guarantee."""

    def draw(self, pair_id, env_id, episode, channel):
        gen = R.channel_generator(pair_id, env_id, episode, channel)
        return torch.rand(8, generator=gen)

    def test_matched_channel_is_identical_across_branches(self):
        # Same pair, same env/episode -> control and intervention see the same physics.
        control = self.draw("pair_003", 17, 9, "friction")
        treated = self.draw("pair_003", 17, 9, "friction")
        assert torch.equal(control, treated)

    @pytest.mark.parametrize("channel", R.MATCHED_CHANNELS)
    def test_every_matched_channel_reproduces(self, channel):
        assert torch.equal(self.draw("p", 3, 4, channel), self.draw("p", 3, 4, channel))

    def test_different_channels_are_independent(self):
        assert not torch.equal(self.draw("p", 3, 4, "friction"), self.draw("p", 3, 4, "push_time"))

    def test_different_envs_are_independent(self):
        assert not torch.equal(self.draw("p", 3, 4, "friction"), self.draw("p", 8, 4, "friction"))

    def test_different_pairs_are_independent(self):
        assert not torch.equal(self.draw("a", 3, 4, "friction"), self.draw("b", 3, 4, "friction"))

    def test_matched_draws_survive_interleaved_global_consumption(self):
        """A branch that consumes extra global randomness still matches."""
        control = self.draw("p", 1, 1, "friction")
        torch.rand(1000)  # intervention branch did more work
        random.random()
        assert torch.equal(self.draw("p", 1, 1, "friction"), control)


class TestRngStateCapture:
    def test_capture_does_not_claim_unwired_counter_rng(self):
        state = R.RngState.capture("p")
        assert state.counter_rng_enabled is False

    def test_audited_caller_can_record_counter_rng_integration(self):
        state = R.RngState.capture("p", counter_rng_enabled=True)
        assert state.counter_rng_enabled is True

    def test_restore_reproduces_python_stream(self):
        state = R.RngState.capture("p")
        expected = [random.random() for _ in range(5)]
        state.restore()
        assert [random.random() for _ in range(5)] == expected

    def test_restore_reproduces_numpy_stream(self):
        state = R.RngState.capture("p")
        expected = np.random.rand(5)
        state.restore()
        assert np.allclose(np.random.rand(5), expected)

    def test_restore_reproduces_torch_cpu_stream(self):
        state = R.RngState.capture("p")
        expected = torch.rand(5)
        state.restore()
        assert torch.equal(torch.rand(5), expected)

    def test_restore_reproduces_all_streams_together(self):
        state = R.RngState.capture("p")
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        state.restore()
        assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected

    def test_capture_is_a_snapshot_not_a_reference(self):
        state = R.RngState.capture("p")
        before = state.torch_cpu_state.clone()
        torch.rand(1000)
        assert torch.equal(state.torch_cpu_state, before)


class TestReceipts:
    def test_same_state_gives_same_receipt(self):
        state = R.RngState.capture("p")
        assert state.receipt().torch_cpu_state_sha256 == state.receipt().torch_cpu_state_sha256

    def test_receipt_changes_after_consumption(self):
        before = R.RngState.capture("p").receipt().torch_cpu_state_sha256
        torch.rand(1000)
        assert R.RngState.capture("p").receipt().torch_cpu_state_sha256 != before

    def test_receipt_binds_the_pair_id(self):
        a = R.RngState.capture("pair_a").receipt().context_stream_key
        b = R.RngState.capture("pair_b").receipt().context_stream_key
        assert a != b

    def test_receipt_records_determinism_flags(self):
        flags = R.RngState.capture("p").receipt().deterministic_flags
        assert "cudnn_deterministic" in flags and "cudnn_benchmark" in flags


class TestPayloadRoundTrip:
    def test_roundtrip_preserves_the_stream(self):
        state = R.RngState.capture("p")
        expected = torch.rand(5)
        R.RngState.from_payload(state.to_payload()).restore()
        assert torch.equal(torch.rand(5), expected)

    def test_roundtrip_preserves_receipt(self):
        state = R.RngState.capture("p")
        assert (
            R.RngState.from_payload(state.to_payload()).receipt().torch_cpu_state_sha256
            == state.receipt().torch_cpu_state_sha256
        )

    def test_roundtrip_survives_torch_save_load(self, tmp_path):
        state = R.RngState.capture("p")
        expected = torch.rand(5)
        path = tmp_path / "rng.pt"
        torch.save(state.to_payload(), path)
        R.RngState.from_payload(torch.load(path, weights_only=False)).restore()
        assert torch.equal(torch.rand(5), expected)


class TestDeterminismFlags:
    def test_reports_cudnn_settings(self):
        assert set(R.current_determinism_flags()) >= {"cudnn_deterministic", "cudnn_benchmark"}

    def test_enable_determinism_sets_flags(self):
        previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        try:
            flags = R.enable_determinism(warn_only=True)
            assert flags["cudnn_deterministic"] is True
            assert flags["cudnn_benchmark"] is False
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
