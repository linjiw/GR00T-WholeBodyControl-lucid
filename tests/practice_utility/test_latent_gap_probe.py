"""Tests for the LUCID latent command-execution gap probe.

Most tests are cheap and structural. One is not: ``TestTransientAttenuation``
trains a small VAE on smooth synthetic motion and then checks LUCID's actual
claim -- that a brief contact-like transient inflates raw joint-space mismatch
far more than it inflates the latent gap. That claim is the reason the latent
representation exists, so it is verified rather than assumed.
"""

import math

import pytest
import torch

from gear_sonic.research.practice_utility import latent_gap_probe as L


def smooth_motion(steps=400, joints=6, seed=0):
    """Sum of low-frequency sinusoids: what real reference motion looks like."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(steps, dtype=torch.float32) / 50.0
    signal = torch.zeros(steps, joints)
    for joint in range(joints):
        for _ in range(3):
            freq = 0.5 + 2.5 * torch.rand(1, generator=generator).item()
            phase = 2 * math.pi * torch.rand(1, generator=generator).item()
            signal[:, joint] += torch.sin(2 * math.pi * freq * t + phase)
    return signal / 3.0


@pytest.fixture(scope="module")
def trained():
    """A small encoder trained on smooth motion. Shared: training costs time."""
    spec = L.WindowSpec(length=16, stride=1)
    corpus = torch.cat(
        [L.build_windows(smooth_motion(seed=s), spec) for s in range(12)], dim=0
    )
    model, history = L.train_encoder(
        corpus, num_joints=6, spec=spec, latent_dim=16, epochs=25, seed=0
    )
    assert history[-1]["recon"] < history[0]["recon"], "encoder failed to learn"
    return model, spec


class TestWindowSpec:
    def test_span_accounts_for_stride(self):
        assert L.WindowSpec(length=4, stride=1).span == 4
        assert L.WindowSpec(length=4, stride=3).span == 10

    @pytest.mark.parametrize("kwargs", [{"length": 1}, {"length": 0}, {"stride": 0}])
    def test_rejects_degenerate_specs(self, kwargs):
        with pytest.raises(ValueError):
            L.WindowSpec(**kwargs)


class TestBuildWindows:
    def test_shape(self):
        windows = L.build_windows(torch.randn(50, 6), L.WindowSpec(length=8))
        assert windows.shape == (43, 8, 6)

    def test_window_contents_are_consecutive(self):
        sequence = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        windows = L.build_windows(sequence, L.WindowSpec(length=4, stride=1))
        assert torch.equal(windows[0].flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert torch.equal(windows[5].flatten(), torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_stride_skips_frames(self):
        sequence = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        windows = L.build_windows(sequence, L.WindowSpec(length=3, stride=2))
        assert torch.equal(windows[0].flatten(), torch.tensor([0.0, 2.0, 4.0]))

    def test_no_lookahead(self):
        """Window i must not contain information past its own final timestep."""
        sequence = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        spec = L.WindowSpec(length=4, stride=1)
        windows = L.build_windows(sequence, spec)
        for i in range(windows.shape[0]):
            assert float(windows[i].max()) <= i + spec.span - 1

    def test_too_short_sequence_yields_nothing(self):
        assert L.build_windows(torch.randn(3, 6), L.WindowSpec(length=8)).shape[0] == 0

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\(T, J\)"):
            L.build_windows(torch.randn(50), L.WindowSpec())


class TestTemporalVAE:
    def test_embed_shape(self):
        model = L.TemporalVAE(6, 8, 16)
        assert model.embed(torch.randn(4, 8, 6)).shape == (4, 16)

    def test_embedding_is_deterministic(self):
        """No sampling: the same motion must always give the same embedding."""
        model = L.TemporalVAE(6, 8, 16).eval()
        windows = torch.randn(4, 8, 6)
        assert torch.equal(model.embed(windows), model.embed(windows))

    def test_forward_is_stochastic(self):
        model = L.TemporalVAE(6, 8, 16)
        windows = torch.randn(4, 8, 6)
        assert not torch.equal(model(windows)[0], model(windows)[0])

    def test_reconstruction_shape_matches_input(self):
        model = L.TemporalVAE(6, 8, 16)
        assert model(torch.randn(4, 8, 6))[0].shape == (4, 8, 6)

    @pytest.mark.parametrize(
        "shape,match",
        [((4, 9, 6), "window length"), ((4, 8, 7), "joint count"), ((4, 8), r"\(B, T, J\)")],
    )
    def test_rejects_mismatched_input(self, shape, match):
        with pytest.raises(ValueError, match=match):
            L.TemporalVAE(6, 8, 16).embed(torch.randn(*shape))

    @pytest.mark.parametrize("kwargs", [{"num_joints": 0}, {"latent_dim": 0}])
    def test_rejects_degenerate_config(self, kwargs):
        params = {"num_joints": 6, "latent_dim": 16}
        params.update(kwargs)
        with pytest.raises(ValueError):
            L.TemporalVAE(window_length=8, **params)


class TestLatentGap:
    def test_identical_embeddings_give_zero(self):
        embedding = torch.randn(8, 16)
        assert float(L.latent_gap(embedding, embedding).abs().max()) < 1e-5

    def test_opposite_embeddings_give_two(self):
        embedding = torch.randn(4, 16)
        assert float(L.latent_gap(embedding, -embedding).min()) > 2.0 - 1e-5

    def test_orthogonal_embeddings_give_one(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        assert float(L.latent_gap(a, b)) == pytest.approx(1.0)

    def test_bounded(self):
        gaps = L.latent_gap(torch.randn(64, 16), torch.randn(64, 16))
        assert float(gaps.min()) >= -1e-6 and float(gaps.max()) <= 2.0 + 1e-6

    def test_scale_invariant(self):
        """Direction, not magnitude: a vigorous motion is not penalized."""
        a, b = torch.randn(4, 16), torch.randn(4, 16)
        assert torch.allclose(L.latent_gap(a, b), L.latent_gap(10.0 * a, 0.1 * b), atol=1e-5)

    def test_zero_vector_does_not_produce_nan(self):
        assert torch.isfinite(L.latent_gap(torch.zeros(2, 16), torch.randn(2, 16))).all()

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="align"):
            L.latent_gap(torch.randn(4, 16), torch.randn(4, 8))


class TestGapSeries:
    def test_perfect_tracking_gives_near_zero_gap(self, trained):
        model, spec = trained
        motion = smooth_motion(steps=120, seed=99)
        series = L.gap_series(model, motion, motion, spec)
        assert float(series["latent"].abs().max()) < 1e-4
        assert float(series["raw"].abs().max()) == pytest.approx(0.0, abs=1e-5)

    def test_reports_warmup(self, trained):
        model, spec = trained
        motion = smooth_motion(steps=60, seed=1)
        assert L.gap_series(model, motion, motion, spec)["warmup_steps"] == spec.span - 1

    def test_short_episode_yields_empty_series(self, trained):
        model, spec = trained
        motion = smooth_motion(steps=4, seed=1)
        assert L.gap_series(model, motion, motion, spec)["latent"].numel() == 0

    def test_rejects_mismatched_streams(self, trained):
        model, spec = trained
        with pytest.raises(ValueError, match="align"):
            L.gap_series(model, torch.randn(60, 6), torch.randn(60, 5), spec)

    def test_sustained_deviation_raises_the_gap(self, trained):
        model, spec = trained
        commanded = smooth_motion(steps=120, seed=7)
        executed = commanded + 0.35            # persistent tracking offset
        series = L.gap_series(model, commanded, executed, spec)
        assert float(series["latent"].mean()) > 1e-3


class TestTransientAttenuation:
    """LUCID's core claim, verified rather than assumed."""

    def test_transient_inflates_raw_mismatch_far_more_than_latent_gap(self, trained):
        model, spec = trained
        commanded = smooth_motion(steps=200, seed=42)

        # A contact-like transient: one frame, large, then gone.
        spiked = commanded.clone()
        spiked[100] += 0.8

        # A sustained deviation of far smaller per-frame magnitude.
        drifted = commanded.clone()
        drifted[80:120] += 0.2

        spike = L.gap_series(model, commanded, spiked, spec)
        drift = L.gap_series(model, commanded, drifted, spec)

        raw_ratio = float(spike["raw"].max()) / max(float(drift["raw"].max()), 1e-9)
        latent_ratio = float(spike["latent"].max()) / max(float(drift["latent"].max()), 1e-9)

        # Raw error rates the one-frame spike as comparable to (or worse than)
        # the sustained drift; the latent gap should down-rank it markedly.
        assert latent_ratio < raw_ratio, (
            f"latent gap did not attenuate the transient relative to raw error "
            f"(latent ratio {latent_ratio:.3f} vs raw ratio {raw_ratio:.3f})"
        )

    def test_encoder_reconstructs_smooth_motion_better_than_noise(self, trained):
        """A sanity check that the instrument learned motion structure at all."""
        model, spec = trained
        smooth = L.build_windows(smooth_motion(steps=200, seed=5), spec)
        noise = torch.randn_like(smooth)
        with torch.no_grad():
            smooth_error = (model(smooth)[0] - smooth).pow(2).mean()
            noise_error = (model(noise)[0] - noise).pow(2).mean()
        assert float(smooth_error) < float(noise_error)


class TestRawMismatch:
    def test_zero_for_identical(self):
        windows = torch.randn(4, 8, 6)
        assert float(L.raw_mismatch(windows, windows).abs().max()) == pytest.approx(0.0)

    def test_grows_with_deviation(self):
        a = torch.zeros(1, 8, 6)
        small = L.raw_mismatch(a, a + 0.1)
        large = L.raw_mismatch(a, a + 1.0)
        assert float(large) > float(small)

    def test_rejects_mismatch(self):
        with pytest.raises(ValueError, match="align"):
            L.raw_mismatch(torch.randn(4, 8, 6), torch.randn(4, 8, 5))


class TestSummarizeGap:
    def test_median_and_quantile(self):
        summary = L.summarize_gap(torch.tensor([0.0, 0.1, 0.2, 0.3, 10.0]))
        assert summary.median == pytest.approx(0.2)
        assert summary.p90 > summary.median

    def test_p90_emphasizes_sustained_near_failure(self):
        """Degradation occupying >10% of an episode moves p90 far more than the mean."""
        calm = torch.full((100,), 0.1)
        degrading = calm.clone()
        degrading[-15:] = 1.5
        p90_shift = L.summarize_gap(degrading).p90 - L.summarize_gap(calm).p90
        mean_shift = L.summarize_gap(degrading).mean - L.summarize_gap(calm).mean
        assert p90_shift > 5 * mean_shift

    def test_p90_is_blind_to_degradation_below_its_own_tail(self):
        """The flip side, and a real limitation of a p90 scheduling statistic.

        Five bad steps in a hundred sit inside the top decile, so p90 does not
        move at all while the mean does. A quantile-driven curriculum is
        therefore insensitive to brief excursions by construction -- which is
        worth knowing when auditing the gap as a curriculum signal.
        """
        calm = torch.full((100,), 0.1)
        brief = calm.clone()
        brief[-5:] = 1.5
        assert L.summarize_gap(brief).p90 == pytest.approx(L.summarize_gap(calm).p90)
        assert L.summarize_gap(brief).mean > L.summarize_gap(calm).mean

    def test_slope_detects_growth(self):
        assert L.summarize_gap(torch.linspace(0.0, 1.0, 50)).slope > 0

    def test_slope_detects_decay(self):
        assert L.summarize_gap(torch.linspace(1.0, 0.0, 50)).slope < 0

    def test_slope_is_zero_for_flat(self):
        assert L.summarize_gap(torch.full((50,), 0.3)).slope == pytest.approx(0.0, abs=1e-9)

    def test_variance_separates_steady_from_erratic(self):
        steady = L.summarize_gap(torch.full((50,), 0.3))
        erratic = L.summarize_gap(torch.tensor([0.0, 0.6] * 25))
        assert erratic.variance > steady.variance

    def test_empty_and_single_are_safe(self):
        assert L.summarize_gap(torch.tensor([])).num_windows == 0
        assert L.summarize_gap(torch.tensor([0.4])).median == pytest.approx(0.4)

    def test_rejects_invalid_quantile(self):
        with pytest.raises(ValueError, match="quantile"):
            L.summarize_gap(torch.rand(10), quantile=1.0)


class TestTerminationFill:
    def test_pads_to_the_horizon(self):
        filled = L.fill_after_termination(torch.full((10,), 0.1), terminated_at=10, horizon=100)
        assert filled.numel() == 100
        assert float(filled[-1]) == L.TERMINATED_GAP

    def test_an_early_fall_scores_worse_not_better(self):
        """Truncating would reward falling early with a small average gap."""
        early_fall = L.fill_after_termination(torch.full((10,), 0.1), 10, 100)
        completed = L.fill_after_termination(torch.full((100,), 0.3), None, 100)
        assert float(early_fall.mean()) > float(completed.mean())

    def test_completed_episode_is_unchanged(self):
        series = torch.full((100,), 0.2)
        assert torch.equal(L.fill_after_termination(series, None, 100), series)

    def test_longer_than_horizon_is_truncated(self):
        assert L.fill_after_termination(torch.full((200,), 0.2), None, 100).numel() == 100

    def test_rejects_negative_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            L.fill_after_termination(torch.rand(5), None, -1)


class TestCorruption:
    def test_changes_the_window(self):
        windows = torch.zeros(4, 8, 6)
        assert not torch.equal(L.corrupt_windows(windows), windows)

    def test_preserves_shape(self):
        windows = torch.randn(4, 8, 6)
        assert L.corrupt_windows(windows).shape == windows.shape

    def test_no_corruption_when_disabled(self):
        windows = torch.randn(4, 8, 6)
        assert torch.equal(L.corrupt_windows(windows, noise_std=0.0, spike_prob=0.0), windows)

    @pytest.mark.parametrize("kwargs", [{"noise_std": -1.0}, {"spike_prob": 1.5},
                                        {"spike_scale": -1.0}])
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            L.corrupt_windows(torch.randn(2, 8, 6), **kwargs)


class TestTrainingAndFingerprint:
    def test_encoder_is_frozen_after_training(self, trained):
        model, _ = trained
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_fingerprint_is_stable(self, trained):
        model, _ = trained
        assert L.encoder_fingerprint(model) == L.encoder_fingerprint(model)

    def test_fingerprint_distinguishes_encoders(self, trained):
        model, _ = trained
        assert L.encoder_fingerprint(model) != L.encoder_fingerprint(L.TemporalVAE(6, 16, 16))

    def test_rejects_empty_corpus(self):
        with pytest.raises(ValueError, match="no training windows"):
            L.train_encoder(torch.zeros(0, 8, 6), num_joints=6)

    def test_rejects_wrong_rank_corpus(self):
        with pytest.raises(ValueError, match=r"\(N, T, J\)"):
            L.train_encoder(torch.zeros(8, 6), num_joints=6)
