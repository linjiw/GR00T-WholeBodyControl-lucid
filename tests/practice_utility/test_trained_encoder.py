"""Validate the *actual frozen encoder artifact* on real held-out motion.

The unit tests in ``test_latent_gap_probe`` prove the machinery works on
synthetic data with a throwaway encoder. This file checks the instrument the
experiments will actually use, on clips it has never seen, and it is the file
that would catch a badly-trained encoder before it silently poisoned every
latent-gap measurement in the programme.

Skipped when the artifact is absent, so the suite still runs on a fresh
checkout.
"""

import json

import numpy as np
import pytest
import torch

from gear_sonic.research.practice_utility import latent_gap_probe as L
from gear_sonic.research.practice_utility.paths import LUCID_ROOT, relocate

ARTIFACT = LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"
POOL = LUCID_ROOT / "manifests/pool_debug512.json"
SPLIT = LUCID_ROOT / "manifests/split_debug512_performer.json"

pytestmark = pytest.mark.skipif(
    not (ARTIFACT.exists() and POOL.exists() and SPLIT.exists()),
    reason="frozen encoder artifact or manifests not present",
)

CONTROL_HZ = 50.0


def resample(sequence, source_fps, target_fps):
    if abs(source_fps - target_fps) < 1e-6:
        return sequence
    frames = sequence.shape[0]
    duration = frames / source_fps
    target = max(2, int(round(duration * target_fps)))
    src = np.linspace(0.0, duration, frames)
    dst = np.linspace(0.0, duration, target)
    return np.stack([np.interp(dst, src, sequence[:, j]) for j in range(sequence.shape[1])], axis=1)


@pytest.fixture(scope="module")
def encoder():
    blob = torch.load(ARTIFACT, weights_only=False)
    model = L.TemporalVAE(blob["num_joints"], blob["window_length"], blob["latent_dim"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, blob, L.WindowSpec(blob["window_length"], blob["window_stride"])


@pytest.fixture(scope="module")
def held_out_clips():
    """Clips from the TEST partition: the encoder has never seen them."""
    import joblib

    pool = json.loads(POOL.read_text())
    split = json.loads(SPLIT.read_text())
    clips = []
    for record in pool["motions"]:
        if split["assignment"][record["motion_key"]] != "test":
            continue
        clip = joblib.load(relocate(record["path"]))[record["motion_key"]]
        dof = resample(
            np.asarray(clip["dof"], dtype=np.float32), float(clip.get("fps", 30)), CONTROL_HZ
        ).astype(np.float32)
        if dof.shape[0] >= 80:
            clips.append(torch.from_numpy(dof))
        if len(clips) >= 25:
            break
    assert clips, "no usable held-out clips"
    return clips


class TestArtifactIntegrity:
    def test_fingerprint_matches_the_receipt(self, encoder):
        model, blob, _ = encoder
        assert L.encoder_fingerprint(model) == blob["encoder_fingerprint"]

    def test_encoder_is_frozen(self, encoder):
        model, _, _ = encoder
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_trained_on_the_adaptation_split_only(self, encoder):
        _, blob, _ = encoder
        assert blob["train_partition"] == "adaptation"
        assert set(blob["withheld_partitions"]) == {"dev", "test"}

    def test_records_pool_and_split_provenance(self, encoder):
        _, blob, _ = encoder
        assert len(blob["pool_sha256"]) == 64 and len(blob["split_sha256"]) == 64

    def test_trained_at_the_control_rate(self, encoder):
        _, blob, _ = encoder
        assert blob["control_hz"] == CONTROL_HZ

    def test_learned_motion_structure_not_noise(self, encoder):
        _, blob, _ = encoder
        assert blob["holdout_recon"] < 0.05 * blob["noise_recon_control"]

    def test_did_not_overfit(self, encoder):
        """Held-out reconstruction must track training reconstruction."""
        _, blob, _ = encoder
        assert blob["holdout_recon"] < 2.0 * blob["final_train_recon"]

    def test_actually_trained(self, encoder):
        _, blob, _ = encoder
        assert blob["final_train_recon"] < 0.25 * blob["first_train_recon"]


class TestOnHeldOutMotion:
    def test_perfect_tracking_gives_zero_gap(self, encoder, held_out_clips):
        model, _, spec = encoder
        for clip in held_out_clips[:5]:
            series = L.gap_series(model, clip, clip, spec)
            assert float(series["latent"].abs().max()) < 1e-4

    def test_sustained_deviation_raises_the_gap(self, encoder, held_out_clips):
        model, _, spec = encoder
        raised = 0
        for clip in held_out_clips[:10]:
            offset = L.gap_series(model, clip, clip + 0.25, spec)
            if float(offset["latent"].mean()) > 1e-3:
                raised += 1
        assert raised >= 8

    def test_transient_is_attenuated_relative_to_raw_error(self, encoder, held_out_clips):
        """LUCID's central claim, on the real instrument and unseen motion.

        A one-frame contact-like spike is compared against a sustained
        deviation of much smaller per-frame magnitude. Raw joint-space error
        cannot separate them; the latent gap must rank the transient lower.
        """
        model, _, spec = encoder
        raw_ratios, latent_ratios = [], []
        for clip in held_out_clips:
            middle = clip.shape[0] // 2
            spiked = clip.clone()
            spiked[middle] += 0.8
            drifted = clip.clone()
            drifted[middle - 20 : middle + 20] += 0.2

            spike = L.gap_series(model, clip, spiked, spec)
            drift = L.gap_series(model, clip, drifted, spec)
            if spike["latent"].numel() == 0:
                continue
            raw_ratios.append(float(spike["raw"].max()) / max(float(drift["raw"].max()), 1e-9))
            latent_ratios.append(
                float(spike["latent"].max()) / max(float(drift["latent"].max()), 1e-9)
            )

        assert len(latent_ratios) >= 10
        raw_ratio = float(np.mean(raw_ratios))
        latent_ratio = float(np.mean(latent_ratios))
        assert latent_ratio < 0.5 * raw_ratio, (
            f"the frozen encoder does not attenuate transients on held-out motion "
            f"(latent ratio {latent_ratio:.3f} vs raw {raw_ratio:.3f}); it is not "
            f"fit to serve as the latent-gap instrument"
        )

    def test_gap_is_bounded(self, encoder, held_out_clips):
        model, _, spec = encoder
        for clip in held_out_clips[:5]:
            series = L.gap_series(model, clip, clip + torch.randn_like(clip), spec)
            assert float(series["latent"].min()) >= -1e-6
            assert float(series["latent"].max()) <= 2.0 + 1e-6

    def test_embedding_is_reproducible(self, encoder, held_out_clips):
        model, _, spec = encoder
        windows = L.build_windows(held_out_clips[0], spec)
        assert torch.equal(model.embed(windows), model.embed(windows))
