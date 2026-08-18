"""Tests for branch capsule save/load and paired forking.

Two guarantees are load-bearing:

* a forked control and intervention start **identical** in every substantive
  field, so later differences are attributable to the intervention; and
* resuming against a different motion pool or config is **refused**, not
  silently accepted.
"""

import pytest
import torch

from gear_sonic.research.practice_utility import branch_capsule as B
from gear_sonic.research.practice_utility.rng_capsule import RngState


def provenance(**overrides):
    base = dict(
        resolved_config_sha256="cfg" + "0" * 61,
        motion_pool_manifest_sha256="pool" + "0" * 60,
        dev_suite_sha256="dev" + "0" * 61,
        source_commit="c374bae5b9039cd0ee71377e654d11ce1bc69e1d",
        checkpoint_sha256="e6bdab3f" + "0" * 56,
    )
    base.update(overrides)
    return B.Provenance(**base)


def sampler_state():
    return {
        "adp_samp_num_episodes": torch.full((12,), 10.0),
        "adp_samp_num_failures": torch.full((12,), 2.0),
    }


def write(tmp_path, name="branch.capsule.pt", **overrides):
    kwargs = dict(
        branch_id="pair0_control",
        pair_id="pair0",
        role="control",
        global_step=1000,
        model_state={"w": torch.arange(6.0)},
        optimizer_state={"step": 1000, "exp_avg": torch.ones(6)},
        trainer_state={"epoch": 3},
        env_state={"episode_count": 42},
        native_sampler_state=sampler_state(),
        rng_state=RngState.capture("pair0"),
        provenance=provenance(),
    )
    kwargs.update(overrides)
    path = tmp_path / name
    digest = B.save_capsule(path, **kwargs)
    return path, digest


class TestSaveLoad:
    def test_roundtrip_preserves_fields(self, tmp_path):
        path, _ = write(tmp_path)
        loaded = B.load_capsule(path)
        assert loaded["branch_id"] == "pair0_control"
        assert loaded["global_step"] == 1000
        assert torch.equal(loaded["model_state"]["w"], torch.arange(6.0))

    def test_sampler_counters_survive(self, tmp_path):
        path, _ = write(tmp_path)
        state = B.load_capsule(path)["native_sampler_state"]
        assert torch.equal(state["adp_samp_num_episodes"], torch.full((12,), 10.0))

    def test_restores_rng_by_default(self, tmp_path):
        state = RngState.capture("pair0")
        expected = torch.rand(5)
        path, _ = write(tmp_path, rng_state=state)
        torch.rand(500)                      # move the stream elsewhere
        B.load_capsule(path)
        assert torch.equal(torch.rand(5), expected)

    def test_can_defer_rng_restore(self, tmp_path):
        state = RngState.capture("pair0")
        expected = torch.rand(5)
        path, _ = write(tmp_path, rng_state=state)
        torch.rand(500)
        loaded = B.load_capsule(path, restore_rng=False)
        assert not torch.equal(torch.rand(5), expected)
        loaded["rng_state"].restore()
        assert torch.equal(torch.rand(5), expected)

    def test_digest_is_stable(self, tmp_path):
        _, first = write(tmp_path, name="a.pt")
        _, second = write(tmp_path, name="b.pt")
        assert first == second

    def test_digest_tracks_identity(self, tmp_path):
        _, control = write(tmp_path, name="a.pt", role="control")
        _, treated = write(tmp_path, name="b.pt", role="intervention", branch_id="pair0_intervention")
        assert control != treated

    def test_no_partial_file_left_behind(self, tmp_path):
        path, _ = write(tmp_path)
        assert list(tmp_path.glob("*.partial")) == []
        assert path.exists()

    def test_creates_missing_directories(self, tmp_path):
        path, _ = write(tmp_path / "nested" / "deeper", name="c.pt")
        assert path.exists()


class TestValidation:
    def test_rejects_unknown_role(self, tmp_path):
        with pytest.raises(ValueError, match="unknown branch role"):
            write(tmp_path, role="treatment")

    def test_requires_sampler_counters(self, tmp_path):
        with pytest.raises(B.CapsuleIntegrityError, match="missing"):
            write(tmp_path, native_sampler_state={"adp_samp_num_episodes": torch.zeros(3)})

    def test_detects_tampering(self, tmp_path):
        path, _ = write(tmp_path)
        payload = torch.load(path, weights_only=False)
        payload["global_step"] = 999999
        torch.save(payload, path)
        with pytest.raises(B.CapsuleIntegrityError, match="integrity check"):
            B.load_capsule(path)

    def test_detects_missing_keys(self, tmp_path):
        path, _ = write(tmp_path)
        payload = torch.load(path, weights_only=False)
        del payload["native_sampler_state"]
        torch.save(payload, path)
        with pytest.raises(B.CapsuleIntegrityError, match="missing keys"):
            B.load_capsule(path)

    def test_detects_schema_drift(self, tmp_path):
        path, _ = write(tmp_path)
        payload = torch.load(path, weights_only=False)
        payload["schema_version"] = 99
        torch.save(payload, path)
        with pytest.raises(B.CapsuleIntegrityError, match="schema"):
            B.load_capsule(path)


class TestProvenanceGuard:
    def test_matching_provenance_loads(self, tmp_path):
        path, _ = write(tmp_path)
        assert B.load_capsule(path, expected_provenance=provenance())["global_step"] == 1000

    def test_refuses_a_different_motion_pool(self, tmp_path):
        path, _ = write(tmp_path)
        with pytest.raises(B.CapsuleIntegrityError, match="motion_pool_manifest_sha256"):
            B.load_capsule(
                path,
                expected_provenance=provenance(motion_pool_manifest_sha256="other" + "0" * 59),
            )

    def test_refuses_a_different_config(self, tmp_path):
        path, _ = write(tmp_path)
        with pytest.raises(B.CapsuleIntegrityError, match="resolved_config_sha256"):
            B.load_capsule(
                path, expected_provenance=provenance(resolved_config_sha256="x" * 64)
            )

    def test_checkpoint_drift_can_be_ignored_after_branching(self, tmp_path):
        """Branches legitimately diverge from the shared origin checkpoint."""
        path, _ = write(tmp_path)
        loaded = B.load_capsule(
            path,
            expected_provenance=provenance(checkpoint_sha256="d" * 64),
            ignore_provenance_fields=("checkpoint_sha256",),
        )
        assert loaded["pair_id"] == "pair0"

    def test_mismatches_lists_every_bad_field(self):
        a = provenance()
        b = provenance(source_commit="deadbeef", dev_suite_sha256="z" * 64)
        assert set(a.mismatches(b)) == {"source_commit", "dev_suite_sha256"}


class TestForkPair:
    def test_writes_both_branches(self, tmp_path):
        path, _ = write(tmp_path)
        written = B.fork_pair(path, "pair_007", tmp_path / "branches")
        assert set(written) == {"control", "intervention"}
        for p in written.values():
            assert (tmp_path / "branches").exists() and p.endswith(".capsule.pt")

    def test_forks_are_identical_where_it_matters(self, tmp_path):
        path, _ = write(tmp_path)
        written = B.fork_pair(path, "pair_007", tmp_path / "branches")
        B.assert_fork_identical(written["control"], written["intervention"])

    def test_forks_carry_the_new_pair_id_and_roles(self, tmp_path):
        path, _ = write(tmp_path)
        written = B.fork_pair(path, "pair_007", tmp_path / "branches")
        control = B.load_capsule(written["control"], restore_rng=False)
        treated = B.load_capsule(written["intervention"], restore_rng=False)
        assert control["pair_id"] == treated["pair_id"] == "pair_007"
        assert control["role"] == "control" and treated["role"] == "intervention"
        assert control["branch_id"] == "pair_007_control"

    def test_forks_share_the_rng_state(self, tmp_path):
        path, _ = write(tmp_path)
        written = B.fork_pair(path, "pair_007", tmp_path / "branches")
        control = B.load_capsule(written["control"], restore_rng=False)
        treated = B.load_capsule(written["intervention"], restore_rng=False)
        assert torch.equal(
            control["rng"]["torch_cpu_state"], treated["rng"]["torch_cpu_state"]
        )

    def test_fork_does_not_consume_randomness(self, tmp_path):
        path, _ = write(tmp_path)
        expected = torch.rand(5)
        torch.manual_seed(1234)
        before = torch.rand(5)
        torch.manual_seed(1234)
        B.fork_pair(path, "pair_007", tmp_path / "branches")
        assert torch.equal(torch.rand(5), before)
        assert expected is not None

    def test_detects_a_divergent_fork(self, tmp_path):
        path, _ = write(tmp_path)
        written = B.fork_pair(path, "pair_007", tmp_path / "branches")
        payload = torch.load(written["intervention"], weights_only=False)
        payload["model_state"]["w"] = torch.zeros(6)
        payload["capsule_sha256"] = B._capsule_hash(payload)
        torch.save(payload, written["intervention"])
        with pytest.raises(B.CapsuleIntegrityError, match="differ in model_state"):
            B.assert_fork_identical(written["control"], written["intervention"])

    def test_detects_divergent_sampler_state(self, tmp_path):
        path, _ = write(tmp_path)
        written = B.fork_pair(path, "pair_007", tmp_path / "branches")
        payload = torch.load(written["intervention"], weights_only=False)
        payload["native_sampler_state"]["adp_samp_num_failures"] = torch.zeros(12)
        payload["capsule_sha256"] = B._capsule_hash(payload)
        torch.save(payload, written["intervention"])
        with pytest.raises(B.CapsuleIntegrityError, match="native_sampler_state"):
            B.assert_fork_identical(written["control"], written["intervention"])

    def test_detects_mismatched_global_step(self, tmp_path):
        a, _ = write(tmp_path, name="a.pt", role="control", branch_id="p_control")
        b, _ = write(
            tmp_path, name="b.pt", role="intervention",
            branch_id="p_intervention", global_step=2000,
        )
        with pytest.raises(B.CapsuleIntegrityError, match="global steps"):
            B.assert_fork_identical(str(a), str(b))

    def test_detects_pair_mismatch(self, tmp_path):
        a, _ = write(tmp_path, name="a.pt", role="control", pair_id="pairA")
        b, _ = write(
            tmp_path, name="b.pt", role="intervention",
            pair_id="pairB", branch_id="pairB_intervention",
        )
        with pytest.raises(B.CapsuleIntegrityError, match="different pairs"):
            B.assert_fork_identical(str(a), str(b))


class TestStateComparison:
    def test_nested_dicts(self):
        assert B._states_equal({"a": {"b": torch.ones(3)}}, {"a": {"b": torch.ones(3)}})
        assert not B._states_equal({"a": {"b": torch.ones(3)}}, {"a": {"b": torch.zeros(3)}})

    def test_shape_mismatch_is_not_equal(self):
        assert not B._states_equal(torch.ones(3), torch.ones(4))

    def test_lists(self):
        assert B._states_equal([torch.ones(2), 3], [torch.ones(2), 3])
        assert not B._states_equal([torch.ones(2)], [torch.ones(2), 3])

    def test_scalars(self):
        assert B._states_equal(5, 5) and not B._states_equal(5, 6)
