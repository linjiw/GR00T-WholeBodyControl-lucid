"""Tests for group-disjoint, family-stratified motion splits.

The property under test is not "did every clip get a label" but "can a held-out
score be read as generalization". That means no performer, duplicate
trajectory, or (in content mode) action name may straddle a partition.
"""

import numpy as np
import pytest

from gear_sonic.research.practice_utility import motion_pool as M
from gear_sonic.research.practice_utility import split as S


def record(key, family="walk", content=None):
    parsed = M.parse_motion_key(key)
    return M.MotionRecord(
        motion_key=key,
        path=f"/tmp/{key}.pkl",
        parsed=parsed,
        num_frames=90,
        fps=30.0,
        num_dofs=29,
        content_sha256=content or f"sha_{key}",
        family=family,
        family_evidence=family,
    )


def scan(records):
    return M.PoolScan(records=list(records), source_root="/tmp/pool")


def synthetic_pool(num_performers=60, clips_per_performer=3):
    """Performers with distinct action names, so channels stay separable."""
    families = ["walk", "run", "jump", "dance", "carry", "idle"]
    records = []
    for p in range(num_performers):
        for c in range(clips_per_performer):
            family = families[(p + c) % len(families)]
            records.append(record(f"{family}_move{p}_{c}_00{c}__A{p:03d}", family=family))
    return scan(records)


class TestBuildGroups:
    def test_performer_mode_groups_by_performer(self):
        pool = scan([record("walk_a_001__A001"), record("jog_b_001__A001"),
                     record("walk_c_001__A002")])
        groups = S.build_groups(pool.records, "performer")
        assert len(groups) == 2

    def test_mirror_pair_stays_together(self):
        pool = scan([record("walk_a_001__A001"), record("walk_a_001__A001_M")])
        assert len(S.build_groups(pool.records, "performer")) == 1

    def test_exact_duplicates_always_group_even_across_performers(self):
        pool = scan([record("walk_a_001__A001", content="same"),
                     record("walk_b_001__A002", content="same")])
        for mode in ("performer", "content", "performer_and_content"):
            assert len(S.build_groups(pool.records, mode)) == 1

    def test_content_mode_groups_by_action_name(self):
        pool = scan([record("walk_a_001__A001"), record("walk_a_002__A002"),
                     record("jog_b_001__A003")])
        groups = S.build_groups(pool.records, "content")
        assert len(groups) == 2   # walk_a (2 performers) + jog_b

    def test_content_mode_ignores_performer(self):
        pool = scan([record("walk_a_001__A001"), record("jog_b_001__A001")])
        assert len(S.build_groups(pool.records, "content")) == 2

    def test_combined_mode_merges_transitively(self):
        # A001 links walk_a and jog_b; A002 links jog_b onward -> one component.
        pool = scan([record("walk_a_001__A001"), record("jog_b_001__A001"),
                     record("jog_b_002__A002")])
        assert len(S.build_groups(pool.records, "performer_and_content")) == 1

    def test_groups_partition_every_clip_exactly_once(self):
        pool = synthetic_pool()
        groups = S.build_groups(pool.records, "performer")
        flat = [k for keys in groups.values() for k in keys]
        assert sorted(flat) == sorted(r.motion_key for r in pool.records)

    def test_grouping_is_deterministic(self):
        pool = synthetic_pool()
        assert S.build_groups(pool.records, "performer") == S.build_groups(pool.records, "performer")


class TestBuildSplit:
    def test_hits_target_shares(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha")
        for name, ratio in S.DEFAULT_RATIOS.items():
            assert result.stats["partitions"][name]["share"] == pytest.approx(ratio, abs=0.05)

    def test_assigns_every_clip(self):
        pool = synthetic_pool()
        assert len(S.build_split(pool, "poolsha").assignment) == pool.num_motions

    def test_partitions_are_disjoint(self):
        result = S.build_split(synthetic_pool(), "poolsha")
        seen = [set(result.partition(p)) for p in S.DEFAULT_RATIOS]
        assert set.intersection(*seen) == set()

    def test_no_performer_straddles_in_performer_mode(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha", linkage="performer")
        by_performer = {}
        for key, partition in result.assignment.items():
            by_performer.setdefault(M.parse_motion_key(key).performer, set()).add(partition)
        assert all(len(v) == 1 for v in by_performer.values())

    def test_no_action_name_straddles_in_content_mode(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha", linkage="content")
        by_name = {}
        for key, partition in result.assignment.items():
            by_name.setdefault(M.parse_motion_key(key).canonical_name, set()).add(partition)
        assert all(len(v) == 1 for v in by_name.values())

    def test_families_are_represented_everywhere(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha")
        families = {r.family for r in pool.records}
        for name in S.DEFAULT_RATIOS:
            assert set(result.stats["partitions"][name]["family_counts"]) == families

    def test_is_deterministic_for_a_seed(self):
        pool = synthetic_pool()
        assert (
            S.build_split(pool, "poolsha", seed=7).split_sha256
            == S.build_split(pool, "poolsha", seed=7).split_sha256
        )

    def test_seed_changes_the_assignment(self):
        pool = synthetic_pool()
        a = S.build_split(pool, "poolsha", seed=1)
        b = S.build_split(pool, "poolsha", seed=2)
        assert a.assignment != b.assignment

    def test_custom_ratios(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha", ratios={"train": 0.5, "holdout": 0.5})
        assert result.stats["partitions"]["train"]["share"] == pytest.approx(0.5, abs=0.06)

    def test_split_hash_tracks_the_pool(self):
        pool = synthetic_pool()
        a = S.build_split(pool, "sha_a")
        b = S.build_split(pool, "sha_b")
        assert a.split_sha256 != b.split_sha256

    def test_partition_accessor_rejects_unknown_name(self):
        with pytest.raises(KeyError):
            S.build_split(synthetic_pool(), "poolsha").partition("nope")

    @pytest.mark.parametrize("ratios", [{"a": 0.5, "b": 0.4}, {"a": 1.0, "b": 0.0}, {}])
    def test_rejects_invalid_ratios(self, ratios):
        with pytest.raises(S.SplitError):
            S.build_split(synthetic_pool(), "poolsha", ratios=ratios)

    def test_rejects_empty_pool(self):
        with pytest.raises(S.SplitError):
            S.build_split(scan([]), "poolsha")


class TestGiantComponentGuard:
    """Closing performer and content at once collapses on real pools."""

    def densely_linked_pool(self, n=40):
        # Every performer shares an action with the next, chaining all of them.
        records = []
        for p in range(n):
            records.append(record(f"shared{p}_001__A{p:03d}"))
            records.append(record(f"shared{(p + 1) % n}_001__A{p:03d}"))
        return scan(records)

    def test_refuses_a_degenerate_split(self):
        with pytest.raises(S.SplitError, match="giant component"):
            S.build_split(self.densely_linked_pool(), "poolsha",
                          linkage="performer_and_content")

    def test_error_explains_the_remedy(self):
        with pytest.raises(S.SplitError, match="separately"):
            S.build_split(self.densely_linked_pool(), "poolsha",
                          linkage="performer_and_content")

    def test_each_channel_alone_still_splits(self):
        pool = self.densely_linked_pool()
        for mode in ("performer", "content"):
            assert S.build_split(pool, "poolsha", linkage=mode) is not None

    def test_threshold_is_configurable(self):
        pool = self.densely_linked_pool()
        with pytest.raises(S.SplitError):
            S.build_split(pool, "poolsha", linkage="performer_and_content",
                          max_group_share=0.1)


class TestVerifySplit:
    def test_detects_an_unassigned_clip(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha")
        result.assignment.pop(next(iter(result.assignment)))
        with pytest.raises(S.SplitError, match="never assigned"):
            S.verify_split(result, pool)

    def test_detects_a_foreign_clip(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha")
        result.assignment["ghost_001__A999"] = "dev"
        with pytest.raises(S.SplitError, match="absent from the pool"):
            S.verify_split(result, pool)

    def test_detects_a_performer_leak(self):
        pool = synthetic_pool()
        result = S.build_split(pool, "poolsha", linkage="performer")
        key = result.partition("adaptation")[0]
        result.assignment[key] = "test"      # tear one clip out of its group
        with pytest.raises(S.SplitError, match="performer leaks"):
            S.verify_split(result, pool)

    def test_detects_a_duplicate_leak(self):
        pool = scan([record(f"walk_x{i}_001__A{i:03d}", content="same" if i < 2 else f"s{i}")
                     for i in range(30)])
        result = S.build_split(pool, "poolsha", linkage="performer")
        dupes = [r.motion_key for r in pool.records if r.content_sha256 == "same"]
        result.assignment[dupes[0]] = "adaptation"
        result.assignment[dupes[1]] = "test"
        with pytest.raises(S.SplitError, match="content_sha256 leaks"):
            S.verify_split(result, pool)


class TestSerialization:
    def test_dict_carries_provenance(self):
        payload = S.build_split(synthetic_pool(), "poolsha").to_dict()
        assert payload["kind"] == "practice_utility_group_disjoint_split"
        assert payload["pool_sha256"] == "poolsha"
        assert len(payload["split_sha256"]) == 64
        assert payload["linkage"] == "performer"

    def test_assignment_is_sorted_for_stable_diffs(self):
        payload = S.build_split(synthetic_pool(), "poolsha").to_dict()
        assert list(payload["assignment"]) == sorted(payload["assignment"])
