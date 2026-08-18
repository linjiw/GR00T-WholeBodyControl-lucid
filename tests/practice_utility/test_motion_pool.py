"""Tests for motion-pool parsing, family assignment, and manifest hashing."""

import numpy as np
import pytest

from gear_sonic.research.practice_utility import motion_pool as M


def clip(frames=90, dofs=29, fps=30, offset=0.0):
    rng = np.random.default_rng(int(offset * 1000) % 2**31)
    return {
        "dof": rng.standard_normal((frames, dofs)).astype("float32") + offset,
        "fps": fps,
    }


def make_loader(mapping):
    """Loader returning ``{motion_key: clip}`` keyed by file path."""
    def loader(path):
        return mapping[str(path).rsplit("/", 1)[-1]]
    return loader


@pytest.fixture
def pool(tmp_path):
    """Two performers, a mirror pair, and one exact duplicate."""
    keys = {
        "walk_forward_001__A001.pkl": {"walk_forward_001__A001": clip(offset=1.0)},
        "walk_forward_001__A001_M.pkl": {"walk_forward_001__A001_M": clip(offset=2.0)},
        "jog_fast_002__A001.pkl": {"jog_fast_002__A001": clip(offset=3.0)},
        "crawl_low_001__A002.pkl": {"crawl_low_001__A002": clip(offset=4.0)},
        "walk_forward_001__A003.pkl": {"walk_forward_001__A003": clip(offset=1.0)},  # dup of A001
    }
    for name in keys:
        (tmp_path / name).write_bytes(b"")
    return tmp_path, make_loader(keys)


class TestParseMotionKey:
    def test_extracts_all_identities(self):
        parsed = M.parse_motion_key("injured_R_leg_walk_ff_start_180_003__A173")
        assert parsed.performer == "A173"
        assert parsed.take == 3
        assert parsed.is_mirror is False
        assert parsed.base_name == "injured_R_leg_walk_ff_start_180_003"

    def test_detects_mirror(self):
        assert M.parse_motion_key("Neutral_throw_ball_001__A057_M").is_mirror is True

    def test_mirror_pair_shares_performer_and_canonical_name(self):
        plain = M.parse_motion_key("walk_001__A057")
        mirrored = M.parse_motion_key("walk_001__A057_M")
        assert plain.performer == mirrored.performer
        assert plain.canonical_name == mirrored.canonical_name

    def test_canonical_name_drops_take_number(self):
        a = M.parse_motion_key("body_check_001__A342")
        b = M.parse_motion_key("body_check_101__A205")
        assert a.canonical_name == "body_check" and b.canonical_name == "body_check"

    def test_canonical_name_is_case_insensitive(self):
        assert M.parse_motion_key("Neutral_Throw_001__A1").canonical_name == "neutral_throw"

    def test_take_is_none_when_absent(self):
        assert M.parse_motion_key("idle_loop__A007").take is None

    @pytest.mark.parametrize("bad", ["no_performer_here", "walk__B012", "walk__A", ""])
    def test_rejects_malformed_keys(self, bad):
        with pytest.raises(M.MotionKeyError):
            M.parse_motion_key(bad)


class TestMotionFamily:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("crawl_forward_001", "crawl"),
            ("turn_high_jump_360_R_opt_1_003", "jump"),
            ("dance_vouge_boogle_180_R_fast_002", "dance"),
            ("injured_R_leg_walk_ff_start_180_003", "injured"),
            ("big_heavy_two_hands_put_down_front_high_R_001", "carry"),
            ("h_b_w_crouch_270_start_004", "crouch"),
            ("baby_full_diaper_jog_ff_270_stop_R_001", "run"),
            ("arc_walk_left_stop_002", "walk"),
            ("open_door_handle_001", "interact"),
            ("bicep_exercise_002", "exercise"),
            ("shoulder_clap_R_003", "gesture"),
            ("boss_dust_brushing_004", "groom"),
            ("looking_around_002", "search"),
            ("idle_loop_001", "idle"),
        ],
    )
    def test_assigns_expected_family(self, name, expected):
        assert M.motion_family(name)[0] == expected

    def test_returns_the_triggering_token(self):
        assert M.motion_family("injured_torso_idle_loop_002")[1] == "injured"

    def test_locomotion_regime_outranks_upper_body_activity(self):
        """A gesture performed while walking is, dynamically, walking."""
        assert M.motion_family("walk_forward_clap_001")[0] == "walk"

    def test_impairment_outranks_gait(self):
        assert M.motion_family("injured_leg_jog_001")[0] == "injured"

    @pytest.mark.parametrize(
        "inflected,base", [("reaching", "reach"), ("looking", "look"),
                           ("dancing", "dance"), ("rubbing", "groom")]
    )
    def test_matches_inflected_forms(self, inflected, base):
        family, token = M.motion_family(f"{inflected}_something_001")
        assert family != M.FALLBACK_FAMILY

    def test_unknown_falls_back(self):
        assert M.motion_family("zzz_qqq_001")[0] == M.FALLBACK_FAMILY

    def test_fallback_has_no_evidence_token(self):
        assert M.motion_family("zzz_qqq_001")[1] == ""


class TestScanPool:
    def test_finds_every_clip(self, pool):
        scan = M.scan_pool(pool[0], loader=pool[1])
        assert scan.num_motions == 5

    def test_records_are_sorted(self, pool):
        scan = M.scan_pool(*pool)
        assert [r.motion_key for r in scan.records] == sorted(r.motion_key for r in scan.records)

    def test_captures_frames_fps_and_dofs(self, pool):
        record = M.scan_pool(*pool).records[0]
        assert (record.num_frames, record.fps, record.num_dofs) == (90, 30.0, 29)

    def test_duration_derived_from_fps(self, pool):
        assert M.scan_pool(*pool).records[0].duration_seconds == pytest.approx(3.0)

    def test_detects_exact_duplicates(self, pool):
        scan = M.scan_pool(*pool)
        assert len(scan.duplicate_groups) == 1
        assert scan.duplicate_groups[0] == ["walk_forward_001__A001", "walk_forward_001__A003"]

    def test_mirror_is_not_an_exact_duplicate(self, pool):
        """A mirrored clip is different data and must not be silently dropped."""
        scan = M.scan_pool(*pool)
        flat = {k for group in scan.duplicate_groups for k in group}
        assert "walk_forward_001__A001_M" not in flat

    def test_summary_reports_structure(self, pool):
        summary = M.scan_pool(*pool).summary()
        assert summary["num_motions"] == 5
        assert summary["num_performers"] == 3
        assert summary["num_mirrored"] == 1
        assert summary["duplicate_group_count"] == 1

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            M.scan_pool(tmp_path / "nope")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no .pkl"):
            M.scan_pool(tmp_path)

    def test_clip_without_dof_raises(self, tmp_path):
        (tmp_path / "x__A001.pkl").write_bytes(b"")
        with pytest.raises(ValueError, match="no 'dof'"):
            M.scan_pool(tmp_path, loader=lambda p: {"x__A001": {"fps": 30}})

    def test_unparsable_keys_are_reported_not_silently_dropped(self, tmp_path):
        (tmp_path / "weird.pkl").write_bytes(b"")
        scan = M.scan_pool(tmp_path, loader=lambda p: {"weird_name": clip()})
        assert scan.unparsed == ["weird_name"] and scan.num_motions == 0

    def test_limit_truncates(self, pool):
        assert M.scan_pool(pool[0], loader=pool[1], limit=2).num_motions == 2


class TestDropExactDuplicates:
    def test_removes_the_duplicate(self, pool):
        deduped = M.drop_exact_duplicates(M.scan_pool(*pool))
        assert deduped.num_motions == 4

    def test_prefers_the_unmirrored_clip(self, tmp_path):
        payload = {
            "a__A001.pkl": {"a__A001_M": clip(offset=1.0)},
            "b__A001.pkl": {"b__A001": clip(offset=1.0)},
        }
        for name in payload:
            (tmp_path / name).write_bytes(b"")
        deduped = M.drop_exact_duplicates(M.scan_pool(tmp_path, loader=make_loader(payload)))
        assert [r.motion_key for r in deduped.records] == ["b__A001"]

    def test_is_idempotent(self, pool):
        once = M.drop_exact_duplicates(M.scan_pool(*pool))
        assert M.drop_exact_duplicates(once).num_motions == once.num_motions


class TestHashing:
    def test_pool_hash_is_stable(self, pool):
        assert M.pool_sha256(M.scan_pool(*pool)) == M.pool_sha256(M.scan_pool(*pool))

    def test_pool_hash_changes_with_content(self, tmp_path, pool):
        before = M.pool_sha256(M.scan_pool(*pool))
        assert M.pool_sha256(M.drop_exact_duplicates(M.scan_pool(*pool))) != before

    def test_manifest_round_trips(self, pool):
        manifest = M.scan_pool(*pool).to_manifest("test_pool")
        assert manifest.num_motions == 5
        assert len(manifest.manifest_sha256) == 64
