"""Per-motion paired comparison: the pairing guards, and the arithmetic."""

import json

import pytest

from gear_sonic.research.practice_utility import motion_paired as MP


def write_metrics(tmp_path, name, failed_idxes, keys=None, motion_count=102):
    keys = keys or [f"m{i}" for i in failed_idxes]
    payload = {
        "eval/success/success_rate": (motion_count - len(failed_idxes)) / motion_count,
        "eval/failed_metrics_dict": {
            "failed_idxes": list(failed_idxes),
            "failed_keys": list(keys),
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def outcome(tmp_path, name, failed, seed=8600, mode="a", preset="id_clean", keys=None):
    return MP.read_outcome(write_metrics(tmp_path, name, failed, keys), seed, mode, preset)


class TestReadOutcome:
    def test_recovers_the_panel_size_and_rate(self, tmp_path):
        out = outcome(tmp_path, "r", list(range(32)))
        assert out.motion_count == 102
        assert out.success_rate == pytest.approx(70 / 102)
        assert sum(out.successes()) == 70

    def test_success_vector_is_in_panel_order(self, tmp_path):
        out = outcome(tmp_path, "r", [0, 2, 101])
        vector = out.successes()
        assert vector[0] == 0 and vector[1] == 1 and vector[2] == 0 and vector[101] == 0
        assert len(vector) == 102

    def test_a_run_with_no_failures_cannot_be_sized(self, tmp_path):
        # Real, but not pairable by this route -- refuse rather than guess.
        with pytest.raises(ValueError, match="no failures"):
            outcome(tmp_path, "r", [])

    def test_inconsistent_rate_and_failure_count_is_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": 0.5,
            "eval/failed_metrics_dict": {"failed_idxes": [1, 2, 3], "failed_keys": ["a", "b", "c"]},
        }))
        out = MP.read_outcome(path, 8600, "a", "id_clean")
        # 3 failures at a 0.5 rate is a 6-motion panel; self-consistent, so allowed.
        assert out.motion_count == 6


class TestPairingGuards:
    def test_mismatched_panel_order_raises(self, tmp_path):
        left = outcome(tmp_path, "l", [1, 5], keys=["walk", "jog"])
        right = outcome(tmp_path, "r", [1, 7], keys=["run", "hop"], mode="b")
        with pytest.raises(ValueError, match="panel order differs"):
            MP.assert_comparable(left, right)

    def test_matching_shared_keys_pass(self, tmp_path):
        left = outcome(tmp_path, "l", [1, 5], keys=["walk", "jog"])
        right = outcome(tmp_path, "r", [1, 7], keys=["walk", "hop"], mode="b")
        MP.assert_comparable(left, right)

    def test_different_seeds_cannot_be_paired(self, tmp_path):
        left = outcome(tmp_path, "l", [1, 5])
        right = outcome(tmp_path, "r", [1, 5], seed=8601, mode="b")
        with pytest.raises(ValueError, match="checkpoint seed"):
            MP.assert_comparable(left, right)

    def test_different_presets_cannot_be_paired(self, tmp_path):
        left = outcome(tmp_path, "l", [1, 5])
        right = outcome(tmp_path, "r", [1, 5], mode="b", preset="dr_full")
        with pytest.raises(ValueError, match="preset"):
            MP.assert_comparable(left, right)

    def test_different_panel_sizes_cannot_be_paired(self, tmp_path):
        left = outcome(tmp_path, "l", [1, 5])
        right = MP.read_outcome(
            write_metrics(tmp_path, "r", [1, 5], motion_count=50), 8600, "b", "id_clean"
        )
        with pytest.raises(ValueError, match="panel sizes differ"):
            MP.assert_comparable(left, right)


class TestPairedDifference:
    def _arm(self, tmp_path, tag, failed_by_seed, mode):
        return [
            MP.read_outcome(
                write_metrics(tmp_path, f"{tag}{seed}", failed), seed, mode, "id_clean"
            )
            for seed, failed in failed_by_seed.items()
        ]

    def test_delta_matches_the_success_rate_difference(self, tmp_path):
        treatment = self._arm(tmp_path, "t", {8600: list(range(20)),
                                              8601: list(range(20)),
                                              8602: list(range(20))}, "t")
        reference = self._arm(tmp_path, "r", {8600: list(range(30)),
                                              8601: list(range(30)),
                                              8602: list(range(30))}, "r")
        out = MP.paired_difference(treatment, reference, samples=500)
        assert out.delta_pts == pytest.approx(100.0 * 10 / 102)
        assert out.num_seeds == 3 and out.num_motions == 102

    def test_identical_arms_give_a_zero_delta_and_an_interval_covering_zero(self, tmp_path):
        failed = {8600: [1, 2], 8601: [3, 4], 8602: [5, 6]}
        treatment = self._arm(tmp_path, "t", failed, "t")
        reference = self._arm(tmp_path, "r", failed, "r")
        out = MP.paired_difference(treatment, reference, samples=500)
        assert out.delta_pts == 0.0
        assert out.ci_low_pts <= 0.0 <= out.ci_high_pts
        assert out.excludes_zero is False
        assert out.treatment_only_wins == out.reference_only_wins == 0

    def test_a_large_consistent_difference_excludes_zero(self, tmp_path):
        treatment = self._arm(tmp_path, "t", {s: list(range(5)) for s in (8600, 8601, 8602)}, "t")
        reference = self._arm(tmp_path, "r", {s: list(range(60)) for s in (8600, 8601, 8602)}, "r")
        out = MP.paired_difference(treatment, reference, samples=2000)
        assert out.delta_pts > 0
        assert out.excludes_zero is True
        assert out.ci_low_pts > 0

    def test_discordant_counts_are_the_paired_evidence(self, tmp_path):
        # Treatment fails 0,1; reference fails 1,2. Motion 0 favours reference,
        # motion 2 favours treatment, motion 1 is agreed.
        treatment = self._arm(tmp_path, "t", {8600: [0, 1]}, "t")
        reference = self._arm(tmp_path, "r", {8600: [1, 2]}, "r")
        out = MP.paired_difference(treatment, reference, samples=200)
        assert out.treatment_only_wins == 1
        assert out.reference_only_wins == 1
        assert out.both_agree == 100
        assert out.delta_pts == pytest.approx(0.0)

    def test_a_seed_present_in_only_one_arm_is_dropped(self, tmp_path):
        treatment = self._arm(tmp_path, "t", {8600: [1], 8601: [1], 8602: [1]}, "t")
        reference = self._arm(tmp_path, "r", {8600: [1], 8601: [1]}, "r")
        out = MP.paired_difference(treatment, reference, samples=200)
        assert out.num_seeds == 2

    def test_no_shared_seed_raises(self, tmp_path):
        treatment = self._arm(tmp_path, "t", {8600: [1]}, "t")
        reference = self._arm(tmp_path, "r", {8601: [1]}, "r")
        with pytest.raises(ValueError, match="no checkpoint seed"):
            MP.paired_difference(treatment, reference, samples=100)

    def test_the_bootstrap_is_deterministic(self, tmp_path):
        treatment = self._arm(tmp_path, "t", {s: [1, 2] for s in (8600, 8601)}, "t")
        reference = self._arm(tmp_path, "r", {s: [3, 4] for s in (8600, 8601)}, "r")
        a = MP.paired_difference(treatment, reference, samples=400)
        b = MP.paired_difference(treatment, reference, samples=400)
        assert (a.ci_low_pts, a.ci_high_pts) == (b.ci_low_pts, b.ci_high_pts)

    def test_the_interval_widens_when_seeds_disagree(self, tmp_path):
        # Same mean delta, but one version has the whole effect in one seed.
        agree_t = self._arm(tmp_path, "at", {s: list(range(10)) for s in (8600, 8601, 8602)}, "t")
        agree_r = self._arm(tmp_path, "ar", {s: list(range(30)) for s in (8600, 8601, 8602)}, "r")
        split_t = self._arm(tmp_path, "st", {8600: list(range(10)), 8601: list(range(30)),
                                             8602: list(range(30))}, "t")
        split_r = self._arm(tmp_path, "sr", {8600: list(range(70)), 8601: list(range(30)),
                                             8602: list(range(30))}, "r")
        agree = MP.paired_difference(agree_t, agree_r, samples=3000)
        split = MP.paired_difference(split_t, split_r, samples=3000)
        assert agree.delta_pts == pytest.approx(split.delta_pts)
        agree_width = agree.ci_high_pts - agree.ci_low_pts
        split_width = split.ci_high_pts - split.ci_low_pts
        assert split_width > agree_width


class TestFromReceipt:
    def test_groups_runs_by_arm_and_preset(self, tmp_path):
        paths = {
            (m, p, s): write_metrics(tmp_path, f"{m}{p}{s}", [1, 2, 3])
            for m in ("fixed", "lucid") for p in ("id_clean", "dr_full") for s in (8600, 8601)
        }
        receipt = {"runs": {
            f"{m}_{p}_{s}": {"mode": m, "preset": p, "checkpoint_seed": s, "metrics_path": str(path)}
            for (m, p, s), path in paths.items()
        }}
        out = MP.outcomes_from_receipt(receipt)
        assert set(out) == {("fixed", "id_clean"), ("fixed", "dr_full"),
                            ("lucid", "id_clean"), ("lucid", "dr_full")}
        assert len(out[("fixed", "id_clean")]) == 2

    def test_a_missing_metrics_file_is_skipped_not_faked(self, tmp_path):
        receipt = {"runs": {"a": {"mode": "fixed", "preset": "id_clean",
                                  "checkpoint_seed": 8600, "metrics_path": str(tmp_path / "nope.json")}}}
        assert MP.outcomes_from_receipt(receipt) == {}

    def test_preset_filter_is_honoured(self, tmp_path):
        path = write_metrics(tmp_path, "x", [1, 2])
        receipt = {"runs": {
            "a": {"mode": "f", "preset": "id_clean", "checkpoint_seed": 8600, "metrics_path": str(path)},
            "b": {"mode": "f", "preset": "dr_full", "checkpoint_seed": 8600, "metrics_path": str(path)},
        }}
        out = MP.outcomes_from_receipt(receipt, presets={"dr_full"})
        assert set(out) == {("f", "dr_full")}


class TestAucWeights:
    def test_weights_sum_to_one_and_match_the_trapezoid(self):
        weights = MP.auc_weights([0.0, 0.5, 1.0, 1.25])
        assert sum(weights) == pytest.approx(1.0)
        assert weights == pytest.approx([0.2, 0.4, 0.3, 0.1])

    def test_an_even_grid_gives_the_classic_half_end_weights(self):
        weights = MP.auc_weights([0.0, 1.0, 2.0])
        assert weights == pytest.approx([0.25, 0.5, 0.25])

    def test_a_non_increasing_grid_is_rejected(self):
        with pytest.raises(ValueError):
            MP.auc_weights([0.0, 0.5, 0.5])
        with pytest.raises(ValueError):
            MP.auc_weights([1.0])


class TestAucScores:
    GRID = {"id_clean": 0.0, "dr_050": 0.5, "dr_full": 1.0, "dr_125": 1.25}

    def _by_preset(self, tmp_path, failed_per_preset, seeds=(8600,)):
        return {
            preset: [
                MP.read_outcome(
                    write_metrics(tmp_path, f"{preset}{s}", failed), s, "m", preset
                )
                for s in seeds
            ]
            for preset, failed in failed_per_preset.items()
        }

    def test_an_all_success_arm_scores_one_per_motion(self, tmp_path):
        by_preset = self._by_preset(tmp_path, {p: [0] for p in self.GRID})
        scores = MP.auc_scores(by_preset, self.GRID)
        # Motion 0 fails everywhere -> 0; every other motion succeeds -> 1.
        assert scores[8600][0] == pytest.approx(0.0)
        assert scores[8600][5] == pytest.approx(1.0)

    def test_the_mean_score_equals_the_trapezoid_auc(self, tmp_path):
        by_preset = self._by_preset(
            tmp_path,
            {"id_clean": list(range(10)), "dr_050": list(range(40)),
             "dr_full": list(range(60)), "dr_125": list(range(80))},
        )
        scores = MP.auc_scores(by_preset, self.GRID)
        rates = [(102 - n) / 102 for n in (10, 40, 60, 80)]
        expected = sum(w * r for w, r in zip(MP.auc_weights([0.0, 0.5, 1.0, 1.25]), rates))
        assert sum(scores[8600]) / 102 == pytest.approx(expected)

    def test_a_seed_missing_one_cell_is_dropped_not_imputed(self, tmp_path):
        by_preset = self._by_preset(tmp_path, {p: [1] for p in self.GRID}, seeds=(8600, 8601))
        by_preset["dr_125"] = [r for r in by_preset["dr_125"] if r.seed == 8600]
        scores = MP.auc_scores(by_preset, self.GRID)
        assert set(scores) == {8600}

    def test_paired_auc_difference_recovers_a_known_gap(self, tmp_path):
        good = self._by_preset(tmp_path, {p: [0] for p in self.GRID}, seeds=(8600, 8601))
        bad = {
            preset: [
                MP.read_outcome(write_metrics(tmp_path, f"b{preset}{s}", list(range(21))),
                                s, "b", preset)
                for s in (8600, 8601)
            ]
            for preset in self.GRID
        }
        out = MP.paired_scores(
            MP.auc_scores(good, self.GRID), MP.auc_scores(bad, self.GRID), samples=2000
        )
        assert out.delta_pts == pytest.approx(100.0 * 20 / 102)
        assert out.excludes_zero is True


class TestFailureListLocation:
    """`failed_idxes` is top level, not inside `eval/failed_metrics_dict`.

    The nested dict holds per-metric arrays for the failed motions and carries
    its own `motion_keys`, which reads exactly like a panel listing. Taking the
    failures from there yields an empty list for a run that really failed 10 of
    102 motions -- a silently wrong answer, not an error.
    """

    def test_top_level_keys_are_used(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": 92 / 102,
            "failed_idxes": list(range(10)),
            "failed_keys": [f"m{i}" for i in range(10)],
            "eval/failed_metrics_dict": {"motion_keys": ["decoy"], "mpjpe_g": [1.0]},
        }))
        out = MP.read_outcome(path, 8600, "origin", "id_clean")
        assert out.motion_count == 102
        assert out.failed_indices == frozenset(range(10))

    def test_nested_keys_are_the_fallback(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": 100 / 102,
            "eval/failed_metrics_dict": {
                "failed_idxes": [3, 7],
                "failed_keys": ["a", "b"],
            },
        }))
        out = MP.read_outcome(path, 8600, "origin", "id_clean")
        assert out.failed_indices == frozenset({3, 7})

    def test_a_rate_implying_failures_with_none_listed_raises(self, tmp_path):
        # The exact shape of the bug: rate says 10 failed, list says none.
        path = tmp_path / "m.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": 92 / 102,
            "eval/failed_metrics_dict": {"motion_keys": ["decoy"]},
        }))
        with pytest.raises(ValueError, match="wrong key"):
            MP.read_outcome(path, 8600, "origin", "id_clean")

    def test_a_perfect_run_still_reports_the_no_failures_reason(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": 1.0,
            "failed_idxes": [], "failed_keys": [],
        }))
        with pytest.raises(ValueError, match="no failures"):
            MP.read_outcome(path, 8600, "origin", "id_clean")


class TestPerfectArmIsNotDropped:
    """A 100%-success arm used to vanish instead of winning."""

    def _receipt(self, tmp_path, name, failed, n=102):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": (n - len(failed)) / n,
            "failed_idxes": list(failed),
            "failed_keys": [f"m{i}" for i in failed],
        }))
        return {"mode": name, "preset": "id_clean", "checkpoint_seed": 8600,
                "metrics_path": str(path), "summary": {"motion_count": n}}

    def test_a_flawless_run_is_read_not_swallowed(self, tmp_path):
        receipt = {"runs": {"perfect": self._receipt(tmp_path, "perfect", [])}}
        out = MP.outcomes_from_receipt(receipt)
        run = out[("perfect", "id_clean")][0]
        assert run.motion_count == 102
        assert run.success_rate == 1.0
        assert sum(run.successes()) == 102

    def test_recorded_count_must_agree_with_the_reported_rate(self, tmp_path):
        run = self._receipt(tmp_path, "bad", [1, 2, 3])
        run["summary"]["motion_count"] = 50   # inconsistent with the 102-based rate
        with pytest.raises(ValueError, match="could not be read"):
            MP.outcomes_from_receipt({"runs": {"bad": run}})

    def test_an_unreadable_run_raises_instead_of_silently_shrinking_the_view(self, tmp_path):
        good = self._receipt(tmp_path, "good", [1, 2])
        bad = dict(good, mode="bad", metrics_path=str(tmp_path / "missing.json"))
        # A missing FILE is still skipped quietly (it was never evaluated)...
        assert set(MP.outcomes_from_receipt({"runs": {"g": good, "b": bad}})) == {("good", "id_clean")}
        # ...but a file that exists and cannot be parsed must raise.
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({"eval/success/success_rate": 0.5}))
        bad2 = dict(good, mode="bad2", metrics_path=str(broken), summary={})
        with pytest.raises(ValueError, match="could not be read"):
            MP.outcomes_from_receipt({"runs": {"g": good, "b": bad2}})
