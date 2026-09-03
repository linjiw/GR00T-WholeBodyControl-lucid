"""Gate A: bin selection must be a function of the data, and the verdict frozen.

The gate exists to make "is a curriculum necessary here?" answerable before
anyone spends seeds on curricula, and to make ``curriculum_unnecessary`` a
reportable finding rather than an outcome to be tuned away. These tests pin the
two things that would let it drift: which bin gets chosen, and what the chosen
bin's numbers are allowed to conclude.
"""

import pytest

from gear_sonic.research.practice_utility import learnability_gate as G


def grid(**presets):
    """presets: name -> {mode: success_rate}."""
    return dict(presets)


class TestBinSelection:
    def test_the_hardest_rankable_bin_is_chosen_by_the_reference_arm(self):
        chosen, _ = G.select_hard_bin(grid(
            easy={"origin": 0.90, "fixed": 0.60},
            middling={"origin": 0.66, "fixed": 0.45},
            hard={"origin": 0.56, "fixed": 0.32},
        ))
        assert chosen.preset == "hard"
        assert chosen.reference_pts == pytest.approx(56.0)

    def test_selection_ignores_how_well_the_treatment_does(self):
        # Same reference values, treatment reversed: the choice must not move.
        a, _ = G.select_hard_bin(grid(
            easy={"origin": 0.90, "fixed": 0.10},
            hard={"origin": 0.56, "fixed": 0.90},
        ))
        b, _ = G.select_hard_bin(grid(
            easy={"origin": 0.90, "fixed": 0.90},
            hard={"origin": 0.56, "fixed": 0.10},
        ))
        assert a.preset == b.preset == "hard"

    def test_the_60ms_cells_can_never_be_ranking_bins(self):
        chosen, candidates = G.select_hard_bin(grid(
            latency_60ms={"origin": 0.00, "fixed": 0.30},
            lat_60ms={"origin": 0.00, "fixed": 0.30},
            hard={"origin": 0.56, "fixed": 0.32},
        ))
        assert chosen.preset == "hard"
        banned = {c.preset for c in candidates if c.banned}
        assert banned == {"latency_60ms", "lat_60ms"}
        assert all(not c.rankable for c in candidates if c.banned)

    def test_a_floor_saturated_bin_is_excluded_even_if_not_named(self):
        chosen, candidates = G.select_hard_bin(grid(
            lat_50ms={"origin": 0.0, "fixed": 0.0, "lucid": 0.0},
            hard={"origin": 0.56, "fixed": 0.32},
        ))
        assert chosen.preset == "hard"
        floor = next(c for c in candidates if c.preset == "lat_50ms")
        assert floor.saturated and "floor" in floor.reason

    def test_a_ceiling_saturated_bin_is_excluded(self):
        chosen, candidates = G.select_hard_bin(grid(
            trivial={"origin": 1.0, "fixed": 1.0},
            hard={"origin": 0.56, "fixed": 0.32},
        ))
        assert chosen.preset == "hard"
        assert next(c for c in candidates if c.preset == "trivial").saturated

    def test_a_bin_with_no_spread_cannot_rank(self):
        chosen, candidates = G.select_hard_bin(grid(
            flat={"origin": 0.40, "fixed": 0.405},
            hard={"origin": 0.56, "fixed": 0.32},
        ))
        assert chosen.preset == "hard"
        flat = next(c for c in candidates if c.preset == "flat")
        assert not flat.rankable and "spread" in flat.reason

    def test_nothing_rankable_is_reported_rather_than_guessed(self):
        chosen, candidates = G.select_hard_bin(grid(
            latency_60ms={"origin": 0.0, "fixed": 0.0},
            flat={"origin": 0.40, "fixed": 0.401},
        ))
        assert chosen is None
        assert candidates and all(not c.rankable for c in candidates)

    def test_a_preset_without_the_reference_arm_is_not_a_candidate(self):
        _, candidates = G.select_hard_bin(grid(
            orphan={"fixed": 0.2},
            hard={"origin": 0.56, "fixed": 0.32},
        ))
        assert {c.preset for c in candidates} == {"hard"}

    def test_ties_break_deterministically_on_the_name(self):
        chosen, _ = G.select_hard_bin(grid(
            bbb={"origin": 0.50, "fixed": 0.20},
            aaa={"origin": 0.50, "fixed": 0.40},
        ))
        assert chosen.preset == "aaa"


class TestGateAVerdict:
    THRESHOLDS = G.GateAThresholds(learned_margin_pts=5.0, curriculum_margin_pts=5.0)

    def _score(self, values, curricula=()):
        return G.score_gate_a(
            {"hard": values}, "hard",
            curriculum_modes=curricula, thresholds=self.THRESHOLDS,
        )

    def test_direct_mixed_learning_it_with_no_curriculum_gain_is_unnecessary(self):
        # The finding the guidance insists must be reportable: curriculum is
        # solving a problem that does not arise.
        out = self._score(
            {"origin": 0.40, "fixed": 0.60, "lucid": 0.62}, curricula=("lucid",)
        )
        assert out.direct_mixed_learns_it is True
        assert out.curriculum_adds_on_it is False
        assert out.verdict == "curriculum_unnecessary"
        assert "unnecessary" in out.rationale

    def test_direct_mixed_failing_the_bin_keeps_curriculum_plausible(self):
        out = self._score({"origin": 0.40, "fixed": 0.42}, curricula=())
        assert out.direct_mixed_learns_it is False
        assert out.verdict == "curriculum_plausible"
        assert "does not learn this bin" in out.rationale

    def test_a_curriculum_clearly_beating_direct_mixed_keeps_it_plausible(self):
        out = self._score(
            {"origin": 0.40, "fixed": 0.60, "lucid": 0.70}, curricula=("lucid",)
        )
        assert out.direct_mixed_learns_it is True
        assert out.curriculum_adds_on_it is True
        assert out.verdict == "curriculum_plausible"

    def test_a_marginal_curriculum_gain_does_not_rescue_necessity(self):
        # +4 pts is below the frozen 5 pt margin.
        out = self._score(
            {"origin": 0.40, "fixed": 0.60, "lucid": 0.64}, curricula=("lucid",)
        )
        assert out.verdict == "curriculum_unnecessary"

    def test_the_best_curriculum_arm_is_the_one_compared(self):
        out = self._score(
            {"origin": 0.40, "fixed": 0.60, "a": 0.61, "b": 0.72},
            curricula=("a", "b"),
        )
        assert out.best_curriculum_arm == "b"
        assert out.curriculum_minus_direct_pts == pytest.approx(12.0)

    def test_a_missing_arm_is_not_evaluable_rather_than_a_pass(self):
        out = self._score({"origin": 0.40})
        assert out.verdict == "not_evaluable"
        assert out.direct_mixed_learns_it is False

    def test_thresholds_are_carried_into_the_result(self):
        out = self._score({"origin": 0.40, "fixed": 0.60})
        assert out.to_dict()["thresholds"]["learned_margin_pts"] == 5.0

    def test_deltas_are_in_success_points(self):
        out = self._score({"origin": 0.40, "fixed": 0.60})
        assert out.direct_mixed_minus_origin_pts == pytest.approx(20.0)


class TestReceiptCollapse:
    def test_mode_summary_receipts_collapse_to_preset_by_mode(self):
        receipt = {"mode_summary": {
            "id_clean": {"origin": {"metrics": {"success_rate": {"mean": 0.9}}},
                         "fixed": {"metrics": {"success_rate": {"mean": 0.5}}}},
            "dr_full": {"origin": {"metrics": {"success_rate": {"mean": 0.6}}}},
        }}
        out = G.per_preset_by_mode([receipt])
        assert out["id_clean"] == {"origin": 0.9, "fixed": 0.5}
        assert out["dr_full"] == {"origin": 0.6}

    def test_several_receipts_merge_without_clobbering(self):
        a = {"mode_summary": {"x": {"origin": {"metrics": {"success_rate": {"mean": 0.9}}}}}}
        b = {"mode_summary": {"x": {"fixed": {"metrics": {"success_rate": {"mean": 0.4}}}}}}
        assert G.per_preset_by_mode([a, b])["x"] == {"origin": 0.9, "fixed": 0.4}

    def test_a_missing_mean_is_dropped_not_zeroed(self):
        receipt = {"mode_summary": {"x": {"m": {"metrics": {"success_rate": {}}}}}}
        assert G.per_preset_by_mode([receipt]) == {}
