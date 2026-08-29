"""Capability metrics: the arithmetic, and the edge cases that would mislead."""

import json

import pytest

from gear_sonic.research.practice_utility import robustness_metrics as RM


def write_cell(tmp_path, name, progress, keys=None):
    """A metrics_eval.json whose per-episode arrays are self-consistent."""
    progress = [float(g) for g in progress]
    terminated = [g < 1.0 for g in progress]
    keys = keys or [f"m{i}" for i in range(len(progress))]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "eval/success/success_rate": sum(0 if t else 1 for t in terminated) / len(progress),
        "eval/success/progress_rate": sum(progress) / len(progress),
        "eval/all_metrics_dict": {
            "terminated": terminated, "progress": progress, "motion_keys": keys,
        },
    }))
    return path


def cell(tmp_path, name, progress, preset=None, difficulty=None, keys=None):
    return RM.read_cell(write_cell(tmp_path, name, progress, keys), preset or name, difficulty)


GRID = {"phys_000": 0.0, "phys_025": 0.25, "phys_050": 0.5, "phys_075": 0.75, "phys_100": 1.0}


class TestReadCell:
    def test_reads_per_episode_arrays(self, tmp_path):
        c = cell(tmp_path, "a", [1.0, 1.0, 0.5, 0.25])
        assert c.num_episodes == 4
        assert c.success == (1, 1, 0, 0)
        assert c.success_rate == pytest.approx(0.5)
        assert c.progress_rate == pytest.approx(0.6875)

    def test_rejects_arrays_that_disagree_with_the_recorded_rate(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "eval/success/success_rate": 0.99,          # inconsistent on purpose
            "eval/success/progress_rate": 0.6875,
            "eval/all_metrics_dict": {
                "terminated": [False, False, True, True],
                "progress": [1.0, 1.0, 0.5, 0.25],
                "motion_keys": ["a", "b", "c", "d"],
            },
        }))
        with pytest.raises(ValueError, match="two views of the same tensor disagree"):
            RM.read_cell(path, "bad")

    def test_rejects_success_that_is_not_full_progress(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "eval/all_metrics_dict": {
                "terminated": [False], "progress": [0.5], "motion_keys": ["a"],
            },
        }))
        with pytest.raises(ValueError, match="success must mean progress == 1"):
            RM.read_cell(path, "bad")

    def test_rejects_a_file_with_no_per_episode_arrays(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"eval/success/success_rate": 0.5}))
        with pytest.raises(ValueError, match="no per-episode"):
            RM.read_cell(path, "bare")


class TestDifficultyCurve:
    def _cells(self, tmp_path, values):
        keys = [f"m{i}" for i in range(8)]
        return {
            p: cell(tmp_path, p, [v] * 8, preset=p, difficulty=GRID[p], keys=keys)
            for p, v in values.items()
        }

    def test_auc_weights_sum_to_one_and_read_on_the_success_scale(self, tmp_path):
        flat = self._cells(tmp_path, {p: 0.6 for p in GRID})
        out = RM.difficulty_curve(flat, GRID)
        assert sum(out.weights.values()) == pytest.approx(1.0)
        assert out.capability_auc == pytest.approx(0.6)

    def test_the_two_policies_the_mean_cannot_distinguish(self, tmp_path):
        # (100, 100, 40) vs (80, 80, 80) -- the user's example, extended to the
        # five-point grid. Similar means, very different tails.
        spiky = self._cells(tmp_path, {"phys_000": 1.0, "phys_025": 1.0, "phys_050": 1.0,
                                       "phys_075": 0.4, "phys_100": 0.4})
        flat = self._cells(tmp_path, {p: 0.8 for p in GRID})
        a, b = RM.difficulty_curve(spiky, GRID), RM.difficulty_curve(flat, GRID)
        assert abs(a.capability_auc - b.capability_auc) < 0.03   # AUC barely separates them
        assert a.worst_value == pytest.approx(0.4)               # the worst cell does
        assert b.worst_value == pytest.approx(0.8)

    def test_a_missing_cell_is_an_error_not_a_gap(self, tmp_path):
        partial = self._cells(tmp_path, {p: 0.5 for p in list(GRID)[:3]})
        with pytest.raises(ValueError, match="incomplete"):
            RM.difficulty_curve(partial, GRID)

    def test_cells_from_different_panels_cannot_be_combined(self, tmp_path):
        cells = self._cells(tmp_path, {p: 0.5 for p in GRID})
        cells["phys_100"] = cell(tmp_path, "other", [0.5] * 8, preset="phys_100",
                                 keys=[f"x{i}" for i in range(8)])
        with pytest.raises(ValueError, match="different panels"):
            RM.difficulty_curve(cells, GRID)

    def test_success_and_progress_scores_differ(self, tmp_path):
        cells = self._cells(tmp_path, {p: 0.5 for p in GRID})
        assert RM.difficulty_curve(cells, GRID, score="progress").capability_auc == pytest.approx(0.5)
        assert RM.difficulty_curve(cells, GRID, score="success").capability_auc == pytest.approx(0.0)


class TestCvar:
    def test_averages_the_worst_fraction(self, tmp_path):
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        out = RM.cvar(values, alpha=0.2)
        assert out["cvar"] == pytest.approx(0.05)          # mean of 0.0 and 0.1
        assert out["mean"] == pytest.approx(0.45)

    def test_a_fractional_tail_is_interpolated_not_rounded(self):
        values = [0.0, 0.0, 0.0, 1.0, 1.0]                  # alpha*n = 1.5
        out = RM.cvar(values, alpha=0.3)
        assert out["tail_size"] == pytest.approx(1.5)
        assert out["cvar"] == pytest.approx(0.0)
        # and a tail that straddles the boundary picks up a partial weight
        out2 = RM.cvar([0.0, 1.0, 1.0, 1.0, 1.0], alpha=0.3)
        assert out2["cvar"] == pytest.approx((0.0 + 0.5 * 1.0) / 1.5)

    def test_a_flawless_policy_has_cvar_one(self):
        assert RM.cvar([1.0] * 20, alpha=0.2)["cvar"] == pytest.approx(1.0)

    def test_binary_success_is_refused(self):
        # The degeneracy this refusal exists for: two policies failing 30% and
        # 90% of episodes both have a success-CVaR of exactly 0.
        with pytest.raises(ValueError, match="degenerates"):
            RM.cvar([0, 1, 1, 1], alpha=0.2, score="success")

    def test_empty_and_bad_alpha_are_errors(self):
        with pytest.raises(ValueError):
            RM.cvar([], alpha=0.2)
        with pytest.raises(ValueError):
            RM.cvar([0.5], alpha=0.0)

    def test_pooled_tail_spans_every_cell(self, tmp_path):
        keys = [f"m{i}" for i in range(4)]
        cells = [cell(tmp_path, "easy", [1.0] * 4, keys=keys),
                 cell(tmp_path, "hard", [0.0] * 4, keys=keys)]
        out = RM.pooled_tail(cells, alpha=0.5)
        assert out["n"] == 8
        assert out["cvar"] == pytest.approx(0.0)
        assert out["pooled_over"] == ["easy", "hard"]


class TestRetentionMatrix:
    def _matrix(self, tmp_path, table):
        keys = [f"m{i}" for i in range(4)]
        return {
            row: {p: cell(tmp_path, f"{row}_{p}", [v] * 4, preset=p, keys=keys)
                  for p, v in cols.items()}
            for row, cols in table.items()
        }

    G = {"easy": 0.0, "medium": 0.5, "hard": 1.0}

    def test_forgetting_is_detected(self, tmp_path):
        # The user's traditional-curriculum example: easy peaks at 0.95, ends 0.74.
        cells = self._matrix(tmp_path, {
            "h05": {"easy": 0.91, "medium": 0.42, "hard": 0.04},
            "h10": {"easy": 0.95, "medium": 0.71, "hard": 0.18},
            "h15": {"easy": 0.92, "medium": 0.88, "hard": 0.53},
            "h20": {"easy": 0.82, "medium": 0.90, "hard": 0.76},
            "h30": {"easy": 0.74, "medium": 0.87, "hard": 0.91},
        })
        out = RM.retention_matrix(cells, self.G, ["h05", "h10", "h15", "h20", "h30"])
        assert out.forgetting_column == "easy"
        assert out.forgetting == pytest.approx(21.0)
        assert out.accumulating is False

    def test_accumulation_is_distinguished(self, tmp_path):
        # The LUCID-Mix example: every column holds its peak.
        cells = self._matrix(tmp_path, {
            "h05": {"easy": 0.91, "medium": 0.42, "hard": 0.04},
            "h10": {"easy": 0.95, "medium": 0.72, "hard": 0.21},
            "h15": {"easy": 0.95, "medium": 0.88, "hard": 0.57},
            "h20": {"easy": 0.94, "medium": 0.91, "hard": 0.80},
            "h30": {"easy": 0.94, "medium": 0.92, "hard": 0.91},
        })
        out = RM.retention_matrix(cells, self.G, ["h05", "h10", "h15", "h20", "h30"])
        assert out.forgetting == pytest.approx(1.0)
        assert out.accumulating is True

    def test_the_two_examples_are_separated_automatically(self, tmp_path):
        """The point of the scalar: no eyeballing."""
        forget = RM.retention_matrix(self._matrix(tmp_path, {
            "a": {"easy": 0.95, "medium": 0.7, "hard": 0.2},
            "b": {"easy": 0.74, "medium": 0.87, "hard": 0.91},
        }), self.G, ["a", "b"])
        keep = RM.retention_matrix(self._matrix(tmp_path, {
            "c": {"easy": 0.95, "medium": 0.7, "hard": 0.2},
            "d": {"easy": 0.94, "medium": 0.92, "hard": 0.91},
        }), self.G, ["c", "d"])
        assert forget.accumulating is False and keep.accumulating is True
        assert forget.forgetting > keep.forgetting

    def test_a_missing_checkpoint_cell_is_an_error(self, tmp_path):
        cells = self._matrix(tmp_path, {"a": {"easy": 0.9, "medium": 0.5, "hard": 0.1}})
        del cells["a"]["hard"]
        with pytest.raises(ValueError, match="missing difficulty cells"):
            RM.retention_matrix(cells, self.G, ["a"])

    def test_one_checkpoint_says_so_rather_than_claiming_no_forgetting(self, tmp_path):
        cells = self._matrix(tmp_path, {"a": {"easy": 0.9, "medium": 0.5, "hard": 0.1}})
        out = RM.retention_matrix(cells, self.G, ["a"])
        assert out.forgetting == pytest.approx(0.0)
        assert any("single checkpoint" in n for n in out.notes)


class TestTargetExpectation:
    def test_weights_by_the_deployment_prior_not_the_trapezoid(self, tmp_path):
        keys = [f"m{i}" for i in range(4)]
        cells = {p: cell(tmp_path, p, [v] * 4, preset=p, keys=keys)
                 for p, v in {"phys_000": 1.0, "phys_050": 0.5, "phys_100": 0.0}.items()}
        prior = {"phys_000": 0.6, "phys_050": 0.3, "phys_100": 0.1}
        out = RM.target_expectation(cells, prior)
        assert out["expectation"] == pytest.approx(0.75)

    def test_a_prior_that_does_not_sum_to_one_is_rejected(self, tmp_path):
        keys = ["m0"]
        cells = {"phys_000": cell(tmp_path, "phys_000", [1.0], keys=keys)}
        with pytest.raises(ValueError, match="sum to 1"):
            RM.target_expectation(cells, {"phys_000": 0.9})

    def test_a_prior_naming_an_unevaluated_cell_is_rejected(self, tmp_path):
        keys = ["m0"]
        cells = {"phys_000": cell(tmp_path, "phys_000", [1.0], keys=keys)}
        with pytest.raises(ValueError, match="not evaluated"):
            RM.target_expectation(cells, {"phys_000": 0.5, "phys_100": 0.5})
