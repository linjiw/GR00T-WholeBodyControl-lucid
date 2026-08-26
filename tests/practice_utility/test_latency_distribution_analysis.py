import pytest

from scripts.practice_utility import analyze_latency_distribution_sweep as A


def test_exact_sign_flip_test_has_three_seed_resolution_limit():
    assert A.sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(0.25)


def test_surface_distinguishes_absolute_from_relative_performance():
    def mode(value):
        return {
            "metrics": {
                "success_rate": {"mean": value},
                "progress_rate": {"mean": value},
            }
        }

    discovery = {
        "cell_summary": {
            "high_absolute": {
                "cell": {},
                "modes": {"lucid": mode(0.8), "fixed": mode(0.9), "off": mode(0.9)},
            },
            "relative_win": {
                "cell": {},
                "modes": {"lucid": mode(0.6), "fixed": mode(0.4), "off": mode(0.5)},
            },
        }
    }
    rows = A.surface(discovery)
    assert A.absolute_best(rows)["lucid"]["success"]["cell_ids"] == ["high_absolute"]
    relative = next(row for row in rows if row["cell_id"] == "relative_win")
    assert relative["lucid_min_success_margin"] == pytest.approx(0.1)


def test_paired_seed_deltas_preserve_checkpoint_pairing():
    grouped = {
        1: {
            "lucid": {"summary": {"success_rate": 0.7}},
            "fixed": {"summary": {"success_rate": 0.5}},
        },
        2: {
            "lucid": {"summary": {"success_rate": 0.4}},
            "fixed": {"summary": {"success_rate": 0.6}},
        },
    }
    assert A.paired_seed_deltas(grouped, "fixed", "success_rate") == {
        1: pytest.approx(0.2),
        2: pytest.approx(-0.2),
    }
