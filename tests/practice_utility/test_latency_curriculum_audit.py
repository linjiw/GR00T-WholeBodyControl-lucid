import pytest

from scripts.practice_utility import audit_latency_curriculum as A


def test_pearson_detects_curriculum_realization():
    assert A.pearson([0.0, 0.5, 1.0], [0.0, 2.0, 4.0]) == pytest.approx(1.0)


def test_applied_lambda_is_aligned_to_the_following_rollout():
    curriculum = [
        {"global_step": 1, "lambda": 0.2},
        {"global_step": 2, "lambda": 0.4},
    ]
    observer = [{"global_step": 1}, {"global_step": 2}, {"global_step": 3}]
    assert A.applied_lambda_by_step("lucid", curriculum, observer) == {
        1: 0.0,
        2: 0.2,
        3: 0.4,
    }


def test_fixed_arm_starts_at_full_lambda():
    assert A.applied_lambda_by_step("fixed", [], [{"global_step": 1}]) == {1: 1.0}
