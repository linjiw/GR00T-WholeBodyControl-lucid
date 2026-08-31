"""Expand-don't-replace support arms (fixed_u / fixed_u150) and the
controller-side anti-collapse guards (monotonic ratchet, competence latch).

The campaign evidence these encode: frontier capability tracks recency-weighted
time at the frontier, the equal-split mixture (25% frontier mass) lost to fixed
at the frontier, and both observed anti-gate collapses cut lambda at peak
competence with zero guard trips.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gear_sonic.research.practice_utility import tace as TACE
from gear_sonic.research.practice_utility.dr_controller import (
    LucidDRController,
    PIConfig,
)
from gear_sonic.research.practice_utility.dr_curriculum import LucidCurriculumCallback
from scripts.practice_utility import run_curriculum_comparison as R


def args():
    return SimpleNamespace(
        checkpoint="/tmp/model.pt",
        num_envs=1024,
        iterations=32,
        warmup_iterations=10,
        max_delay=8,
        delta_target=0.778,
        kp=1.0,
        ki=0.02,
        alpha=0.05,
        integral_max=1.0,
        return_floor=8.0,
        exp="manager/universal_token/all_modes/sonic_release",
        encoder="/tmp/encoder.pt",
        motion_file="data/motion_lib_bones_seed/robot_filtered",
        smpl_motion_file="data/motion_lib_bones_seed/smpl_filtered",
        consolidation_fraction=0.0,
        spread_strata=1,
        latency_cap=0.5,
        return_guard="absolute",
        return_relative_drop=0.25,
        return_window=8,
        margin_horizon=12,
        margin_band_lo=1.10,
        margin_band_hi=1.30,
        yardstick_envs=64,
    )


# ------------------------------------------------------------- sizes helper --


def test_expand_sizes_hold_three_quarters_at_the_frontier():
    sizes = R.expand_stratum_sizes(1024, 8, 0.75)
    assert sizes == [37, 37, 37, 37, 36, 36, 36, 768]
    assert sum(sizes) == 1024


def test_expand_sizes_cover_every_env_at_any_scale():
    for num_focus in (128, 256, 512, 1000, 1024):
        sizes = R.expand_stratum_sizes(num_focus, 8, 0.75)
        assert sum(sizes) == num_focus
        assert sizes[-1] == int(round(0.75 * num_focus))
        assert all(s >= 1 for s in sizes)


def test_expand_sizes_refuse_a_tail_too_thin_for_the_strata():
    with pytest.raises(ValueError, match="too thin"):
        R.expand_stratum_sizes(16, 8, 0.75)


# ---------------------------------------------------------- cohort assignment --


def test_explicit_sizes_partition_the_focus_pool_exactly():
    sizes = (37, 37, 37, 37, 36, 36, 36, 768)
    assignment = TACE.assign_cohorts(
        1024, 0.0, 8600, reserved_focus_ids=(0,), num_strata=8, stratum_sizes=sizes
    )
    realized = tuple(len(ids) for ids in assignment.focus_strata)
    assert realized == sizes
    everyone = [env for stratum in assignment.focus_strata for env in stratum]
    assert sorted(everyone) == list(range(1024))
    # The observer env reads the frontier: reserved ids sit in the top stratum.
    assert 0 in assignment.focus_strata[-1]


def test_explicit_sizes_are_deterministic_and_seed_dependent():
    sizes = (32, 96)
    first = TACE.assign_cohorts(128, 0.0, 8600, num_strata=2, stratum_sizes=sizes)
    again = TACE.assign_cohorts(128, 0.0, 8600, num_strata=2, stratum_sizes=sizes)
    other = TACE.assign_cohorts(128, 0.0, 8601, num_strata=2, stratum_sizes=sizes)
    assert first.focus_strata == again.focus_strata
    assert first.focus_strata != other.focus_strata


def test_none_sizes_reproduce_the_round_robin_split_exactly():
    with_none = TACE.assign_cohorts(
        128, 0.0, 8600, reserved_focus_ids=(0,), num_strata=4, stratum_sizes=None
    )
    legacy = TACE.assign_cohorts(128, 0.0, 8600, reserved_focus_ids=(0,), num_strata=4)
    assert with_none.focus_strata == legacy.focus_strata


def test_explicit_sizes_validate_length_sum_and_top_capacity():
    with pytest.raises(ValueError, match="entries"):
        TACE.assign_cohorts(128, 0.0, 8600, num_strata=4, stratum_sizes=(64, 64))
    with pytest.raises(ValueError, match="sum"):
        TACE.assign_cohorts(128, 0.0, 8600, num_strata=2, stratum_sizes=(64, 65))
    with pytest.raises(ValueError, match="reserved"):
        TACE.assign_cohorts(
            8, 0.0, 8600, reserved_focus_ids=(0, 1), num_strata=2, stratum_sizes=(7, 1)
        )
    with pytest.raises(ValueError, match=">= 1"):
        TACE.assign_cohorts(128, 0.0, 8600, num_strata=2, stratum_sizes=(0, 128))


def test_sizes_count_the_focus_cohort_not_the_whole_population():
    # 16 yardstick envs leave a 112-env focus cohort; sizes speak focus.
    assignment = TACE.assign_cohorts(
        128, 0.0, 8600, num_strata=2, num_yardstick=16, stratum_sizes=(28, 84)
    )
    assert tuple(len(ids) for ids in assignment.focus_strata) == (28, 84)
    with pytest.raises(ValueError, match="sum"):
        TACE.assign_cohorts(128, 0.0, 8600, num_strata=2, num_yardstick=16, stratum_sizes=(32, 96))


def test_callback_validates_sizes_against_strata():
    with pytest.raises(ValueError, match="entries"):
        LucidCurriculumCallback(
            enabled=False, mode="fixed", spread_strata=4, stratum_sizes=(64, 64)
        )


# ------------------------------------------------------------------ launcher --


def test_fixed_u_is_an_in_envelope_mixture_with_a_fat_frontier():
    cmd = R.build_command(args(), "fixed_u", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.mode=fixed" in cmd
    assert "++callbacks.lucid_curriculum.fixed_lambda=1.0" in cmd
    assert "++callbacks.lucid_curriculum.spread_strata=8" in cmd
    assert "++callbacks.lucid_curriculum.stratum_sizes=[37,37,37,37,36,36,36,768]" in cmd
    assert not any("allow_extrapolation" in part for part in cmd)


def test_fixed_u150_extends_the_frontier_and_keeps_the_tail():
    a = args()
    a.max_delay = 12
    cmd = R.build_command(a, "fixed_u150", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.fixed_lambda=1.5" in cmd
    assert "++callbacks.lucid_curriculum.allow_extrapolation=true" in cmd
    assert "++callbacks.lucid_curriculum.spread_strata=8" in cmd
    assert "++callbacks.lucid_curriculum.stratum_sizes=[37,37,37,37,36,36,36,768]" in cmd


def test_fixed_u150_refuses_an_undersized_delay_buffer():
    with pytest.raises(SystemExit, match="max-delay 12"):
        R.build_command(args(), "fixed_u150", 8600, "b", Path("/tmp/artifact"))


def test_consolidation_reaches_unanchored_arms_and_never_off():
    a = args()
    a.consolidation_fraction = 0.1
    for mode in ("lucid_s4_rg", "fixed_u", "fixed"):
        cmd = R.build_command(a, mode, 8600, "b", Path("/tmp/artifact"))
        assert "++callbacks.lucid_curriculum.consolidation_fraction=0.1" in cmd, mode
    cmd = R.build_command(a, "off", 8600, "b", Path("/tmp/artifact"))
    assert not any("consolidation_fraction" in part for part in cmd)


def test_ratchet_arm_is_named_opt_in_and_keeps_the_relative_guard():
    cmd = R.build_command(args(), "lucid_ratchet_rg", 8600, "b", Path("/tmp/artifact"))
    assert "++callbacks.lucid_curriculum.mode=lucid" in cmd
    assert "++callbacks.lucid_curriculum.return_guard=relative" in cmd
    assert "++callbacks.lucid_curriculum.monotonic=true" in cmd
    assert not any("competence_latch" in part for part in cmd)


def test_frozen_margin_arm_does_not_enable_post_queue_anti_collapse_guards():
    cmd = R.build_command(args(), "lucid_margin_s4_rg", 8600, "b", Path("/tmp/artifact"))
    assert not any("monotonic" in part or "competence_latch" in part for part in cmd)


# ---------------------------------------------------------------- controller --


def _drive_to_competence(controller, epochs=60, value=10.0):
    for _ in range(epochs):
        controller.update(gaps=[0.1] * 8, mean_return=value)


def test_monotonic_mode_refuses_the_pi_laws_decreases():
    controller = LucidDRController(
        PIConfig(delta_target=0.5, ki=0.0, monotonic=True), initial_lambda=0.5
    )
    step = controller.update(gaps=[2.0] * 8, mean_return=1.0)  # gap above target
    assert step.lambda_after == pytest.approx(0.5)
    assert step.latch_active is True
    up = controller.update(gaps=[0.0] * 8, mean_return=1.0)  # gap below target
    assert up.lambda_after > 0.5
    assert up.latch_active is False


def test_monotonic_mode_still_lets_the_guard_lower_lambda():
    controller = LucidDRController(
        PIConfig(
            delta_target=0.5,
            monotonic=True,
            return_guard="relative",
            return_window=8,
        ),
        initial_lambda=1.0,
    )
    for value in (10.0, 10.0, 10.0, 10.0):
        controller.update(gaps=[0.1] * 8, mean_return=value)
    tripped = []
    for value in (1.0, 1.0):
        tripped.append(controller.update(gaps=[0.1] * 8, mean_return=value))
    assert any(step.guard_tripped for step in tripped)
    assert controller.lambda_value < 1.0


def test_latch_binds_only_while_the_policy_is_thriving():
    config = PIConfig(delta_target=0.5, ki=0.0, competence_latch=True, latch_window=50)
    controller = LucidDRController(config, initial_lambda=1.0)
    _drive_to_competence(controller, epochs=60, value=10.0)
    held = controller.update(gaps=[2.0] * 8, mean_return=10.0)
    assert held.latch_active is True
    assert held.lambda_after == pytest.approx(1.0)
    # A genuinely failing policy releases the latch and lambda may fall.
    for _ in range(30):
        controller.update(gaps=[2.0] * 8, mean_return=2.0)
    assert controller.lambda_value < 1.0


def test_latch_waits_for_a_full_window_and_compares_like_aggregates():
    config = PIConfig(delta_target=0.5, ki=0.0, competence_latch=True, latch_window=50)
    controller = LucidDRController(config, initial_lambda=1.0)
    for _ in range(48):
        controller.update(gaps=[0.1] * 8, mean_return=10.0)
    early = controller.update(gaps=[2.0] * 8, mean_return=10.0)
    assert early.latch_active is False

    # A noisy best single iteration must not make a healthy trailing mean miss
    # the 0.95 threshold: both the current and best references are window means.
    controller = LucidDRController(config, initial_lambda=1.0)
    for _ in range(49):
        controller.update(gaps=[0.1] * 8, mean_return=10.0)
    held = controller.update(gaps=[2.0] * 8, mean_return=100.0)
    assert held.latch_active is True
    assert held.lambda_after == pytest.approx(1.0)


def test_latch_is_off_by_default_so_existing_arms_reproduce():
    controller = LucidDRController(PIConfig(delta_target=0.5, ki=0.0), initial_lambda=1.0)
    _drive_to_competence(controller, epochs=20, value=10.0)
    step = controller.update(gaps=[2.0] * 8, mean_return=10.0)
    assert step.latch_active is False
    assert step.lambda_after < 1.0


def test_latch_state_survives_a_save_load_round_trip():
    config = PIConfig(delta_target=0.5, competence_latch=True, latch_window=50)
    controller = LucidDRController(config, initial_lambda=0.8)
    _drive_to_competence(controller, epochs=60, value=7.5)
    state = controller.state_dict()
    restored = LucidDRController(config, initial_lambda=0.0)
    restored.load_state_dict(state)
    assert restored.latch_best_mean == controller.latch_best_mean
    assert list(restored.latch_returns) == list(controller.latch_returns)


def test_fixed_u150_telemetry_uses_the_applied_frontier_not_the_capped_controller():
    callback = LucidCurriculumCallback(
        enabled=True,
        mode="fixed",
        fixed_lambda=1.5,
        allow_extrapolation=True,
        spread_strata=8,
    )
    callback.assignment = TACE.assign_cohorts(1024, 0.0, 8600, num_strata=8)
    callback._apply(1.5)
    assert callback.controller.lambda_value == pytest.approx(1.0)
    assert callback._tace_telemetry()["stratum_lambdas"] == pytest.approx(
        [0.1875, 0.375, 0.5625, 0.75, 0.9375, 1.125, 1.3125, 1.5]
    )
