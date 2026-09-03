"""Practice-allocation arms: where is extra training productive?

These arms are the open-loop instrument for the question that comes before any
scheduler. Each one keeps the architecture, reward, motion, origin checkpoint,
environment count and iteration budget fixed, and changes only what a fixed 25%
share of the same environments practises. What the tests below pin is exactly
the property that makes a difference between two arms interpretable:

* the share is REALLOCATED, never added, so no arm trains on more episodes;
* every arm reallocates the SAME share, so the arms differ only in content;
* the practised levels come from the measured single-channel sweep rather than
  from an intuition about which physics is hard;
* nothing in the arm can move during a run, read a signal, or contract support.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gear_sonic.research.practice_utility.dr_curriculum import LucidCurriculumCallback
from scripts.practice_utility import run_curriculum_comparison as R
from scripts.practice_utility import run_curriculum_robustness_eval as E


def args(**overrides):
    base = SimpleNamespace(
        checkpoint="/tmp/model.pt",
        num_envs=1024,
        iterations=1500,
        warmup_iterations=10,
        max_delay=12,
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
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


# --------------------------------------------------------- the allocation --


def test_every_practice_arm_reallocates_the_same_share():
    """Equal share, so a difference between arms is about content, not budget."""
    sizes = {
        arm: R.expand_stratum_sizes(1024, 2, R.ARM_TOP_FRACTION[arm]) for arm in R.PRACTICE_ARMS
    }
    assert all(size == [768, 256] for size in sizes.values()), sizes
    for size in sizes.values():
        assert sum(size) == 1024  # reallocated out of the same pool, never added


def test_practice_arms_pin_lambda_one_and_two_strata():
    for arm in R.PRACTICE_ARMS:
        mode, anchor, _ = R.ARMS[arm]
        assert mode == "fixed"  # inert controller: no signal can move this arm
        assert anchor == 0.0
        assert R.ARM_SPREAD_STRATA[arm] == 2
        assert R.ARM_FIXED_LAMBDA.get(arm, 1.0) == 1.0  # the base cohort stays at the envelope


def test_the_null_arm_practises_nothing():
    """The matched control: dispatcher active, content identical to the base."""
    assert R.PRACTICE_CHANNELS["prac_null"] == {}
    assert R.practice_vectors("prac_null") == [{}, {}]
    assert R.ARM_PRACTICE_MAX["prac_null"] == 1.0


def test_practice_vectors_leave_the_retained_stratum_untouched():
    for arm in R.PRACTICE_ARMS:
        vectors = R.practice_vectors(arm)
        assert len(vectors) == R.ARM_SPREAD_STRATA[arm]
        assert vectors[0] == {}, "the retained stratum must train where the control arm does"
        assert vectors[1] == R.PRACTICE_CHANNELS[arm]


def test_practised_channels_are_real_scalable_terms():
    for arm, channels in R.PRACTICE_CHANNELS.items():
        assert set(channels) <= R.EXPECTED_SCALABLE_TERMS, arm
        assert "randomize_action_delay" not in channels, "no arm widens latency"


def test_practice_levels_match_the_measured_sweep():
    """The levels are read off the attribution sweep, not chosen.

    Origin (fixed@s8600) single-channel success: push 3x = 0.746 (hard, far from
    the floor), mass 3x = 0.949, CoM 3x = 0.988, joint 3x = 0.990 (already
    manageable), push 2x = 0.912 and friction 1.5x = 0.973 (each cheap alone).
    """
    assert R.PRACTICE_CHANNELS["prac_push"] == {"push_robot": 3.0}
    assert R.PRACTICE_CHANNELS["prac_easy"] == {
        "randomize_rigid_body_mass": 3.0,
        "base_com": 3.0,
        "add_joint_default_pos": 3.0,
    }
    assert R.PRACTICE_CHANNELS["prac_fric"] == {"physics_material": 1.5}
    assert R.PRACTICE_CHANNELS["prac_pushfric"] == {"push_robot": 3.0, "physics_material": 1.5}


def test_the_four_practice_arms_form_a_two_by_two():
    """Amendment A1: the interaction must be estimable, not confounded with dose.

    prac_pushfric practises push at exactly the level prac_push does and friction at
    exactly the level prac_fric does, so the only thing that varies between the four
    arms is which factors are present.
    """
    null = R.PRACTICE_CHANNELS["prac_null"]
    push = R.PRACTICE_CHANNELS["prac_push"]
    fric = R.PRACTICE_CHANNELS["prac_fric"]
    both = R.PRACTICE_CHANNELS["prac_pushfric"]
    assert null == {}
    assert both == {**push, **fric}
    assert set(push) & set(fric) == set()


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="not a practice-allocation arm"):
        R.practice_vectors("fixed")


# ------------------------------------------------------------- the launch --


def test_launch_emits_the_frozen_allocation_and_the_matched_sizes():
    for arm in R.PRACTICE_ARMS:
        command = " ".join(R.build_command(args(), mode=arm, seed=8600, branch_id="b", artifact_dir=Path("/tmp/a")))
        assert "spread_strata=2" in command
        assert "stratum_sizes=[768,256]" in command
        assert "practice_vectors_json=" in command
        payload = command.split("practice_vectors_json='")[1].split("'")[0]
        assert json.loads(payload) == R.practice_vectors(arm)


def test_only_the_extrapolating_arms_ask_for_extrapolation():
    for arm in R.PRACTICE_ARMS:
        command = " ".join(R.build_command(args(), mode=arm, seed=8600, branch_id="b", artifact_dir=Path("/tmp/a")))
        wants = "allow_extrapolation=true" in command
        assert wants == (R.ARM_PRACTICE_MAX[arm] > 1.0), arm


def test_the_delay_buffer_check_reads_the_latency_ceiling_not_the_practice_level():
    """A push arm reaches 3x on push and 1x on latency; the buffer sizes on latency."""
    for arm in R.PRACTICE_ARMS:
        assert R.ARM_DELAY_CEILING[arm] == 1.0
        R.build_command(args(max_delay=8), mode=arm, seed=8600, branch_id="b", artifact_dir=Path("/tmp/a"))


# ---------------------------------------------------------- the callback --


def parse(vectors, strata=2):
    return LucidCurriculumCallback(
        enabled=True,
        mode="fixed",
        spread_strata=strata,
        practice_vectors_json=json.dumps(vectors),
    )


def test_the_callback_parses_one_vector_per_stratum():
    callback = parse([{}, {"push_robot": 3.0}])
    assert callback.practice_vectors == ({}, {"push_robot": 3.0})


def test_a_vector_count_that_disagrees_with_the_strata_is_refused():
    with pytest.raises(ValueError, match="spread_strata"):
        parse([{}, {}, {}], strata=2)


def test_a_negative_intensity_is_refused():
    with pytest.raises(ValueError, match="must be >= 0"):
        parse([{}, {"push_robot": -1.0}])


def test_no_allocation_leaves_the_callback_exactly_as_it_was():
    callback = LucidCurriculumCallback(enabled=True, mode="fixed")
    assert callback.practice_vectors is None


# ------------------------------------------------------------ the cells --


def test_the_above_practice_cell_sits_above_every_practised_level():
    highest = max(
        value for channels in R.PRACTICE_CHANNELS.values() for value in channels.values()
    )
    assert E.PRESET_CHANNEL["ch_push_350"]["push_robot"] > highest


def test_the_pair_cells_name_two_terms_and_bracket_the_practised_corner():
    for preset, scales in E.PRESET_PAIR.items():
        assert len(scales) == 2
        assert preset in E.PRESETS
        assert set(scales) <= R.EXPECTED_SCALABLE_TERMS
    practised = R.PRACTICE_CHANNELS["prac_pushfric"]
    assert E.PRESET_PAIR["ch_push_fric_300_150"] == practised
    above = E.PRESET_PAIR["ch_push_fric_350_150"]
    assert above["push_robot"] > practised["push_robot"]
    assert above["physics_material"] == practised["physics_material"]


def test_every_scaled_cell_stays_inside_the_extrapolation_cap():
    from gear_sonic.research.practice_utility import dr_scaling as DS

    for preset, scales in E.PRESET_SCALED.items():
        for term, value in scales.items():
            assert 0.0 <= value <= DS.MAX_EXTRAPOLATION, (preset, term, value)


def test_the_pair_cells_reach_the_evaluator_as_channel_overrides():
    metadata = E.requested_preset_metadata(["ch_push_fric_300_150"])
    row = metadata["ch_push_fric_300_150"]
    assert row["channel_dr_scales"] == {"push_robot": 3.0, "physics_material": 1.5}
    assert row["non_latency_dr_scale"] == 1.0 and row["fixed_latency_steps"] == 0
    override = E.channel_override("ch_push_fric_300_150")
    assert "physics_material:1.5" in override and "push_robot:3.0" in override
