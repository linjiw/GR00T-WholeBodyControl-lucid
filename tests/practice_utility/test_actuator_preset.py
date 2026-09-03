"""The actuator event preset: schedulable, and comparable to the baseline it extends.

A preset is configuration, so it fails quietly. The two ways it could fail here
are both silent: a term declared "startup" would be invisible to the curriculum
while still appearing in the config, and a shared term whose settings drifted
from lucid_curriculum.yaml would turn a controlled comparison into two different
experiments. Both are checked from the YAML, with no simulator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gear_sonic.research.practice_utility import dr_scaling as DS

EVENTS = Path(__file__).resolve().parents[2] / "gear_sonic/config/manager_env/events"
BASELINE = EVENTS / "tracking/lucid_curriculum.yaml"
ACTUATOR = EVENTS / "tracking/lucid_actuator.yaml"
ACTUATOR_TERMS = (
    "randomize_joint_effort_limit",
    "randomize_joint_friction",
    "randomize_joint_armature",
    "randomize_joint_velocity_limit",
)


def load(path):
    return yaml.safe_load(path.read_text())


def defaults_of(doc):
    return {str(d).split("@")[0].removeprefix("terms/") for d in doc.get("defaults", [])}


def term_doc(name):
    return load(EVENTS / f"terms/{name}.yaml")[name]


# --------------------------------------------------- schedulable, not startup

@pytest.mark.parametrize("name", ACTUATOR_TERMS)
def test_each_actuator_term_runs_at_reset_so_a_curriculum_can_move_it(name):
    """A startup term is applied once and a runtime curriculum cannot move it.

    That is the exact defect lucid_curriculum.yaml was written to fix for four
    other channels; it would be silent here too.
    """
    doc = term_doc(name)
    assert doc["mode"] in DS.RUNTIME_MODES, f"{name} is mode {doc['mode']!r}"


@pytest.mark.parametrize("name", ACTUATOR_TERMS)
def test_each_actuator_term_exposes_a_range_the_scaler_knows(name):
    """The whole integration rests on this: the machinery keys on the param name."""
    params = term_doc(name)["params"]
    ranges = [k for k in params if k in DS.RANGE_NOMINALS]
    assert len(ranges) == 1, f"{name} exposes {ranges}"
    low, high = params[ranges[0]]
    assert high > low, f"{name} range is not ordered"
    nominal = DS.RANGE_NOMINALS[ranges[0]]
    assert low <= nominal <= high or high <= nominal, (
        f"{name} range {[low, high]} does not straddle or approach its nominal {nominal}")


@pytest.mark.parametrize("name", ACTUATOR_TERMS)
def test_each_actuator_term_points_at_the_module_that_implements_it(name):
    func = term_doc(name)["func"]
    module, _, attr = func.partition(":")
    assert module.endswith("events_actuator"), func
    assert attr == name, f"{func} does not implement {name}"

    from gear_sonic.research.practice_utility import events_actuator as EA
    assert callable(getattr(EA, attr))


# ------------------------------------------- comparable to the baseline preset

def test_the_actuator_preset_adds_exactly_the_actuator_terms():
    added = defaults_of(load(ACTUATOR)) - defaults_of(load(BASELINE))
    assert added == set(ACTUATOR_TERMS)


def test_the_actuator_preset_drops_nothing_from_the_baseline():
    missing = defaults_of(load(BASELINE)) - defaults_of(load(ACTUATOR))
    assert missing == set()


def test_the_shared_terms_are_configured_identically_in_both_presets():
    """Otherwise a comparison between the two presets is two experiments, not one."""
    base, act = load(BASELINE), load(ACTUATOR)
    shared = defaults_of(base)
    for term in sorted(shared):
        assert base.get(term) == act.get(term), f"{term} differs between the presets"
    assert base["_target_"] == act["_target_"]


def test_the_baseline_preset_is_untouched_by_this_work():
    """No landed result may change because these channels were added."""
    base = load(BASELINE)
    for name in ACTUATOR_TERMS:
        assert name not in base
        assert name not in defaults_of(base)


def test_the_config_class_has_a_slot_for_every_actuator_term():
    """Hydra cannot fill a field the configclass does not declare."""
    source = (Path(__file__).resolve().parents[2]
              / "gear_sonic/research/practice_utility/events_reset_safe.py").read_text()
    for name in ACTUATOR_TERMS:
        assert f"{name} = None" in source, f"LucidEventCfg has no slot for {name}"


# ------------------------------------------------------- the evaluation cells

def _eval_module():
    from scripts.practice_utility import run_curriculum_robustness_eval as R
    return R


def test_every_actuator_cell_selects_the_preset_that_has_those_terms():
    """A cell naming a term absent from its preset makes the evaluator fail closed."""
    R = _eval_module()
    for cell in R.PRESET_ACTUATOR:
        assert R.PRESETS[cell] == "tracking/lucid_actuator", cell


def test_the_physics_cells_did_not_move_to_the_new_preset():
    """Every landed physics result must stay comparable to itself."""
    R = _eval_module()
    for cell in list(R.PRESET_CHANNEL) + list(R.PRESET_PAIR) + list(R.PRESET_PHYSICS_ONLY):
        assert R.PRESETS[cell] == "tracking/lucid_curriculum", cell


def test_the_off_cell_collapses_every_actuator_channel_to_its_nominal():
    R = _eval_module()
    assert set(R.PRESET_ACTUATOR["act_off"]) == set(R.ACTUATOR_TERMS)
    assert set(R.PRESET_ACTUATOR["act_off"].values()) == {0.0}


def test_every_other_actuator_cell_varies_exactly_one_channel():
    """A severity ladder that moved two channels at once could not attribute a drop."""
    R = _eval_module()
    for cell, scales in R.PRESET_ACTUATOR.items():
        if cell == "act_off":
            continue
        assert set(scales) == set(R.ACTUATOR_TERMS), cell
        active = [n for n, v in scales.items() if v != 0.0]
        assert len(active) == 1, f"{cell} varies {active}"


def test_each_actuator_channel_has_a_ladder_not_a_single_point():
    R = _eval_module()
    per_channel = {}
    for cell, scales in R.PRESET_ACTUATOR.items():
        for name, value in scales.items():
            if value != 0.0:
                per_channel.setdefault(name, []).append(value)
    assert set(per_channel) == set(R.ACTUATOR_TERMS)
    for name, rungs in per_channel.items():
        assert len(rungs) >= 2, f"{name} has no ladder"
        assert len(set(rungs)) == len(rungs), f"{name} has duplicate rungs"


def test_the_actuator_cells_reach_the_evaluator_as_channel_overrides():
    R = _eval_module()
    metadata = R.requested_preset_metadata(["act_effort_150"])
    row = metadata["act_effort_150"]
    assert row["event_preset"] == "tracking/lucid_actuator"
    assert row["channel_dr_scales"]["randomize_joint_effort_limit"] == 1.5
    override = R.channel_override("act_effort_150")
    assert "randomize_joint_effort_limit:1.5" in override


# ------------------------------------------------- the point-versus-range arms

def _train_module():
    from scripts.practice_utility import run_curriculum_comparison as T
    return T


def _args(**over):
    from types import SimpleNamespace
    base = dict(checkpoint="/tmp/m.pt", num_envs=1024, iterations=1500, warmup_iterations=10,
                max_delay=12, delta_target=0.778, kp=1.0, ki=0.02, alpha=0.05, integral_max=1.0,
                return_floor=8.0, exp="e", encoder="/tmp/e.pt", motion_file="m",
                smpl_motion_file="s", consolidation_fraction=0.0, spread_strata=1,
                latency_cap=0.5, return_guard="absolute", return_relative_drop=0.25,
                return_window=8, margin_horizon=12, margin_band_lo=1.1, margin_band_hi=1.3,
                yardstick_envs=64, actuator_channel="effort_limit", actuator_target=0.5,
                gate_threshold=0.80, gate_window=100, gate_dwell=50, gate_min_episodes=200,
                gate_guard_action="freeze", box_channel_budget=300,
                ramp_begin_iteration=0, ramp_end_iteration=1500)
    base.update(over)
    return SimpleNamespace(**base)


def _command(mode, **over):
    from pathlib import Path
    T = _train_module()
    return " ".join(T.build_command(_args(**over), mode=mode, seed=8600,
                                    branch_id="b", artifact_dir=Path("/tmp/a")))


def test_only_the_actuator_arms_switch_event_preset():
    T = _train_module()
    for mode in T.ACTUATOR_ARMS:
        assert "manager_env/events=tracking/lucid_actuator" in _command(mode), mode
    for mode in ("fixed", "off", "gate_150", "prac_push"):
        assert "manager_env/events=tracking/lucid_curriculum" in _command(mode), mode


def test_the_point_arm_trains_every_environment_at_the_target():
    """The whole claim rests on this: a point target has no easy episodes in it."""
    assert "effort_limit_scale_range=[0.5,0.5]" in _command("act_point")


def test_the_range_arm_keeps_the_easy_end_that_makes_it_self_curricularizing():
    assert "effort_limit_scale_range=[0.5,1.0]" in _command("act_range")


def test_the_scheduled_arms_end_where_the_point_arm_sits():
    """Otherwise staging would be compared against a different final distribution."""
    for mode in ("act_ramp", "act_gate"):
        assert "effort_limit_scale_range=[0.5,0.5]" in _command(mode), mode


def test_the_control_arm_leaves_every_actuator_channel_at_its_nominal():
    T = _train_module()
    command = _command("act_off")
    for _term, param, nominal in T.ACTUATOR_RANGE_PARAM.values():
        assert f"{param}=[{nominal},{nominal}]" in command


def test_every_arm_holds_the_channels_it_is_not_testing_at_nominal():
    """A difference between arms must be attributable to one channel."""
    T = _train_module()
    command = _command("act_point")
    for name, (_term, param, nominal) in T.ACTUATOR_RANGE_PARAM.items():
        if name == "effort_limit":
            continue
        assert f"{param}=[{nominal},{nominal}]" in command, name


def test_the_target_channel_is_a_launch_argument_not_a_constant():
    """Which channel is worth training on is decided by the frozen-policy screen."""
    command = _command("act_point", actuator_channel="joint_friction", actuator_target=6.0)
    assert "joint_friction_range=[6.0,6.0]" in command
    assert "effort_limit_scale_range=[1.0,1.0]" in command


def test_an_unknown_channel_or_arm_is_refused():
    T = _train_module()
    with pytest.raises(ValueError, match="not an actuator-barrier arm"):
        T.actuator_overrides("fixed", "effort_limit", 0.5)
    with pytest.raises(ValueError, match="unknown actuator channel"):
        T.actuator_overrides("act_point", "nonsense", 0.5)
