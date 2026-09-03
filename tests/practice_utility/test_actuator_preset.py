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
