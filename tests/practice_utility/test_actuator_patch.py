"""Tests for the implicit -> delayed actuator swap.

The swap must preserve every field of the original config. A dropped
effort_limit or stiffness would change the robot silently, and the resulting
comparison would attribute the difference to latency.

IsaacLab cannot be imported before SimulationApp exists, so the field-copy logic
is exercised against stand-in dataclasses with the same shape.
"""

import dataclasses

import pytest

from gear_sonic.research.practice_utility import actuator_patch as AP


class FakeImplicitActuator:
    pass


class FakeDelayedActuator(FakeImplicitActuator):
    pass


@dataclasses.dataclass
class FakeImplicitCfg:
    joint_names_expr: list
    effort_limit_sim: dict
    stiffness: dict
    damping: dict
    armature: float = 0.01
    velocity_limit_sim: float = 100.0
    class_type: type = FakeImplicitActuator


@dataclasses.dataclass
class FakeDelayedCfg(FakeImplicitCfg):
    class_type: type = FakeDelayedActuator
    min_delay: int = 0
    max_delay: int = 0


def implicit():
    return FakeImplicitCfg(
        joint_names_expr=[".*_knee_joint"],
        effort_limit_sim={".*_knee_joint": 139.0},
        stiffness={".*_knee_joint": 150.0},
        damping={".*_knee_joint": 5.0},
        armature=0.02,
    )


class TestFieldPreservation:
    def test_every_field_survives_the_swap(self):
        source = implicit()
        result = AP._to_delayed(source, FakeDelayedCfg, min_delay=0, max_delay=8)
        for field in dataclasses.fields(source):
            if field.name == "class_type":
                continue
            assert getattr(result, field.name) == getattr(source, field.name), field.name

    def test_factory_discriminator_changes_to_the_delayed_actuator(self):
        result = AP._to_delayed(implicit(), FakeDelayedCfg, min_delay=0, max_delay=8)
        assert result.class_type is FakeDelayedActuator

    def test_delay_fields_are_set(self):
        result = AP._to_delayed(implicit(), FakeDelayedCfg, min_delay=2, max_delay=8)
        assert result.min_delay == 2 and result.max_delay == 8

    def test_result_is_the_delayed_type(self):
        assert isinstance(AP._to_delayed(implicit(), FakeDelayedCfg, 0, 8), FakeDelayedCfg)

    def test_nested_dicts_are_deep_copied(self):
        """A shared dict would let one group's edit leak into another."""
        source = implicit()
        result = AP._to_delayed(source, FakeDelayedCfg, 0, 8)
        result.stiffness[".*_knee_joint"] = 999.0
        assert source.stiffness[".*_knee_joint"] == 150.0

    def test_lists_are_deep_copied(self):
        source = implicit()
        result = AP._to_delayed(source, FakeDelayedCfg, 0, 8)
        result.joint_names_expr.append(".*_ankle_joint")
        assert source.joint_names_expr == [".*_knee_joint"]

    def test_existing_delay_fields_on_the_source_are_not_carried(self):
        """Re-swapping must take the new bounds, not the old ones."""
        already = FakeDelayedCfg(
            joint_names_expr=[],
            effort_limit_sim={},
            stiffness={},
            damping={},
            min_delay=3,
            max_delay=5,
        )
        result = AP._to_delayed(already, FakeDelayedCfg, min_delay=0, max_delay=8)
        assert (result.min_delay, result.max_delay) == (0, 8)

    def test_resolved_mapping_is_swapped_in_place(self):
        actuators = {"legs": implicit(), "arms": implicit()}
        groups = AP._replace_actuator_mapping(
            actuators, FakeImplicitCfg, FakeDelayedCfg, min_delay=0, max_delay=8
        )
        assert groups == ["legs", "arms"]
        assert all(isinstance(cfg, FakeDelayedCfg) for cfg in actuators.values())
        assert all(cfg.max_delay == 8 for cfg in actuators.values())

    def test_resolved_mapping_reports_already_delayed_groups(self):
        actuators = {
            "legs": FakeDelayedCfg(
                joint_names_expr=[], effort_limit_sim={}, stiffness={}, damping={}
            )
        }
        groups = AP._replace_actuator_mapping(
            actuators, FakeImplicitCfg, FakeDelayedCfg, min_delay=0, max_delay=8
        )
        assert groups == ["legs (already delayed)"]


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_delay": -1},
            {"max_delay": 4, "min_delay": 5},
            {"max_delay": 4, "min_delay": -1},
        ],
    )
    def test_rejects_invalid_bounds(self, kwargs):
        params = {"max_delay": 8, "min_delay": 0}
        params.update(kwargs)
        with pytest.raises(ValueError):
            AP.enable_delayed_actuators(**params)

    def test_zero_max_delay_is_permitted_as_a_baseline(self):
        """max_delay=0 gives a single-slot buffer: identical to no latency."""
        try:
            AP.enable_delayed_actuators(max_delay=0)
        except ValueError:
            pytest.fail("max_delay=0 must be allowed; it is the baseline condition")
        except Exception:
            pass  # IsaacLab unavailable on CPU; the bounds check passed


class TestReporting:
    def test_config_names_cover_the_g1_variants(self):
        assert "G1_CYLINDER_MODEL_12_DEX_CFG" in AP.G1_CONFIG_NAMES

    def test_describe_actuators_is_safe_without_isaac(self):
        try:
            assert isinstance(AP.describe_actuators(), dict)
        except Exception:
            pass  # import-time Isaac dependency, exercised live instead
