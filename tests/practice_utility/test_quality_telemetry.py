"""Tests for live quality telemetry collection.

The behaviour that matters most: an unavailable signal is *recorded as missing*,
never reported as zero. A zero would read as "no slip" when it means "not
measured", which is exactly the kind of silent hole that makes a harm gate
useless.
"""

import json

import pytest
import torch

from gear_sonic.research.practice_utility import quality_metrics as QM
from gear_sonic.research.practice_utility import quality_telemetry as QT

BODIES = [
    "pelvis", "torso_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_elbow_link", "right_elbow_link",
]
FOOT_IDS = [2, 3]
NUM_ENVS, NUM_JOINTS = 4, 29


class Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeRobot:
    def __init__(self, foot_speed=0.0, torque_ratio=0.0, joint_at_limit=False):
        self.body_names = list(BODIES)
        velocity = torch.zeros(NUM_ENVS, len(BODIES), 3)
        velocity[:, FOOT_IDS, 0] = foot_speed
        limits = torch.zeros(NUM_ENVS, NUM_JOINTS, 2)
        limits[..., 0], limits[..., 1] = -1.0, 1.0
        effort = torch.full((NUM_ENVS, NUM_JOINTS), 100.0)
        self.data = Data(
            body_lin_vel_w=velocity,
            applied_torque=torch.full((NUM_ENVS, NUM_JOINTS), 100.0 * torque_ratio),
            joint_vel=torch.full((NUM_ENVS, NUM_JOINTS), 2.0),
            joint_pos=torch.full((NUM_ENVS, NUM_JOINTS), 1.0 if joint_at_limit else 0.0),
            soft_joint_pos_limits=limits,
            joint_effort_limits=effort,
        )

    def find_bodies(self, names):
        return ([i for i, n in enumerate(self.body_names) if n in set(names)], names)


class FakeSensor:
    def __init__(self, foot_force=0.0, torso_force=0.0):
        forces = torch.zeros(NUM_ENVS, len(BODIES), 3)
        forces[:, FOOT_IDS, 2] = foot_force
        forces[:, 1, 2] = torso_force          # torso == an undesired contact
        self.data = Data(net_forces_w=forces)


class FakeScene:
    def __init__(self, robot=None, sensor=None):
        self._entities = {}
        if robot is not None:
            self._entities["robot"] = robot
        if sensor is not None:
            self._entities["contact_forces"] = sensor

    def __getitem__(self, key):
        return self._entities[key]


class FakeEnv:
    def __init__(self, robot=None, sensor=None):
        self.scene = FakeScene(robot, sensor)


class NestedEnv:
    """Mimics a wrapper holding the real env on ``.env``."""

    def __init__(self, robot=None, sensor=None):
        self.env = FakeEnv(robot, sensor)


class TestFootSlip:
    def test_no_contact_means_no_slip(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        collector.observe(FakeEnv(FakeRobot(foot_speed=5.0), FakeSensor(foot_force=0.0)))
        assert collector.snapshot()["foot_slip_total_m"] == 0.0

    def test_contact_with_motion_is_slip(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        collector.observe(FakeEnv(FakeRobot(foot_speed=1.0), FakeSensor(foot_force=100.0)))
        # two feet, 1 m/s each, one step of 0.02 s
        assert collector.snapshot()["foot_slip_total_m"] == pytest.approx(0.04)

    def test_contact_without_motion_is_not_slip(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        collector.observe(FakeEnv(FakeRobot(foot_speed=0.0), FakeSensor(foot_force=100.0)))
        assert collector.snapshot()["foot_slip_total_m"] == 0.0

    def test_force_below_threshold_is_not_contact(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        collector.observe(FakeEnv(FakeRobot(foot_speed=1.0), FakeSensor(foot_force=5.0)))
        assert collector.snapshot()["foot_slip_total_m"] == 0.0

    def test_slip_accumulates_over_steps(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        env = FakeEnv(FakeRobot(foot_speed=1.0), FakeSensor(foot_force=100.0))
        for _ in range(5):
            collector.observe(env)
        assert collector.snapshot()["foot_slip_total_m"] == pytest.approx(0.2)


class TestContacts:
    def test_impulse_integrates_force(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        collector.observe(FakeEnv(FakeRobot(), FakeSensor(foot_force=100.0)))
        # two feet at 100 N for 0.02 s
        assert collector.snapshot()["contact_impulse_total"] == pytest.approx(4.0)

    def test_peak_force_is_tracked(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(), FakeSensor(foot_force=100.0)))
        collector.observe(FakeEnv(FakeRobot(), FakeSensor(foot_force=900.0)))
        assert collector.snapshot()["contact_force_peak"] == pytest.approx(900.0)

    def test_foot_contact_is_not_undesired(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(), FakeSensor(foot_force=500.0)))
        assert collector.snapshot()["undesired_contact_rate"] == 0.0

    def test_torso_contact_is_undesired(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(), FakeSensor(torso_force=500.0)))
        assert collector.snapshot()["undesired_contact_rate"] == pytest.approx(1.0)


class TestActuator:
    def test_torque_at_limit_saturates(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(torque_ratio=1.0), FakeSensor()))
        assert collector.snapshot()["torque_saturation"] == pytest.approx(1.0)

    def test_low_torque_does_not_saturate(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(torque_ratio=0.1), FakeSensor()))
        assert collector.snapshot()["torque_saturation"] == 0.0

    def test_joint_at_limit_reads_full_proximity(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(joint_at_limit=True), FakeSensor()))
        assert collector.snapshot()["joint_limit_proximity"] == pytest.approx(1.0)

    def test_joint_mid_range_reads_zero_proximity(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(joint_at_limit=False), FakeSensor()))
        assert collector.snapshot()["joint_limit_proximity"] == pytest.approx(0.0)

    def test_energy_is_absolute_power(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(torque_ratio=0.5), FakeSensor()))
        # 29 joints x |50 Nm x 2 rad/s|
        assert collector.snapshot()["energy_proxy"] == pytest.approx(29 * 100.0)


class TestActionSmoothness:
    def test_first_step_has_no_rate(self):
        collector = QT.QualityTelemetryCollector()
        env = FakeEnv(FakeRobot(), FakeSensor())
        collector.observe(env, torch.zeros(NUM_ENVS, NUM_JOINTS))
        assert collector.snapshot()["action_rate"] == 0.0

    def test_constant_action_has_no_rate(self):
        collector = QT.QualityTelemetryCollector()
        env = FakeEnv(FakeRobot(), FakeSensor())
        for _ in range(4):
            collector.observe(env, torch.ones(NUM_ENVS, NUM_JOINTS))
        assert collector.snapshot()["action_rate"] == 0.0

    def test_changing_action_has_rate(self):
        collector = QT.QualityTelemetryCollector()
        env = FakeEnv(FakeRobot(), FakeSensor())
        for i in range(4):
            collector.observe(env, torch.full((NUM_ENVS, NUM_JOINTS), float(i)))
        assert collector.snapshot()["action_rate"] > 0.0

    def test_alternating_action_has_acceleration(self):
        collector = QT.QualityTelemetryCollector()
        env = FakeEnv(FakeRobot(), FakeSensor())
        for i in range(6):
            collector.observe(env, torch.full((NUM_ENVS, NUM_JOINTS), float(i % 2)))
        assert collector.snapshot()["action_acceleration"] > 0.0


class TestMissingSignalsAreNotZeros:
    def test_absent_sensor_is_reported_missing(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(), sensor=None))
        missing = collector.snapshot()["missing_signals"]
        assert "foot_slip" in missing and "contact" in missing

    def test_absent_robot_is_reported_missing(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(robot=None, sensor=FakeSensor()))
        assert "robot" in collector.snapshot()["missing_signals"]

    def test_absent_robot_does_not_count_a_step(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(robot=None))
        assert collector.snapshot()["steps"] == 0

    def test_absent_effort_limits_flag_saturation_missing(self):
        robot = FakeRobot()
        del robot.data.joint_effort_limits
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(robot, FakeSensor()))
        assert "torque_saturation" in collector.snapshot()["missing_signals"]

    def test_missing_flags_survive_a_reset(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(FakeEnv(FakeRobot(), sensor=None))
        collector.reset()
        assert "foot_slip" in collector.snapshot()["missing_signals"]

    def test_observe_never_raises(self):
        collector = QT.QualityTelemetryCollector()
        collector.observe(object())          # nothing resembling an env
        assert "robot" in collector.snapshot()["missing_signals"]


class TestEnvAccess:
    def test_finds_entities_on_a_plain_env(self):
        assert QT._scene_entity(FakeEnv(FakeRobot(), FakeSensor()), "robot") is not None

    def test_finds_entities_through_a_wrapper(self):
        assert QT._scene_entity(NestedEnv(FakeRobot(), FakeSensor()), "contact_forces") is not None

    def test_returns_none_for_an_unknown_entity(self):
        assert QT._scene_entity(FakeEnv(FakeRobot()), "camera") is None

    def test_nested_collection_works_end_to_end(self):
        collector = QT.QualityTelemetryCollector(step_dt=0.02)
        collector.observe(NestedEnv(FakeRobot(foot_speed=1.0), FakeSensor(foot_force=100.0)))
        assert collector.snapshot()["foot_slip_total_m"] == pytest.approx(0.04)

    def test_find_bodies_falls_back_to_names(self):
        """Older asset wrappers expose body_names but no find_bodies helper."""

        class NoFinder(FakeRobot):
            find_bodies = None

        assert QT._find_bodies(NoFinder(), list(QT.FOOT_BODIES)) == FOOT_IDS

    def test_find_bodies_falls_back_when_the_helper_raises(self):
        class Broken(FakeRobot):
            def find_bodies(self, names):
                raise RuntimeError("asset not initialized")

        assert QT._find_bodies(Broken(), list(QT.FOOT_BODIES)) == FOOT_IDS

    def test_find_bodies_returns_empty_without_any_naming(self):
        class Nameless:
            body_names = None

        assert QT._find_bodies(Nameless(), ["x"]) == []


class TestCallback:
    def test_disabled_collects_nothing(self, tmp_path):
        callback = QT.PracticeQualityCallback(enabled=False, output_dir=str(tmp_path))
        callback.observe(FakeEnv(FakeRobot(), FakeSensor()), torch.zeros(4, 29))
        callback.on_step_end(None, type("S", (), {"global_step": 1})(), None)
        assert callback.history == [] and not list(tmp_path.glob("*.jsonl"))

    def test_enabled_writes_one_record_per_iteration(self, tmp_path):
        state = type("S", (), {"global_step": 1})()
        callback = QT.PracticeQualityCallback(
            enabled=True, output_dir=str(tmp_path), branch_id="b0")
        env = FakeEnv(FakeRobot(foot_speed=1.0), FakeSensor(foot_force=100.0))
        for _ in range(3):
            callback.observe(env, torch.zeros(4, 29))
        callback.on_step_end(None, state, None)
        assert len(callback.history) == 1
        record = json.loads((tmp_path / "quality_b0.jsonl").read_text().strip())
        assert record["steps"] == 3 and record["foot_slip_total_m"] > 0

    def test_accumulator_resets_between_iterations(self, tmp_path):
        callback = QT.PracticeQualityCallback(enabled=True, output_dir=str(tmp_path))
        env = FakeEnv(FakeRobot(foot_speed=1.0), FakeSensor(foot_force=100.0))
        callback.observe(env)
        callback.on_step_end(None, type("S", (), {"global_step": 1})(), None)
        callback.on_step_end(None, type("S", (), {"global_step": 2})(), None)
        assert callback.history[1]["steps"] == 0

    def test_sample_every_subsamples(self):
        callback = QT.PracticeQualityCallback(enabled=True, sample_every=3)
        env = FakeEnv(FakeRobot(), FakeSensor())
        for _ in range(9):
            callback.observe(env)
        assert callback.collector.snapshot()["steps"] == 3

    def test_thresholds_report_flags_a_breach(self):
        callback = QT.PracticeQualityCallback(enabled=True)
        env = FakeEnv(FakeRobot(foot_speed=10.0), FakeSensor(foot_force=100.0))
        for _ in range(50):
            callback.observe(env)
        callback.on_step_end(None, type("S", (), {"global_step": 1})(), None)
        assert callback.thresholds_report(QM.QualityThresholds())["foot_slip_exceeds"] is True

    def test_thresholds_report_is_empty_without_history(self):
        assert QT.PracticeQualityCallback(enabled=True).thresholds_report(
            QM.QualityThresholds()) == {}
