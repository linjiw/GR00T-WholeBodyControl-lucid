"""Tests for the live rollout observer (LUCID gap + quality telemetry)."""

import json

import pytest
import torch

from gear_sonic.research.practice_utility import latent_gap_probe as L
from gear_sonic.research.practice_utility import observer as OB

NUM_ENVS, NUM_JOINTS = 4, 6
BODIES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]


class Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeRobot:
    def __init__(self, command=None, executed=None, with_target=True):
        self.body_names = list(BODIES)
        command = command if command is not None else torch.zeros(NUM_ENVS, NUM_JOINTS)
        executed = executed if executed is not None else torch.zeros(NUM_ENVS, NUM_JOINTS)
        fields = dict(
            joint_pos=executed,
            joint_vel=torch.zeros(NUM_ENVS, NUM_JOINTS),
            applied_torque=torch.zeros(NUM_ENVS, NUM_JOINTS),
            body_lin_vel_w=torch.zeros(NUM_ENVS, len(BODIES), 3),
            soft_joint_pos_limits=torch.stack(
                [-torch.ones(NUM_ENVS, NUM_JOINTS), torch.ones(NUM_ENVS, NUM_JOINTS)], dim=-1),
            joint_effort_limits=torch.full((NUM_ENVS, NUM_JOINTS), 100.0),
        )
        if with_target:
            fields["joint_pos_target"] = command
        self.data = Data(**fields)

    def find_bodies(self, names):
        return ([i for i, n in enumerate(self.body_names) if n in set(names)], names)


class FakeSensor:
    def __init__(self):
        self.data = Data(net_forces_w=torch.zeros(NUM_ENVS, len(BODIES), 3))


class FakeScene:
    def __init__(self, robot, sensor):
        self._e = {"robot": robot, "contact_forces": sensor}

    def __getitem__(self, key):
        return self._e[key]


class FakeEnv:
    """Env whose step() advances a simple command/execution trajectory."""

    def __init__(self, lag=0.0, with_target=True):
        self.lag = lag
        self.with_target = with_target
        self.t = 0
        self.robot = FakeRobot(with_target=with_target)
        self.sensor = FakeSensor()
        self.scene = FakeScene(self.robot, self.sensor)
        self.step_calls = 0

    def step(self, actions):
        self.step_calls += 1
        self.t += 1
        command = torch.full((NUM_ENVS, NUM_JOINTS), float(self.t) * 0.01)
        executed = command - self.lag
        if self.with_target:
            self.robot.data.joint_pos_target = command
        self.robot.data.joint_pos = executed
        extras = {"env_actions": command.clone()}
        return ({}, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras)


class State:
    def __init__(self, step):
        self.global_step = step


@pytest.fixture(scope="module")
def encoder_artifact(tmp_path_factory):
    """A small trained encoder saved in artifact form."""
    spec = L.WindowSpec(length=8, stride=1)
    corpus = []
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        t = torch.arange(200, dtype=torch.float32) / 50.0
        signal = torch.stack([torch.sin(2 * torch.pi * (1 + j) * t) for j in range(NUM_JOINTS)], 1)
        signal = signal + 0.01 * torch.randn(signal.shape, generator=generator)
        corpus.append(L.build_windows(signal, spec))
    model, _ = L.train_encoder(torch.cat(corpus), NUM_JOINTS, spec, latent_dim=8, epochs=4, seed=0)
    path = tmp_path_factory.mktemp("enc") / "enc.pt"
    torch.save({
        "state_dict": model.state_dict(), "num_joints": NUM_JOINTS,
        "window_length": 8, "window_stride": 1, "latent_dim": 8,
        "encoder_fingerprint": L.encoder_fingerprint(model),
    }, path)
    return str(path)


class TestInstallation:
    def test_disabled_does_not_patch(self):
        env = FakeEnv()
        callback = OB.PracticeObserverCallback(enabled=False)
        callback.on_train_begin(None, State(0), None, env=env)
        assert OB.PracticeObserverCallback.is_patched(env) is False

    def test_enabled_patches_step(self):
        env = FakeEnv()
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        assert OB.PracticeObserverCallback.is_patched(env) is True

    def test_patched_step_still_returns_its_result(self):
        env = FakeEnv()
        OB.PracticeObserverCallback(enabled=True).on_train_begin(None, State(0), None, env=env)
        result = env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        assert len(result) == 4 and env.step_calls == 1

    def test_uninstall_removes_the_patch(self):
        env = FakeEnv()
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        callback.uninstall()
        assert OB.PracticeObserverCallback.is_patched(env) is False

    def test_train_end_uninstalls(self):
        env = FakeEnv()
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        callback.on_train_end(None, State(5), None, env=env)
        assert OB.PracticeObserverCallback.is_patched(env) is False

    def test_observer_error_never_kills_the_run(self):
        """Telemetry losing a measurement is acceptable; losing the run is not."""
        env = FakeEnv()
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        callback.observe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))          # must not raise
        assert "observer_error" in callback.quality.snapshot()["missing_signals"]


class TestCommandSource:
    def test_prefers_the_pd_target(self):
        env = FakeEnv(with_target=True)
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        assert callback.command_source == "joint_pos_target"

    def test_falls_back_to_env_actions(self):
        env = FakeEnv(with_target=False)
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        assert callback.command_source == "env_actions"

    def test_source_is_recorded_with_the_measurement(self):
        env = FakeEnv()
        callback = OB.PracticeObserverCallback(enabled=True)
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(20):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        record = callback.on_step_end(None, State(1), None, env=env) or callback.history[-1]
        assert callback.history[-1]["command_source"] == "joint_pos_target"


class TestGapCollection:
    def run(self, callback, env, steps=40):
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(steps):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        callback.on_step_end(None, State(1), None, env=env)
        return callback.history[-1]

    def test_raw_gap_is_zero_under_perfect_tracking(self):
        record = self.run(OB.PracticeObserverCallback(enabled=True, window_length=8),
                          FakeEnv(lag=0.0))
        assert record["raw_median"] == pytest.approx(0.0, abs=1e-5)

    def test_raw_gap_grows_with_lag(self):
        tight = self.run(OB.PracticeObserverCallback(enabled=True, window_length=8),
                         FakeEnv(lag=0.01))
        loose = self.run(OB.PracticeObserverCallback(enabled=True, window_length=8),
                         FakeEnv(lag=0.5))
        assert loose["raw_median"] > tight["raw_median"]

    def test_warmup_produces_no_samples(self):
        callback = OB.PracticeObserverCallback(enabled=True, window_length=8)
        env = FakeEnv()
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(3):                       # fewer steps than the window span
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        callback.on_step_end(None, State(1), None, env=env)
        assert callback.history[-1]["num_gap_samples"] == 0

    def test_latent_gap_requires_an_encoder(self):
        record = self.run(OB.PracticeObserverCallback(enabled=True, window_length=8), FakeEnv())
        assert record["num_gap_samples"] == 0        # raw only, no encoder loaded
        assert "raw_median" in record

    def test_latent_gap_collected_with_an_encoder(self, encoder_artifact):
        callback = OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact)
        record = self.run(callback, FakeEnv(lag=0.2))
        assert record["num_gap_samples"] > 0
        assert "latent_median" in record
        assert record["encoder_fingerprint"] is not None

    def test_encoder_dictates_the_window_geometry(self, encoder_artifact):
        """A gap is only comparable against gaps from the same instrument."""
        callback = OB.PracticeObserverCallback(
            enabled=True, encoder_path=encoder_artifact, window_length=999)
        callback.on_train_begin(None, State(0), None, env=FakeEnv())
        assert callback.spec.length == 8

    def test_latent_gap_is_bounded(self, encoder_artifact):
        callback = OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact)
        record = self.run(callback, FakeEnv(lag=1.0))
        assert 0.0 - 1e-6 <= record["latent_median"] <= 2.0 + 1e-6

    def test_buffers_reset_between_iterations(self, encoder_artifact):
        callback = OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact)
        env = FakeEnv(lag=0.2)
        self.run(callback, env)
        callback.on_step_end(None, State(2), None, env=env)
        assert callback.history[-1]["num_gap_samples"] == 0


class TestQualityIntegration:
    def test_quality_is_collected_alongside(self):
        callback = OB.PracticeObserverCallback(enabled=True, window_length=8)
        env = FakeEnv()
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(10):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        callback.on_step_end(None, State(1), None, env=env)
        assert callback.history[-1]["steps"] == 10

    def test_quality_can_be_disabled(self):
        callback = OB.PracticeObserverCallback(
            enabled=True, window_length=8, collect_quality=False)
        env = FakeEnv()
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(10):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        callback.on_step_end(None, State(1), None, env=env)
        assert callback.history[-1]["steps"] == 0

    def test_sample_every_subsamples(self):
        callback = OB.PracticeObserverCallback(enabled=True, window_length=8, sample_every=5)
        env = FakeEnv()
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(20):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        callback.on_step_end(None, State(1), None, env=env)
        assert callback.history[-1]["steps"] == 4

    def test_writes_a_jsonl_record(self, tmp_path):
        callback = OB.PracticeObserverCallback(
            enabled=True, window_length=8, output_dir=str(tmp_path), branch_id="b0")
        env = FakeEnv()
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(12):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        callback.on_step_end(None, State(1), None, env=env)
        record = json.loads((tmp_path / "observer_b0.jsonl").read_text().strip())
        assert record["branch_id"] == "b0" and record["global_step"] == 1


class TestDrainIsOrderIndependent:
    """Callback order is dict order in the Hydra config, so a consumer must not
    depend on running before the observer's own on_step_end."""

    def collect(self, callback, env, steps=40):
        callback.on_train_begin(None, State(0), None, env=env)
        for _ in range(steps):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        return callback

    def test_drain_before_flush_returns_the_current_epoch(self, encoder_artifact):
        callback = self.collect(
            OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact),
            FakeEnv(lag=0.2))
        assert len(callback.drain_gaps()) > 0

    def test_drain_after_flush_still_returns_that_epoch(self, encoder_artifact):
        """The bug: sixteen live iterations reported num_gap_samples = 0."""
        env = FakeEnv(lag=0.2)
        callback = self.collect(
            OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact), env)
        before = len(callback.drain_gaps())
        callback.on_step_end(None, State(1), None, env=env)      # observer flushes first
        assert len(callback.drain_gaps()) == before

    def test_a_fresh_epoch_supersedes_the_stale_one(self, encoder_artifact):
        env = FakeEnv(lag=0.2)
        callback = self.collect(
            OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact), env)
        callback.on_step_end(None, State(1), None, env=env)
        stale = callback.drain_gaps()
        for _ in range(40):
            env.step(torch.zeros(NUM_ENVS, NUM_JOINTS))
        assert callback.drain_gaps() != stale

    def test_drain_never_returns_none(self, encoder_artifact):
        callback = OB.PracticeObserverCallback(enabled=True, encoder_path=encoder_artifact)
        assert callback.drain_gaps() == []


class TestBuffer:
    def test_respects_capacity(self):
        buffer = OB.CommandExecutionBuffer(capacity=4)
        for i in range(10):
            buffer.append(torch.full((3,), float(i)), torch.full((3,), float(i)))
        assert len(buffer.commanded) == 4
        assert float(buffer.commanded[-1][0]) == 9.0

    def test_not_ready_until_full(self):
        buffer = OB.CommandExecutionBuffer(capacity=4)
        buffer.append(torch.zeros(3), torch.zeros(3))
        assert buffer.ready is False
        for _ in range(3):
            buffer.append(torch.zeros(3), torch.zeros(3))
        assert buffer.ready is True

    def test_stacks_in_time_order(self):
        buffer = OB.CommandExecutionBuffer(capacity=3)
        for i in range(3):
            buffer.append(torch.full((2,), float(i)), torch.zeros(2))
        command, _ = buffer.stacks()
        assert command.shape == (3, 2) and float(command[0, 0]) == 0.0
