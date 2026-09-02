"""Box mode wired into the curriculum callback: vector frontier, one probe."""

import json

import pytest
import torch

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import observer as OBS
from gear_sonic.research.practice_utility import survival_observer as SO
from gear_sonic.research.practice_utility.dr_curriculum import (
    LucidCurriculumCallback,
    clear_curricula,
)


class Term:
    def __init__(self, mode, params, func=None):
        self.mode = mode
        self.params = params
        self.func = func or (lambda env, env_ids, **p: None)


class Manager:
    def __init__(self, terms, num_envs):
        self._terms = terms
        self.active_terms = list(terms)
        self._term_cfgs = list(terms.values())
        self.num_envs = num_envs


class Scene:
    def __init__(self, num_envs):
        self.num_envs = num_envs


class FakeEnv:
    def __init__(self, num_envs=32):
        self.event_manager = Manager(
            {
                "randomize_rigid_body_mass": Term(
                    "reset", {"mass_distribution_params": [0.8, 1.2], "operation": "scale"}
                ),
                "push_robot": Term("interval", {"velocity_range": {"x": [-0.5, 0.5]}}),
                # Startup-mode: captured but never re-applied, so never a channel.
                "physics_material": Term("startup", {"static_friction_range": [0.3, 1.6]}),
            },
            num_envs,
        )
        self.scene = Scene(num_envs)


class State:
    def __init__(self, step, mean_reward=10.0):
        self.global_step = step
        self.log_history = [{"mean_reward": mean_reward}]


class StubSurvival:
    """Survival observer stand-in: fixed probe survival, records its strata."""

    def __init__(self, survival=0.95, episodes=10, branch_id="b0"):
        self.branch_id = branch_id
        self.probe_index = None
        self.stratum_masks = ()
        self.survival = survival
        self.episodes = episodes
        self.flushed = []

    def set_strata(self, masks, probe_index):
        self.stratum_masks = tuple(masks)
        self.probe_index = probe_index

    def ensure_flushed(self, step):
        self.flushed.append(step)

    def current_probe(self):
        return self.survival, self.episodes

    def current_population(self):
        return 0.5


@pytest.fixture(autouse=True)
def _clean():
    OBS.clear_observers()
    SO.clear_survival_observers()
    clear_curricula()
    yield
    OBS.clear_observers()
    SO.clear_survival_observers()
    clear_curricula()


def box(**overrides):
    params = dict(
        enabled=True,
        mode="box",
        branch_id="b0",
        survival_branch_id="b0",
        allow_extrapolation=True,
        spread_strata=8,
        anchor_seed=1,
        gate_lambda_max=1.5,
        gate_probe_max=1.5,
        gate_step=0.25,
        gate_probe_offset=0.25,
        gate_window=2,
        gate_dwell=0,
        gate_min_episodes=1,
        gate_evidence_grace=5,
        warmup_iterations=0,
    )
    params.update(overrides)
    return LucidCurriculumCallback(**params)


def test_box_past_one_requires_the_extrapolation_flag():
    with pytest.raises(ValueError, match="allow_extrapolation"):
        LucidCurriculumCallback(enabled=True, mode="box", spread_strata=8, gate_lambda_max=1.5)
    with pytest.raises(ValueError, match="spread_strata"):
        LucidCurriculumCallback(enabled=True, mode="box", allow_extrapolation=True, gate_lambda_max=1.5)


def test_bind_builds_the_box_over_the_scalable_terms_only():
    SO.register_survival_observer(StubSurvival())
    env = FakeEnv()
    cb = box()
    cb.on_train_begin(None, State(0), None, env=env)
    assert cb.box is not None
    assert cb.box.config.channels == ("push_robot", "randomize_rigid_body_mass")
    assert cb.box.frontier == {"push_robot": 1.0, "randomize_rigid_body_mass": 1.0}
    # The probe stratum raises the active channel only, and it is the last stratum.
    top = cb._stratum_lambdas_absolute[-1]
    assert top == {"push_robot": 1.25, "randomize_rigid_body_mass": 1.0}
    assert cb._stratum_lambdas_absolute[-2] == {"push_robot": 1.0, "randomize_rigid_body_mass": 1.0}
    assert cb._stratum_lambdas_absolute[0]["push_robot"] == pytest.approx(1.0 / 7.0)


def test_channels_widen_one_at_a_time_on_probe_evidence():
    stub = StubSurvival(survival=0.95)
    SO.register_survival_observer(stub)
    env = FakeEnv()
    cb = box()
    cb.on_train_begin(None, State(0), None, env=env)
    assert stub.probe_index == 7 and len(stub.stratum_masks) == 8
    cb.on_step_end(None, State(1), None, env=env)
    cb.on_step_end(None, State(2), None, env=env)
    row = cb.history[-1]
    assert row["mode"] == "box" and row["fired"] is True
    assert row["frontier_vector"] == {"push_robot": 1.25, "randomize_rigid_body_mass": 1.0}
    assert row["lambda"] == pytest.approx(1.125)
    # The live event config carries the vector: push widened, mass untouched.
    terms = env.event_manager._terms
    assert terms["push_robot"].params["velocity_range"]["x"] == pytest.approx([-0.625, 0.625])
    assert terms["randomize_rigid_body_mass"].params["mass_distribution_params"] == [0.8, 1.2]
    # The probe has moved on to mass.
    cb.on_step_end(None, State(3), None, env=env)
    assert cb.history[-1]["active_channel"] == "randomize_rigid_body_mass"
    assert cb.history[-1]["probe_vector"] == {"push_robot": 1.25, "randomize_rigid_body_mass": 1.25}
    cb.on_step_end(None, State(4), None, env=env)
    assert cb.history[-1]["frontier_vector"] == {"push_robot": 1.25, "randomize_rigid_body_mass": 1.25}
    assert terms["randomize_rigid_body_mass"].params["mass_distribution_params"] == pytest.approx([0.75, 1.25])
    # Telemetry is JSON and the tace strata are vectors.
    json.dumps(cb.history[-1], default=str)
    assert isinstance(cb.history[-1]["tace"]["stratum_lambdas"][-1], dict)


def test_a_failing_channel_is_skipped_while_the_other_advances():
    class Split(StubSurvival):
        def __init__(self, cb_ref):
            super().__init__()
            self.cb_ref = cb_ref

        def current_probe(self):
            active = self.cb_ref[0].box.active_channel
            return (0.1, 10) if active == "push_robot" else (0.95, 10)

    holder = []
    stub = Split(holder)
    SO.register_survival_observer(stub)
    env = FakeEnv()
    cb = box()
    holder.append(cb)
    cb.on_train_begin(None, State(0), None, env=env)
    for step in range(1, 9):
        cb.on_step_end(None, State(step), None, env=env)
    final = cb.history[-1]["frontier_vector"]
    assert final["push_robot"] == 1.0
    assert final["randomize_rigid_body_mass"] >= 1.25
    assert all(row["applied_decrease"] is False for row in cb.history)


def test_no_channel_ever_decreases():
    SO.register_survival_observer(StubSurvival(survival=0.95))
    env = FakeEnv()
    cb = box()
    cb.on_train_begin(None, State(0), None, env=env)
    previous = None
    for step in range(1, 12):
        cb.on_step_end(None, State(step), None, env=env)
        vector = cb.history[-1]["frontier_vector"]
        if previous is not None:
            assert all(vector[k] >= previous[k] for k in vector)
        previous = vector
    assert cb.history[-1]["at_ceiling"] is True
    assert cb.history[-1]["frontier_vector"] == {"push_robot": 1.5, "randomize_rigid_body_mass": 1.5}


def test_box_state_survives_a_restart_before_bind():
    SO.register_survival_observer(StubSurvival(survival=0.95))
    env = FakeEnv()
    cb = box()
    cb.on_train_begin(None, State(0), None, env=env)
    for step in range(1, 3):
        cb.on_step_end(None, State(step), None, env=env)
    state = cb.state_dict()
    assert state["box"]["gates"]["push_robot"]["frontier"] == 1.25
    resumed = box()
    resumed.load_state_dict(state)  # before bind: parked
    assert resumed.box is None
    resumed.on_train_begin(None, State(3), None, env=FakeEnv())
    assert resumed.box.frontier == {"push_robot": 1.25, "randomize_rigid_body_mass": 1.0}
    assert resumed._frontier_lambda == pytest.approx(1.125)


def test_dead_signal_path_aborts_instead_of_holding():
    SO.register_survival_observer(StubSurvival(survival=None, episodes=0))
    env = FakeEnv()
    cb = box(gate_evidence_grace=2)
    cb.on_train_begin(None, State(0), None, env=env)
    with pytest.raises(RuntimeError, match="no probe episode"):
        for step in range(1, 6):
            cb.on_step_end(None, State(step), None, env=env)


def test_per_channel_ceilings_bound_probe_and_frontier_per_channel():
    SO.register_survival_observer(StubSurvival(survival=0.95))
    env = FakeEnv()
    cb = box(
        gate_lambda_max=2.0,
        gate_probe_max=2.0,
        box_lambda_max={"push_robot": 1.25},  # mass absent -> default 2.0
    )
    cb.on_train_begin(None, State(0), None, env=env)
    assert cb.box.config.ceiling("push_robot") == 1.25
    assert cb.box.config.ceiling("randomize_rigid_body_mass") == 2.0
    # The probe never exceeds a channel's own ceiling.
    assert cb.box.config.probe_ceiling("push_robot") == 1.25
    for step in range(1, 16):
        cb.on_step_end(None, State(step), None, env=env)
    final = cb.history[-1]["frontier_vector"]
    assert final["push_robot"] == pytest.approx(1.25)
    assert final["randomize_rigid_body_mass"] == pytest.approx(2.0)
    probes = [row["probe_vector"]["push_robot"] for row in cb.history]
    assert max(probes) <= 1.25 + 1e-9


def test_box_ceiling_above_gate_lambda_max_is_refused():
    with pytest.raises(ValueError, match="box_lambda_max"):
        box(gate_lambda_max=1.5, box_lambda_max={"push_robot": 2.0})
