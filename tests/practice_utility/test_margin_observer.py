"""The margin observer and its wiring into the curriculum, on fakes."""

import json

import pytest
import torch

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import margin_observer as MO
from gear_sonic.research.practice_utility import margin_signal as MS
from gear_sonic.research.practice_utility import observer as OBS
from gear_sonic.research.practice_utility import tace as TACE
from gear_sonic.research.practice_utility.dr_curriculum import (
    LucidCurriculumCallback,
    clear_curricula,
)
from tests.practice_utility.test_margin_signal import Cmd, manager as term_manager
from tests.practice_utility.test_tace import Recorder, State, Term
from tests.practice_utility.test_tace import Manager as EventManager


class CommandManager:
    def __init__(self, cmd): self._cmd = cmd
    def get_term(self, name): return self._cmd


class InnerEnv:
    """Looks enough like a ManagerBasedRLEnv for the observer."""

    def __init__(self, n):
        self.cmd = Cmd(n)
        self.command_manager = CommandManager(self.cmd)
        self.termination_manager = term_manager()
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.event_manager = EventManager({
            "randomize_rigid_body_mass": Term("reset", {"mass_distribution_params": [0.8, 1.2]}, Recorder()),
        }, num_envs=n)
        self.scene = type("Scene", (), {"num_envs": n})()
        self.num_envs = n
        self.dones_to_return = torch.zeros(n, dtype=torch.bool)
        self.time_outs_to_return = torch.zeros(n, dtype=torch.bool)

    def step(self, actions):
        # Age advances; envs flagged done reset to 0 after the step.
        self.episode_length_buf += 1
        dones = self.dones_to_return.clone()
        extras = {"time_outs": self.time_outs_to_return.clone()}
        self.episode_length_buf[dones] = 0
        return None, None, dones.long(), extras


@pytest.fixture(autouse=True)
def _clean():
    OBS.clear_observers(); MO.clear_margin_observers(); clear_curricula()
    yield
    OBS.clear_observers(); MO.clear_margin_observers(); clear_curricula()


def _run_episode(env, observer, n_steps, done_at_end=True, feet_err=0.0):
    env.cmd.robot_body_pos_w[:, 2] = torch.tensor([feet_err, 0.0, 0.0])
    for i in range(n_steps):
        env.dones_to_return[:] = done_at_end and (i == n_steps - 1)
        env.step(None)


class TestObserver:
    def test_installs_reads_thresholds_and_counts_steps(self):
        env = InnerEnv(4)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="m", horizon=4)
        obs.on_train_begin(None, None, None, env=env)
        assert obs.thresholds.foot == 0.5
        assert MO.get_active_margin_observer("m") is obs
        _run_episode(env, obs, 3, done_at_end=False)
        assert obs._steps == 3

    def test_prefix_mean_lands_in_the_iteration_record(self):
        env = InnerEnv(2)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="m", horizon=4)
        obs.on_train_begin(None, None, None, env=env)
        obs.set_cohorts(torch.tensor([True, False]), torch.tensor([False, True]))
        # one full episode of 5 steps for both envs at feet error 0.25 (margin 0.5)
        _run_episode(env, obs, 5, feet_err=0.25)
        obs.on_step_end(None, State(1), None, env=env)
        rec = obs.history[-1]
        assert rec["episodes_ended"] == 2
        assert rec["q_focus"] == pytest.approx(0.5)
        assert rec["q_yardstick"] == pytest.approx(0.5)
        assert rec["ratio"] == pytest.approx(1.0)
        assert rec["culprit_share"]["feet"] == pytest.approx(1.0)
        assert rec["coverage_short_of_horizon"]["focus"] == 0.0

    def test_ratio_separates_a_degraded_focus_cohort(self):
        env = InnerEnv(2)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="m", horizon=4, tau=1.0)
        obs.on_train_begin(None, None, None, env=env)
        obs.set_cohorts(torch.tensor([True, False]), torch.tensor([False, True]))
        # focus env 0 tracks worse (0.4 m) than yardstick env 1 (0.2 m)
        env.cmd.robot_body_pos_w[0, 2] = torch.tensor([0.4, 0.0, 0.0])
        env.cmd.robot_body_pos_w[1, 2] = torch.tensor([0.2, 0.0, 0.0])
        for i in range(5):
            env.dones_to_return[:] = i == 4
            env.step(None)
        obs.on_step_end(None, State(1), None, env=env)
        assert obs.current_ratio() == pytest.approx(2.0)
        assert obs.current_error() == pytest.approx(1.30 - 2.0)   # above the band: lower

    def test_flush_is_idempotent_per_step(self):
        env = InnerEnv(1)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="m", horizon=4)
        obs.on_train_begin(None, None, None, env=env)
        _run_episode(env, obs, 3)
        obs.ensure_flushed(7); obs.ensure_flushed(7); obs.on_step_end(None, State(7), None, env=env)
        assert len(obs.history) == 1

    def test_writes_jsonl(self, tmp_path):
        env = InnerEnv(1)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="m", horizon=4, output_dir=str(tmp_path))
        obs.on_train_begin(None, None, None, env=env)
        _run_episode(env, obs, 2)
        obs.on_step_end(None, State(1), None, env=env)
        rows = [json.loads(l) for l in (tmp_path / "margin_m.jsonl").read_text().splitlines()]
        assert rows[0]["episodes_ended"] == 1 and rows[0]["thresholds"]["foot_pos_xyz"] == 0.5

    def test_uninstall_restores_step(self):
        env = InnerEnv(1)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="m")
        obs.on_train_begin(None, None, None, env=env)
        assert "step" in vars(env)
        obs.on_train_end(None, None, None)
        assert "step" not in vars(env)


class TestYardstickCohort:
    def test_assignment_reserves_a_disjoint_yardstick(self):
        plan = TACE.assign_cohorts(64, 0.25, seed=3, reserved_focus_ids=(0,), num_strata=4, num_yardstick=8)
        y = set(plan.yardstick_ids)
        assert len(y) == 8 and y.isdisjoint(plan.anchor_ids) and 0 not in y
        assert all(y.isdisjoint(s) for s in plan.focus_strata)
        assert plan.num_focus == 64 - 16 - 8
        assert plan.focus_mask().sum() == plan.num_focus and plan.yardstick_mask().sum() == 8
        assert not (plan.focus_mask() & plan.yardstick_mask()).any()

    def test_dispatch_sends_yardstick_envs_to_lambda_zero_params(self):
        rec = Recorder()
        plan = TACE.assign_cohorts(8, 0.0, seed=1, num_strata=1, num_yardstick=2)
        d = TACE.CohortDispatch(rec, "t", {"velocity_range": [-1.0, 1.0]}, plan.mask(), plan.stratum_masks())
        d.set_yardstick(plan.yardstick_mask(), {"velocity_range": [0.0, 0.0]})
        d(None, torch.arange(8), velocity_range=[-0.5, 0.5])
        yard = [c for c in rec.calls if c[1]["velocity_range"] == [0.0, 0.0]]
        assert len(yard) == 1 and sorted(yard[0][0].tolist()) == sorted(plan.yardstick_ids)
        assert d.telemetry()["env_counts"]["yardstick"] == 2
        sampled = sorted(torch.cat([c[0] for c in rec.calls]).tolist())
        assert sampled == list(range(8))   # every env exactly once


class TestCurriculumWiring:
    def _curriculum(self, **kw):
        return LucidCurriculumCallback(
            enabled=True, mode="lucid", branch_id="b", observer_branch_id="obs",
            initial_lambda=0.5, spread_strata=4, anchor_seed=1, signal="margin",
            yardstick_envs=4, margin_branch_id="b", alpha=0.1, kp=1.0, ki=0.0, **kw,
        )

    def test_margin_needs_a_yardstick(self):
        with pytest.raises(ValueError, match="yardstick"):
            LucidCurriculumCallback(enabled=True, mode="lucid", signal="margin", yardstick_envs=0)

    def test_bind_installs_yardstick_and_tells_the_observer(self):
        env = InnerEnv(32)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="b", horizon=4)
        obs.on_train_begin(None, None, None, env=env)
        cur = self._curriculum()
        cur.on_train_begin(None, None, None, env=env)
        assert cur.assignment.num_yardstick == 4
        assert obs.yardstick_mask.sum() == 4 and obs.focus_mask.sum() == 28
        d = cur.dispatchers["randomize_rigid_body_mass"]
        yp = object.__getattribute__(d, "_yardstick_params")
        assert yp["mass_distribution_params"] == [1.0, 1.0]         # lambda = 0 collapses to nominal

    def test_lambda_holds_in_band_rises_below_falls_above(self):
        env = InnerEnv(32)
        obs = MO.MarginObserverCallback(enabled=True, branch_id="b", horizon=4, tau=1.0)
        obs.on_train_begin(None, None, None, env=env)
        cur = self._curriculum()
        cur.on_train_begin(None, None, None, env=env)
        focus = obs.focus_mask.nonzero().flatten(); yard = obs.yardstick_mask.nonzero().flatten()

        def episode(focus_err, yard_err, step):
            env.cmd.robot_body_pos_w[focus, 2] = torch.tensor([focus_err, 0.0, 0.0])
            env.cmd.robot_body_pos_w[yard, 2] = torch.tensor([yard_err, 0.0, 0.0])
            for i in range(5):
                env.dones_to_return[:] = i == 4
                env.step(None)
            cur.on_step_end(None, State(step, mean_reward=10.0), None, env=env)
            return cur.history[-1]

        r1 = episode(0.24, 0.20, 1)          # ratio 1.2: inside the band -> hold
        assert r1["margin_ratio"] == pytest.approx(1.2) and r1["lambda"] == pytest.approx(0.5)
        r2 = episode(0.20, 0.20, 2)          # ratio 1.0: below -> rise
        assert r2["lambda"] > 0.5
        r3 = episode(0.40, 0.20, 3)          # ratio 2.0: above -> fall
        assert r3["lambda"] < r2["lambda"]
        assert r3["signal"] == "margin"

    def test_no_observer_means_hold_not_raise(self):
        env = InnerEnv(16)
        cur = self._curriculum()
        cur.on_train_begin(None, None, None, env=env)
        cur.on_step_end(None, State(1, mean_reward=10.0), None, env=env)
        assert cur.history[-1]["lambda"] == pytest.approx(0.5)
        assert cur.history[-1]["margin_observer_present"] is False

    def test_gap_signal_path_is_unchanged(self):
        env = InnerEnv(16)
        OBS._ACTIVE_OBSERVERS["obs"] = type("S", (), {"branch_id": "obs", "tracked_env": 0,
                                                      "drain_gaps": lambda self: [0.05] * 8})()
        cur = LucidCurriculumCallback(enabled=True, mode="lucid", branch_id="b", observer_branch_id="obs",
                                      initial_lambda=0.5, delta_target=0.1, alpha=0.1)
        cur.on_train_begin(None, None, None, env=env)
        cur.on_step_end(None, State(1, mean_reward=10.0), None, env=env)
        assert cur.history[-1]["signal"] == "gap" and cur.history[-1]["lambda"] > 0.5
