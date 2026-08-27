"""TACE: cohort assignment, the dispatcher, and its wiring into the curriculum."""

import json

import pytest
import torch

from gear_sonic.research.practice_utility import dr_scaling as DS
from gear_sonic.research.practice_utility import observer as OBS
from gear_sonic.research.practice_utility import tace as TACE
from gear_sonic.research.practice_utility.dr_curriculum import (
    LucidCurriculumCallback,
    clear_curricula,
)


# ----------------------------------------------------------------- fakes --


class Term:
    def __init__(self, mode, params, func=None):
        self.mode = mode
        self.params = params
        self.func = func


class Manager:
    def __init__(self, terms, num_envs=16):
        self._terms = terms
        self.active_terms = list(terms)
        self._term_cfgs = list(terms.values())
        self.num_envs = num_envs


class Scene:
    def __init__(self, num_envs):
        self.num_envs = num_envs


class FakeEnv:
    def __init__(self, manager, num_envs=16):
        self.event_manager = manager
        self.scene = Scene(num_envs)


class Recorder:
    """An event function that records exactly what it was called with."""

    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, env, env_ids, **params):
        self.calls.append((None if env_ids is None else env_ids.clone(), dict(params)))
        return self.ret


class MaterialTerm:
    def __init__(self, n=8):
        self.material_buckets = torch.stack(
            [torch.linspace(0.3, 1.6, n), torch.linspace(0.3, 1.2, n), torch.linspace(0.0, 0.5, n)],
            dim=1,
        )
        self.seen_buckets = []

    def __call__(self, env, env_ids, **params):
        self.seen_buckets.append(self.material_buckets.clone())
        return None


class State:
    def __init__(self, step, max_steps=None, mean_reward=10.0):
        self.global_step = step
        self.max_steps = max_steps
        self.log_history = [{"mean_reward": mean_reward}]


class StubObserver:
    def __init__(self, gaps, branch_id="b0", tracked_env=0):
        self._gaps = list(gaps)
        self.branch_id = branch_id
        self.tracked_env = tracked_env

    def drain_gaps(self):
        return list(self._gaps)


def manager(num_envs=16, mass_func=None, push_func=None):
    return Manager(
        {
            "randomize_rigid_body_mass": Term(
                "reset",
                {"mass_distribution_params": [0.8, 1.2], "operation": "scale"},
                mass_func or Recorder(),
            ),
            "push_robot": Term(
                "interval", {"velocity_range": {"x": [-0.5, 0.5]}}, push_func or Recorder()
            ),
            "physics_material": Term(
                "startup", {"static_friction_range": [0.3, 1.6], "num_buckets": 8}, Recorder()
            ),
        },
        num_envs=num_envs,
    )


@pytest.fixture(autouse=True)
def _clean():
    OBS.clear_observers()
    clear_curricula()
    yield
    OBS.clear_observers()
    clear_curricula()


# ------------------------------------------------------------ assignment --


class TestAssignment:
    def test_exact_cohort_sizes(self):
        a = TACE.assign_cohorts(256, 0.25, seed=7)
        assert a.num_anchor == 64 and a.num_focus == 192
        assert a.mask().sum().item() == 64

    def test_seed_reproduces_and_changes(self):
        assert TACE.assign_cohorts(64, 0.5, 1).anchor_ids == TACE.assign_cohorts(64, 0.5, 1).anchor_ids
        assert TACE.assign_cohorts(64, 0.5, 1).anchor_ids != TACE.assign_cohorts(64, 0.5, 2).anchor_ids

    def test_reserved_envs_are_never_anchors(self):
        for seed in range(20):
            a = TACE.assign_cohorts(8, 0.9, seed, reserved_focus_ids=(0, 3))
            assert 0 not in a.anchor_ids and 3 not in a.anchor_ids
            assert a.num_anchor == 6  # capped by the reservation

    def test_ratio_endpoints(self):
        assert TACE.assign_cohorts(10, 0.0, 0).num_anchor == 0
        assert TACE.assign_cohorts(10, 1.0, 0, reserved_focus_ids=(0,)).num_anchor == 9

    @pytest.mark.parametrize("ratio", [-0.1, 1.1])
    def test_rejects_bad_ratio(self, ratio):
        with pytest.raises(ValueError):
            TACE.assign_cohorts(10, ratio, 0)


# ------------------------------------------------------------ dispatcher --


class TestDispatch:
    def test_splits_env_ids_and_params_by_cohort(self):
        rec = Recorder()
        mask = torch.tensor([True, False, True, False])
        d = TACE.CohortDispatch(rec, "mass", {"mass_distribution_params": [0.8, 1.2]}, mask)
        # focus params are whatever the (scaled) cfg.params say
        d(None, torch.tensor([0, 1, 2, 3]), mass_distribution_params=[0.95, 1.05], operation="scale")
        assert len(rec.calls) == 2
        focus_ids, focus_params = rec.calls[0]
        anchor_ids, anchor_params = rec.calls[1]
        assert focus_ids.tolist() == [1, 3] and focus_params["mass_distribution_params"] == [0.95, 1.05]
        assert anchor_ids.tolist() == [0, 2] and anchor_params["mass_distribution_params"] == [0.8, 1.2]
        assert anchor_params["operation"] == "scale"  # non-range params pass through
        assert d.env_counts == {"anchor": 2, "focus": 2}

    def test_single_cohort_makes_a_single_call(self):
        rec = Recorder()
        d = TACE.CohortDispatch(rec, "mass", {"mass_distribution_params": [0.8, 1.2]}, torch.zeros(4, dtype=torch.bool))
        d(None, torch.tensor([0, 1]), mass_distribution_params=[1.0, 1.0])
        assert len(rec.calls) == 1 and rec.calls[0][0].tolist() == [0, 1]

    @pytest.mark.parametrize("ids", [None, slice(None)])
    def test_none_and_slice_mean_every_env(self, ids):
        rec = Recorder()
        env = FakeEnv(manager(num_envs=4), num_envs=4)
        d = TACE.CohortDispatch(rec, "mass", {}, torch.tensor([True, False, False, True]))
        d(env, ids)
        assert sorted(rec.calls[0][0].tolist() + rec.calls[1][0].tolist()) == [0, 1, 2, 3]

    def test_integer_returns_are_summed(self):
        """randomize_action_delay reports actuator count; a split must not halve it."""
        d = TACE.CohortDispatch(Recorder(ret=5), "delay", {}, torch.tensor([True, False]))
        assert d(None, torch.tensor([0, 1])) == 10

    def test_consolidation_routes_every_env_to_anchor(self):
        rec = Recorder()
        d = TACE.CohortDispatch(rec, "mass", {"mass_distribution_params": [0.8, 1.2]}, torch.tensor([True, False]))
        d.all_envs_mode = True
        d(None, torch.tensor([0, 1]), mass_distribution_params=[1.0, 1.0])
        assert len(rec.calls) == 1
        assert rec.calls[0][0].tolist() == [0, 1]
        assert rec.calls[0][1]["mass_distribution_params"] == [0.8, 1.2]

    def test_material_anchor_sees_full_buckets_and_restores_live(self):
        term = MaterialTerm()
        full = term.material_buckets.clone()
        d = TACE.CohortDispatch(term, "physics_material", {"static_friction_range": [0.3, 1.6]}, torch.tensor([True, False]))
        # curriculum shrinks the live buckets (via proxied attribute)
        d.material_buckets = torch.full_like(full, 0.95)
        d(None, torch.tensor([0, 1]), static_friction_range=[0.95, 0.95])
        focus_seen, anchor_seen = term.seen_buckets
        assert torch.allclose(focus_seen, torch.full_like(full, 0.95))
        assert torch.equal(anchor_seen, full)
        assert torch.allclose(term.material_buckets, torch.full_like(full, 0.95))  # restored

    def test_install_wraps_only_runtime_terms_and_is_idempotent(self):
        m = manager()
        base = DS.capture_baseline(m)
        a = TACE.assign_cohorts(16, 0.25, 0)
        first = TACE.install(m, base, a)
        second = TACE.install(m, base, a)
        assert sorted(first) == ["push_robot", "randomize_rigid_body_mass"]
        assert first["push_robot"] is second["push_robot"]
        assert not isinstance(m._terms["physics_material"].func, TACE.CohortDispatch)
        assert TACE.uninstall(m) == ["push_robot", "randomize_rigid_body_mass"]
        assert isinstance(m._terms["push_robot"].func, Recorder)

    def test_apply_lambda_still_resamples_through_the_proxy(self):
        term = MaterialTerm()
        m = Manager({"physics_material": Term("reset", {"static_friction_range": [0.3, 1.6], "dynamic_friction_range": [0.3, 1.2], "restitution_range": [0.0, 0.5], "num_buckets": 8}, term)})
        base = DS.capture_baseline(m)
        TACE.install(m, base, TACE.assign_cohorts(16, 0.25, 0))
        report = DS.apply_lambda(m, base, 0.0)
        assert report.material_terms_resampled == ["physics_material"]
        assert torch.allclose(term.material_buckets[:, 0], torch.full((8,), 0.95))


# ------------------------------------------------------------ curriculum --


def curriculum(**overrides):
    params = dict(enabled=True, mode="lucid", delta_target=0.10, alpha=0.05, branch_id="b0")
    params.update(overrides)
    return LucidCurriculumCallback(**params)


class TestCurriculumWiring:
    def test_anchor_ratio_zero_installs_nothing(self):
        env = FakeEnv(manager())
        cb = curriculum(anchor_ratio=0.0)
        cb.on_train_begin(None, State(0), None, env=env)
        assert cb.assignment is None and cb.dispatchers == {}
        assert isinstance(env.event_manager._terms["push_robot"].func, Recorder)

    def test_anchor_cohort_is_installed_seeded_and_excludes_tracked_env(self):
        OBS.register_observer(StubObserver([0.01] * 8, tracked_env=5))
        env = FakeEnv(manager())
        cb = curriculum(anchor_ratio=0.5, anchor_seed=11, observer_branch_id="b0")
        cb.on_train_begin(None, State(0), None, env=env)
        assert cb.assignment.num_anchor == 8
        assert 0 not in cb.assignment.anchor_ids and 5 not in cb.assignment.anchor_ids
        assert sorted(cb.dispatchers) == ["push_robot", "randomize_rigid_body_mass"]
        assert cb.assignment.anchor_ids == TACE.assign_cohorts(16, 0.5, 11, (0, 5)).anchor_ids

    def test_anchor_envs_sample_the_full_envelope_while_focus_follows_lambda(self):
        OBS.register_observer(StubObserver([0.01] * 8))
        mass = Recorder()
        env = FakeEnv(manager(mass_func=mass))
        cb = curriculum(anchor_ratio=0.25, anchor_seed=3, observer_branch_id="b0", initial_lambda=0.0)
        cb.on_train_begin(None, State(0), None, env=env)
        term = env.event_manager._terms["randomize_rigid_body_mass"]
        term.func(env, torch.arange(16), **term.params)  # what the event manager does at reset
        focus_ids, focus_params = mass.calls[0]
        anchor_ids, anchor_params = mass.calls[1]
        assert focus_params["mass_distribution_params"] == pytest.approx([1.0, 1.0])
        assert anchor_params["mass_distribution_params"] == [0.8, 1.2]
        assert set(anchor_ids.tolist()) == set(cb.assignment.anchor_ids)
        assert len(focus_ids) == 12 and len(anchor_ids) == 4

    def test_records_tace_telemetry_and_state(self):
        OBS.register_observer(StubObserver([0.01] * 8))
        env = FakeEnv(manager())
        cb = curriculum(anchor_ratio=0.25, observer_branch_id="b0")
        cb.on_train_begin(None, State(0), None, env=env)
        cb.on_step_end(None, State(1), None, env=env)
        tace = cb.history[-1]["tace"]
        assert tace["num_anchor"] == 4 and tace["num_focus"] == 12
        assert "randomize_rigid_body_mass" in tace["dispatch"]
        state = cb.state_dict()
        assert state["tace"]["num_anchor"] == 4 and state["consolidating"] is False
        json.dumps(state)

    def test_lambda_still_climbs_on_focus_evidence(self):
        OBS.register_observer(StubObserver([0.01] * 8))
        env = FakeEnv(manager())
        cb = curriculum(anchor_ratio=0.25, observer_branch_id="b0")
        cb.on_train_begin(None, State(0), None, env=env)
        for step in range(1, 6):
            cb.on_step_end(None, State(step), None, env=env)
        assert cb.controller.lambda_value > 0.0

    def test_consolidation_switches_every_cohort_to_target(self):
        OBS.register_observer(StubObserver([0.01] * 8))
        env = FakeEnv(manager())
        cb = curriculum(anchor_ratio=0.25, observer_branch_id="b0", consolidation_fraction=0.2)
        cb.on_train_begin(None, State(0, max_steps=10), None, env=env)
        cb.on_step_end(None, State(7, max_steps=10), None, env=env)
        assert not cb.history[-1].get("consolidation")
        cb.on_step_end(None, State(8, max_steps=10), None, env=env)
        assert cb.history[-1]["consolidation"] is True and cb.history[-1]["lambda"] == 1.0
        assert all(d.all_envs_mode for d in cb.dispatchers.values())
        assert env.event_manager._terms["randomize_rigid_body_mass"].params["mass_distribution_params"] == [0.8, 1.2]

    def test_consolidation_needs_a_known_budget(self):
        env = FakeEnv(manager())
        cb = curriculum(consolidation_fraction=0.5)
        cb.on_train_begin(None, State(0), None, env=env)
        cb.on_step_end(None, State(100), None, env=env)
        assert not cb.history[-1].get("consolidation")


class TestYoked:
    def _schedule(self, tmp_path, values):
        path = tmp_path / "curriculum_b0.jsonl"
        with path.open("w") as h:
            for i, v in enumerate(values):
                h.write(json.dumps({"global_step": i + 1, "lambda": v}) + "\n")
        return str(path)

    def test_requires_a_schedule(self):
        with pytest.raises(ValueError, match="yoked"):
            LucidCurriculumCallback(enabled=True, mode="yoked")

    def test_replays_lambda_by_iteration_and_holds_the_last(self, tmp_path):
        sched = self._schedule(tmp_path, [0.0, 0.1, 0.3, 0.6])
        env = FakeEnv(manager())
        cb = curriculum(mode="yoked", yoked_schedule_path=sched, warmup_iterations=0)
        cb.on_train_begin(None, State(0), None, env=env)
        got = []
        for step in range(0, 7):
            cb.on_step_end(None, State(step), None, env=env)
            got.append(cb.history[-1]["lambda"])
        assert got == [0.0, 0.1, 0.3, 0.6, 0.6, 0.6, 0.6]
        mass = env.event_manager._terms["randomize_rigid_body_mass"].params["mass_distribution_params"]
        assert mass == pytest.approx([1.0 - 0.6 * 0.2, 1.0 + 0.6 * 0.2])

    def test_yoked_ignores_gap_evidence(self, tmp_path):
        OBS.register_observer(StubObserver([0.0] * 20))  # would drive lucid up hard
        sched = self._schedule(tmp_path, [0.2, 0.2, 0.2])
        env = FakeEnv(manager())
        cb = curriculum(mode="yoked", yoked_schedule_path=sched, observer_branch_id="b0", warmup_iterations=0)
        cb.on_train_begin(None, State(0), None, env=env)
        for step in range(5):
            cb.on_step_end(None, State(step), None, env=env)
        assert cb.controller.lambda_value == 0.2
