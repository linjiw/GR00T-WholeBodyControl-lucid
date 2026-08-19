"""Tests for reset-safe, per-environment domain-randomization terms.

The property under test is the one that makes a runtime curriculum possible at
all: calling a term on every episode reset must re-derive from the nominal, not
stack on the previous draw. SONIC's startup-mode terms fail that, which is why
they are startup-mode.
"""

import pytest
import torch

from gear_sonic.research.practice_utility import events_reset_safe as E

NUM_ENVS, NUM_BODIES, NUM_JOINTS = 8, 4, 6


class FakePhysxView:
    def __init__(self, num_envs=NUM_ENVS, num_bodies=NUM_BODIES):
        # CoM layout is (envs, bodies, 7): position then orientation.
        self.coms = torch.zeros(num_envs, num_bodies, 7)
        self.writes = []

    def get_coms(self):
        return self.coms

    def set_coms(self, coms, env_ids):
        self.writes.append(torch.as_tensor(env_ids).clone())
        rows = torch.as_tensor(env_ids).long()
        self.coms[rows] = coms[rows]


class FakeAsset:
    def __init__(self):
        self.num_bodies = NUM_BODIES
        self.device = torch.device("cpu")
        self.root_physx_view = FakePhysxView()
        self.joint_names = [f"j{i}" for i in range(NUM_JOINTS)]

        class Data:
            pass

        self.data = Data()
        self.data.default_joint_pos = torch.zeros(NUM_ENVS, NUM_JOINTS)


class FakeTerm:
    def __init__(self):
        self._joint_names = [f"j{i}" for i in range(NUM_JOINTS)]
        self._offset = torch.zeros(NUM_ENVS, NUM_JOINTS)


class FakeActionManager:
    def __init__(self):
        self.term = FakeTerm()

    def get_term(self, name):
        if name != "joint_pos":
            raise KeyError(name)
        return self.term


class FakeScene:
    def __init__(self, asset):
        self.num_envs = NUM_ENVS
        self._asset = asset

    def __getitem__(self, key):
        return self._asset


class FakeEnv:
    def __init__(self, with_action_manager=True):
        self.asset = FakeAsset()
        self.scene = FakeScene(self.asset)
        self.action_manager = FakeActionManager() if with_action_manager else None


class Cfg:
    def __init__(self, body_ids=slice(None), joint_ids=slice(None)):
        self.name = "robot"
        self.body_ids = body_ids
        self.joint_ids = joint_ids


COM_RANGE = {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}


class TestComDoesNotCompound:
    def test_repeated_resets_stay_within_the_configured_range(self):
        """The defect: '+=' on current CoM accumulates without bound."""
        env = FakeEnv()
        for _ in range(50):
            E.randomize_rigid_body_com(env, None, COM_RANGE, Cfg())
        offsets = env.asset.root_physx_view.coms[..., :3]
        assert float(offsets[..., 0].abs().max()) <= 0.025 + 1e-6
        assert float(offsets[..., 1].abs().max()) <= 0.05 + 1e-6

    def test_naive_accumulation_would_have_drifted(self):
        """Contrast: the original pattern over the same number of resets."""
        coms = torch.zeros(NUM_ENVS, NUM_BODIES, 3)
        for _ in range(50):
            coms += torch.empty(NUM_ENVS, 1, 3).uniform_(-0.05, 0.05)
        assert float(coms.abs().max()) > 0.05      # far outside the range

    def test_nominal_is_captured_before_any_randomization(self):
        env = FakeEnv()
        env.asset.root_physx_view.coms[..., :3] = 0.7      # a non-zero nominal
        E.randomize_rigid_body_com(env, None, COM_RANGE, Cfg())
        nominal = getattr(env.asset, E.NOMINAL_COM)
        assert float(nominal[..., 0].min()) == pytest.approx(0.7)

    def test_randomization_is_centred_on_the_nominal(self):
        env = FakeEnv()
        env.asset.root_physx_view.coms[..., :3] = 0.7
        for _ in range(30):
            E.randomize_rigid_body_com(env, None, COM_RANGE, Cfg())
        offsets = env.asset.root_physx_view.coms[..., 0] - 0.7
        assert float(offsets.abs().max()) <= 0.025 + 1e-6

    def test_partial_env_ids_touch_only_those_rows(self):
        """The original broadcast len(env_ids) samples across all num_envs rows."""
        env = FakeEnv()
        selected = torch.tensor([1, 3])
        E.randomize_rigid_body_com(env, selected, COM_RANGE, Cfg())
        touched = env.asset.root_physx_view.coms[..., :3].abs().sum(dim=(1, 2))
        assert float(touched[0]) == 0.0 and float(touched[2]) == 0.0
        assert float(touched[1]) > 0.0 or float(touched[3]) > 0.0

    def test_write_is_scoped_to_the_selected_envs(self):
        env = FakeEnv()
        E.randomize_rigid_body_com(env, torch.tensor([2, 5]), COM_RANGE, Cfg())
        assert env.asset.root_physx_view.writes[-1].tolist() == [2, 5]

    def test_empty_env_ids_is_a_no_op(self):
        env = FakeEnv()
        E.randomize_rigid_body_com(env, torch.tensor([], dtype=torch.long), COM_RANGE, Cfg())
        assert env.asset.root_physx_view.writes == []

    def test_respects_body_subset(self):
        env = FakeEnv()
        E.randomize_rigid_body_com(env, None, COM_RANGE, Cfg(body_ids=[1]))
        moved = env.asset.root_physx_view.coms[..., :3].abs().sum(dim=(0, 2))
        assert float(moved[0]) == 0.0 and float(moved[1]) > 0.0


class TestJointDefaultsDoNotCompound:
    def test_repeated_resets_stay_within_range(self):
        env = FakeEnv()
        for _ in range(50):
            E.randomize_joint_default_pos(env, None, Cfg(), (-0.01, 0.01), "add")
        assert float(env.asset.data.default_joint_pos.abs().max()) <= 0.01 + 1e-6

    def test_nominal_survives_repeated_calls(self):
        env = FakeEnv()
        env.asset.data.default_joint_pos.fill_(0.3)
        for _ in range(20):
            E.randomize_joint_default_pos(env, None, Cfg(), (-0.01, 0.01), "add")
        nominal = getattr(env.asset, E.NOMINAL_JOINT_POS)
        assert float(nominal.min()) == pytest.approx(0.3)
        assert float((env.asset.data.default_joint_pos - 0.3).abs().max()) <= 0.01 + 1e-6

    def test_export_nominal_is_the_true_nominal(self):
        env = FakeEnv()
        env.asset.data.default_joint_pos.fill_(0.3)
        for _ in range(5):
            E.randomize_joint_default_pos(env, None, Cfg(), (-0.01, 0.01), "add")
        assert float(env.asset.data.default_joint_pos_nominal.min()) == pytest.approx(0.3)

    def test_partial_env_ids_leave_others_untouched(self):
        env = FakeEnv()
        E.randomize_joint_default_pos(env, torch.tensor([0, 1]), Cfg(), (-0.01, 0.01), "add")
        assert float(env.asset.data.default_joint_pos[4].abs().max()) == 0.0

    def test_action_offset_is_kept_in_step(self):
        env = FakeEnv()
        E.randomize_joint_default_pos(env, None, Cfg(), (-0.01, 0.01), "add")
        offset = env.action_manager.term._offset
        assert torch.allclose(offset, env.asset.data.default_joint_pos, atol=1e-6)

    def test_missing_action_manager_is_survivable(self):
        env = FakeEnv(with_action_manager=False)
        E.randomize_joint_default_pos(env, None, Cfg(), (-0.01, 0.01), "add")
        assert float(env.asset.data.default_joint_pos.abs().max()) > 0.0

    def test_no_params_is_a_no_op(self):
        env = FakeEnv()
        E.randomize_joint_default_pos(env, None, Cfg(), None)
        assert float(env.asset.data.default_joint_pos.abs().max()) == 0.0

    def test_abs_operation_replaces_rather_than_adds(self):
        env = FakeEnv()
        env.asset.data.default_joint_pos.fill_(0.5)
        E.randomize_joint_default_pos(env, None, Cfg(), (0.1, 0.1), "abs")
        assert float(env.asset.data.default_joint_pos.max()) == pytest.approx(0.1)


class TestMaterialBuckets:
    class Term:
        def __init__(self, n=64):
            self.material_buckets = torch.stack([
                torch.empty(n).uniform_(0.3, 1.6),
                torch.empty(n).uniform_(0.3, 1.2),
                torch.empty(n).uniform_(0.0, 0.5),
            ], dim=1)

    def test_resampling_changes_the_distribution(self):
        """Without this, scaling friction ranges at runtime does nothing."""
        term = self.Term()
        before = term.material_buckets.clone()
        assert E.resample_material_buckets(term, (0.9, 0.95), (0.9, 0.95), (0.0, 0.0))
        assert not torch.allclose(before, term.material_buckets)

    def test_resampled_values_respect_the_new_range(self):
        term = self.Term()
        E.resample_material_buckets(term, (0.9, 0.95), (0.9, 0.95), (0.0, 0.0))
        static = term.material_buckets[:, 0]
        assert float(static.min()) >= 0.9 - 1e-6 and float(static.max()) <= 0.95 + 1e-6

    def test_bucket_count_is_preserved(self):
        """PhysX caps unique materials; the bucket count must not grow."""
        term = self.Term(n=64)
        E.resample_material_buckets(term, (0.5, 0.6))
        assert term.material_buckets.shape == (64, 3)

    def test_unspecified_channels_keep_their_span(self):
        term = self.Term()
        E.resample_material_buckets(term, static_friction_range=(0.9, 0.95))
        restitution = term.material_buckets[:, 2]
        assert float(restitution.max()) <= 0.5 + 1e-6

    def test_make_consistent_enforces_the_friction_constraint(self):
        term = self.Term()
        E.resample_material_buckets(term, (0.3, 0.4), (1.0, 1.2), make_consistent=True)
        assert bool((term.material_buckets[:, 1] <= term.material_buckets[:, 0] + 1e-6).all())

    def test_a_term_without_buckets_is_reported(self):
        assert E.resample_material_buckets(object()) is False


class TestHelpers:
    @pytest.mark.parametrize("op,expected", [("add", 1.5), ("scale", 0.5), ("abs", 1.5)])
    def test_operations(self, op, expected):
        base = torch.full((3,), 1.0) if op != "scale" else torch.full((3,), 1.0)
        samples = torch.full((3,), 0.5) if op == "scale" else torch.full((3,), 0.5)
        result = E.apply_operation(base, samples, op)
        assert float(result[0]) == pytest.approx(expected if op != "abs" else 0.5)

    def test_unknown_operation_raises(self):
        with pytest.raises(ValueError, match="unsupported operation"):
            E.apply_operation(torch.zeros(2), torch.zeros(2), "interpolate")

    def test_sample_uniform_respects_bounds(self):
        s = E.sample_uniform(torch.tensor(-2.0), torch.tensor(3.0), (5000,))
        assert float(s.min()) >= -2.0 and float(s.max()) <= 3.0

    def test_sample_uniform_broadcasts_per_channel(self):
        low = torch.tensor([0.0, 10.0])
        high = torch.tensor([1.0, 11.0])
        s = E.sample_uniform(low, high, (1000, 2))
        assert float(s[:, 0].max()) <= 1.0 and float(s[:, 1].min()) >= 10.0


class FakeDelayBuffer:
    def __init__(self, history_length=8, num_envs=NUM_ENVS):
        self.history_length = history_length
        self.time_lags = torch.zeros(num_envs, dtype=torch.int)
        self.resets = []

    def set_time_lag(self, lags, env_ids):
        lags = torch.as_tensor(lags)
        if int(lags.max()) > self.history_length:
            raise ValueError("time lag exceeds buffer history length")
        self.time_lags[torch.as_tensor(env_ids).long()] = lags.to(torch.int)

    def reset(self, env_ids):
        self.resets.append(torch.as_tensor(env_ids).clone())


class FakeDelayedActuator:
    def __init__(self, history_length=8):
        self.positions_delay_buffer = FakeDelayBuffer(history_length)
        self.velocities_delay_buffer = FakeDelayBuffer(history_length)
        self.efforts_delay_buffer = FakeDelayBuffer(history_length)


class FakePlainActuator:
    """A stock ImplicitActuator: no delay buffers at all."""


def env_with_actuators(actuators):
    env = FakeEnv()
    env.asset.actuators = actuators
    return env


class TestActionDelay:
    def test_sets_a_per_env_lag(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        touched = E.randomize_action_delay(env, None, (0.0, 8.0), Cfg())
        assert touched == 1
        lags = env.asset.actuators["legs"].positions_delay_buffer.time_lags
        assert int(lags.max()) <= 8 and int(lags.min()) >= 0

    def test_lambda_zero_range_means_no_delay(self):
        """The clean A/B baseline: zero delay must be exactly zero."""
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, None, (0.0, 0.0), Cfg())
        assert int(env.asset.actuators["legs"].positions_delay_buffer.time_lags.max()) == 0

    def test_lags_vary_across_environments(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, None, (0.0, 8.0), Cfg())
        lags = env.asset.actuators["legs"].positions_delay_buffer.time_lags
        assert len(set(lags.tolist())) > 1

    def test_all_three_buffers_are_set(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, None, (2.0, 2.0), Cfg())
        actuator = env.asset.actuators["legs"]
        for name in E.DELAY_BUFFERS:
            assert int(getattr(actuator, name).time_lags.max()) == 2

    def test_buffers_are_reset(self):
        """Otherwise stale targets built with the previous joint-default offset
        keep being applied after a reset."""
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, torch.tensor([1, 2]), (1.0, 3.0), Cfg())
        assert env.asset.actuators["legs"].positions_delay_buffer.resets[-1].tolist() == [1, 2]

    def test_only_selected_envs_change(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, torch.tensor([0, 1]), (4.0, 4.0), Cfg())
        lags = env.asset.actuators["legs"].positions_delay_buffer.time_lags
        assert int(lags[0]) == 4 and int(lags[5]) == 0

    def test_clamps_to_the_buffer_capacity(self):
        """max_delay sizes the buffer at construction; set_time_lag raises above
        it, so a curriculum must not be able to crash the run."""
        env = env_with_actuators({"legs": FakeDelayedActuator(history_length=4)})
        E.randomize_action_delay(env, None, (0.0, 100.0), Cfg())
        assert int(env.asset.actuators["legs"].positions_delay_buffer.time_lags.max()) <= 4

    def test_plain_actuators_report_zero_rather_than_failing_silently(self):
        """A config that forgot DelayedImplicitActuatorCfg must be detectable."""
        env = env_with_actuators({"legs": FakePlainActuator()})
        assert E.randomize_action_delay(env, None, (0.0, 8.0), Cfg()) == 0

    def test_counts_every_delayed_actuator_group(self):
        env = env_with_actuators({
            "legs": FakeDelayedActuator(), "arms": FakeDelayedActuator(),
            "hands": FakePlainActuator(),
        })
        assert E.randomize_action_delay(env, None, (0.0, 4.0), Cfg()) == 2

    def test_no_actuators_is_survivable(self):
        env = env_with_actuators({})
        assert E.randomize_action_delay(env, None, (0.0, 8.0), Cfg()) == 0

    def test_empty_env_ids_is_a_no_op(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        assert E.randomize_action_delay(
            env, torch.tensor([], dtype=torch.long), (0.0, 8.0), Cfg()) == 0

    def test_inverted_range_is_tolerated(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, None, (5.0, 2.0), Cfg())
        assert int(env.asset.actuators["legs"].positions_delay_buffer.time_lags.max()) == 5

    def test_fractional_range_is_rounded_to_physics_steps(self):
        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, None, (1.4, 1.4), Cfg())
        assert int(env.asset.actuators["legs"].positions_delay_buffer.time_lags.max()) == 1


class TestDelayIsCurriculumScalable:
    def test_lambda_maps_to_the_papers_training_range(self):
        """LUCID v1 trains over 0-40 ms; at 200 Hz that is 0-8 physics steps."""
        from gear_sonic.research.practice_utility.dr_scaling import RANGE_NOMINALS, scale_range

        assert RANGE_NOMINALS["delay_range"] == 0.0
        assert scale_range([0.0, 8.0], 0.0, 0.0) == [0.0, 0.0]
        assert scale_range([0.0, 8.0], 0.5, 0.0) == pytest.approx([0.0, 4.0])
        assert scale_range([0.0, 8.0], 1.0, 0.0) == pytest.approx([0.0, 8.0])

    def test_scaled_range_drives_the_sampled_lag(self):
        from gear_sonic.research.practice_utility.dr_scaling import scale_range

        env = env_with_actuators({"legs": FakeDelayedActuator()})
        E.randomize_action_delay(env, None, scale_range([0.0, 8.0], 0.25, 0.0), Cfg())
        assert int(env.asset.actuators["legs"].positions_delay_buffer.time_lags.max()) <= 2


class TestStickyResampling:
    """Holding a draw for K episodes decouples randomization *width* (lambda,
    slow) from randomization *frequency* (resample_every, fast)."""

    class Scene:
        num_envs = 12

    class Env:
        def __init__(self):
            self.scene = TestStickyResampling.Scene()

    def test_every_one_is_a_pass_through(self):
        env, ids = self.Env(), torch.arange(12)
        assert E.due_for_resample(env, ids, 1, "k").tolist() == ids.tolist()

    def test_every_zero_is_a_pass_through(self):
        env, ids = self.Env(), torch.arange(12)
        assert E.due_for_resample(env, ids, 0, "k").tolist() == ids.tolist()

    def test_average_redraw_rate_matches_one_over_every(self):
        torch.manual_seed(0)
        env, ids = self.Env(), torch.arange(12)
        total = sum(E.due_for_resample(env, ids, 4, "k").numel() for _ in range(40))
        assert total / 40 == pytest.approx(3.0, abs=0.35)   # 12 envs / every 4

    def test_phases_are_staggered_not_lockstep(self):
        """A zero-init counter redraws the whole batch at once -- a synchronised
        physics shock, the opposite of what sticky DR is for."""
        torch.manual_seed(0)
        env, ids = self.Env(), torch.arange(12)
        sizes = [E.due_for_resample(env, ids, 4, "k").numel() for _ in range(8)]
        assert max(sizes) < 12

    def test_each_env_redraws_exactly_once_per_period(self):
        torch.manual_seed(0)
        env, ids = self.Env(), torch.arange(12)
        seen = []
        for _ in range(4):
            seen.extend(E.due_for_resample(env, ids, 4, "k").tolist())
        assert sorted(seen) == list(range(12))

    def test_channels_keep_independent_counters(self):
        torch.manual_seed(0)
        env, ids = self.Env(), torch.arange(12)
        for _ in range(3):
            E.due_for_resample(env, ids, 4, "mass")
        fresh = E.due_for_resample(env, ids, 1, "push")
        assert fresh.tolist() == ids.tolist()

    def test_partial_env_ids_only_advance_their_own_counters(self):
        torch.manual_seed(0)
        env = self.Env()
        for _ in range(10):
            E.due_for_resample(env, torch.tensor([0, 1]), 4, "k")
        counters = getattr(env, E.RESAMPLE_COUNTER)["k"]
        assert int(counters[0]) >= 10 and int(counters[5]) < 10

    def test_sticky_wrapper_forwards_only_due_envs(self):
        torch.manual_seed(0)
        env = self.Env()
        calls = []

        def inner(env, env_ids, scale):
            calls.append((env_ids.tolist(), scale))

        wrapped = E.sticky(inner, "mass", every=4)
        for _ in range(4):
            wrapped(env, torch.arange(12), scale=2.0)
        forwarded = sorted(i for call, _ in calls for i in call)
        assert forwarded == list(range(12))

    def test_sticky_wrapper_skips_when_nothing_is_due(self):
        torch.manual_seed(0)
        env = self.Env()
        calls = []
        wrapped = E.sticky(lambda e, ids: calls.append(ids), "k", every=1000)
        for _ in range(3):
            wrapped(env, torch.arange(12))
        assert len(calls) <= 1        # at most the initially-phased envs

    def test_sticky_wrapper_preserves_arguments(self):
        env = self.Env()
        seen = {}

        def inner(env, env_ids, a, b=None):
            seen["a"], seen["b"] = a, b

        E.sticky(inner, "k", every=1)(env, torch.arange(12), 7, b=9)
        assert seen == {"a": 7, "b": 9}

    def test_sticky_wrapper_handles_none_env_ids(self):
        env = self.Env()
        calls = []
        E.sticky(lambda e, ids: calls.append(ids.numel()), "k", every=1)(env, None)
        assert calls == [12]
