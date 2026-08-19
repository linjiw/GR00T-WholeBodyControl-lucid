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
