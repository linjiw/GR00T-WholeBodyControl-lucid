"""The termination-margin signal, tested against the kill boundary it mirrors."""

import math

import pytest
import torch

from gear_sonic.research.practice_utility import margin_signal as MS

BODIES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
          "left_wrist_yaw_link", "right_wrist_yaw_link"]


class Cmd:
    """A duck-typed TrackingCommand with N envs and the six bodies above."""

    def __init__(self, n):
        self.cmd_body_names = list(BODIES)
        self.body_pos_relative_w = torch.zeros(n, len(BODIES), 3)
        self.robot_body_pos_w = torch.zeros(n, len(BODIES), 3)
        self.anchor_pos_w = torch.zeros(n, 3)
        self.robot_anchor_pos_w = torch.zeros(n, 3)
        self.anchor_quat_w = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
        self.robot_anchor_quat_w = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
        self.running_ref_root_height = torch.full((n,), 0.8)


class Term:
    def __init__(self, params): self.params = params


class Mgr:
    def __init__(self, terms):
        self.active_terms = list(terms); self._term_cfgs = [Term(p) for p in terms.values()]


def manager(foot=0.5, ee=0.5, anchor=0.5, ori=1.0, adaptive=True):
    return Mgr({
        "anchor_pos": {"threshold": anchor, "threshold_adaptive": adaptive, "down_threshold": 0.75, "root_height_threshold": 0.5},
        "anchor_ori_full": {"threshold": ori},
        "ee_body_pos": {"threshold": ee, "threshold_adaptive": adaptive, "down_threshold": 0.75, "root_height_threshold": 0.5},
        "time_out": {},
        "foot_pos_xyz": {"threshold": foot},
    })


class TestThresholds:
    def test_reads_the_live_values_not_yaml(self):
        t = MS.read_thresholds(manager(foot=0.2, ee=0.15, anchor=0.15, ori=0.2))
        assert (t.foot, t.ee, t.anchor, t.ori) == (0.2, 0.15, 0.15, 0.2)
        assert t.ee_adaptive and t.anchor_adaptive and t.ee_down == 0.75

    def test_a_missing_term_is_an_error(self):
        with pytest.raises(KeyError, match="foot_pos_xyz"):
            MS.read_thresholds(Mgr({"anchor_pos": {"threshold": 0.5}}))


class TestMargins:
    def test_one_is_exactly_the_kill_boundary(self):
        cmd = Cmd(1); t = MS.read_thresholds(manager())
        cmd.robot_body_pos_w[0, 2] = torch.tensor([0.5, 0.0, 0.0])   # left ankle 0.5 m off in x
        m = MS.termination_margins(cmd, t)
        assert m.feet[0] == pytest.approx(1.0)
        assert m.ee[0] == pytest.approx(0.0)          # x error does not count for the z-height term
        assert MS.CULPRITS[m.culprit[0]] == "feet"

    def test_ee_uses_z_only_and_includes_wrists(self):
        cmd = Cmd(1); t = MS.read_thresholds(manager())
        cmd.robot_body_pos_w[0, 4] = torch.tensor([0.0, 0.0, 0.25])  # left wrist 0.25 m high
        m = MS.termination_margins(cmd, t)
        assert m.ee[0] == pytest.approx(0.5)
        assert m.feet[0] == pytest.approx(0.0)
        assert MS.CULPRITS[m.culprit[0]] == "ee"

    def test_adaptive_threshold_loosens_for_a_low_reference_root(self):
        cmd = Cmd(2); t = MS.read_thresholds(manager())
        cmd.running_ref_root_height[1] = 0.3               # crouching reference
        cmd.robot_body_pos_w[:, 2] = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]])
        m = MS.termination_margins(cmd, t)
        assert m.ee[0] == pytest.approx(0.3 / 0.5)
        assert m.ee[1] == pytest.approx(0.3 / 0.75)

    def test_orientation_margin_is_squared_angle_over_threshold(self):
        cmd = Cmd(1); t = MS.read_thresholds(manager(ori=1.0))
        a = math.pi / 2                                    # 90 degrees about z
        cmd.robot_anchor_quat_w[0] = torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        m = MS.termination_margins(cmd, t)
        assert m.ori[0] == pytest.approx(a * a, rel=1e-5)

    def test_pelvis_margin_is_height_only(self):
        cmd = Cmd(1); t = MS.read_thresholds(manager())
        cmd.robot_anchor_pos_w[0] = torch.tensor([3.0, 0.0, -0.25])  # 3 m of xy drift, 0.25 m low
        m = MS.termination_margins(cmd, t)
        assert m.pelvis[0] == pytest.approx(0.5)

    def test_overall_is_the_binding_margin(self):
        cmd = Cmd(1); t = MS.read_thresholds(manager())
        cmd.robot_body_pos_w[0, 3] = torch.tensor([0.1, 0.0, 0.0])
        cmd.robot_anchor_pos_w[0, 2] = -0.4
        m = MS.termination_margins(cmd, t)
        assert m.overall[0] == pytest.approx(0.8)
        assert MS.CULPRITS[m.culprit[0]] == "pelvis"

    def test_a_missing_body_is_an_error(self):
        cmd = Cmd(1); cmd.cmd_body_names = ["pelvis"]
        with pytest.raises(KeyError, match="missing"):
            MS.termination_margins(cmd, MS.read_thresholds(manager()))


class TestPrefixAccumulator:
    def _margins(self, values):
        v = torch.tensor(values, dtype=torch.float32)
        z = torch.zeros_like(v)
        return MS.Margins(feet=v, ee=z, pelvis=z, ori=z)

    def test_prefix_mean_over_the_first_k_steps(self):
        acc = MS.PrefixAccumulator(num_envs=1, horizon=3)
        for age, value in ((0, 9.0), (1, 0.1), (2, 0.2), (3, 0.3), (4, 5.0), (5, 5.0)):
            acc.push(self._margins([value]), torch.tensor([age]))
        ids, means, counts, shares = acc.finish(torch.tensor([True]))
        assert ids.tolist() == [0]
        assert means[0] == pytest.approx(0.2)              # age 0 skipped, ages 4-5 beyond K
        assert counts[0] == 3
        assert shares[0, MS.CULPRITS.index("feet")] == pytest.approx(1.0)

    def test_short_episodes_contribute_what_they_had(self):
        acc = MS.PrefixAccumulator(num_envs=1, horizon=12)
        acc.push(self._margins([0.4]), torch.tensor([1]))
        acc.push(self._margins([0.6]), torch.tensor([2]))
        ids, means, counts, _ = acc.finish(torch.tensor([True]))
        assert means[0] == pytest.approx(0.5) and counts[0] == 2

    def test_finish_clears_only_the_ended_envs(self):
        acc = MS.PrefixAccumulator(num_envs=2, horizon=4)
        acc.push(self._margins([0.5, 0.9]), torch.tensor([1, 1]))
        ids, means, _, _ = acc.finish(torch.tensor([True, False]))
        assert ids.tolist() == [0] and means[0] == pytest.approx(0.5)
        assert acc.counts.tolist() == [0.0, 1.0]

    def test_an_env_that_ended_at_age_zero_is_dropped_not_zeroed(self):
        acc = MS.PrefixAccumulator(num_envs=1, horizon=4)
        acc.push(self._margins([0.5]), torch.tensor([0]))
        ids, means, _, _ = acc.finish(torch.tensor([True]))
        assert ids.numel() == 0 and means.numel() == 0


class TestRatioAndBand:
    def test_ratio_is_one_at_lambda_zero_by_construction(self):
        r = MS.MarginRatio(tau=1.0)
        assert r.update(0.3, 0.3) == pytest.approx(1.0)

    def test_ema_smooths_and_ignores_missing(self):
        r = MS.MarginRatio(tau=2.0)
        r.update(0.2, 0.2)
        val = r.update(0.6, None)                           # yardstick had no episodes this iteration
        assert r.yardstick.value == pytest.approx(0.2)
        assert val == pytest.approx(0.4 / 0.2)

    def test_band_error_signs(self):
        assert MS.band_error(1.0, 1.1, 1.3) == pytest.approx(0.1)     # better than band: raise
        assert MS.band_error(1.2, 1.1, 1.3) == 0.0                     # inside: hold
        assert MS.band_error(1.5, 1.1, 1.3) == pytest.approx(-0.2)    # worse: lower
        assert MS.band_error(None, 1.1, 1.3) == 0.0

    def test_cohort_median(self):
        ids = torch.tensor([0, 1, 2, 3]); vals = torch.tensor([0.1, 0.9, 0.2, 0.8])
        mask = torch.tensor([True, False, True, False])
        assert MS.cohort_median(vals, ids, mask) == pytest.approx(0.15)
        assert MS.cohort_median(vals, ids, torch.zeros(4, dtype=torch.bool)) is None
