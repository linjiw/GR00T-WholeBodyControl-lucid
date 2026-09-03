"""The thermal channel: does it behave like a torque budget, and is it off at zero?

The channel is proposed because it is different in kind from every other one
here: the difficulty is created by the policy's own effort, it accumulates within
an episode, and escaping it needs an economical gait rather than a reflex. These
tests pin exactly those properties, plus the two safety rails any new channel in
this project has to clear — that it is a bit-identical no-op at intensity zero,
and that it can never hand a joint more torque than the hardware has.
"""

from __future__ import annotations

import math

import pytest
import torch

from gear_sonic.research.practice_utility.thermal import (
    ThermalConfig,
    ThermalState,
    sustained_duty_to_onset,
)

PEAK = torch.tensor([139.0, 139.0, 88.0, 50.0], dtype=torch.float64)
DT = 0.02  # the 50 Hz control step


def state(lam=1.0, num_envs=3, **cfg):
    return ThermalState(num_envs, PEAK, lam=lam, config=ThermalConfig(**cfg))


def run(st, duty, seconds):
    """Hold a constant duty for a while; return the final available fraction of peak."""
    steps = int(round(seconds / DT))
    torque = duty * PEAK.unsqueeze(0).expand(st.num_envs, -1)
    for _ in range(steps):
        st.step(torque, DT)
    return (st.available_torque() / PEAK.unsqueeze(0))


# ------------------------------------------------------------ off means off

def test_at_zero_intensity_the_channel_is_a_bit_identical_no_op():
    """A run with the channel enabled at zero must be a valid baseline."""
    st = state(lam=0.0)
    assert float(st.temperature.max()) == 0.0
    for _ in range(500):  # 10 s of continuous peak torque
        limit = st.step(PEAK.unsqueeze(0).expand(3, -1), DT)
        assert torch.equal(limit, PEAK.unsqueeze(0).expand(3, -1))
    assert float(st.temperature.max()) == 0.0
    commanded = torch.tensor([[100.0, -139.0, 40.0, 10.0]] * 3, dtype=torch.float64)
    assert torch.equal(st.clamp(commanded), commanded)


def test_at_zero_intensity_a_reset_draws_exactly_cold():
    st = state(lam=0.0)
    st.reset()
    assert float(st.temperature.max()) == 0.0


# ------------------------------------------------- it behaves like a budget

def test_available_torque_never_exceeds_peak_and_never_falls_below_continuous():
    st = state(lam=3.0)
    fraction = run(st, duty=1.0, seconds=30.0)
    assert float(fraction.max()) <= 1.0 + 1e-12
    assert float(fraction.min()) >= st.config.continuous_over_peak - 1e-12


def test_a_cold_joint_below_the_onset_is_not_derated():
    st = state(lam=1.0)
    fraction = run(st, duty=0.1, seconds=2.0)
    assert float(fraction.min()) == pytest.approx(1.0)


def test_harder_work_derates_sooner():
    """Monotone in duty: the whole point of an endogenous channel."""
    lazy = float(run(state(lam=2.0), duty=0.3, seconds=4.0).min())
    busy = float(run(state(lam=2.0), duty=0.9, seconds=4.0).min())
    assert busy < lazy


def test_higher_intensity_derates_sooner_at_the_same_effort():
    mild = float(run(state(lam=1.0), duty=0.6, seconds=4.0).min())
    harsh = float(run(state(lam=3.0), duty=0.6, seconds=4.0).min())
    assert harsh < mild


def test_a_joint_recovers_when_the_effort_stops():
    st = state(lam=3.0)
    hot = float(run(st, duty=1.0, seconds=10.0).min())
    cooled = float(run(st, duty=0.0, seconds=60.0).min())
    assert cooled > hot


def test_clamping_preserves_sign_and_respects_the_budget():
    st = state(lam=3.0)
    run(st, duty=1.0, seconds=20.0)
    commanded = torch.tensor([[139.0, -139.0, 88.0, -50.0]] * 3, dtype=torch.float64)
    limited = st.clamp(commanded)
    assert torch.equal(limited.sign(), commanded.sign())
    assert bool((limited.abs() <= st.available_torque() + 1e-9).all())
    assert float(limited.abs().max()) < float(commanded.abs().max())


def test_heating_uses_the_delivered_torque_not_the_command():
    """A joint that could not produce the torque must not heat as though it had."""
    st = state(lam=3.0)
    run(st, duty=1.0, seconds=10.0)
    limit = st.available_torque()
    before = st.temperature.clone()
    st.step(st.clamp(PEAK.unsqueeze(0).expand(3, -1)), DT)  # delivered, i.e. limited
    delivered_rise = (st.temperature - before).clone()
    st.temperature = before.clone()
    st.step(PEAK.unsqueeze(0).expand(3, -1), DT)  # as if the command had been met
    commanded_rise = st.temperature - before
    assert float(delivered_rise.max()) < float(commanded_rise.max())
    assert float(limit.max()) < float(PEAK.max())


# ------------------------------------------- the ladder rests on this formula

@pytest.mark.parametrize("lam,duty", [(1.0, 0.6), (2.0, 0.5), (3.0, 0.4), (2.0, 0.8)])
def test_the_analytic_onset_time_matches_the_integrator(lam, duty):
    """The severity ladder was chosen from this formula, so it has to be right."""
    config = ThermalConfig()
    predicted = sustained_duty_to_onset(config, lam, duty)
    assert math.isfinite(predicted)
    st = ThermalState(1, PEAK, lam=lam, config=config)
    st.temperature.zero_()  # the formula describes a COLD start; episodes may begin warm
    torque = duty * PEAK.unsqueeze(0)
    observed = None
    for step in range(int(round(4 * predicted / DT))):
        st.step(torque, DT)
        if float(st.temperature.max()) >= config.onset:
            observed = (step + 1) * DT
            break
    assert observed is not None
    assert observed == pytest.approx(predicted, rel=0.05)


def test_a_duty_whose_steady_state_sits_below_the_onset_never_derates():
    config = ThermalConfig()
    assert sustained_duty_to_onset(config, lam=0.5, duty=0.2) == float("inf")
    st = ThermalState(1, PEAK, lam=0.5, config=config)
    st.temperature.zero_()
    fraction = None
    for _ in range(30000):  # 600 s
        st.step(0.2 * PEAK.unsqueeze(0), DT)
    fraction = (st.available_torque() / PEAK.unsqueeze(0))
    assert float(fraction.min()) == pytest.approx(1.0)


# ------------------------------------------------------------- housekeeping

def test_a_reset_draw_is_bounded_by_the_intensity():
    st = state(lam=0.5, num_envs=64)
    st.reset()
    assert float(st.temperature.max()) <= 0.5 * st.config.initial_temperature_max + 1e-12
    assert float(st.temperature.min()) >= 0.0


def test_the_report_states_the_severity_rather_than_assuming_it():
    st = state(lam=2.0)
    run(st, duty=1.0, seconds=10.0)
    report = st.report()
    assert report["lam"] == 2.0
    assert 0.0 < report["mean_available_fraction_of_peak"] <= 1.0
    assert report["fraction_derated"] > 0.0
    assert report["config"]["depth"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [{"continuous_over_peak": 0.0}, {"onset": 1.0},
                                 {"heat_seconds": 0.0}])
def test_an_impossible_configuration_is_refused(bad):
    with pytest.raises(ValueError):
        state(**bad)


def test_a_negative_intensity_is_refused():
    with pytest.raises(ValueError, match="lam"):
        state(lam=-1.0)
