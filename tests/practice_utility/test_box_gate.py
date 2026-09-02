import pytest

from gear_sonic.research.practice_utility import box_gate as BG


CHANNELS = ("physics_material", "randomize_rigid_body_mass", "push_robot")


def make(**overrides):
    config = dict(
        channels=CHANNELS,
        lambda_max=1.5,
        threshold=0.8,
        window=3,
        step_size=0.25,
        probe_offset=0.25,
        dwell=0,
        min_episodes=1,
        channel_budget=0,
    )
    config.update(overrides)
    return BG.BoxGateController(BG.BoxGateConfig(**config), initial_lambda=1.0)


def feed(gate, survival, n, episodes=10):
    last = None
    for _ in range(n):
        last = gate.update(probe_survival=survival, probe_episodes=episodes, mean_return=5.0)
    return last


def test_probe_raises_only_the_active_channel():
    gate = make()
    assert gate.active_channel == "physics_material"
    assert gate.frontier == {c: 1.0 for c in CHANNELS}
    assert gate.probe == {"physics_material": 1.25, "randomize_rigid_body_mass": 1.0, "push_robot": 1.0}


def test_passing_window_steps_that_channel_and_hands_the_probe_on():
    gate = make()
    step = feed(gate, 0.95, 3)
    assert step.fired and step.active_channel == "physics_material"
    assert step.frontier_after["physics_material"] == pytest.approx(1.25)
    assert step.frontier_after["randomize_rigid_body_mass"] == pytest.approx(1.0)
    # The probe has moved: next update runs on the mass channel, marked as a rotation.
    nxt = gate.update(probe_survival=0.9, probe_episodes=10, mean_return=5.0)
    assert nxt.active_channel == "randomize_rigid_body_mass"
    assert nxt.rotation == "fired" and nxt.rotated_to == "randomize_rigid_body_mass"
    assert nxt.probe == {"physics_material": 1.25, "randomize_rigid_body_mass": 1.25, "push_robot": 1.0}
    # The friction channel's evidence was cleared when the probe left it.
    assert gate.gates["physics_material"].window_length == 0


def test_failing_window_blocks_the_channel_and_rotates():
    gate = make()
    step = feed(gate, 0.2, 3)
    assert not step.fired and step.withheld == "below_threshold"
    assert step.frontier_after == {c: 1.0 for c in CHANNELS}
    nxt = gate.update(probe_survival=0.9, probe_episodes=10, mean_return=5.0)
    assert nxt.active_channel == "randomize_rigid_body_mass"
    assert nxt.rotation == "blocked"
    assert "physics_material" in nxt.blocked


def test_a_round_of_failures_clears_the_blocks_and_retries():
    gate = make()
    for _ in CHANNELS:
        feed(gate, 0.2, 3)
    # All three blocked: the round closes on the next selection.
    step = gate.update(probe_survival=0.9, probe_episodes=10, mean_return=5.0)
    assert step.active_channel == "physics_material"
    assert step.round_index >= 1
    assert step.blocked == ()


def test_channels_that_are_never_probed_never_move():
    gate = make()
    feed(gate, 0.95, 3)  # friction fires
    frontier = gate.frontier
    assert frontier["randomize_rigid_body_mass"] == 1.0 and frontier["push_robot"] == 1.0
    assert gate.gates["push_robot"].window_length == 0


def test_no_channel_ever_decreases_under_the_freeze_guard():
    gate = make(return_window=2, return_relative_drop=0.2)
    feed(gate, 0.95, 3)
    peak = gate.frontier
    for _ in range(6):
        step = gate.update(probe_survival=0.95, probe_episodes=10, mean_return=0.5)
    assert step.guard_tripped or step.withheld == "guard_freeze"
    assert not step.applied_decrease
    assert all(gate.frontier[c] >= peak[c] for c in CHANNELS)


def test_all_channels_at_ceiling_retires_the_probe():
    gate = make(step_size=0.5, probe_offset=0.5)
    for _ in range(3):
        feed(gate, 0.95, 3)
    step = gate.update(probe_survival=0.95, probe_episodes=10, mean_return=5.0)
    assert gate.all_at_ceiling
    assert step.active_channel is None
    assert step.withheld == "all_at_ceiling"
    assert step.probe == gate.frontier == {c: 1.5 for c in CHANNELS}


def test_channel_budget_times_out_a_channel_without_evidence():
    gate = make(channel_budget=4, window=50)
    step = feed(gate, None, 4, episodes=0)
    assert step.withheld == "channel_timeout"
    nxt = gate.update()
    assert nxt.rotation == "timeout" and nxt.active_channel == "randomize_rigid_body_mass"


def test_state_round_trips():
    gate = make()
    feed(gate, 0.95, 3)
    feed(gate, 0.2, 2)
    state = gate.state_dict()
    clone = make()
    clone.load_state_dict(state)
    assert clone.frontier == gate.frontier
    assert clone.active_channel == gate.active_channel
    assert clone.round_index == gate.round_index
    assert clone.blocked == gate.blocked
    a = gate.update(probe_survival=0.2, probe_episodes=10, mean_return=5.0)
    b = clone.update(probe_survival=0.2, probe_episodes=10, mean_return=5.0)
    assert a.to_dict() == b.to_dict()


def test_per_channel_ceilings_bound_each_channel_separately():
    gate = make(lambda_max={"physics_material": 1.25, "randomize_rigid_body_mass": 1.5, "push_robot": 2.0})
    feed(gate, 0.95, 3)  # friction -> 1.25, now at ceiling
    assert gate.gates["physics_material"].at_ceiling
    assert "physics_material" not in gate._eligible()
    step = gate.to_dict() if hasattr(gate, "to_dict") else gate.history[-1].to_dict()
    assert step["lambda_mean"] == pytest.approx((1.25 + 1.0 + 1.0) / 3)


def test_config_rejects_duplicates_and_bad_budget():
    with pytest.raises(ValueError):
        BG.BoxGateConfig(channels=("a", "a"))
    with pytest.raises(ValueError):
        BG.BoxGateConfig(channels=("a",), channel_budget=-1)
