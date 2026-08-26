from pathlib import Path
from types import SimpleNamespace

from scripts.practice_utility import run_latency_distribution_sweep as S


def test_grid_is_frozen_and_contains_the_historical_training_match():
    cells = S.build_cells()
    assert len(cells) == 22
    matched = [cell for cell in cells if cell["training_matched"]]
    assert [cell["cell_id"] for cell in matched] == ["dr100_episode_uniform_independent_08"]


def test_panel_split_is_canonical_content_disjoint():
    pool = {
        "motions": [
            {"motion_key": "a1", "canonical_name": "a"},
            {"motion_key": "a2", "canonical_name": "a"},
            {"motion_key": "b1", "canonical_name": "b"},
            {"motion_key": "c1", "canonical_name": "c"},
        ]
    }
    split = {"assignment": {key: "dev" for key in ("a1", "a2", "b1", "c1")}}
    panels = S.select_panel_keys(pool, split, target=1, salt="fixed")
    by_key = {row["motion_key"]: row["canonical_name"] for row in pool["motions"]}
    discovery_groups = {by_key[key] for key in panels["discovery"]}
    confirmation_groups = {by_key[key] for key in panels["confirmation"]}
    assert not discovery_groups & confirmation_groups


def test_jitter_command_has_reset_interval_and_independent_physics_scale():
    args = SimpleNamespace(num_envs=32, smpl_motion_file="smpl")
    cell = next(
        cell for cell in S.build_cells() if cell["cell_id"] == "dr050_jitter_uniform_common_08"
    )
    command = S.build_command(
        args,
        Path("/tmp/model.pt"),
        "lucid",
        cell,
        9100,
        "branch",
        Path("/tmp/out"),
        "/tmp/motions",
    )
    assert "+manager_env/events=tracking/lucid_eval_latency_jitter" in command
    assert "++callbacks.practice_eval.non_latency_dr_scale=0.5" in command
    assert any(
        "randomize_action_delay_interval.params.delay_range=[0,8]" in item for item in command
    )
    assert any("randomize_action_delay.params.coupling=common" in item for item in command)


def test_discovery_selection_requires_advantage_over_both_references():
    def cell(success_fixed, success_off, progress_fixed, progress_off):
        return {
            "paired": {
                "fixed": {
                    "success_rate": {"mean_delta": success_fixed},
                    "progress_rate": {"mean_delta": progress_fixed},
                },
                "off": {
                    "success_rate": {"mean_delta": success_off},
                    "progress_rate": {"mean_delta": progress_off},
                },
            }
        }

    summary = {
        "eligible": cell(0.0, 0.1, 0.02, 0.03),
        "loses_fixed": cell(-0.01, 0.2, 0.10, 0.10),
    }
    assert S.select_confirmation_cells(summary) == ["eligible"]


def test_capacity_gate_returns_when_enough_memory(monkeypatch):
    monkeypatch.setattr(S.TP, "gpu_snapshot", lambda: {"free_mib": 20000})
    S.wait_for_capacity(18000, wait_minutes=0)


def test_capacity_gate_fails_without_wait_budget(monkeypatch):
    monkeypatch.setattr(S.TP, "gpu_snapshot", lambda: {"free_mib": 10000})
    try:
        S.wait_for_capacity(18000, wait_minutes=0)
    except RuntimeError as error:
        assert "capacity gate failed" in str(error)
    else:
        raise AssertionError("insufficient capacity must fail")
