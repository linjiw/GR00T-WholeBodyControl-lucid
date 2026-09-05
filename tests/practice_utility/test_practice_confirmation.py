import json

import pytest

from gear_sonic.research.practice_utility import practice_confirmation as PC

PRESETS = list(PC.CELL_CHANNELS)


def eval_receipt(seed, mode, values, evaluation_seed=None):
    evaluation_seed = 8700 + seed - 8600 if evaluation_seed is None else evaluation_seed
    return {
        "runs": {
            f"{seed}-{mode}-{preset}": {
                "complete": True,
                "checkpoint_seed": seed,
                "evaluation_seed": evaluation_seed,
                "mode": mode,
                "preset": preset,
                "summary": {
                    "success_rate": values.get(preset, 0.8),
                    "progress_rate": 0.9,
                    "mpjpe_g": 0.1,
                    "mpjpe_l": 0.05,
                    "foot_slip_per_step_m": 0.01,
                    "undesired_contact_rate": 0.02,
                    "torque_saturation": 0.03,
                    "energy_proxy": 1.0,
                    "quality_missing_signals": [],
                },
            }
            for preset in PRESETS
        }
    }


def training_receipt(seed):
    vectors = {
        "prac_null": {},
        "prac_push": {"push_robot": 3.0},
        "prac_easy": {
            "randomize_rigid_body_mass": 3.0,
            "base_com": 3.0,
            "add_joint_default_pos": 3.0,
        },
    }
    return {
        "arms": {
            mode: {
                "seed": seed,
                "mode": mode,
                "tace_final": {
                    "stratum_lambdas": [
                        {name: 1.0 for name in PC.PHYSICS_CHANNELS},
                        {name: vectors[mode].get(name, 1.0) for name in PC.PHYSICS_CHANNELS},
                    ]
                },
            }
            for mode in PC.MODES
        }
    }


def prereg():
    return {
        "claim_scope": "test",
        "independent_unit": "seed",
        "training": {"confirmation_seeds": [8601, 8602]},
        "evaluation": {
            "cells": PRESETS,
            "evaluation_seed_by_checkpoint_seed": {
                "8600": 8700,
                "8601": 8701,
                "8602": 8702,
            },
        },
        "margins": {
            "meaningful_improvement_pts": 5.0,
            "tie_band_pts": 2.0,
            "retention_margin_pts": 2.0,
        },
        "not_claimed": [],
    }


def write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_loader_preserves_seed_in_the_cell_key(tmp_path):
    first = write(tmp_path / "s1.json", eval_receipt(8601, "prac_push", {}))
    second = write(tmp_path / "s2.json", eval_receipt(8602, "prac_push", {}))
    cells = PC.load_evaluations([first, second])
    assert (8601, "prac_push", "ch_push_300") in cells
    assert (8602, "prac_push", "ch_push_300") in cells
    assert len(cells) == 2 * len(PRESETS)


def test_duplicate_cell_is_rejected_instead_of_overwritten(tmp_path):
    first = write(tmp_path / "a.json", eval_receipt(8601, "prac_push", {}))
    second = write(tmp_path / "b.json", eval_receipt(8601, "prac_push", {}))
    with pytest.raises(ValueError, match="duplicate evaluation cell"):
        PC.load_evaluations([first, second])


def test_loader_preserves_every_finite_top_level_reported_scalar(tmp_path):
    metrics = write(
        tmp_path / "metrics.json",
        {
            "eval/all/mpjpe_g": 123.0,
            "eval/success/mpjpe_g": 99.0,
            "eval/quality/energy_proxy": 42.0,
            "nested": {"not": "a scalar"},
            "flag": True,
        },
    )
    receipt = eval_receipt(8601, "prac_push", {})
    for run in receipt["runs"].values():
        run["metrics_path"] = str(metrics)
    cells = PC.load_evaluations([write(tmp_path / "eval.json", receipt)])
    cell = cells[(8601, "prac_push", "ch_push_300")]
    assert cell["metrics_path"] == str(metrics.resolve())
    assert cell["reported_scalars"] == {
        "eval/all/mpjpe_g": 123.0,
        "eval/success/mpjpe_g": 99.0,
        "eval/quality/energy_proxy": 42.0,
    }


def test_exposure_support_is_seed_and_arm_specific(tmp_path):
    exposure = PC.load_exposures([write(tmp_path / "training.json", training_receipt(8601))])
    assert PC.in_support("ch_push_300", exposure[(8601, "prac_push")]) is True
    assert PC.in_support("ch_push_350", exposure[(8601, "prac_push")]) is False
    assert PC.in_support("ch_mass_300", exposure[(8601, "prac_easy")]) is True
    assert PC.in_support("ch_push_300", exposure[(8601, "prac_easy")]) is False


def complete_inputs(tmp_path, push_gain=0.06, targeting_gain=0.00):
    eval_paths = []
    train_paths = []
    for seed in (8600, 8601, 8602):
        train_paths.append(write(tmp_path / f"train-{seed}.json", training_receipt(seed)))
        null = {preset: 0.80 for preset in PRESETS}
        easy = {preset: 0.82 for preset in PRESETS}
        push = {preset: 0.82 + targeting_gain for preset in PRESETS}
        push["ch_push_300"] = null["ch_push_300"] + push_gain
        for mode, values in (("prac_null", null), ("prac_easy", easy), ("prac_push", push)):
            eval_paths.append(
                write(tmp_path / f"eval-{seed}-{mode}.json", eval_receipt(seed, mode, values))
            )
    return PC.load_evaluations(eval_paths), PC.load_exposures(train_paths)


def test_confirmation_uses_only_the_two_new_seeds_for_its_gate(tmp_path):
    cells, exposures = complete_inputs(tmp_path, push_gain=0.06)
    # Make the already-observed discovery seed adverse. The confirmation still passes.
    cells[(8600, "prac_push", "ch_push_300")]["metrics"]["success_rate"] = 0.70
    out = PC.analyze(cells, exposures, prereg())
    decision = out["decisions"]["D1_push_practice_is_productive"]
    assert decision["per_seed"]["8600"] == pytest.approx(-10.0)
    assert decision["confirmation_mean_pts"] == pytest.approx(6.0)
    assert decision["verdict"] == "CONFIRMED"


def test_macro_tie_is_not_described_as_equivalence(tmp_path):
    cells, exposures = complete_inputs(tmp_path, push_gain=0.06, targeting_gain=0.0)
    out = PC.analyze(cells, exposures, prereg())
    decision = out["decisions"]["D2_selecting_push_beats_manageable_placebo"]
    assert decision["verdict"] == "INSIDE_SCREEN_BAND_NOT_EQUIVALENCE"
    assert out["broad_macro_denominator"] == 13


def test_missing_cell_and_wrong_eval_seed_fail_the_instrument_audit(tmp_path):
    cells, exposures = complete_inputs(tmp_path)
    del cells[(8602, "prac_null", "phys_100")]
    cells[(8601, "prac_push", "ch_push_300")]["evaluation_seed"] = 9999
    out = PC.analyze(cells, exposures, prereg())
    assert out["instrument_audit"]["complete"] is False
    assert len(out["instrument_audit"]["missing_cells"]) == 1
    assert len(out["instrument_audit"]["wrong_evaluation_seeds"]) == 1
