"""Seed-aware analysis for the practice-allocation confirmation.

The discovery readout was intentionally single-seed.  This module keeps the
training/checkpoint seed in every key so that loading another receipt can never
silently overwrite an earlier seed with the same mode and evaluation preset.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

MODES = ("prac_null", "prac_push", "prac_easy")
SCALAR_METRICS = (
    "success_rate",
    "progress_rate",
    "mpjpe_g",
    "mpjpe_l",
    "foot_slip_per_step_m",
    "undesired_contact_rate",
    "torque_saturation",
    "energy_proxy",
)
PHYSICS_CHANNELS = (
    "randomize_rigid_body_mass",
    "base_com",
    "add_joint_default_pos",
    "physics_material",
    "push_robot",
)
CELL_CHANNELS: dict[str, dict[str, float]] = {
    "phys_000": {},
    "phys_100": {name: 1.0 for name in PHYSICS_CHANNELS},
    "phys_150": {name: 1.5 for name in PHYSICS_CHANNELS},
    "phys_200": {name: 2.0 for name in PHYSICS_CHANNELS},
    "ch_push_200": {"push_robot": 2.0},
    "ch_push_300": {"push_robot": 3.0},
    "ch_mass_300": {"randomize_rigid_body_mass": 3.0},
    "ch_com_300": {"base_com": 3.0},
    "ch_joint_300": {"add_joint_default_pos": 3.0},
    "ch_fric_150": {"physics_material": 1.5},
    "ch_push_fric_300_150": {"push_robot": 3.0, "physics_material": 1.5},
    "ch_push_350": {"push_robot": 3.5},
    "ch_push_fric_350_150": {"push_robot": 3.5, "physics_material": 1.5},
}


def _json_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(Path(path) for path in sorted(glob.glob(str(candidate / "*.json"))))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise FileNotFoundError(candidate)
    return paths


def load_evaluations(inputs: Iterable[Path]) -> dict[tuple[int, str, str], dict[str, Any]]:
    """Load completed evaluation cells without dropping the checkpoint seed."""
    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    for path in _json_paths(inputs):
        receipt = json.loads(path.read_text())
        for run in (receipt.get("runs") or {}).values():
            if not run.get("complete"):
                continue
            summary = run.get("summary") or {}
            if summary.get("success_rate") is None:
                continue
            key = (int(run["checkpoint_seed"]), str(run["mode"]), str(run["preset"]))
            if key in cells:
                raise ValueError(f"duplicate evaluation cell {key} in {path}")
            cells[key] = {
                "checkpoint_seed": key[0],
                "evaluation_seed": int(run["evaluation_seed"]),
                "mode": key[1],
                "preset": key[2],
                "metrics": {name: summary.get(name) for name in SCALAR_METRICS},
                "quality_missing_signals": summary.get("quality_missing_signals", []),
                "receipt": str(path.resolve()),
            }
    return cells


def load_exposures(inputs: Iterable[Path]) -> dict[tuple[int, str], dict[str, float]]:
    """Load the realized top-stratum vector for every training seed and arm."""
    exposures: dict[tuple[int, str], dict[str, float]] = {}
    for path in _json_paths(inputs):
        receipt = json.loads(path.read_text())
        arms = receipt.get("arms") or {}
        members = arms.values() if isinstance(arms, dict) else arms
        for arm in members:
            if not isinstance(arm, dict) or arm.get("mode") not in MODES:
                continue
            key = (int(arm["seed"]), str(arm["mode"]))
            if key in exposures:
                raise ValueError(f"duplicate training exposure {key} in {path}")
            strata = ((arm.get("tace_final") or {}).get("stratum_lambdas")) or []
            top = strata[-1] if strata else None
            if not isinstance(top, dict):
                raise ValueError(f"missing realized top-stratum vector for {key} in {path}")
            exposures[key] = {
                str(name): float(value) for name, value in top.items() if float(value) > 1.0
            }
    return exposures


def in_support(preset: str, exposure: dict[str, float] | None) -> bool | None:
    """Whether every widened channel in a cell was inside the arm's practice support."""
    if exposure is None:
        return None
    return all(
        exposure.get(channel, 1.0) >= level - 1e-9
        for channel, level in CELL_CHANNELS[preset].items()
    )


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def _paired_success(
    cells: dict[tuple[int, str, str], dict[str, Any]],
    seed: int,
    treatment: str,
    reference: str,
    preset: str,
) -> float | None:
    left = cells.get((seed, treatment, preset))
    right = cells.get((seed, reference, preset))
    if left is None or right is None:
        return None
    return 100.0 * (
        float(left["metrics"]["success_rate"]) - float(right["metrics"]["success_rate"])
    )


def _macro_success(
    cells: dict[tuple[int, str, str], dict[str, Any]],
    seed: int,
    mode: str,
    presets: list[str],
) -> float | None:
    values = []
    for preset in presets:
        cell = cells.get((seed, mode, preset))
        if cell is None:
            return None
        values.append(float(cell["metrics"]["success_rate"]))
    return 100.0 * statistics.fmean(values)


def _seed_summary(values: dict[int, float | None]) -> dict[str, Any]:
    complete = [float(value) for value in values.values() if value is not None]
    return {
        "per_seed": {str(seed): value for seed, value in values.items()},
        **_mean_sd(complete),
        "complete": len(complete) == len(values),
    }


def analyze(
    cells: dict[tuple[int, str, str], dict[str, Any]],
    exposures: dict[tuple[int, str], dict[str, float]],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen two-seed confirmation rules and build a three-seed readout."""
    evaluation = preregistration["evaluation"]
    presets = list(evaluation["cells"])
    confirmation_seeds = [int(seed) for seed in preregistration["training"]["confirmation_seeds"]]
    all_seeds = [8600, *confirmation_seeds]
    improve = float(preregistration["margins"]["meaningful_improvement_pts"])
    tie = float(preregistration["margins"]["tie_band_pts"])
    retention = float(preregistration["margins"]["retention_margin_pts"])

    expected_eval_seeds = {
        int(seed): int(value)
        for seed, value in evaluation["evaluation_seed_by_checkpoint_seed"].items()
    }
    missing_cells = [
        {"checkpoint_seed": seed, "mode": mode, "preset": preset}
        for seed in all_seeds
        for mode in MODES
        for preset in presets
        if (seed, mode, preset) not in cells
    ]
    wrong_eval_seeds = [
        {
            "checkpoint_seed": seed,
            "mode": mode,
            "preset": preset,
            "expected": expected_eval_seeds[seed],
            "observed": cell["evaluation_seed"],
        }
        for (seed, mode, preset), cell in cells.items()
        if seed in all_seeds
        and mode in MODES
        and preset in presets
        and cell["evaluation_seed"] != expected_eval_seeds[seed]
    ]

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for preset in presets:
        table[preset] = {}
        for seed in all_seeds:
            seed_row: dict[str, Any] = {}
            for mode in MODES:
                cell = cells.get((seed, mode, preset))
                if cell is None:
                    continue
                seed_row[mode] = {
                    **cell["metrics"],
                    "quality_missing_signals": cell["quality_missing_signals"],
                    "in_training_support": in_support(preset, exposures.get((seed, mode))),
                    "evaluation_seed": cell["evaluation_seed"],
                }
            table[preset][str(seed)] = seed_row

    macros = {
        seed: {mode: _macro_success(cells, seed, mode, presets) for mode in MODES}
        for seed in all_seeds
    }
    push_deltas = {
        seed: _paired_success(cells, seed, "prac_push", "prac_null", "ch_push_300")
        for seed in all_seeds
    }
    targeting_deltas = {
        seed: (
            None
            if macros[seed]["prac_push"] is None or macros[seed]["prac_easy"] is None
            else macros[seed]["prac_push"] - macros[seed]["prac_easy"]
        )
        for seed in all_seeds
    }

    d1_new = [push_deltas[seed] for seed in confirmation_seeds]
    d1_evaluable = all(value is not None for value in d1_new)
    d1_confirmed = bool(
        d1_evaluable
        and all(float(value) > 0.0 for value in d1_new)
        and statistics.fmean(float(value) for value in d1_new) >= improve
    )

    d2_new = [targeting_deltas[seed] for seed in confirmation_seeds]
    d2_evaluable = all(value is not None for value in d2_new)
    d2_mean = statistics.fmean(float(value) for value in d2_new) if d2_evaluable else None
    d2_pays = bool(
        d2_evaluable and all(float(value) > 0.0 for value in d2_new) and float(d2_mean) >= improve
    )
    d2_inside_band = bool(d2_evaluable and abs(float(d2_mean)) < tie)

    retention_deltas = {
        mode: {
            seed: _paired_success(cells, seed, mode, "prac_null", "phys_100") for seed in all_seeds
        }
        for mode in ("prac_push", "prac_easy")
    }
    retention_tradeoffs = [
        {"mode": mode, "seed": seed, "delta_pts": value}
        for mode, values in retention_deltas.items()
        for seed, value in values.items()
        if seed in confirmation_seeds and value is not None and value < -retention
    ]

    supportive = {
        preset: _seed_summary(
            {
                seed: _paired_success(cells, seed, "prac_push", "prac_null", preset)
                for seed in all_seeds
            }
        )
        for preset in ("ch_push_350", "ch_push_fric_350_150")
    }

    return {
        "kind": "lucid_practice_allocation_confirmation_analysis",
        "schema_version": 1,
        "claim_scope": preregistration["claim_scope"],
        "independent_unit": preregistration["independent_unit"],
        "confirmation_seeds": confirmation_seeds,
        "discovery_seed": 8600,
        "instrument_audit": {
            "expected_cells": len(all_seeds) * len(MODES) * len(presets),
            "observed_expected_cells": len(all_seeds) * len(MODES) * len(presets)
            - len(missing_cells),
            "missing_cells": missing_cells,
            "wrong_evaluation_seeds": wrong_eval_seeds,
            "complete": not missing_cells and not wrong_eval_seeds,
        },
        "realized_practice_vectors": {
            str(seed): {mode: exposures.get((seed, mode)) for mode in MODES} for seed in all_seeds
        },
        "cells": table,
        "macro_success_pts": {str(seed): values for seed, values in macros.items()},
        "decisions": {
            "D1_push_practice_is_productive": {
                **_seed_summary(push_deltas),
                "confirmation_values_pts": {
                    str(seed): push_deltas[seed] for seed in confirmation_seeds
                },
                "confirmation_mean_pts": (
                    statistics.fmean(float(value) for value in d1_new) if d1_evaluable else None
                ),
                "verdict": (
                    "CONFIRMED"
                    if d1_confirmed
                    else "NOT_CONFIRMED" if d1_evaluable else "UNEVALUABLE"
                ),
                "rule": "both new-seed deltas > 0 and their mean >= 5 points",
            },
            "D2_selecting_push_beats_manageable_placebo": {
                **_seed_summary(targeting_deltas),
                "confirmation_values_pts": {
                    str(seed): targeting_deltas[seed] for seed in confirmation_seeds
                },
                "confirmation_mean_pts": d2_mean,
                "verdict": (
                    "TARGETING_PAYS"
                    if d2_pays
                    else (
                        "INSIDE_SCREEN_BAND_NOT_EQUIVALENCE"
                        if d2_inside_band
                        else "INCONCLUSIVE_OR_ADVERSE" if d2_evaluable else "UNEVALUABLE"
                    )
                ),
                "rule": "both new-seed deltas > 0 and mean >= 5 points; |mean| < 2 is only the frozen screen band",
            },
            "D3_retention_tradeoff": {
                "per_seed_pts": {
                    mode: {str(seed): value for seed, value in values.items()}
                    for mode, values in retention_deltas.items()
                },
                "tradeoffs_beyond_margin": retention_tradeoffs,
                "verdict": "TRADEOFF" if retention_tradeoffs else "NO_NEW_SEED_TRADEOFF",
            },
        },
        "supportive_transfer": supportive,
        "metrics_reported_per_cell": list(SCALAR_METRICS),
        "broad_macro_denominator": len(presets),
        "not_claimed": preregistration["not_claimed"],
    }
