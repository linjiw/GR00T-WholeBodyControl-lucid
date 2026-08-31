#!/usr/bin/env python3
"""Analyze the frozen Tier-1 ``lucid_ratchet_rg`` comparison.

The analyzer is deliberately small and deterministic.  It unions split
training/evaluation receipts, computes normalized trapezoidal success and
progress AUCs on the frozen physics grids, audits the ratchet trajectory, and
compares it with fixed DR using the frozen noninferiority margins.  It has no
threshold or arm-selection command-line knobs.

One trained policy seed is useful screening evidence, but it is not an
independent estimate of training-procedure variability.  Consequently a
pass/fail noninferiority verdict is emitted only for exactly three paired
training seeds; all smaller comparisons are explicitly ``screening_only``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility.paths import MANIFESTS, relocate  # noqa: E402

RATCHET_MODE = "lucid_ratchet_rg"
FIXED_MODE = "fixed"

IN_ENVELOPE_GRID = (
    ("phys_000", 0.00),
    ("phys_025", 0.25),
    ("phys_050", 0.50),
    ("phys_075", 0.75),
    ("phys_100", 1.00),
)
FRONTIER_GRID = (
    ("phys_125", 1.25),
    ("phys_150", 1.50),
    ("phys_175", 1.75),
    ("phys_200", 2.00),
)
LATENCY_PRESET = "lat_50ms"
METRICS = ("success_rate", "progress_rate")
ALL_PRESETS = tuple(preset for preset, _ in (*IN_ENVELOPE_GRID, *FRONTIER_GRID)) + (
    "lat_10ms",
    "lat_20ms",
    "lat_30ms",
    "lat_40ms",
    LATENCY_PRESET,
)

EXPECTED_NUM_ENVS = 512
EXPECTED_MAX_DELAY = 12
EXPECTED_PHYSICS_STEP_MS = 5
EXPECTED_PANEL_SHA256 = "e2e61933405e6701b0563eb4df793b6faf5c90d8ae5b7d8fc1e11f47142aefd7"
EXPECTED_PANEL_ALIAS_SHA256 = "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430"
EXPECTED_EVALUATOR_SHA256 = "308e24150e4d4f03d0abf0dc6a427063ac662904bb3a7765488a9bff63cd94ca"
SCREENING_EVAL_SEED = {"8601": 8701}

# Frozen program conventions.  Rates, AUCs, and margins are all fractions.
FRONTIER_MARGIN = 0.02
IN_ENVELOPE_MARGIN = 0.01
LATENCY_MARGIN = FRONTIER_MARGIN
EXPECTED_PAIRED_SEEDS = 3
REQUIRED_WITHIN_MARGIN_SEEDS = 2

HIGH_LAMBDA = 0.95
REACH_BY_STEP = 500
TERMINAL_WINDOW = 1000
TERMINAL_MIN_FRACTION = 0.95
FLOAT_TOLERANCE = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"receipt must contain a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command], cwd=REPO, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _put_unique(
    target: dict[tuple[str, str, str, str], float],
    key: tuple[str, str, str, str],
    value: Any,
    source: Path,
) -> None:
    if value is None:
        return
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite robustness value for {key} in {source}: {value}")
    previous = target.get(key)
    if previous is not None and not math.isclose(
        previous, numeric, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
    ):
        raise ValueError(
            f"conflicting robustness values for {key}: {previous} versus {numeric} in {source}"
        )
    target[key] = numeric


def collect_robustness(
    receipt_paths: Iterable[Path],
) -> dict[tuple[str, str, str, str], float]:
    """Collect ``(mode, metric, preset, training_seed) -> rate`` values.

    Current evaluator receipts carry a ``mode_summary``.  The ``runs`` fallback
    keeps the analyzer useful for a receipt interrupted after individual runs
    were written but before its summary was assembled.  Duplicate identical
    cells are harmless; conflicting duplicates are rejected rather than made
    order-dependent.
    """

    values: dict[tuple[str, str, str, str], float] = {}
    for path in receipt_paths:
        receipt = load_json(path)
        summary = receipt.get("mode_summary") or {}
        for preset, modes in summary.items():
            for mode, block in modes.items():
                for metric in METRICS:
                    per_seed = (block.get("metrics") or {}).get(metric, {}).get(
                        "per_checkpoint_seed"
                    ) or {}
                    for seed, value in per_seed.items():
                        _put_unique(
                            values, (str(mode), metric, str(preset), str(seed)), value, path
                        )

        # Do not double-read a normal receipt.  ``runs`` is only a fallback
        # when no aggregate summary survived.
        if summary:
            continue
        for run in (receipt.get("runs") or {}).values():
            if not run.get("complete", True):
                continue
            mode = run.get("mode")
            preset = run.get("preset")
            seed = run.get("checkpoint_seed")
            if mode is None or preset is None or seed is None:
                continue
            run_summary = run.get("summary") or {}
            for metric in METRICS:
                _put_unique(
                    values,
                    (str(mode), metric, str(preset), str(seed)),
                    run_summary.get(metric),
                    path,
                )
    return values


def audit_instrument(receipt_paths: Iterable[Path]) -> dict[str, Any]:
    """Fail closed unless receipts implement the frozen matched instrument."""

    errors: list[str] = []
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    receipt_records = []
    expected_presets = set(ALL_PRESETS)
    panel_hashes: set[str] = set()
    launcher_hashes: set[str] = set()

    for path in receipt_paths:
        receipt = load_json(path)
        verified = receipt.get("verified")
        if not isinstance(verified, list) or not verified:
            errors.append(f"{path}: evaluator receipt is not verified")

        launcher = str(receipt.get("launcher_sha256") or "")
        launcher_hashes.add(launcher)
        if launcher != EXPECTED_EVALUATOR_SHA256:
            errors.append(f"{path}: unexpected evaluator hash {launcher!r}")

        protocol = receipt.get("protocol") or {}
        if protocol.get("num_envs") != EXPECTED_NUM_ENVS:
            errors.append(f"{path}: num_envs is not {EXPECTED_NUM_ENVS}")
        if protocol.get("max_delay_capacity_steps") != EXPECTED_MAX_DELAY:
            errors.append(f"{path}: max-delay capacity is not {EXPECTED_MAX_DELAY}")
        if protocol.get("physics_step_ms") != EXPECTED_PHYSICS_STEP_MS:
            errors.append(f"{path}: physics step is not {EXPECTED_PHYSICS_STEP_MS} ms")
        if protocol.get("no_learning") is not True:
            errors.append(f"{path}: evaluation is not marked no-learning")

        suite = protocol.get("suite") or {}
        replicate = suite.get("replicate_panel") or {}
        panel_path_text = replicate.get("receipt")
        if (
            suite.get("motion_count") != EXPECTED_NUM_ENVS
            or replicate.get("replicates") != EXPECTED_NUM_ENVS
        ):
            errors.append(f"{path}: replicate panel is not the frozen 512-alias suite")
        if replicate.get("alias_keys_sha256") != EXPECTED_PANEL_ALIAS_SHA256:
            errors.append(f"{path}: replicate-panel alias hash differs")
        if not panel_path_text:
            errors.append(f"{path}: replicate-panel receipt path is missing")
        else:
            panel_path = Path(panel_path_text)
            if not panel_path.is_file():
                errors.append(f"{path}: replicate-panel receipt is missing: {panel_path}")
            else:
                panel_hash = sha256(panel_path)
                panel_hashes.add(panel_hash)
                if panel_hash != EXPECTED_PANEL_SHA256:
                    errors.append(f"{path}: replicate-panel receipt hash differs")

        before = receipt.get("checkpoint_sha256_before") or {}
        after = receipt.get("checkpoint_sha256_after") or {}
        if not before or before != after:
            errors.append(f"{path}: checkpoint hashes are missing or changed during evaluation")
        checkpoint_hashes = set(before.values())

        runs = receipt.get("runs") or {}
        if not runs:
            errors.append(f"{path}: no evaluation runs recorded")
        for branch_id, run in runs.items():
            mode = str(run.get("mode") or "")
            seed = str(run.get("checkpoint_seed"))
            preset = str(run.get("preset") or "")
            key = (mode, seed, preset)
            if mode not in (RATCHET_MODE, FIXED_MODE):
                errors.append(f"{path}: unexpected mode {mode!r}")
            if preset not in expected_presets:
                errors.append(f"{path}: unexpected preset {preset!r}")
            if run.get("complete") is not True or (run.get("runtime") or {}).get("exit_code") != 0:
                errors.append(f"{path}: incomplete cell {branch_id}")
            if run.get("checkpoint_sha256") not in checkpoint_hashes:
                errors.append(f"{path}: cell {branch_id} has an unpinned checkpoint hash")
            if key in cells:
                errors.append(f"duplicate evaluation cell {key} in {path}")
            cells[key] = {
                "evaluation_seed": run.get("evaluation_seed"),
                "receipt": str(path),
            }

        receipt_records.append(
            {
                "path": str(path),
                "launcher_sha256": launcher,
                "run_count": len(runs),
                "verified": bool(verified),
            }
        )

    seeds_by_mode = {
        mode: {seed for candidate_mode, seed, _ in cells if candidate_mode == mode}
        for mode in (RATCHET_MODE, FIXED_MODE)
    }
    if seeds_by_mode[RATCHET_MODE] != seeds_by_mode[FIXED_MODE]:
        errors.append(f"ratchet/fixed training seeds are not paired: {seeds_by_mode}")

    for mode, seeds in seeds_by_mode.items():
        for seed in sorted(seeds):
            observed = {
                preset
                for candidate_mode, candidate_seed, preset in cells
                if candidate_mode == mode and candidate_seed == seed
            }
            if observed != expected_presets:
                errors.append(
                    f"mode={mode} seed={seed} preset set differs: "
                    f"missing={sorted(expected_presets - observed)} extra={sorted(observed - expected_presets)}"
                )

    evaluation_seed_by_training_seed: dict[str, int] = {}
    for seed in sorted(seeds_by_mode[RATCHET_MODE] & seeds_by_mode[FIXED_MODE]):
        eval_seeds = {
            int(cell["evaluation_seed"])
            for (mode, candidate_seed, _), cell in cells.items()
            if candidate_seed == seed and mode in (RATCHET_MODE, FIXED_MODE)
        }
        if len(eval_seeds) != 1:
            errors.append(
                f"training seed {seed} does not have one matched evaluation seed: {eval_seeds}"
            )
            continue
        evaluation_seed = next(iter(eval_seeds))
        evaluation_seed_by_training_seed[seed] = evaluation_seed
        frozen_screening_seed = SCREENING_EVAL_SEED.get(seed)
        if frozen_screening_seed is not None and evaluation_seed != frozen_screening_seed:
            errors.append(
                f"training seed {seed} used eval seed {evaluation_seed}, expected {frozen_screening_seed}"
            )

    audit = {
        "passed": not errors,
        "errors": errors,
        "receipts": receipt_records,
        "expected_presets": list(ALL_PRESETS),
        "cell_count": len(cells),
        "paired_training_seeds": sorted(seeds_by_mode[RATCHET_MODE] & seeds_by_mode[FIXED_MODE]),
        "evaluation_seed_by_training_seed": evaluation_seed_by_training_seed,
        "launcher_sha256": sorted(launcher_hashes),
        "panel_sha256": sorted(panel_hashes),
    }
    if errors:
        raise ValueError("frozen instrument audit failed:\n- " + "\n- ".join(errors))
    return audit


def trapezoid_weights(grid: Sequence[tuple[str, float]]) -> dict[str, float]:
    if len(grid) < 2:
        raise ValueError("an AUC grid needs at least two points")
    xs = [float(x) for _, x in grid]
    if any(right <= left for left, right in zip(xs, xs[1:])):
        raise ValueError(f"AUC grid is not strictly increasing: {grid}")
    width = xs[-1] - xs[0]
    weights = []
    for index in range(len(xs)):
        left = 0.0 if index == 0 else (xs[index] - xs[index - 1]) / 2.0
        right = 0.0 if index == len(xs) - 1 else (xs[index + 1] - xs[index]) / 2.0
        weights.append((left + right) / width)
    return {preset: weight for (preset, _), weight in zip(grid, weights, strict=True)}


def profile(
    values: dict[tuple[str, str, str, str], float],
    mode: str,
    metric: str,
    grid: Sequence[tuple[str, float]],
) -> dict[str, Any]:
    """Return complete per-seed normalized trapezoidal AUCs.

    A seed with a missing cell is reported but never imputed.  Because the
    weights sum to one, each AUC stays on the input rate scale ``[0, 1]``.
    """

    weights = trapezoid_weights(grid)
    presets = [preset for preset, _ in grid]
    candidate_seeds = sorted(
        {
            seed
            for candidate_mode, candidate_metric, preset, seed in values
            if candidate_mode == mode and candidate_metric == metric and preset in presets
        }
    )
    per_seed: dict[str, dict[str, Any]] = {}
    incomplete: dict[str, list[str]] = {}
    for seed in candidate_seeds:
        cells = {
            preset: values[(mode, metric, preset, seed)]
            for preset in presets
            if (mode, metric, preset, seed) in values
        }
        missing = [preset for preset in presets if preset not in cells]
        if missing:
            incomplete[seed] = missing
            continue
        per_seed[seed] = {
            "auc": sum(weights[preset] * cells[preset] for preset in presets),
            "cells": cells,
        }
    aucs = [block["auc"] for block in per_seed.values()]
    return {
        "grid": {preset: intensity for preset, intensity in grid},
        "weights": weights,
        "per_seed": per_seed,
        "mean_auc": statistics.fmean(aucs) if aucs else None,
        "complete_seeds": list(per_seed),
        "incomplete_seeds": incomplete,
    }


def single_cell(
    values: dict[tuple[str, str, str, str], float], mode: str, metric: str, preset: str
) -> dict[str, Any]:
    per_seed = {
        seed: value
        for (candidate_mode, candidate_metric, candidate_preset, seed), value in values.items()
        if candidate_mode == mode and candidate_metric == metric and candidate_preset == preset
    }
    ordered = dict(sorted(per_seed.items()))
    return {
        "preset": preset,
        "per_seed": ordered,
        "mean": statistics.fmean(ordered.values()) if ordered else None,
    }


def _endpoint_values(endpoint: dict[str, Any]) -> dict[str, float]:
    if "mean_auc" in endpoint:
        return {seed: float(block["auc"]) for seed, block in endpoint["per_seed"].items()}
    return {seed: float(value) for seed, value in endpoint["per_seed"].items()}


def noninferiority(
    ratchet_endpoint: dict[str, Any], fixed_endpoint: dict[str, Any], margin: float
) -> dict[str, Any]:
    """Paired fixed-margin comparison at the training-seed level."""

    ratchet = _endpoint_values(ratchet_endpoint)
    fixed = _endpoint_values(fixed_endpoint)
    common = sorted(set(ratchet) & set(fixed))
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in common:
        delta = ratchet[seed] - fixed[seed]
        per_seed[seed] = {
            "ratchet": ratchet[seed],
            "fixed": fixed[seed],
            "delta": delta,
            "delta_pts": 100.0 * delta,
            "within_noninferiority_margin": delta >= -margin - FLOAT_TOLERANCE,
            "strictly_favorable": delta > FLOAT_TOLERANCE,
        }
    deltas = [block["delta"] for block in per_seed.values()]
    within = sum(block["within_noninferiority_margin"] for block in per_seed.values())
    favorable = sum(block["strictly_favorable"] for block in per_seed.values())
    if not common:
        verdict = "not_evaluable"
    elif len(common) != EXPECTED_PAIRED_SEEDS:
        verdict = "screening_only"
    else:
        verdict = "pass" if within >= REQUIRED_WITHIN_MARGIN_SEEDS else "fail"
    return {
        "margin": margin,
        "margin_pts": 100.0 * margin,
        "criterion": (
            "with exactly three paired training seeds, at least two must have "
            "ratchet-minus-fixed >= -margin"
        ),
        "per_seed": per_seed,
        "paired_seeds": common,
        "num_paired_seeds": len(common),
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "mean_delta_pts": 100.0 * statistics.fmean(deltas) if deltas else None,
        "within_margin_seeds": within,
        "strictly_favorable_seeds_descriptive": favorable,
        "required_within_margin_seeds": (
            REQUIRED_WITHIN_MARGIN_SEEDS if len(common) == EXPECTED_PAIRED_SEEDS else None
        ),
        "verdict": verdict,
    }


def joint_noninferiority(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Show which seeds satisfy every named component margin.

    This keeps the joint interpretation explicit while preserving each
    component comparison above.  Strict superiority is never part of the rule.
    """

    if not comparisons:
        return {"verdict": "not_evaluable", "per_seed": {}, "num_paired_seeds": 0}
    seed_sets = [set(block["per_seed"]) for block in comparisons.values()]
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    per_seed = {
        seed: {
            "within_all_margins": all(
                block["per_seed"][seed]["within_noninferiority_margin"]
                for block in comparisons.values()
            ),
            "components": {
                name: block["per_seed"][seed]["within_noninferiority_margin"]
                for name, block in comparisons.items()
            },
        }
        for seed in common
    }
    within = sum(block["within_all_margins"] for block in per_seed.values())
    if not common:
        verdict = "not_evaluable"
    elif len(common) != EXPECTED_PAIRED_SEEDS:
        verdict = "screening_only"
    else:
        verdict = "pass" if within >= REQUIRED_WITHIN_MARGIN_SEEDS else "fail"
    return {
        "criterion": (
            "with exactly three paired training seeds, at least two must satisfy every "
            "named component's frozen noninferiority margin"
        ),
        "components": list(comparisons),
        "per_seed": per_seed,
        "paired_seeds": common,
        "num_paired_seeds": len(common),
        "within_all_margin_seeds": within,
        "verdict": verdict,
    }


def _curriculum_rows(
    arm: dict[str, Any], receipt_path: Path
) -> tuple[list[dict[str, Any]], Path | None]:
    embedded = arm.get("curriculum")
    if isinstance(embedded, list):
        return [row for row in embedded if isinstance(row, dict)], None
    recorded = arm.get("curriculum_path")
    if not recorded:
        return [], None
    path = relocate(recorded)
    if not path.is_absolute():
        path = receipt_path.parent / path
    if not path.is_file():
        return [], path
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"curriculum row {line_number} is not an object: {path}")
        rows.append(value)
    return rows, path


def _gate(value: bool | None) -> str:
    if value is None:
        return "not_evaluable"
    return "pass" if value else "fail"


def mechanism_for_arm(arm: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    rows, curriculum_path = _curriculum_rows(arm, receipt_path)
    ordered = sorted(
        enumerate(rows),
        key=lambda pair: (
            float(pair[1].get("global_step", math.inf)),
            pair[0],
        ),
    )
    rows = [row for _, row in ordered]

    lambda_rows: list[tuple[int, float, dict[str, Any]]] = []
    for row in rows:
        step = row.get("global_step")
        value = row.get("lambda", row.get("lambda_after"))
        if step is None or value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        lambda_rows.append((int(step), numeric, row))

    reached = next(
        ((step, value) for step, value, _ in lambda_rows if value >= HIGH_LAMBDA - FLOAT_TOLERANCE),
        None,
    )
    max_step = max((step for step, _, _ in lambda_rows), default=None)
    if reached is not None:
        reach_pass: bool | None = reached[0] <= REACH_BY_STEP
    elif max_step is not None and max_step >= REACH_BY_STEP:
        reach_pass = False
    else:
        reach_pass = None

    transitions = [
        row
        for row in rows
        if row.get("lambda_before") is not None and row.get("lambda_after") is not None
    ]
    decreases = [
        row
        for row in transitions
        if float(row["lambda_after"]) < float(row["lambda_before"]) - FLOAT_TOLERANCE
    ]
    unguarded_decreases = [row for row in decreases if not bool(row.get("guard_tripped"))]
    blocked = [row for row in transitions if bool(row.get("latch_active"))]
    blocked_hold_violations = [
        row
        for row in blocked
        if not math.isclose(
            float(row["lambda_before"]),
            float(row["lambda_after"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    ]
    guard_trips = sum(bool(row.get("guard_tripped")) for row in rows)
    decrease_invariant = (
        not unguarded_decreases and not blocked_hold_violations if transitions else None
    )

    terminal: dict[str, Any]
    if max_step is None:
        terminal = {
            "requested_iterations": TERMINAL_WINDOW,
            "observed_iterations": 0,
            "high_lambda_iterations": 0,
            "high_lambda_fraction": None,
            "contiguous": False,
            "gate": "not_evaluable",
        }
    else:
        lower = max_step - TERMINAL_WINDOW + 1
        terminal_rows = [(step, value) for step, value, _ in lambda_rows if step >= lower]
        terminal_steps = [step for step, _ in terminal_rows]
        contiguous = (
            len(terminal_steps) == TERMINAL_WINDOW
            and len(set(terminal_steps)) == TERMINAL_WINDOW
            and min(terminal_steps) == lower
            and max(terminal_steps) == max_step
        )
        high = sum(value >= HIGH_LAMBDA - FLOAT_TOLERANCE for _, value in terminal_rows)
        terminal_pass = (
            high / TERMINAL_WINDOW >= TERMINAL_MIN_FRACTION - FLOAT_TOLERANCE
            if contiguous
            else None
        )
        terminal = {
            "start_step": lower,
            "end_step": max_step,
            "requested_iterations": TERMINAL_WINDOW,
            "observed_iterations": len(terminal_rows),
            "high_lambda_threshold": HIGH_LAMBDA,
            "minimum_high_lambda_fraction": TERMINAL_MIN_FRACTION,
            "high_lambda_iterations": high,
            "high_lambda_fraction": high / len(terminal_rows) if terminal_rows else None,
            "contiguous": contiguous,
            "gate": _gate(terminal_pass),
        }

    configured_monotonic = (arm.get("arm_spec") or {}).get("monotonic")
    receipt_bind_rows = arm.get("ratchet_bind_rows")
    summary_consistent = (
        int(receipt_bind_rows) == len(blocked) if receipt_bind_rows is not None else None
    )
    source = {
        "training_receipt": str(receipt_path),
        "curriculum_path": str(curriculum_path) if curriculum_path is not None else None,
        "curriculum_sha256": (
            sha256(curriculum_path)
            if curriculum_path is not None and curriculum_path.is_file()
            else None
        ),
    }
    return {
        "source": source,
        "rows": len(rows),
        "lambda_rows": len(lambda_rows),
        "configured_monotonic": configured_monotonic,
        "configuration_gate": _gate(
            configured_monotonic if configured_monotonic is not None else None
        ),
        "reach_lambda_095_by_step_500": {
            "threshold": HIGH_LAMBDA,
            "deadline_step": REACH_BY_STEP,
            "first_reach_step": reached[0] if reached is not None else None,
            "gate": _gate(reach_pass),
        },
        "pi_decrease_control": {
            "transition_rows": len(transitions),
            "blocked_pi_decrease_rows": len(blocked),
            "blocking_observed": bool(blocked),
            "actual_decrease_rows": len(decreases),
            "guard_trip_rows": guard_trips,
            "unguarded_decrease_rows": len(unguarded_decreases),
            "blocked_hold_violations": len(blocked_hold_violations),
            "guard_is_only_legal_decrease_gate": _gate(decrease_invariant),
            "receipt_ratchet_bind_rows": receipt_bind_rows,
            "receipt_bind_count_consistent": summary_consistent,
        },
        "terminal_1000_high_lambda_exposure": terminal,
    }


def collect_mechanisms(training_paths: Iterable[Path]) -> tuple[dict[str, Any], list[str]]:
    by_seed: dict[str, Any] = {}
    ignored_modes: set[str] = set()
    for path in training_paths:
        receipt = load_json(path)
        for arm in (receipt.get("arms") or {}).values():
            mode = str(arm.get("mode", ""))
            if mode != RATCHET_MODE:
                if mode:
                    ignored_modes.add(mode)
                continue
            seed = arm.get("seed")
            if seed is None:
                raise ValueError(f"ratchet arm without a seed in {path}")
            key = str(seed)
            if key in by_seed:
                raise ValueError(f"duplicate ratchet training arm for seed {key}")
            by_seed[key] = mechanism_for_arm(arm, path)
    return dict(sorted(by_seed.items())), sorted(ignored_modes)


def mechanism_summary(per_seed: dict[str, Any]) -> dict[str, Any]:
    gate_paths = (
        ("configuration_gate",),
        ("reach_lambda_095_by_step_500", "gate"),
        ("pi_decrease_control", "guard_is_only_legal_decrease_gate"),
        ("terminal_1000_high_lambda_exposure", "gate"),
    )

    def read(block: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = block
        for part in path:
            value = value.get(part) if isinstance(value, dict) else None
        return value

    per_seed_pass = {
        seed: all(read(block, path) == "pass" for path in gate_paths)
        for seed, block in per_seed.items()
    }
    return {
        "per_seed_all_gates_pass": per_seed_pass,
        "all_available_seeds_pass": all(per_seed_pass.values()) if per_seed_pass else None,
        "blocking_observed_seeds": [
            seed
            for seed, block in per_seed.items()
            if block["pi_decrease_control"]["blocking_observed"]
        ],
        "interpretation": (
            "A zero blocked-row count means the PI law did not request a logged decrease; "
            "it is not treated as a mechanism failure. Any actual lambda decrease must be "
            "guard-tripped, and at least 95% of the final 1000 logged iterations must be "
            "at lambda>=0.95."
        ),
    }


def preregistered_decision(
    comparisons: dict[str, dict[str, Any]],
    mechanism: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Apply H_R0/H_R1 or H_R0/H_R2 without promoting the latency secondary.

    The seed-8601 result is a targeted screen.  Its pass/fail result only
    controls whether confirmatory training is warranted; it never authorizes a
    directional claim.  At three seeds, H_R2 is component-wise 2-of-3 for the
    four preregistered AUC endpoints.  The latency cell remains secondary.
    """

    components = {
        f"{metric}:{endpoint}": comparisons[metric][endpoint]
        for metric in METRICS
        for endpoint in ("frontier_auc", "in_envelope_auc")
    }
    paired = scope["paired_training_seeds"]
    mechanism_by_seed = mechanism_summary(mechanism)["per_seed_all_gates_pass"]
    mechanism_complete = bool(paired) and all(seed in mechanism_by_seed for seed in paired)
    mechanism_pass = mechanism_complete and all(mechanism_by_seed[seed] for seed in paired)

    if len(paired) in (1, EXPECTED_PAIRED_SEEDS) and not mechanism_complete:
        component_pass = None
        status = "not_evaluable"
        interpretation = (
            "A paired capability result is present, but a ratchet mechanism receipt is missing."
        )
    elif len(paired) == 1:
        seed = paired[0]
        component_pass = all(
            block["per_seed"][seed]["within_noninferiority_margin"] for block in components.values()
        )
        status = "screen_pass" if mechanism_pass and component_pass else "screen_fail"
        interpretation = (
            "Targeted one-seed continuation gate only; no directional or three-seed "
            "noninferiority claim is authorized."
        )
    elif len(paired) == EXPECTED_PAIRED_SEEDS:
        component_pass = all(block["verdict"] == "pass" for block in components.values())
        status = "pass" if mechanism_pass and component_pass else "fail"
        interpretation = (
            "Three-seed H_R2 decision: each preregistered AUC component must pass its "
            "own frozen 2-of-3 margin and every ratchet mechanism gate must pass."
        )
    else:
        component_pass = None
        status = "not_evaluable"
        interpretation = (
            "The frozen design requires one screening seed or exactly three paired seeds."
        )

    return {
        "status": status,
        "paired_training_seeds": paired,
        "mechanism_complete": mechanism_complete,
        "mechanism_pass": mechanism_pass if paired else None,
        "capability_components": list(components),
        "capability_components_pass": component_pass,
        "lat_50ms_is_secondary": True,
        "directional_claim_authorized": scope["directional_claim_authorized"],
        "interpretation": interpretation,
    }


def claim_scope(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = comparisons["success_rate"]["frontier_auc"]
    count = primary["num_paired_seeds"]
    if count == EXPECTED_PAIRED_SEEDS:
        status = "three_seed_decision"
        statement = (
            "Exactly three paired training seeds are available; the frozen 2-of-3 "
            "noninferiority rule is decision-eligible."
        )
    elif count == 1:
        status = "screening_only"
        statement = (
            "Only one paired training seed is available. Results are screening point "
            "estimates confounded with training seed and support no directional claim."
        )
    elif count == 0:
        status = "not_evaluable"
        statement = "No paired ratchet/fixed training seed has a complete frontier profile."
    else:
        status = "screening_only"
        statement = (
            f"{count} paired training seeds are available, not the frozen three-seed design; "
            "results are descriptive only."
        )
    return {
        "status": status,
        "paired_training_seeds": primary["paired_seeds"],
        "num_paired_training_seeds": count,
        "directional_claim_authorized": count == EXPECTED_PAIRED_SEEDS,
        "statement": statement,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-receipt",
        "--robustness-receipts",
        dest="robustness_receipts",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help="one or more frozen robustness receipts; the option may be repeated",
    )
    parser.add_argument(
        "--training-receipt",
        "--training-receipts",
        dest="training_receipts",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help="one or more training receipts; the option may be repeated",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="analysis receipt path (default: LUCID manifests directory)",
    )
    return parser.parse_args(argv)


def analyze(robustness_paths: Sequence[Path], training_paths: Sequence[Path]) -> dict[str, Any]:
    robustness_paths = [Path(path).resolve() for path in robustness_paths]
    training_paths = [Path(path).resolve() for path in training_paths]
    instrument = audit_instrument(robustness_paths)
    values = collect_robustness(robustness_paths)

    arms: dict[str, dict[str, Any]] = {}
    for mode in (RATCHET_MODE, FIXED_MODE):
        arms[mode] = {}
        for metric in METRICS:
            arms[mode][metric] = {
                "in_envelope_auc": profile(values, mode, metric, IN_ENVELOPE_GRID),
                "frontier_auc": profile(values, mode, metric, FRONTIER_GRID),
                LATENCY_PRESET: single_cell(values, mode, metric, LATENCY_PRESET),
            }

    comparisons: dict[str, dict[str, Any]] = {}
    margins = {
        "in_envelope_auc": IN_ENVELOPE_MARGIN,
        "frontier_auc": FRONTIER_MARGIN,
        LATENCY_PRESET: LATENCY_MARGIN,
    }
    for metric in METRICS:
        comparisons[metric] = {
            endpoint: noninferiority(
                arms[RATCHET_MODE][metric][endpoint],
                arms[FIXED_MODE][metric][endpoint],
                margin,
            )
            for endpoint, margin in margins.items()
        }

    success_joint = joint_noninferiority(comparisons["success_rate"])
    progress_joint = joint_noninferiority(comparisons["progress_rate"])
    all_components = {
        f"{metric}:{endpoint}": comparison
        for metric, endpoints in comparisons.items()
        for endpoint, comparison in endpoints.items()
    }
    joint = {
        "success_primary": success_joint,
        "progress_co_primary": progress_joint,
        "success_and_progress": joint_noninferiority(all_components),
    }

    mechanism, ignored_training_modes = collect_mechanisms(training_paths)
    scope = claim_scope(comparisons)
    decision = preregistered_decision(comparisons, mechanism, scope)
    verified = [
        "normalized trapezoidal success/progress AUCs computed without imputation",
        "ratchet-minus-fixed comparisons paired on training seed",
        "ratchet lambda trajectories audited from curriculum JSONL rows",
    ]
    not_yet_verified = []
    if scope["status"] != "three_seed_decision":
        not_yet_verified.append("three-seed noninferiority verdict")
    if not mechanism:
        not_yet_verified.append("ratchet mechanism trajectory (no ratchet arm found)")

    input_records = {
        "robustness_receipts": [
            {"path": str(path), "sha256": sha256(path)} for path in robustness_paths
        ],
        "training_receipts": [
            {"path": str(path), "sha256": sha256(path)} for path in training_paths
        ],
    }
    return {
        "kind": "lucid_ratchet_analysis",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": _git(("rev-parse", "HEAD")),
        "git_status_short": _git(("status", "--short")),
        "inputs": input_records,
        "instrument_audit": instrument,
        "frozen_contract": {
            "ratchet_mode": RATCHET_MODE,
            "fixed_mode": FIXED_MODE,
            "in_envelope_grid": dict(IN_ENVELOPE_GRID),
            "frontier_grid": dict(FRONTIER_GRID),
            "latency_cell": LATENCY_PRESET,
            "frontier_margin": FRONTIER_MARGIN,
            "in_envelope_margin": IN_ENVELOPE_MARGIN,
            "latency_margin": LATENCY_MARGIN,
            "expected_paired_seeds": EXPECTED_PAIRED_SEEDS,
            "required_within_margin_seeds": REQUIRED_WITHIN_MARGIN_SEEDS,
            "high_lambda_threshold": HIGH_LAMBDA,
            "reach_by_step": REACH_BY_STEP,
            "terminal_window": TERMINAL_WINDOW,
            "terminal_min_high_lambda_fraction": TERMINAL_MIN_FRACTION,
        },
        "claim_scope": scope,
        "preregistered_decision": decision,
        "arms": arms,
        "ratchet_vs_fixed": comparisons,
        "joint_noninferiority": joint,
        "mechanism": {
            "per_seed": mechanism,
            "summary": mechanism_summary(mechanism),
            "ignored_training_modes": ignored_training_modes,
        },
        "verified": verified,
        "not_yet_verified": not_yet_verified,
    }


def _print_summary(receipt: dict[str, Any], out: Path) -> None:
    scope = receipt["claim_scope"]
    print(scope["statement"])
    print(f"{'metric':<16}{'endpoint':<22}{'delta pts':>12}{'within':>10}{'verdict':>16}")
    for metric, endpoints in receipt["ratchet_vs_fixed"].items():
        for endpoint, block in endpoints.items():
            delta = block["mean_delta_pts"]
            delta_text = "n/a" if delta is None else f"{delta:.3f}"
            within = f"{block['within_margin_seeds']}/{block['num_paired_seeds']}"
            print(f"{metric:<16}{endpoint:<22}{delta_text:>12}{within:>10}{block['verdict']:>16}")
    print(f"analysis receipt {out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = analyze(args.robustness_receipts, args.training_receipts)
    out = args.out or MANIFESTS / f"lucid_ratchet_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _print_summary(receipt, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
