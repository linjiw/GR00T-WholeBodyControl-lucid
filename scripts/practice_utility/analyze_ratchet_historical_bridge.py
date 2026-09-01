#!/usr/bin/env python3
"""Build the strict post-H_R2 historical ``lucid_rg`` comparison.

This analyzer is intentionally retrospective and nonbinding.  It does not
re-open, amend, or reinterpret H_R2.  Instead, it requires a terminal H_R2
analysis, revalidates that analysis from its SHA-pinned inputs, and then adds
exactly three historical ``lucid_rg`` checkpoints scored with the same frozen
14-cell instrument.  Every result emitted here is descriptive: there are no
margins, p-values, superiority rules, or pass/fail outcomes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility.paths import MANIFESTS, relocate  # noqa: E402
from scripts.practice_utility import analyze_ratchet as ratchet  # noqa: E402

LUCID_MODE = "lucid_rg"
RATCHET_MODE = ratchet.RATCHET_MODE
FIXED_MODE = ratchet.FIXED_MODE
MODES = (LUCID_MODE, RATCHET_MODE, FIXED_MODE)
SEEDS = ("8600", "8601", "8602")
EXPECTED_RECEIPTS_PER_MODE = 3
EXPECTED_CELLS_PER_MODE = len(SEEDS) * len(ratchet.ALL_PRESETS)
EXPECTED_TOTAL_CELLS = len(MODES) * EXPECTED_CELLS_PER_MODE

LEGACY_FRONTIER_GRID = (
    ("phys_100", 1.00),
    ("phys_125", 1.25),
    ("phys_150", 1.50),
    ("phys_175", 1.75),
    ("phys_200", 2.00),
)

# These identify the only historical 1024-env, 8000-iteration ``lucid_rg``
# cells admitted to this bridge.  They deliberately exclude the ne128
# fine-tuning generation and the invalid artifact-side seed-8600 config.
EXPECTED_LUCID_CHECKPOINT_SHA256 = {
    "8600": "95aadf780c6bdf90e3d78e90b7ef14ee8a3b03a8362e776f39ea1408dc71fd2a",
    "8601": "e8ece9de91b5d73ea7ef920cc27047068ee1a25ea804d8c7001cf603fb31d70e",
    "8602": "aced3185ca7804d39e67d6223dd47f033808ea449500c1690b8f5d8f41613bf3",
}
EXPECTED_LUCID_CONFIG_SHA256 = {
    "8600": "4c0b49de050a4c09b687e339cdbed11e4f2a5a3b2130edd3e08649681ce369ff",
    "8601": "9997fe633cf33c319314a8fb28f239c8d70a15e9470209b828f1e591abce3568",
    "8602": "a3cd711fd0456fad745dc9a6b732a38461d63489818f4b5a22c754e9cfb9efb9",
}
EXPECTED_LUCID_CURRICULUM_SHA256 = {
    "8600": "e37dbdd0da02b42c81dac055d1f41e1a11911a84d062e6be11baeacd092413aa",
    "8601": "3e98983a34b8896fd45a8a72d032ad22048c4f517a7135f25018b0579b0b6e0d",
    "8602": "27d861498121a4b879d6cc47b1016f50e321bcd93db4a5458761e59a603d0537",
}
EXPECTED_H_R2_AMENDMENT_SHA256 = "2064bf7a16ca159092c6ebeabfbf09bc2fe3c1b30ce359a64505503a83786044"
PREDECLARED_COLLAPSED_SEED = {"8600": False, "8601": True, "8602": False}

EXPECTED_TRAINING_ITERATIONS = 8000
EXPECTED_WARMUP_ITERATIONS = 10
HIGH_LAMBDA = ratchet.HIGH_LAMBDA
TERMINAL_WINDOW = ratchet.TERMINAL_WINDOW
FLOAT_TOLERANCE = ratchet.FLOAT_TOLERANCE


def load_json(path: Path) -> dict[str, Any]:
    return ratchet.load_json(path)


def sha256(path: Path) -> str:
    return ratchet.sha256(path)


def _resolve_recorded_path(recorded: Any, relative_to: Path) -> Path:
    if not recorded:
        raise ValueError(f"missing recorded path in {relative_to}")
    path = relocate(str(recorded))
    if not path.is_absolute():
        path = relative_to.parent / path
    return path.resolve()


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash differs: {observed} != {expected} ({path})")


def _input_records(analysis: dict[str, Any], key: str) -> list[dict[str, str]]:
    records = (analysis.get("inputs") or {}).get(key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"H_R2 analysis has no {key} input records")
    normalized = []
    for record in records:
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"malformed H_R2 {key} input record: {record!r}")
        path = Path(record["path"]).resolve()
        expected = str(record["sha256"])
        _require_file_hash(path, expected, f"H_R2 {key} input")
        normalized.append({"path": str(path), "sha256": expected})
    return normalized


def audit_terminal_h_r2(path: Path) -> tuple[dict[str, Any], list[Path], list[Path], str]:
    """Require a terminal, reproducible three-seed H_R2 analysis."""

    path = path.resolve()
    before = sha256(path)
    analysis = load_json(path)
    errors: list[str] = []
    instrument = analysis.get("instrument_audit") or {}
    scope = analysis.get("claim_scope") or {}
    decision = analysis.get("preregistered_decision") or {}

    if analysis.get("kind") != "lucid_ratchet_analysis":
        errors.append("kind is not lucid_ratchet_analysis")
    if instrument.get("passed") is not True or instrument.get("cell_count") != 84:
        errors.append("instrument audit is not the complete 84-cell H_R2 instrument")
    if instrument.get("paired_training_seeds") != list(SEEDS):
        errors.append("H_R2 instrument does not contain exactly seeds 8600/8601/8602")
    if scope.get("status") != "three_seed_decision":
        errors.append("H_R2 claim scope is not terminal three_seed_decision")
    if scope.get("paired_training_seeds") != list(SEEDS):
        errors.append("H_R2 claim scope seed set differs")
    if scope.get("noninferiority_decision_eligible") is not True:
        errors.append("H_R2 is not marked decision-eligible")
    if scope.get("superiority_claim_authorized") is not False:
        errors.append("H_R2 improperly authorizes superiority")
    if decision.get("status") not in ("pass", "fail"):
        errors.append("H_R2 decision is not terminal pass/fail")
    if decision.get("paired_training_seeds") != list(SEEDS):
        errors.append("H_R2 decision seed set differs")
    if decision.get("noninferiority_decision_eligible") is not True:
        errors.append("H_R2 decision is not marked eligible")
    if decision.get("superiority_claim_authorized") is not False:
        errors.append("H_R2 decision improperly authorizes superiority")
    if decision.get("mechanism_complete") is not True:
        errors.append("H_R2 mechanism telemetry is incomplete")
    if decision.get("mechanism_pass") is not True:
        errors.append("H_R2 H_R0 mechanism gates did not all pass")
    mechanism_summary = (analysis.get("mechanism") or {}).get("summary") or {}
    if mechanism_summary.get("all_available_seeds_pass") is not True:
        errors.append("H_R2 mechanism summary did not pass for every seed")
    if errors:
        raise ValueError("terminal H_R2 audit failed:\n- " + "\n- ".join(errors))

    robustness_records = _input_records(analysis, "robustness_receipts")
    training_records = _input_records(analysis, "training_receipts")
    robustness_paths = [Path(record["path"]) for record in robustness_records]
    training_paths = [Path(record["path"]) for record in training_records]
    if len(robustness_paths) != 6:
        raise ValueError(
            f"H_R2 must bind exactly six evaluation receipts, got {len(robustness_paths)}"
        )
    if len(training_paths) != 3:
        raise ValueError(
            f"H_R2 must bind exactly three ratchet training receipts, got {len(training_paths)}"
        )

    # Replay the existing analyzer and compare only scientific fields.  Volatile
    # timestamps and repository-status metadata are intentionally excluded.
    replay = ratchet.analyze(robustness_paths, training_paths)
    scientific_fields = (
        "instrument_audit",
        "frozen_contract",
        "claim_scope",
        "preregistered_decision",
        "arms",
        "ratchet_vs_fixed",
        "joint_noninferiority",
        "mechanism",
        "verified",
        "not_yet_verified",
    )
    mismatches = [field for field in scientific_fields if replay.get(field) != analysis.get(field)]
    if mismatches:
        raise ValueError(f"terminal H_R2 analysis does not replay: {mismatches}")
    if sha256(path) != before:
        raise ValueError("H_R2 analysis changed while it was being audited")
    return analysis, robustness_paths, training_paths, before


def _audit_resolved_config(
    receipt: dict[str, Any], receipt_path: Path, mode: str, seed: str
) -> dict[str, Any]:
    config = (receipt.get("protocol") or {}).get("resolved_training_config")
    if not isinstance(config, dict):
        raise ValueError(f"{receipt_path}: resolved training config is missing")
    recorded_sha = str(config.get("sha256") or "")
    source = _resolve_recorded_path(config.get("source"), receipt_path)
    _require_file_hash(source, recorded_sha, f"{mode} seed {seed} source config")
    installed_values = config.get("installed")
    if not isinstance(installed_values, list) or not installed_values:
        raise ValueError(f"{receipt_path}: resolved training config has no installed copies")
    installed = []
    for value in installed_values:
        candidate = _resolve_recorded_path(value, receipt_path)
        _require_file_hash(candidate, recorded_sha, f"{mode} seed {seed} installed config")
        installed.append(str(candidate))
    return {"source": str(source), "sha256": recorded_sha, "installed": installed}


def _audit_panel_identity(panel_path: Path) -> dict[str, Any]:
    panel = load_json(panel_path)
    required = {
        "kind": "lucid_replicate_panel",
        "schema_version": 1,
        "replicates": ratchet.EXPECTED_NUM_ENVS,
        "alias_keys_sha256": ratchet.EXPECTED_PANEL_ALIAS_SHA256,
    }
    for key, expected in required.items():
        if panel.get(key) != expected:
            raise ValueError(f"replicate-panel {key} differs: {panel.get(key)!r}")
    if not panel.get("verified"):
        raise ValueError("replicate-panel receipt is not verified")
    fields = (
        "motion_key",
        "source_clip_sha256",
        "pool_sha256",
        "split_sha256",
        "partition",
    )
    if any(not panel.get(field) for field in fields):
        raise ValueError("replicate-panel receipt lacks complete pool/split identity")
    motion_file = _resolve_recorded_path(panel.get("motion_file"), panel_path)
    if not motion_file.is_dir():
        raise ValueError(f"replicate-panel motion tree is missing: {motion_file}")
    return {
        "path": str(panel_path),
        "sha256": sha256(panel_path),
        "motion_file": str(motion_file),
        **{field: panel[field] for field in fields},
        "replicates": panel["replicates"],
        "alias_keys_sha256": panel["alias_keys_sha256"],
    }


def audit_exact_instrument(
    h_r2_paths: Sequence[Path], historical_paths: Sequence[Path]
) -> dict[str, Any]:
    """Fail closed unless the union is the exact 3 x 3 x 14 instrument."""

    if len(historical_paths) != EXPECTED_RECEIPTS_PER_MODE:
        raise ValueError(
            "historical bridge requires exactly three lucid_rg evaluation receipts, "
            f"got {len(historical_paths)}"
        )
    if len(h_r2_paths) != 6:
        raise ValueError(
            f"H_R2 bridge requires exactly six immutable receipts, got {len(h_r2_paths)}"
        )
    all_paths = [Path(path).resolve() for path in (*h_r2_paths, *historical_paths)]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("duplicate evaluation receipt path")

    # Preserve the parent analyzer's audited H_R2 semantics before extending
    # its mode allowlist for the descriptive historical arm.
    h_r2_audit = ratchet.audit_instrument(h_r2_paths)
    if h_r2_audit.get("cell_count") != 84:
        raise ValueError("H_R2 receipt union is not exactly 84 cells")

    expected_cells = {
        (mode, seed, preset) for mode in MODES for seed in SEEDS for preset in ratchet.ALL_PRESETS
    }
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
    panel_hashes: set[str] = set()
    panel_identity: dict[str, Any] | None = None
    receipt_records = []

    for path in all_paths:
        receipt = load_json(path)
        if receipt.get("kind") != "lucid_frozen_checkpoint_robustness_evaluation":
            raise ValueError(f"{path}: unexpected robustness receipt kind")
        if receipt.get("schema_version") != 1:
            raise ValueError(f"{path}: unexpected robustness receipt schema")
        if not isinstance(receipt.get("verified"), list) or not receipt["verified"]:
            raise ValueError(f"{path}: evaluator receipt is not verified")
        if receipt.get("launcher_sha256") != ratchet.EXPECTED_EVALUATOR_SHA256:
            raise ValueError(f"{path}: evaluator hash differs")

        protocol = receipt.get("protocol") or {}
        if protocol.get("num_envs") != ratchet.EXPECTED_NUM_ENVS:
            raise ValueError(f"{path}: num_envs differs")
        if protocol.get("max_delay_capacity_steps") != ratchet.EXPECTED_MAX_DELAY:
            raise ValueError(f"{path}: max-delay capacity differs")
        if protocol.get("physics_step_ms") != ratchet.EXPECTED_PHYSICS_STEP_MS:
            raise ValueError(f"{path}: physics step differs")
        if protocol.get("no_learning") is not True:
            raise ValueError(f"{path}: evaluation is not no-learning")
        suite = protocol.get("suite") or {}
        replicate = suite.get("replicate_panel") or {}
        if suite.get("motion_count") != ratchet.EXPECTED_NUM_ENVS:
            raise ValueError(f"{path}: motion count differs")
        if replicate.get("replicates") != ratchet.EXPECTED_NUM_ENVS:
            raise ValueError(f"{path}: replicate count differs")
        if replicate.get("alias_keys_sha256") != ratchet.EXPECTED_PANEL_ALIAS_SHA256:
            raise ValueError(f"{path}: alias-key hash differs")
        panel_path = _resolve_recorded_path(replicate.get("receipt"), path)
        _require_file_hash(panel_path, ratchet.EXPECTED_PANEL_SHA256, "replicate-panel receipt")
        panel_hashes.add(sha256(panel_path))
        candidate_panel = _audit_panel_identity(panel_path)
        if panel_identity is None:
            panel_identity = candidate_panel
        elif candidate_panel != panel_identity:
            raise ValueError(f"{path}: replicate-panel identity differs across receipts")
        suite_motion = _resolve_recorded_path(suite.get("motion_file"), path)
        if str(suite_motion) != candidate_panel["motion_file"]:
            raise ValueError(f"{path}: evaluated motion tree differs from the frozen panel")
        for key in ("pool_sha256", "split_sha256", "partition"):
            if suite.get(key) != candidate_panel[key]:
                raise ValueError(f"{path}: suite {key} differs from the frozen panel")
        if suite.get("motion_keys_sha256") != candidate_panel["alias_keys_sha256"]:
            raise ValueError(f"{path}: suite motion-key hash differs from the frozen panel")
        if suite.get("split_linkage") != "replicate-panel":
            raise ValueError(f"{path}: suite is not linked as a replicate panel")
        for key in (
            "motion_key",
            "source_clip_sha256",
            "replicates",
            "alias_keys_sha256",
        ):
            if replicate.get(key) != candidate_panel[key]:
                raise ValueError(f"{path}: replicate-panel {key} link differs")

        before = receipt.get("checkpoint_sha256_before")
        after = receipt.get("checkpoint_sha256_after")
        if not isinstance(before, dict) or len(before) != 1 or before != after:
            raise ValueError(f"{path}: checkpoint before/after maps differ or are not singular")
        checkpoint_text, checkpoint_sha = next(iter(before.items()))
        checkpoint = _resolve_recorded_path(checkpoint_text, path)
        _require_file_hash(checkpoint, str(checkpoint_sha), "evaluated checkpoint")
        if checkpoint.stat().st_mode & 0o222:
            raise ValueError(f"{path}: evaluated checkpoint is not frozen read-only: {checkpoint}")

        runs = receipt.get("runs")
        if not isinstance(runs, dict) or len(runs) != len(ratchet.ALL_PRESETS):
            raise ValueError(f"{path}: receipt does not contain exactly 14 runs")
        observed_modes = {str(run.get("mode") or "") for run in runs.values()}
        observed_seeds = {str(run.get("checkpoint_seed")) for run in runs.values()}
        if len(observed_modes) != 1 or len(observed_seeds) != 1:
            raise ValueError(f"{path}: each receipt must contain one mode and one training seed")
        mode = next(iter(observed_modes))
        seed = next(iter(observed_seeds))
        if mode not in MODES or seed not in SEEDS:
            raise ValueError(f"{path}: unexpected mode/seed {mode}/{seed}")
        expected_eval_seed = ratchet.EXPECTED_EVALUATION_SEED[seed]
        if set(map(str, protocol.get("modes") or [])) != {mode}:
            raise ValueError(f"{path}: protocol mode differs from runs")
        if {str(value) for value in protocol.get("checkpoint_seeds") or []} != {seed}:
            raise ValueError(f"{path}: protocol training seed differs from runs")
        expected_map = {seed: expected_eval_seed}
        observed_map = {
            str(key): int(value)
            for key, value in (protocol.get("evaluation_seed_by_checkpoint_seed") or {}).items()
        }
        if observed_map != expected_map:
            raise ValueError(f"{path}: evaluation-seed mapping differs")

        config = _audit_resolved_config(receipt, path, mode, seed)
        config_key = (mode, seed)
        if config_key in configs:
            raise ValueError(f"duplicate config provenance for {mode} seed {seed}")
        configs[config_key] = config
        checkpoint_record = {"path": str(checkpoint), "sha256": str(checkpoint_sha)}
        if config_key in checkpoints:
            raise ValueError(f"duplicate checkpoint provenance for {mode} seed {seed}")
        checkpoints[config_key] = checkpoint_record

        run_metric_keys: set[tuple[str, str, str, str]] = set()
        mode_summary = receipt.get("mode_summary") or {}
        for branch, run in runs.items():
            preset = str(run.get("preset") or "")
            key = (mode, seed, preset)
            if preset not in ratchet.ALL_PRESETS:
                raise ValueError(f"{path}: unexpected preset {preset}")
            if key in cells:
                raise ValueError(f"duplicate evaluation cell {key}")
            if run.get("complete") is not True or (run.get("runtime") or {}).get("exit_code") != 0:
                raise ValueError(f"{path}: incomplete cell {branch}")
            if int(run.get("evaluation_seed", -1)) != expected_eval_seed:
                raise ValueError(f"{path}: cell {branch} has the wrong evaluation seed")
            if str(run.get("checkpoint_sha256") or "") != str(checkpoint_sha):
                raise ValueError(f"{path}: cell {branch} has the wrong checkpoint hash")
            if _resolve_recorded_path(run.get("checkpoint"), path) != checkpoint:
                raise ValueError(f"{path}: cell {branch} has the wrong checkpoint path")
            summary = run.get("summary") or {}
            if summary.get("motion_count") != ratchet.EXPECTED_NUM_ENVS:
                raise ValueError(f"{path}: cell {branch} did not score all 512 motions")
            aggregate = ((mode_summary.get(preset) or {}).get(mode) or {}).get("metrics") or {}
            for metric in ratchet.METRICS:
                value = summary.get(metric)
                per_seed = (aggregate.get(metric) or {}).get("per_checkpoint_seed") or {}
                aggregate_value = per_seed.get(seed)
                try:
                    numeric = float(value)
                    aggregate_numeric = float(aggregate_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: missing metric {metric} for {branch}") from error
                if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                    raise ValueError(f"{path}: invalid metric {metric} for {branch}")
                if not math.isclose(
                    numeric, aggregate_numeric, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
                ):
                    raise ValueError(f"{path}: run/aggregate mismatch for {branch} {metric}")
                run_metric_keys.add((mode, seed, preset, metric))
            cells[key] = {
                "receipt": str(path),
                "checkpoint_sha256": str(checkpoint_sha),
                "config_sha256": config["sha256"],
                "evaluation_seed": expected_eval_seed,
            }

        aggregate_metric_keys = {
            (str(candidate_mode), str(candidate_seed), str(preset), metric)
            for preset, modes in mode_summary.items()
            for candidate_mode, block in modes.items()
            for metric in ratchet.METRICS
            for candidate_seed in (
                ((block.get("metrics") or {}).get(metric) or {}).get("per_checkpoint_seed", {})
            )
        }
        if aggregate_metric_keys != run_metric_keys:
            raise ValueError(f"{path}: aggregate/run metric keyspace differs")
        receipt_records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "mode": mode,
                "training_seed": seed,
                "cell_count": len(runs),
            }
        )

    observed_cells = set(cells)
    if observed_cells != expected_cells:
        raise ValueError(
            "exact 126-cell keyspace differs: "
            f"missing={sorted(expected_cells - observed_cells)} "
            f"extra={sorted(observed_cells - expected_cells)}"
        )
    counts = {mode: sum(key[0] == mode for key in cells) for mode in MODES}
    if counts != {mode: EXPECTED_CELLS_PER_MODE for mode in MODES}:
        raise ValueError(f"per-mode cell counts differ: {counts}")
    if panel_hashes != {ratchet.EXPECTED_PANEL_SHA256}:
        raise ValueError(f"more than one panel was used: {sorted(panel_hashes)}")
    return {
        "passed": True,
        "cell_count": len(cells),
        "expected_cell_count": EXPECTED_TOTAL_CELLS,
        "per_mode_cell_count": counts,
        "expected_per_mode_cell_count": EXPECTED_CELLS_PER_MODE,
        "modes": list(MODES),
        "training_seeds": list(SEEDS),
        "presets": list(ratchet.ALL_PRESETS),
        "evaluation_seed_by_training_seed": dict(ratchet.EXPECTED_EVALUATION_SEED),
        "evaluator_sha256": ratchet.EXPECTED_EVALUATOR_SHA256,
        "panel_sha256": ratchet.EXPECTED_PANEL_SHA256,
        "panel_identity": panel_identity,
        "receipts": receipt_records,
        "config_provenance": {
            f"{mode}:s{seed}": block for (mode, seed), block in sorted(configs.items())
        },
        "checkpoint_provenance": {
            f"{mode}:s{seed}": block for (mode, seed), block in sorted(checkpoints.items())
        },
        "h_r2_parent_audit": h_r2_audit,
    }


def _read_curriculum(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank curriculum row {line_number}: {path}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"curriculum row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def trajectory_summary(rows: Sequence[dict[str, Any]], *, expect_ratchet: bool) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAINING_ITERATIONS:
        raise ValueError(f"curriculum must have exactly 8000 rows, got {len(rows)}")
    steps = [row.get("global_step") for row in rows]
    if steps != list(range(1, EXPECTED_TRAINING_ITERATIONS + 1)):
        raise ValueError("curriculum global_step values are not exactly contiguous 1..8000")
    lambdas = []
    for row in rows:
        try:
            value = float(row["lambda"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid lambda at curriculum step {row.get('global_step')}"
            ) from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"lambda outside [0,1] at curriculum step {row['global_step']}")
        lambdas.append(value)
    post_warmup = rows[EXPECTED_WARMUP_ITERATIONS:]
    required = ("lambda_before", "lambda_after", "guard_tripped")
    if any(
        any(field not in row or row[field] is None for field in required) for row in post_warmup
    ):
        raise ValueError("curriculum lacks complete post-warmup transition telemetry")
    if expect_ratchet and any(
        "latch_active" not in row or row["latch_active"] is None for row in post_warmup
    ):
        raise ValueError("ratchet curriculum lacks latch_active telemetry")

    decreases = [
        row
        for row in post_warmup
        if float(row["lambda_after"]) < float(row["lambda_before"]) - FLOAT_TOLERANCE
    ]
    guard_trips = sum(bool(row["guard_tripped"]) for row in post_warmup)
    blocked = sum(bool(row.get("latch_active")) for row in post_warmup)
    first_reach = next(
        (index for index, value in enumerate(lambdas, start=1) if value >= HIGH_LAMBDA), None
    )
    terminal = lambdas[-TERMINAL_WINDOW:]
    terminal_high = sum(value >= HIGH_LAMBDA for value in terminal)
    return {
        "rows": len(rows),
        "final_lambda": lambdas[-1],
        "first_lambda_ge_095_step": first_reach,
        "iterations_at_lambda_ge_095": sum(value >= HIGH_LAMBDA for value in lambdas),
        "actual_decrease_rows": len(decreases),
        "unguarded_decrease_rows": sum(not bool(row["guard_tripped"]) for row in decreases),
        "guard_trip_rows": guard_trips,
        "blocked_pi_decrease_rows": blocked if expect_ratchet else None,
        "terminal_1000_high_lambda_iterations": terminal_high,
        "terminal_1000_high_lambda_fraction": terminal_high / TERMINAL_WINDOW,
    }


def _bridge_config_path(receipt: dict[str, Any], arm: dict[str, Any], path: Path) -> Path:
    explicit = arm.get("resolved_config") or receipt.get("resolved_config")
    if explicit:
        return _resolve_recorded_path(explicit, path)
    run_dir = (arm.get("arm_spec") or {}).get("run_dir")
    if not run_dir:
        raise ValueError(f"{path}: historical bridge has no true run config path")
    run_path = relocate(str(run_dir))
    if not run_path.is_absolute():
        run_path = REPO / run_path
    return (run_path / "config.yaml").resolve()


def audit_historical_bridges(paths: Sequence[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(paths) != EXPECTED_RECEIPTS_PER_MODE:
        raise ValueError(
            f"exactly three historical training bridges are required, got {len(paths)}"
        )
    records: dict[str, Any] = {}
    mechanisms: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        receipt = load_json(path)
        if receipt.get("kind") != "lucid_historical_training_cell_bridge":
            raise ValueError(f"{path}: unexpected historical bridge kind")
        if receipt.get("schema_version") != 1:
            raise ValueError(f"{path}: unexpected historical bridge schema")
        if not isinstance(receipt.get("verified"), list) or not receipt["verified"]:
            raise ValueError(f"{path}: historical bridge is not verified")
        config = receipt.get("config") or {}
        arms = receipt.get("arms")
        if not isinstance(arms, dict) or len(arms) != 1:
            raise ValueError(f"{path}: historical bridge must contain exactly one arm")
        arm = next(iter(arms.values()))
        seed = str(arm.get("seed"))
        if seed not in SEEDS or arm.get("mode") != LUCID_MODE:
            raise ValueError(f"{path}: unexpected historical mode/seed")
        if seed in records:
            raise ValueError(f"duplicate historical bridge for seed {seed}")
        if (
            config.get("from_scratch") is not True
            or config.get("num_envs") != 1024
            or config.get("iterations") != EXPECTED_TRAINING_ITERATIONS
            or config.get("warmup_iterations") != EXPECTED_WARMUP_ITERATIONS
            or [str(value) for value in config.get("seeds") or []] != [seed]
            or config.get("modes") != [LUCID_MODE]
            or config.get("consolidation_fraction") != 0
            or config.get("max_delay_steps") != 8
        ):
            raise ValueError(f"{path}: historical training configuration differs")
        spec = arm.get("arm_spec") or {}
        if (
            arm.get("complete") is not True
            or arm.get("checkpoint_exported") is not True
            or arm.get("iterations_parsed") != EXPECTED_TRAINING_ITERATIONS
            or arm.get("curriculum_rows") != EXPECTED_TRAINING_ITERATIONS
            or spec.get("curriculum_mode") != "lucid"
            or float(spec.get("anchor_ratio", math.nan)) != 0.0
            or spec.get("spread_strata") != 1
            or spec.get("return_guard") != "relative"
            or spec.get("monotonic") is not False
            or spec.get("allow_extrapolation") is not False
            or spec.get("margin") is not None
        ):
            raise ValueError(f"{path}: historical lucid_rg arm contract differs")

        hashes = receipt.get("sha256") or {}
        checkpoint = _resolve_recorded_path(arm.get("checkpoint"), path)
        curriculum = _resolve_recorded_path(arm.get("curriculum_path"), path)
        true_config = _bridge_config_path(receipt, arm, path)
        expected_checkpoint = EXPECTED_LUCID_CHECKPOINT_SHA256[seed]
        expected_curriculum = EXPECTED_LUCID_CURRICULUM_SHA256[seed]
        expected_config = EXPECTED_LUCID_CONFIG_SHA256[seed]
        if hashes.get("checkpoint") != expected_checkpoint:
            raise ValueError(f"{path}: historical checkpoint pin differs for seed {seed}")
        if hashes.get("curriculum") != expected_curriculum:
            raise ValueError(f"{path}: historical curriculum pin differs for seed {seed}")
        if hashes.get("resolved_config") != expected_config:
            raise ValueError(f"{path}: historical true-config pin differs for seed {seed}")
        _require_file_hash(checkpoint, expected_checkpoint, f"historical seed {seed} checkpoint")
        _require_file_hash(curriculum, expected_curriculum, f"historical seed {seed} curriculum")
        _require_file_hash(true_config, expected_config, f"historical seed {seed} true config")
        if checkpoint.stat().st_mode & 0o222:
            raise ValueError(f"{path}: historical checkpoint is not frozen read-only")

        trajectory = trajectory_summary(_read_curriculum(curriculum), expect_ratchet=False)
        collapsed = PREDECLARED_COLLAPSED_SEED[seed]
        trace_consistent = (
            trajectory["final_lambda"] < HIGH_LAMBDA
            and trajectory["terminal_1000_high_lambda_fraction"] < 0.95
            if collapsed
            else trajectory["final_lambda"] >= HIGH_LAMBDA
            and trajectory["terminal_1000_high_lambda_fraction"] >= 0.95
        )
        if not trace_consistent:
            raise ValueError(
                f"{path}: trajectory contradicts predeclared collapse label for {seed}"
            )
        mechanisms[seed] = {
            **trajectory,
            "predeclared_historical_collapse": collapsed,
            "collapse_label_trace_consistent": True,
        }
        records[seed] = {
            "path": str(path),
            "sha256": sha256(path),
            "checkpoint": {"path": str(checkpoint), "sha256": expected_checkpoint},
            "curriculum": {"path": str(curriculum), "sha256": expected_curriculum},
            "true_config": {"path": str(true_config), "sha256": expected_config},
        }
    if set(records) != set(SEEDS):
        raise ValueError(f"historical bridge seed set differs: {sorted(records)}")
    return dict(sorted(records.items())), dict(sorted(mechanisms.items()))


def _frozen_file_record(
    container: dict[str, Any], key: str, manifest_path: Path
) -> tuple[Path, str]:
    record = container.get(key)
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"{manifest_path}: frozen {key} record is incomplete")
    path = _resolve_recorded_path(record["path"], manifest_path)
    expected = str(record["sha256"])
    _require_file_hash(path, expected, f"frozen {key}")
    return path, expected


def _freeze_source_config_path(
    source: dict[str, Any], arm: dict[str, Any], source_receipt: Path, frozen_config: Path
) -> Path:
    """Resolve a source config across the original and detached confirmation worktrees."""

    candidate = _bridge_config_path(source, arm, source_receipt)
    if candidate.is_file():
        return candidate
    run_dir_text = (arm.get("arm_spec") or {}).get("run_dir")
    if run_dir_text:
        run_dir = Path(str(run_dir_text))
        if not run_dir.is_absolute():
            parts = run_dir.parts
            parent_parts = frozen_config.parent.parts
            if len(parent_parts) >= len(parts) and parent_parts[-len(parts) :] == parts:
                return frozen_config
    return candidate


def audit_h_r2_freeze_manifests(
    paths: Sequence[Path],
    h_r2_training_paths: Sequence[Path],
    instrument: dict[str, Any],
) -> dict[str, Any]:
    """Bind every H_R2 eval policy/config to a frozen training source."""

    if len(paths) != 6:
        raise ValueError(f"exactly six H_R2 freeze manifests are required, got {len(paths)}")
    h_r2_training_records = {
        str(path.resolve()): sha256(path.resolve()) for path in h_r2_training_paths
    }
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    ratchet_source_receipts: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        manifest = load_json(path)
        if manifest.get("kind") != "lucid_frozen_training_checkpoint":
            raise ValueError(f"{path}: unexpected freeze-manifest kind")
        if manifest.get("schema_version") != 1:
            raise ValueError(f"{path}: unexpected freeze-manifest schema")
        mode = str(manifest.get("mode") or "")
        seed = str(manifest.get("seed"))
        key = (mode, seed)
        if mode not in (RATCHET_MODE, FIXED_MODE) or seed not in SEEDS:
            raise ValueError(f"{path}: unexpected frozen mode/seed {mode}/{seed}")
        if key in observed:
            raise ValueError(f"duplicate H_R2 freeze manifest for {mode} seed {seed}")
        if (
            manifest.get("state") != "frozen_for_evaluation"
            or manifest.get("evaluation_only") is not True
            or manifest.get("resume_forbidden") is not True
            or manifest.get("iterations") != EXPECTED_TRAINING_ITERATIONS
            or not manifest.get("verified")
        ):
            raise ValueError(f"{path}: checkpoint is not an immutable 8000-iteration freeze")

        checkpoint, checkpoint_sha = _frozen_file_record(manifest, "checkpoint", path)
        config, config_sha = _frozen_file_record(manifest, "config", path)
        curriculum, curriculum_sha = _frozen_file_record(manifest, "curriculum", path)
        source_receipt, source_receipt_sha = _frozen_file_record(manifest, "training_receipt", path)
        checkpoint_record = manifest["checkpoint"]
        if checkpoint_record.get("read_only") is not True or checkpoint.stat().st_mode & 0o222:
            raise ValueError(f"{path}: frozen checkpoint regained write bits")
        if (manifest.get("curriculum") or {}).get("rows") != EXPECTED_TRAINING_ITERATIONS:
            raise ValueError(f"{path}: frozen curriculum row count differs")

        source = load_json(source_receipt)
        source_config = source.get("config") or {}
        arms = source.get("arms")
        if not isinstance(arms, dict) or len(arms) != 1:
            raise ValueError(f"{path}: frozen source receipt must contain exactly one arm")
        arm = next(iter(arms.values()))
        if str(arm.get("seed")) != seed or arm.get("mode") != mode:
            raise ValueError(f"{path}: frozen source arm mode/seed differs")
        if (
            arm.get("complete") is not True
            or arm.get("checkpoint_exported") is not True
            or arm.get("iterations_parsed") != EXPECTED_TRAINING_ITERATIONS
            or arm.get("curriculum_rows") != EXPECTED_TRAINING_ITERATIONS
            or source_config.get("from_scratch") is not True
            or source_config.get("num_envs") != 1024
            or source_config.get("iterations") != EXPECTED_TRAINING_ITERATIONS
            or source_config.get("warmup_iterations") != EXPECTED_WARMUP_ITERATIONS
            or [str(value) for value in source_config.get("seeds") or []] != [seed]
            or source_config.get("modes") != [mode]
            or source_config.get("consolidation_fraction") != 0
            or source_config.get("max_delay_steps") != 8
            or not source.get("verified")
        ):
            raise ValueError(f"{path}: frozen source training contract differs")
        source_checkpoint = _resolve_recorded_path(arm.get("checkpoint"), source_receipt)
        source_curriculum = _resolve_recorded_path(arm.get("curriculum_path"), source_receipt)
        source_config_path = _freeze_source_config_path(source, arm, source_receipt, config)
        _require_file_hash(source_checkpoint, checkpoint_sha, "source training checkpoint")
        _require_file_hash(source_curriculum, curriculum_sha, "source training curriculum")
        _require_file_hash(source_config_path, config_sha, "source training config")

        eval_checkpoint = instrument["checkpoint_provenance"][f"{mode}:s{seed}"]
        eval_config = instrument["config_provenance"][f"{mode}:s{seed}"]
        if eval_checkpoint["sha256"] != checkpoint_sha:
            raise ValueError(f"{path}: evaluated checkpoint differs from frozen training")
        if eval_config["sha256"] != config_sha:
            raise ValueError(f"{path}: evaluated config differs from frozen training")
        if mode == RATCHET_MODE:
            expected_source_sha = h_r2_training_records.get(str(source_receipt))
            if expected_source_sha != source_receipt_sha:
                raise ValueError(f"{path}: ratchet source is not an H_R2 analysis input")
            ratchet_source_receipts.add(str(source_receipt))

        observed[key] = {
            "path": str(path),
            "sha256": sha256(path),
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "config": {"path": str(config), "sha256": config_sha},
            "curriculum": {"path": str(curriculum), "sha256": curriculum_sha},
            "training_receipt": {
                "path": str(source_receipt),
                "sha256": source_receipt_sha,
            },
        }
    expected = {(mode, seed) for mode in (RATCHET_MODE, FIXED_MODE) for seed in SEEDS}
    if set(observed) != expected:
        raise ValueError(f"H_R2 freeze-manifest keyspace differs: {sorted(observed)}")
    if ratchet_source_receipts != set(h_r2_training_records):
        raise ValueError("H_R2 ratchet training inputs are not in one-to-one frozen correspondence")
    return {f"{mode}:s{seed}": block for (mode, seed), block in sorted(observed.items())}


def audit_h_r2_amendment(
    path: Path,
    freezes: dict[str, Any],
    h_r2_robustness_paths: Sequence[Path],
    h_r2_training_paths: Sequence[Path],
) -> dict[str, Any]:
    """Bind reused controls and the targeted screen to the prospective amendment."""

    path = Path(path).resolve()
    _require_file_hash(path, EXPECTED_H_R2_AMENDMENT_SHA256, "H_R2 amendment")
    amendment = load_json(path)
    if amendment.get("kind") != "lucid_monotone_ratchet_confirmation_amendment":
        raise ValueError("unexpected H_R2 amendment kind")
    if amendment.get("schema_version") != 1:
        raise ValueError("unexpected H_R2 amendment schema")
    evaluation = amendment.get("evaluation") or {}
    observed_seed_map = {
        str(seed): int(value)
        for seed, value in (evaluation.get("evaluation_seed_by_training_seed") or {}).items()
    }
    if (
        evaluation.get("evaluator_sha256") != ratchet.EXPECTED_EVALUATOR_SHA256
        or evaluation.get("panel_sha256") != ratchet.EXPECTED_PANEL_SHA256
        or evaluation.get("num_envs") != ratchet.EXPECTED_NUM_ENVS
        or evaluation.get("presets") != list(ratchet.ALL_PRESETS)
        or observed_seed_map != ratchet.EXPECTED_EVALUATION_SEED
    ):
        raise ValueError("H_R2 amendment evaluation contract differs")

    frozen_inputs = amendment.get("frozen_inputs") or {}

    def frozen_input(name: str) -> dict[str, str]:
        record = frozen_inputs.get(name)
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"H_R2 amendment lacks frozen input {name}")
        input_path = _resolve_recorded_path(record["path"], path)
        expected = str(record["sha256"])
        _require_file_hash(input_path, expected, f"H_R2 amendment input {name}")
        return {"path": str(input_path), "sha256": expected}

    links = {
        "fixed:s8600": (
            frozen_input("fixed_seed_8600_checkpoint_bundle"),
            frozen_input("fixed_seed_8600_config"),
            frozen_input("fixed_seed_8600_bridge"),
        ),
        "fixed:s8601": (
            frozen_input("fixed_seed_8601_checkpoint"),
            frozen_input("fixed_seed_8601_config"),
            frozen_input("fixed_seed_8601_bridge"),
        ),
        "lucid_ratchet_rg:s8601": (
            frozen_input("ratchet_seed_8601_checkpoint"),
            frozen_input("ratchet_seed_8601_config"),
            frozen_input("screen_training"),
        ),
    }
    for freeze_key, (checkpoint, config, training) in links.items():
        freeze = freezes[freeze_key]
        if checkpoint["sha256"] != freeze["checkpoint"]["sha256"]:
            raise ValueError(f"{freeze_key}: freeze checkpoint differs from H_R2 amendment")
        if config["sha256"] != freeze["config"]["sha256"]:
            raise ValueError(f"{freeze_key}: freeze config differs from H_R2 amendment")
        if training["sha256"] != freeze["training_receipt"]["sha256"]:
            raise ValueError(f"{freeze_key}: source receipt differs from H_R2 amendment")

    robustness_hashes = {sha256(Path(value).resolve()) for value in h_r2_robustness_paths}
    training_hashes = {sha256(Path(value).resolve()) for value in h_r2_training_paths}
    screen_ratchet = frozen_input("screen_ratchet_evaluation")
    screen_fixed = frozen_input("screen_fixed_evaluation")
    screen_training = frozen_input("screen_training")
    if screen_ratchet["sha256"] not in robustness_hashes:
        raise ValueError("amendment ratchet screen receipt is not an H_R2 input")
    if screen_fixed["sha256"] not in robustness_hashes:
        raise ValueError("amendment fixed screen receipt is not an H_R2 input")
    if screen_training["sha256"] not in training_hashes:
        raise ValueError("amendment screen training receipt is not an H_R2 input")
    panel = frozen_input("panel_receipt")
    if panel["sha256"] != ratchet.EXPECTED_PANEL_SHA256:
        raise ValueError("amendment panel differs from the exact H_R2 panel")
    return {
        "path": str(path),
        "sha256": EXPECTED_H_R2_AMENDMENT_SHA256,
        "prospective_reused_links": links,
        "screen_ratchet_evaluation": screen_ratchet,
        "screen_fixed_evaluation": screen_fixed,
        "panel_receipt": panel,
    }


def _ratchet_trajectories(training_paths: Iterable[Path]) -> dict[str, Any]:
    trajectories: dict[str, Any] = {}
    for path in training_paths:
        receipt = load_json(path)
        for arm in (receipt.get("arms") or {}).values():
            if arm.get("mode") != RATCHET_MODE:
                continue
            seed = str(arm.get("seed"))
            if seed in trajectories:
                raise ValueError(f"duplicate H_R2 ratchet training arm for seed {seed}")
            curriculum_path = _resolve_recorded_path(arm.get("curriculum_path"), path)
            trajectories[seed] = trajectory_summary(
                _read_curriculum(curriculum_path), expect_ratchet=True
            )
    if set(trajectories) != set(SEEDS):
        raise ValueError(f"H_R2 ratchet training seed set differs: {sorted(trajectories)}")
    return dict(sorted(trajectories.items()))


def _mode_profiles(values: dict[tuple[str, str, str, str], float], mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ratchet.METRICS:
        result[metric] = {
            "in_envelope_auc": ratchet.profile(values, mode, metric, ratchet.IN_ENVELOPE_GRID),
            "frontier_auc": ratchet.profile(values, mode, metric, ratchet.FRONTIER_GRID),
            "legacy_phys_100_200_auc": ratchet.profile(values, mode, metric, LEGACY_FRONTIER_GRID),
            ratchet.LATENCY_PRESET: ratchet.single_cell(
                values, mode, metric, ratchet.LATENCY_PRESET
            ),
        }
    return result


def _endpoint_seed_values(endpoint: dict[str, Any]) -> dict[str, float]:
    if "mean_auc" in endpoint:
        return {seed: float(block["auc"]) for seed, block in endpoint["per_seed"].items()}
    return {seed: float(value) for seed, value in endpoint["per_seed"].items()}


def descriptive_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_values = _endpoint_seed_values(left)
    right_values = _endpoint_seed_values(right)
    if set(left_values) != set(SEEDS) or set(right_values) != set(SEEDS):
        raise ValueError("descriptive delta requires exactly the three matched training seeds")
    per_seed = {
        seed: {
            "left": left_values[seed],
            "right": right_values[seed],
            "delta": left_values[seed] - right_values[seed],
            "delta_pts": 100.0 * (left_values[seed] - right_values[seed]),
        }
        for seed in SEEDS
    }
    deltas = [block["delta"] for block in per_seed.values()]
    return {
        "per_seed": per_seed,
        "mean_delta": statistics.fmean(deltas),
        "mean_delta_pts": 100.0 * statistics.fmean(deltas),
        "binding": False,
        "inference": "none",
    }


def collapsed_seed_interaction(
    ratchet_arm: dict[str, Any], lucid_arm: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    collapsed_seed = next(
        seed for seed, collapsed in PREDECLARED_COLLAPSED_SEED.items() if collapsed
    )
    noncollapsed = [seed for seed in SEEDS if seed != collapsed_seed]
    for metric in ratchet.METRICS:
        result[metric] = {}
        for endpoint in (
            "in_envelope_auc",
            "frontier_auc",
            "legacy_phys_100_200_auc",
            ratchet.LATENCY_PRESET,
        ):
            deltas = descriptive_delta(ratchet_arm[metric][endpoint], lucid_arm[metric][endpoint])
            per_seed = deltas["per_seed"]
            collapsed_delta = per_seed[collapsed_seed]["delta"]
            noncollapsed_mean = statistics.fmean(per_seed[seed]["delta"] for seed in noncollapsed)
            interaction = collapsed_delta - noncollapsed_mean
            result[metric][endpoint] = {
                "definition": (
                    "(ratchet-minus-historical-lucid) at predeclared collapsed seed 8601 "
                    "minus the mean ratchet-minus-historical-lucid delta at seeds 8600/8602"
                ),
                "collapsed_seed": collapsed_seed,
                "noncollapsed_seeds": noncollapsed,
                "ratchet_minus_lucid_per_seed": {seed: per_seed[seed]["delta"] for seed in SEEDS},
                "collapsed_seed_delta": collapsed_delta,
                "noncollapsed_seed_mean_delta": noncollapsed_mean,
                "interaction": interaction,
                "interaction_pts": 100.0 * interaction,
                "binding": False,
                "inference": "none; n=1 collapsed seed versus 2 noncollapsed seeds",
            }
    return result


def analyze(
    h_r2_analysis_path: Path,
    h_r2_amendment_path: Path,
    h_r2_freeze_manifest_paths: Sequence[Path],
    historical_robustness_paths: Sequence[Path],
    historical_bridge_paths: Sequence[Path],
) -> dict[str, Any]:
    h_r2_analysis_path = Path(h_r2_analysis_path).resolve()
    h_r2_amendment_path = Path(h_r2_amendment_path).resolve()
    h_r2_freeze_manifest_paths = [Path(path).resolve() for path in h_r2_freeze_manifest_paths]
    historical_robustness_paths = [Path(path).resolve() for path in historical_robustness_paths]
    historical_bridge_paths = [Path(path).resolve() for path in historical_bridge_paths]

    h_r2, h_r2_robustness, h_r2_training, h_r2_sha = audit_terminal_h_r2(h_r2_analysis_path)
    bridge_records, historical_mechanism = audit_historical_bridges(historical_bridge_paths)
    instrument = audit_exact_instrument(h_r2_robustness, historical_robustness_paths)
    h_r2_freezes = audit_h_r2_freeze_manifests(
        h_r2_freeze_manifest_paths, h_r2_training, instrument
    )
    h_r2_amendment = audit_h_r2_amendment(
        h_r2_amendment_path, h_r2_freezes, h_r2_robustness, h_r2_training
    )

    # Bind each newly scored historical checkpoint/config to the independently
    # audited training bridge for that seed.
    for seed in SEEDS:
        eval_checkpoint = instrument["checkpoint_provenance"][f"{LUCID_MODE}:s{seed}"]
        eval_config = instrument["config_provenance"][f"{LUCID_MODE}:s{seed}"]
        bridge = bridge_records[seed]
        if eval_checkpoint["sha256"] != bridge["checkpoint"]["sha256"]:
            raise ValueError(f"historical evaluation checkpoint does not match seed {seed} bridge")
        if eval_config["sha256"] != bridge["true_config"]["sha256"]:
            raise ValueError(f"historical evaluation config does not match seed {seed} bridge")

    all_robustness = [*h_r2_robustness, *historical_robustness_paths]
    values = ratchet.collect_robustness(all_robustness)
    arms = {mode: _mode_profiles(values, mode) for mode in MODES}
    comparisons = {
        "ratchet_minus_historical_lucid": {},
        "fixed_minus_historical_lucid": {},
    }
    for metric in ratchet.METRICS:
        comparisons["ratchet_minus_historical_lucid"][metric] = {}
        comparisons["fixed_minus_historical_lucid"][metric] = {}
        for endpoint in arms[LUCID_MODE][metric]:
            comparisons["ratchet_minus_historical_lucid"][metric][endpoint] = descriptive_delta(
                arms[RATCHET_MODE][metric][endpoint], arms[LUCID_MODE][metric][endpoint]
            )
            comparisons["fixed_minus_historical_lucid"][metric][endpoint] = descriptive_delta(
                arms[FIXED_MODE][metric][endpoint], arms[LUCID_MODE][metric][endpoint]
            )

    ratchet_mechanism = _ratchet_trajectories(h_r2_training)
    mechanism_table = {
        seed: {
            "predeclared_historical_collapse": PREDECLARED_COLLAPSED_SEED[seed],
            LUCID_MODE: historical_mechanism[seed],
            RATCHET_MODE: ratchet_mechanism[seed],
            FIXED_MODE: {
                "evidence_kind": "open_loop_mode_contract_not_rederived_training_telemetry",
                "schedule": "lambda=1 throughout",
            },
        }
        for seed in SEEDS
    }

    after = sha256(h_r2_analysis_path)
    if after != h_r2_sha:
        raise ValueError("H_R2 analysis changed during historical bridge analysis")
    return {
        "kind": "lucid_ratchet_historical_bridge_analysis",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": ratchet._git(("rev-parse", "HEAD")),
        "git_status_short": ratchet._git(("status", "--short")),
        "claim_scope": {
            "classification": "posthoc_descriptive",
            "binding": False,
            "alters_H_R2": False,
            "inference": "none",
            "noninferiority_claim_authorized": False,
            "superiority_claim_authorized": False,
            "latent_gap_rehabilitated": False,
            "statement": (
                "This retrospective bridge asks whether ratchet behavior is consistent with "
                "preventing the historical latent controller's wrong-way lambda decreases. "
                "It is not part of H_R2 and cannot alter its verdict. Different code-era "
                "training makes this mechanistic context, not a causal flag-only comparison."
            ),
        },
        "activation": {
            "condition": (
                "terminal H_R2 analysis exists with complete, passing H_R0 mechanism gates; "
                "activation is independent of whether capability H_R2 passes or fails"
            ),
            "satisfied": True,
            "h_r0_mechanism_pass": True,
            "h_r2_status_observed": h_r2["preregistered_decision"]["status"],
            "h_r2_analysis_path": str(h_r2_analysis_path),
            "h_r2_analysis_sha256_before": h_r2_sha,
            "h_r2_analysis_sha256_after": after,
            "h_r2_unchanged": True,
        },
        "inputs": {
            "h_r2_robustness_receipts": (h_r2.get("inputs") or {})["robustness_receipts"],
            "h_r2_training_receipts": (h_r2.get("inputs") or {})["training_receipts"],
            "h_r2_freeze_manifests": h_r2_freezes,
            "h_r2_confirmation_amendment": h_r2_amendment,
            "historical_robustness_receipts": [
                {"path": str(path), "sha256": sha256(path)} for path in historical_robustness_paths
            ],
            "historical_training_bridges": bridge_records,
        },
        "instrument_audit": instrument,
        "frozen_descriptive_contract": {
            "modes": list(MODES),
            "training_seeds": list(SEEDS),
            "presets": list(ratchet.ALL_PRESETS),
            "cell_count": EXPECTED_TOTAL_CELLS,
            "in_envelope_grid": dict(ratchet.IN_ENVELOPE_GRID),
            "frontier_grid": dict(ratchet.FRONTIER_GRID),
            "legacy_phys_100_200_grid": dict(LEGACY_FRONTIER_GRID),
            "latency_cell": ratchet.LATENCY_PRESET,
            "metrics": list(ratchet.METRICS),
            "predeclared_historical_collapse": dict(PREDECLARED_COLLAPSED_SEED),
            "historical_config_provenance_rule": (
                "Use the SHA-pinned true run config for each B8000 lucid_rg cell; the "
                "incorrect artifact-side seed-8600 config is categorically excluded."
            ),
            "excluded": [
                "training return as an outcome",
                "noninferiority or superiority decisions",
                "p-values or confidence claims at three seeds",
                "checkpoint or cell selection after scoring",
            ],
        },
        "arms": arms,
        "descriptive_comparisons": comparisons,
        "mechanism_table": mechanism_table,
        "collapsed_seed_interaction": collapsed_seed_interaction(
            arms[RATCHET_MODE], arms[LUCID_MODE]
        ),
        "verified": [
            "terminal H_R2 analysis replayed from its unchanged SHA-pinned inputs",
            "exact 126-cell mode/seed/preset keyspace audited without imputation",
            "historical checkpoints, true configs, and curriculum traces matched frozen hashes",
            "all deltas and the collapsed-seed interaction are descriptive and nonbinding",
        ],
        "limitations": [
            "historical lucid_rg and ratchet policies were trained in different code eras",
            "only one predeclared historical seed collapsed",
            "the panel contains aliases of the single training motion",
            "a causal monotonic-flag claim would require fresh current-commit lucid_rg controls",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-r2-analysis", type=Path, required=True)
    parser.add_argument("--h-r2-amendment", type=Path, required=True)
    parser.add_argument(
        "--h-r2-freeze-manifest",
        "--h-r2-freeze-manifests",
        dest="h_r2_freeze_manifests",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
    )
    parser.add_argument(
        "--historical-robustness-receipt",
        "--historical-robustness-receipts",
        dest="historical_robustness_receipts",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
    )
    parser.add_argument(
        "--historical-training-bridge",
        "--historical-training-bridges",
        dest="historical_training_bridges",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _print_summary(receipt: dict[str, Any], out: Path) -> None:
    print("posthoc descriptive historical bridge; H_R2 remains unchanged")
    print(f"audited cells: {receipt['instrument_audit']['cell_count']}")
    for metric in ratchet.METRICS:
        block = receipt["collapsed_seed_interaction"][metric]["frontier_auc"]
        print(f"{metric} frontier collapsed-seed interaction: {block['interaction_pts']:.3f} pts")
    print(f"analysis receipt {out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = analyze(
        args.h_r2_analysis,
        args.h_r2_amendment,
        args.h_r2_freeze_manifests,
        args.historical_robustness_receipts,
        args.historical_training_bridges,
    )
    out = args.out or MANIFESTS / (
        f"lucid_ratchet_historical_bridge_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _print_summary(receipt, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
