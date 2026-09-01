#!/usr/bin/env python3
"""Fail-closed Tier-2 support-extension screening analysis.

This analyzer compares three fresh seed-8600 training arms (``fixed``,
``fixed_150``, and ``fixed_u150``) on one frozen 512-alias evaluation panel.
The historical fixed checkpoint is used only as an instrument bridge.  Every
input is role-specific: two receipts with mode ``fixed`` are never inferred or
merged by mode name.

The result is deliberately screening-grade.  Passing a frozen threshold can
select a candidate for confirmation; it cannot authorize a directional or
superiority claim from one training seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import dr_scaling as DS  # noqa: E402
from gear_sonic.research.practice_utility.paths import MANIFESTS, relocate  # noqa: E402

EVALUATION_ROLES = ("historical_fixed", "fresh_fixed", "fixed_150", "fixed_u150")
TRAINING_ROLES = ("fresh_fixed", "fixed_150", "fixed_u150")
ROLE_MODE = {
    "historical_fixed": "fixed",
    "fresh_fixed": "fixed",
    "fixed_150": "fixed_150",
    "fixed_u150": "fixed_u150",
}

EXPECTED_TRAINING_SEED = 8600
EXPECTED_EVALUATION_SEED = 8700
EXPECTED_EVALUATION_ENVS = 512
EXPECTED_TRAINING_ENVS = 1024
EXPECTED_TRAINING_ITERATIONS = 8000
EXPECTED_WARMUP_ITERATIONS = 10
EXPECTED_MAX_DELAY_STEPS = 12
EXPECTED_PHYSICS_STEP_MS = 5
FLOAT_TOLERANCE = 1e-12

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
LATENCY_STEPS = {
    "lat_10ms": 2,
    "lat_20ms": 4,
    "lat_30ms": 6,
    "lat_40ms": 8,
    "lat_50ms": 10,
    "lat_60ms": 12,
}
PHYSICS_LEVELS = dict((*IN_ENVELOPE_GRID, *FRONTIER_GRID))
EXPECTED_PRESETS = tuple(PHYSICS_LEVELS) + tuple(LATENCY_STEPS)
METRICS = ("success_rate", "progress_rate")
HARD_SUCCESS_PRESETS = ("phys_150", "phys_175", "phys_200")

EXPECTED_STRATUM_SIZES = [37, 37, 37, 37, 36, 36, 36, 768]
EXPECTED_STRATUM_LAMBDAS = [0.1875, 0.375, 0.5625, 0.75, 0.9375, 1.125, 1.3125, 1.5]
EXPECTED_SCALABLE_TERMS = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
EXPECTED_NON_LATENCY_TERMS = EXPECTED_SCALABLE_TERMS - {"randomize_action_delay"}

EVALUATOR_RELATIVE_PATH = "scripts/practice_utility/run_curriculum_robustness_eval.py"
ANALYZER_RELATIVE_PATH = "scripts/practice_utility/analyze_support_screen.py"
TRAINER_RELATIVE_PATH = "scripts/practice_utility/run_curriculum_comparison.py"
REQUIRED_FROZEN_INPUTS = {
    "panel_receipt",
    "h_r2_analysis",
    "motion",
    "encoder",
    "historical_fixed_training",
    "historical_fixed_freeze_manifest",
    "historical_fixed_checkpoint",
    "historical_fixed_config",
}

# Frozen screening thresholds. Rates and AUCs are fractions, not percentage points.
BRIDGE_FRONTIER_TOLERANCE = 0.02
BRIDGE_IN_ENVELOPE_TOLERANCE = 0.01
BRIDGE_LAT50_TOLERANCE = 0.02
CANDIDATE_FRONTIER_SUCCESS_GAIN = 0.02
CANDIDATE_FRONTIER_PROGRESS_FLOOR = -0.02
CANDIDATE_IN_ENVELOPE_FLOOR = -0.01
H_X1_HARD_SUCCESS_GAIN = 0.03
LAT60_REPORTED_GAIN = 0.05
PREFERENCE_FRONTIER_GAIN = 0.02
PREFERENCE_NOMINAL_LOSS = 0.01

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, rejecting arrays and scalar stand-ins."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact(observed: Any, expected: Any, label: str) -> None:
    _require(observed == expected, f"{label}: expected {expected!r}, observed {observed!r}")


def _require_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label}: invalid sha256",
    )
    return value


def _require_rate(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}: expected a numeric rate, observed {value!r}",
    )
    numeric = float(value)
    _require(math.isfinite(numeric), f"{label}: rate is not finite")
    _require(0.0 <= numeric <= 1.0, f"{label}: rate {numeric} is outside [0, 1]")
    return numeric


def _require_close(observed: Any, expected: float, label: str) -> None:
    _require(
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE),
        f"{label}: expected {expected!r}, observed {observed!r}",
    )


def _materialized(recorded: str | Path) -> Path:
    """Resolve an on-disk receipt reference, honoring the program data-root relocation."""
    return relocate(recorded).resolve()


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"{label}: file is missing: {path}")
    observed = sha256(path)
    _require_exact(observed, expected, f"{label}.sha256")


def _require_verified(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{label}: expected a nonempty list")
    return value


def _require_read_only(path: Path, label: str) -> None:
    _require(path.is_file(), f"{label}: file is missing: {path}")
    _require(
        not (stat.S_IMODE(path.stat().st_mode) & 0o222),
        f"{label}: frozen file has write bits: {path}",
    )


def _require_clean_status(value: Any, label: str) -> None:
    _require(value in (None, "", []), f"{label}: expected a clean Git status, observed {value!r}")


def _require_nested_close(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _require_close(observed, float(expected), label)
        return
    if isinstance(expected, dict):
        _require(isinstance(observed, dict), f"{label}: expected an object")
        _require_exact(set(observed), set(expected), f"{label}.keys")
        for key, value in expected.items():
            _require_nested_close(observed[key], value, f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        _require(isinstance(observed, (list, tuple)), f"{label}: expected a sequence")
        _require_exact(len(observed), len(expected), f"{label}.length")
        for index, value in enumerate(expected):
            _require_nested_close(observed[index], value, f"{label}[{index}]")
        return
    _require_exact(observed, expected, label)


def _same_file_or_content(left: Path, right: Path) -> bool:
    if left == right:
        return True
    return left.is_file() and right.is_file() and sha256(left) == sha256(right)


def audit_preregistration(path: Path, expected_sha: str) -> dict[str, Any]:
    """Bind every analysis input to the immutable, pre-GPU Tier-2 contract."""
    expected_sha = _require_sha(expected_sha, "expected_preregistration_sha")
    path = Path(path).resolve()
    _require_file_hash(path, expected_sha, "preregistration")
    prereg = load_json(path)
    _require_exact(
        prereg.get("kind"),
        "lucid_tier2_support_screen_preregistration",
        "preregistration.kind",
    )
    _require_exact(prereg.get("schema_version"), 1, "preregistration.schema_version")
    _require(prereg.get("frozen") is True, "preregistration is not frozen")
    _require(prereg.get("written_before_gpu") is True, "preregistration was not pre-GPU")

    code = prereg.get("code_state") or {}
    worktree_raw = code.get("worktree")
    _require(
        isinstance(worktree_raw, str) and worktree_raw,
        "preregistration.code_state.worktree is missing",
    )
    worktree = Path(worktree_raw).resolve()
    _require_exact(worktree, REPO.resolve(), "preregistration.code_state.worktree")
    git_sha = code.get("git_sha")
    _require(
        isinstance(git_sha, str) and _GIT_SHA_RE.fullmatch(git_sha) is not None,
        "preregistration.code_state.git_sha is invalid",
    )
    _require_exact(_git(("rev-parse", "HEAD")), git_sha, "preregistration live git_sha")
    _require(
        code.get("clean_detached_worktree_required") is True,
        "preregistration does not require a clean detached worktree",
    )
    file_hashes = code.get("file_sha256")
    _require(isinstance(file_hashes, dict), "preregistration.code_state.file_sha256 is missing")
    for relative, digest in file_hashes.items():
        _require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"preregistration has unsafe code path {relative!r}",
        )
        expected = _require_sha(digest, f"preregistration.code_state.file_sha256.{relative}")
        _require_file_hash(worktree / relative, expected, f"preregistered code {relative}")
    evaluator_sha = _require_sha(
        file_hashes.get(EVALUATOR_RELATIVE_PATH), "preregistration evaluator file sha"
    )
    analyzer_sha = _require_sha(
        file_hashes.get(ANALYZER_RELATIVE_PATH), "preregistration analyzer file sha"
    )
    trainer_sha = _require_sha(
        file_hashes.get(TRAINER_RELATIVE_PATH), "preregistration trainer file sha"
    )

    frozen = prereg.get("frozen_inputs")
    _require(isinstance(frozen, dict), "preregistration.frozen_inputs is missing")
    _require(
        REQUIRED_FROZEN_INPUTS.issubset(frozen),
        "preregistration is missing one or more required frozen inputs",
    )
    inputs: dict[str, dict[str, str]] = {}
    for key, entry in frozen.items():
        _require(isinstance(entry, dict), f"preregistration.frozen_inputs.{key} must be an object")
        raw_path = entry.get("path")
        _require(
            isinstance(raw_path, str) and raw_path,
            f"preregistration.frozen_inputs.{key}.path is missing",
        )
        materialized = _materialized(raw_path)
        expected = _require_sha(entry.get("sha256"), f"preregistration.frozen_inputs.{key}")
        _require_file_hash(materialized, expected, f"preregistration.frozen_inputs.{key}")
        inputs[key] = {"path": str(materialized), "sha256": expected}

    design = prereg.get("design") or {}
    training = design.get("training") or {}
    for key, expected in {
        "from_scratch": True,
        "seed": EXPECTED_TRAINING_SEED,
        "num_envs": EXPECTED_TRAINING_ENVS,
        "iterations": EXPECTED_TRAINING_ITERATIONS,
        "warmup_iterations": EXPECTED_WARMUP_ITERATIONS,
        "order": ["fresh_fixed", "fixed_150", "fixed_u150"],
        "role_to_mode": {
            "fresh_fixed": "fixed",
            "fixed_150": "fixed_150",
            "fixed_u150": "fixed_u150",
        },
        "max_delay_capacity_steps": EXPECTED_MAX_DELAY_STEPS,
        "resume_allowed": False,
    }.items():
        _require_exact(training.get(key), expected, f"preregistration.design.training.{key}")
    evaluation = design.get("evaluation") or {}
    for key, expected in {
        "num_envs": EXPECTED_EVALUATION_ENVS,
        "checkpoint_seed": EXPECTED_TRAINING_SEED,
        "evaluation_seed": EXPECTED_EVALUATION_SEED,
        "roles": list(EVALUATION_ROLES),
        "presets": list(EXPECTED_PRESETS),
        "total_cells": len(EVALUATION_ROLES) * len(EXPECTED_PRESETS),
    }.items():
        _require_exact(evaluation.get(key), expected, f"preregistration.design.evaluation.{key}")
    analysis = prereg.get("analysis") or {}
    _require_exact(
        analysis.get("script"), ANALYZER_RELATIVE_PATH, "preregistration.analysis.script"
    )
    _require(
        analysis.get("screening_only") is True, "preregistration analysis is not screening-only"
    )
    _require(
        analysis.get("directional_claim_authorized") is False
        and analysis.get("superiority_claim_authorized") is False,
        "preregistration analysis authorizes a confirmatory claim",
    )

    eval_binding = prereg.get("evaluation") or {}
    _require_exact(
        _materialized(eval_binding.get("panel_receipt")),
        Path(inputs["panel_receipt"]["path"]),
        "preregistration.evaluation.panel_receipt",
    )
    _require_exact(
        eval_binding.get("panel_sha256"),
        inputs["panel_receipt"]["sha256"],
        "preregistration.evaluation.panel_sha256",
    )
    _require_exact(
        eval_binding.get("evaluator_sha256"),
        evaluator_sha,
        "preregistration.evaluation.evaluator_sha256",
    )
    return {
        "path": str(path),
        "sha256": expected_sha,
        "git_sha": git_sha,
        "evaluator_sha256": evaluator_sha,
        "analyzer_sha256": analyzer_sha,
        "trainer_sha256": trainer_sha,
        "frozen_inputs": inputs,
    }


def _expected_preset_metadata() -> dict[str, dict[str, Any]]:
    metadata = {
        preset: {
            "event_preset": "tracking/lucid_curriculum",
            "non_latency_dr_scale": level,
            "fixed_latency_steps": 0,
        }
        for preset, level in PHYSICS_LEVELS.items()
    }
    metadata.update(
        {
            preset: {
                "event_preset": "tracking/lucid_eval_clean",
                "fixed_latency_steps": steps,
            }
            for preset, steps in LATENCY_STEPS.items()
        }
    )
    return metadata


def audit_panel(
    panel_path: Path,
    *,
    expected_path: Path,
    expected_sha: str,
    expected_motion_path: Path,
    expected_motion_sha: str,
) -> dict[str, Any]:
    """Validate the supplied 512-alias panel receipt and return frozen identity."""
    panel_path = Path(panel_path).resolve()
    _require_exact(panel_path, Path(expected_path).resolve(), "panel preregistered path")
    _require_file_hash(panel_path, expected_sha, "panel preregistered bytes")
    panel = load_json(panel_path)
    _require_exact(panel.get("kind"), "lucid_replicate_panel", "panel.kind")
    _require_exact(panel.get("schema_version"), 1, "panel.schema_version")
    _require_exact(panel.get("replicates"), EXPECTED_EVALUATION_ENVS, "panel.replicates")
    _require_sha(panel.get("alias_keys_sha256"), "panel.alias_keys_sha256")
    _require_sha(panel.get("source_clip_sha256"), "panel.source_clip_sha256")
    _require(
        isinstance(panel.get("motion_key"), str) and panel["motion_key"],
        "panel.motion_key is missing",
    )
    _require_verified(panel.get("verified"), "panel.verified")
    source_raw = panel.get("source_clip")
    motion_file_raw = panel.get("motion_file")
    _require(isinstance(source_raw, str) and source_raw, "panel.source_clip is missing")
    _require(isinstance(motion_file_raw, str) and motion_file_raw, "panel.motion_file is missing")
    source = _materialized(source_raw)
    motion_file = _materialized(motion_file_raw)
    _require_exact(source, Path(expected_motion_path).resolve(), "panel preregistered source clip")
    _require_file_hash(source, expected_motion_sha, "panel preregistered source clip")
    _require_file_hash(source, panel["source_clip_sha256"], "panel source clip")
    _require_exact(source.stem, panel["motion_key"], "panel source clip motion_key")
    _require(motion_file.is_dir(), f"panel.motion_file is not a directory: {motion_file}")
    entries = sorted(motion_file.iterdir())
    _require_exact(len(entries), EXPECTED_EVALUATION_ENVS, "panel alias directory entries")
    _require(
        all(entry.suffix == ".pkl" and entry.is_symlink() and entry.is_file() for entry in entries),
        "panel alias directory must contain only 512 live .pkl symlinks",
    )
    alias_keys = [entry.stem for entry in entries]
    alias_digest = hashlib.sha256(("\n".join(alias_keys) + "\n").encode()).hexdigest()
    _require_exact(alias_digest, panel["alias_keys_sha256"], "panel live alias stem digest")
    targets = {entry.resolve() for entry in entries}
    _require_exact(targets, {source}, "panel live alias canonical target")
    return {
        "path": str(panel_path),
        "sha256": sha256(panel_path),
        "motion_file": str(motion_file),
        "source_clip": str(source),
        "motion_key": panel["motion_key"],
        "source_clip_sha256": panel["source_clip_sha256"],
        "replicates": panel["replicates"],
        "alias_keys_sha256": panel["alias_keys_sha256"],
        "pool_sha256": panel.get("pool_sha256"),
        "split_sha256": panel.get("split_sha256"),
        "partition": panel.get("partition"),
    }


def audit_h_r2(path: Path, *, expected_path: Path, expected_sha: str) -> dict[str, Any]:
    """Require the completed, three-seed H_R2 ratchet analysis gate."""
    path = Path(path).resolve()
    _require_exact(path, Path(expected_path).resolve(), "H_R2 preregistered path")
    _require_file_hash(path, expected_sha, "H_R2 preregistered bytes")
    receipt = load_json(path)
    _require_exact(receipt.get("kind"), "lucid_ratchet_analysis", "H_R2.kind")
    _require(
        (receipt.get("instrument_audit") or {}).get("passed") is True,
        "H_R2 instrument audit did not pass",
    )
    decision = receipt.get("preregistered_decision") or {}
    _require_exact(decision.get("status"), "pass", "H_R2 decision.status")
    _require_exact(
        decision.get("paired_training_seeds"),
        ["8600", "8601", "8602"],
        "H_R2 paired_training_seeds",
    )
    _require(decision.get("mechanism_pass") is True, "H_R2 mechanism did not pass")
    _require(
        decision.get("capability_components_pass") is True,
        "H_R2 capability components did not pass",
    )
    _require(
        decision.get("noninferiority_claim_authorized") is True,
        "H_R2 noninferiority claim is not authorized",
    )
    _require(
        decision.get("superiority_claim_authorized") is False,
        "H_R2 unexpectedly authorizes superiority",
    )
    scope = receipt.get("claim_scope") or {}
    _require_exact(scope.get("status"), "three_seed_decision", "H_R2 claim_scope.status")
    _require(
        scope.get("noninferiority_decision_eligible") is True, "H_R2 is not a three-seed decision"
    )
    summary = (receipt.get("mechanism") or {}).get("summary") or {}
    _require(summary.get("all_available_seeds_pass") is True, "H_R2 mechanism summary did not pass")
    return {"path": str(path), "sha256": sha256(path), "status": decision["status"]}


def _require_float_sequence(observed: Any, expected: Sequence[float], label: str) -> None:
    _require(isinstance(observed, list), f"{label}: expected a list")
    _require_exact(len(observed), len(expected), f"{label}.length")
    for index, target in enumerate(expected):
        _require_close(observed[index], target, f"{label}[{index}]")


def _audit_fixed_u150_tace(
    tace: Mapping[str, Any],
    label: str,
    *,
    require_positive_counts: bool,
    expected_anchor_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute every lower-stratum dispatcher parameter from its live baseline."""
    for key, expected in {
        "num_anchor": 0,
        "num_focus": EXPECTED_TRAINING_ENVS,
        "anchor_ratio": 0.0,
        "num_strata": 8,
        "stratum_sizes": EXPECTED_STRATUM_SIZES,
        "consolidating": False,
    }.items():
        _require_exact(tace.get(key), expected, f"{label}.{key}")
    _require_float_sequence(
        tace.get("stratum_lambdas"), EXPECTED_STRATUM_LAMBDAS, f"{label}.stratum_lambdas"
    )

    dispatch = tace.get("dispatch")
    _require(isinstance(dispatch, dict), f"{label}.dispatch is missing")
    _require_exact(set(dispatch), EXPECTED_SCALABLE_TERMS, f"{label}.dispatch terms")
    focus_keys = {f"focus_s{index}" for index in range(8)}
    baselines: dict[str, Any] = {}
    for term in sorted(EXPECTED_SCALABLE_TERMS):
        telemetry = dispatch[term]
        _require(isinstance(telemetry, dict), f"{label}.dispatch.{term} must be an object")
        _require_exact(telemetry.get("term"), term, f"{label}.dispatch.{term}.term")
        _require_exact(telemetry.get("num_strata"), 8, f"{label}.dispatch.{term}.num_strata")
        baseline = telemetry.get("anchor_params")
        _require(
            isinstance(baseline, dict) and bool(baseline),
            f"{label}.dispatch.{term}.anchor_params is missing",
        )
        _require(
            set(baseline).issubset(DS.RANGE_NOMINALS) and bool(set(baseline)),
            f"{label}.dispatch.{term}.anchor_params has unsupported fields",
        )
        if expected_anchor_params is not None:
            _require_nested_close(
                baseline,
                expected_anchor_params[term],
                f"{label}.dispatch.{term}.anchor_params stability",
            )
        baselines[term] = baseline
        params = telemetry.get("stratum_params")
        _require(isinstance(params, list), f"{label}.dispatch.{term}.stratum_params is missing")
        _require_exact(len(params), 8, f"{label}.dispatch.{term}.stratum_params length")
        for index, dose in enumerate(EXPECTED_STRATUM_LAMBDAS[:-1]):
            expected = DS.scaled_term_params(baseline, dose, allow_extrapolation=True)
            expected, _ = DS.clamp_params_physical(expected)
            _require_nested_close(
                params[index], expected, f"{label}.dispatch.{term}.stratum_params[{index}]"
            )
        _require(params[-1] is None, f"{label}.dispatch.{term}: top stratum is not passthrough")
        counts = telemetry.get("env_counts")
        _require(isinstance(counts, dict), f"{label}.dispatch.{term}.env_counts is missing")
        _require(
            all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in counts.values()
            ),
            f"{label}.dispatch.{term}.env_counts has invalid cumulative counts",
        )
        if require_positive_counts:
            _require(focus_keys.issubset(counts), f"{label}.dispatch.{term}: missing focus counts")
            for index in range(8):
                _require(
                    counts[f"focus_s{index}"] > 0,
                    f"{label}.dispatch.{term}.env_counts.focus_s{index} is not positive",
                )
    return baselines


def _audit_curriculum(role: str, curriculum_path: Path) -> dict[str, Any]:
    """Stream and independently verify the full manipulation-bearing JSONL."""
    curriculum_path = Path(curriculum_path).resolve()
    _require(curriculum_path.is_file(), f"{role}.curriculum is missing: {curriculum_path}")
    fixed_lambda = 1.0 if role in ("historical_fixed", "fresh_fixed") else 1.5
    warmup_rows = 0
    postwarmup_rows = 0
    tace_rows = 0
    final_tace: Mapping[str, Any] | None = None
    anchor_params: dict[str, Any] | None = None
    rows = 0
    with curriculum_path.open() as stream:
        for expected_step, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"{role}.curriculum row {expected_step} is blank")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{role}.curriculum row {expected_step} is invalid JSON"
                ) from error
            _require(
                isinstance(row, dict), f"{role}.curriculum row {expected_step} is not an object"
            )
            rows += 1
            _require_exact(row.get("global_step"), expected_step, f"{role}.curriculum.global_step")
            _require_exact(row.get("mode"), "fixed", f"{role}.curriculum row {expected_step}.mode")
            _require_close(
                row.get("lambda"), fixed_lambda, f"{role}.curriculum row {expected_step}.lambda"
            )
            _require_exact(
                set(row.get("scalable_terms") or []),
                EXPECTED_SCALABLE_TERMS,
                f"{role}.curriculum row {expected_step}.scalable_terms",
            )
            _require(
                not bool(row.get("consolidation")),
                f"{role}.curriculum row {expected_step} consolidated",
            )
            if expected_step <= EXPECTED_WARMUP_ITERATIONS:
                warmup_rows += 1
                _require(
                    row.get("warmup_hold") is True,
                    f"{role}.curriculum row {expected_step} is not warmup",
                )
                _require(
                    row.get("tace") is None,
                    f"{role}.warmup row {expected_step} unexpectedly has TACE",
                )
                continue

            postwarmup_rows += 1
            _require(
                not bool(row.get("warmup_hold")),
                f"{role}.curriculum row {expected_step} stayed warmup",
            )
            if role in ("fixed_150", "fixed_u150"):
                _require(
                    row.get("allow_extrapolation") is True,
                    f"{role}.curriculum row {expected_step} disabled extrapolation",
                )
                _require_exact(
                    row.get("physical_clamp"),
                    ["physics_material"],
                    f"{role}.curriculum row {expected_step}.physical_clamp",
                )
            else:
                _require(
                    not bool(row.get("allow_extrapolation")),
                    f"{role}.curriculum row {expected_step} enabled extrapolation",
                )
                _require(
                    row.get("physical_clamp") in (None, []),
                    f"{role}.curriculum row {expected_step} has a physical clamp",
                )

            if role == "fixed_u150":
                tace = row.get("tace")
                _require(
                    isinstance(tace, dict), f"{role}.curriculum row {expected_step} lacks TACE"
                )
                anchor_params = _audit_fixed_u150_tace(
                    tace,
                    f"{role}.curriculum row {expected_step}.tace",
                    # Per-stratum counters are created lazily when an event
                    # term next runs on resetting envs. Early post-warmup rows
                    # can therefore carry only aggregate anchor/focus counts.
                    require_positive_counts=False,
                    expected_anchor_params=anchor_params,
                )
                tace_rows += 1
                final_tace = tace
            else:
                _require(row.get("tace") is None, f"{role}.curriculum row {expected_step} has TACE")

    _require_exact(rows, EXPECTED_TRAINING_ITERATIONS, f"{role}.curriculum rows")
    _require_exact(warmup_rows, EXPECTED_WARMUP_ITERATIONS, f"{role}.curriculum warmup rows")
    expected_post = EXPECTED_TRAINING_ITERATIONS - EXPECTED_WARMUP_ITERATIONS
    _require_exact(postwarmup_rows, expected_post, f"{role}.curriculum postwarmup rows")
    _require_exact(
        tace_rows, expected_post if role == "fixed_u150" else 0, f"{role}.curriculum TACE rows"
    )
    return {
        "path": str(curriculum_path),
        "sha256": sha256(curriculum_path),
        "rows": rows,
        "warmup_rows": warmup_rows,
        "postwarmup_rows": postwarmup_rows,
        "tace_rows": tace_rows,
        "final_tace": final_tace,
    }


def audit_training(role: str, path: Path, *, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one new seed-8600, 1024x8000 from-scratch training receipt."""
    _require(role in TRAINING_ROLES, f"unknown training role: {role}")
    mode = ROLE_MODE[role]
    path = Path(path).resolve()
    receipt = load_json(path)
    _require_exact(receipt.get("kind"), "lucid_three_arm_training_comparison", f"{role}.kind")
    _require_exact(receipt.get("schema_version"), 1, f"{role}.schema_version")
    _require_verified(receipt.get("verified"), f"{role}.verified")
    _require_exact(receipt.get("git_sha"), preregistration["git_sha"], f"{role}.git_sha")
    _require_clean_status(receipt.get("git_status_short"), f"{role}.git_status_short")
    _require_exact(
        receipt.get("launcher_sha256"),
        preregistration["trainer_sha256"],
        f"{role}.launcher_sha256",
    )

    config = receipt.get("config") or {}
    exact_config = {
        "checkpoint": None,
        "num_envs": EXPECTED_TRAINING_ENVS,
        "iterations": EXPECTED_TRAINING_ITERATIONS,
        "warmup_iterations": EXPECTED_WARMUP_ITERATIONS,
        "seeds": [EXPECTED_TRAINING_SEED],
        "modes": [mode],
        "from_scratch": True,
        "event_preset": "tracking/lucid_curriculum",
        "termination_thresholds": "default",
        "consolidation_fraction": 0,
        "max_delay_steps": EXPECTED_MAX_DELAY_STEPS,
        "max_delay_ms": EXPECTED_MAX_DELAY_STEPS * EXPECTED_PHYSICS_STEP_MS,
        "arms": {mode: ["fixed", 0.0, None]},
    }
    for key, expected in exact_config.items():
        _require_exact(config.get(key), expected, f"{role}.config.{key}")
    _require_exact(
        config.get("arm_order"),
        [{"seed": EXPECTED_TRAINING_SEED, "modes": [mode]}],
        f"{role}.config.arm_order",
    )

    arms = receipt.get("arms")
    runtime = receipt.get("runtime")
    commands = receipt.get("commands")
    _require(
        isinstance(arms, dict) and len(arms) == 1, f"{role}: expected exactly one training arm"
    )
    _require(
        isinstance(runtime, dict) and set(runtime) == set(arms), f"{role}: runtime/arm keys differ"
    )
    _require(
        isinstance(commands, dict) and set(commands) == set(arms),
        f"{role}: command/arm keys differ",
    )
    branch_id, arm = next(iter(arms.items()))
    _require(isinstance(arm, dict), f"{role}: arm must be an object")
    _require_exact(arm.get("branch_id"), branch_id, f"{role}.arm.branch_id")
    _require_exact(arm.get("seed"), EXPECTED_TRAINING_SEED, f"{role}.arm.seed")
    _require_exact(arm.get("mode"), mode, f"{role}.arm.mode")
    _require(arm.get("complete") is True, f"{role}: training arm is incomplete")
    _require_exact(
        arm.get("iterations_parsed"), EXPECTED_TRAINING_ITERATIONS, f"{role}.iterations_parsed"
    )
    _require_exact(
        arm.get("curriculum_rows"), EXPECTED_TRAINING_ITERATIONS, f"{role}.curriculum_rows"
    )
    _require_exact(arm.get("actuator_groups_swapped"), 5, f"{role}.actuator_groups_swapped")
    _require_exact(arm.get("consolidation_rows"), 0, f"{role}.consolidation_rows")
    _require(arm.get("checkpoint_exported") is True, f"{role}: checkpoint was not exported")
    _require_exact(
        set(arm.get("scalable_terms") or []), EXPECTED_SCALABLE_TERMS, f"{role}.scalable_terms"
    )
    _require_exact((runtime[branch_id] or {}).get("exit_code"), 0, f"{role}.runtime.exit_code")
    delay = arm.get("live_delay_final") or {}
    expected_live_max = 8 if role == "fresh_fixed" else 12
    _require_exact(
        delay.get("action_delay_actuator_groups"), 5, f"{role}.live_delay.actuator_groups"
    )
    _require_exact(
        delay.get("action_delay_num_lags"),
        EXPECTED_TRAINING_ENVS * 5,
        f"{role}.live_delay.num_lags",
    )
    _require_exact(delay.get("action_delay_min_steps"), 0, f"{role}.live_delay.min_steps")
    _require_exact(
        delay.get("action_delay_max_steps"), expected_live_max, f"{role}.live_delay.max_steps"
    )
    _require(
        isinstance(delay.get("action_delay_nonzero_fraction"), (int, float))
        and 0.0 < float(delay["action_delay_nonzero_fraction"]) <= 1.0,
        f"{role}.live_delay.nonzero_fraction is invalid",
    )
    histogram = delay.get("action_delay_histogram")
    _require(
        isinstance(histogram, list) and len(histogram) == expected_live_max + 1,
        f"{role}.live_delay.histogram does not cover 0..{expected_live_max}",
    )
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in histogram
        ),
        f"{role}.live_delay.histogram has invalid counts",
    )
    _require_exact(sum(histogram), EXPECTED_TRAINING_ENVS * 5, f"{role}.live_delay.histogram total")

    checkpoint_raw = arm.get("checkpoint")
    _require(
        isinstance(checkpoint_raw, str) and checkpoint_raw, f"{role}: checkpoint path is missing"
    )
    checkpoint = _materialized(checkpoint_raw)
    _require(checkpoint.is_file(), f"{role}: checkpoint file is missing: {checkpoint}")
    curriculum_raw = arm.get("curriculum_path")
    _require(
        isinstance(curriculum_raw, str) and curriculum_raw,
        f"{role}: curriculum path is missing",
    )
    curriculum = _audit_curriculum(role, _materialized(curriculum_raw))

    spec = arm.get("arm_spec") or {}
    fixed_lambda = 1.0 if role == "fresh_fixed" else 1.5
    allow_extrapolation = role != "fresh_fixed"
    spread_strata = 8 if role == "fixed_u150" else 1
    expected_anchor_seed = EXPECTED_TRAINING_SEED if role == "fixed_u150" else None
    fixed_fields = {
        "curriculum_mode": "fixed",
        "anchor_ratio": 0.0,
        "anchor_seed": expected_anchor_seed,
        "yoked_source": None,
        "yoked_cross_seed": False,
        "term_lambda_overrides": {},
        "spread_strata": spread_strata,
        "return_guard": "absolute",
        "fixed_lambda": fixed_lambda,
        "allow_extrapolation": allow_extrapolation,
        "physical_clamp": ["physics_material"] if allow_extrapolation else None,
        "signal": "gap",
        "margin": None,
        "term_lambda_caps": {},
        "max_delay_steps": EXPECTED_MAX_DELAY_STEPS,
    }
    for key, expected in fixed_fields.items():
        _require_exact(spec.get(key), expected, f"{role}.arm_spec.{key}")
    _require_close(arm.get("final_lambda"), fixed_lambda, f"{role}.final_lambda")

    if role == "fixed_u150":
        _require_exact(
            spec.get("stratum_sizes"), EXPECTED_STRATUM_SIZES, f"{role}.arm_spec.stratum_sizes"
        )
        _require_float_sequence(
            spec.get("stratum_lambdas"),
            EXPECTED_STRATUM_LAMBDAS,
            f"{role}.arm_spec.stratum_lambdas",
        )
        _require_close(spec.get("top_fraction"), 0.75, f"{role}.arm_spec.top_fraction")
        tace = arm.get("tace_final") or {}
        _audit_fixed_u150_tace(
            tace,
            f"{role}.tace_final",
            require_positive_counts=True,
            expected_anchor_params=None,
        )
        _require_nested_close(tace, curriculum["final_tace"], f"{role}.tace_final/curriculum")
    else:
        _require(spec.get("stratum_sizes") is None, f"{role}: unexpected stratum_sizes")
        _require(spec.get("stratum_lambdas") is None, f"{role}: unexpected stratum_lambdas")
        _require(spec.get("top_fraction") is None, f"{role}: unexpected top_fraction")
        _require(arm.get("tace_final") is None, f"{role}: unexpected TACE dispatcher telemetry")

    return {
        "role": role,
        "mode": mode,
        "path": str(path),
        "sha256": sha256(path),
        "experiment_id": receipt.get("experiment_id"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "curriculum": {key: value for key, value in curriculum.items() if key != "final_tace"},
    }


def audit_historical_training(path: Path, *, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the preregistered legacy fixed cell used only for bridge validity."""
    frozen = preregistration["frozen_inputs"]
    path = Path(path).resolve()
    _require_exact(
        path, Path(frozen["historical_fixed_training"]["path"]), "historical training path"
    )
    _require_file_hash(path, frozen["historical_fixed_training"]["sha256"], "historical training")
    receipt = load_json(path)
    _require(
        receipt.get("kind")
        in (
            "lucid_historical_training_cell_bridge",
            "lucid_three_arm_training_comparison",
        ),
        "historical training kind is invalid",
    )
    _require_exact(receipt.get("schema_version"), 1, "historical training schema_version")
    _require_verified(receipt.get("verified"), "historical training.verified")
    config = receipt.get("config") or {}
    for key, expected in {
        "checkpoint": None,
        "from_scratch": True,
        "num_envs": EXPECTED_TRAINING_ENVS,
        "iterations": EXPECTED_TRAINING_ITERATIONS,
        "warmup_iterations": EXPECTED_WARMUP_ITERATIONS,
        "seeds": [EXPECTED_TRAINING_SEED],
        "modes": ["fixed"],
        "termination_thresholds": "default",
        "consolidation_fraction": 0,
    }.items():
        _require_exact(config.get(key), expected, f"historical training.config.{key}")
    arms = receipt.get("arms")
    _require(isinstance(arms, dict) and len(arms) == 1, "historical training must have one arm")
    branch_id, arm = next(iter(arms.items()))
    _require(isinstance(arm, dict), "historical training arm must be an object")
    for key, expected in {
        "seed": EXPECTED_TRAINING_SEED,
        "mode": "fixed",
        "complete": True,
        "checkpoint_exported": True,
        "iterations_parsed": EXPECTED_TRAINING_ITERATIONS,
        "curriculum_rows": EXPECTED_TRAINING_ITERATIONS,
    }.items():
        _require_exact(arm.get(key), expected, f"historical training.arm.{key}")
    _require_exact(arm.get("branch_id"), branch_id, "historical training.arm.branch_id")
    spec = arm.get("arm_spec") or {}
    for key, expected in {
        "curriculum_mode": "fixed",
        "anchor_ratio": 0,
        "spread_strata": 1,
        "fixed_lambda": 1,
        "allow_extrapolation": False,
    }.items():
        _require_exact(spec.get(key), expected, f"historical training.arm_spec.{key}")
    checkpoint = _materialized(arm.get("checkpoint"))
    curriculum = _audit_curriculum("historical_fixed", _materialized(arm.get("curriculum_path")))
    _require_exact(
        checkpoint,
        Path(frozen["historical_fixed_checkpoint"]["path"]),
        "historical training checkpoint path",
    )
    _require_file_hash(
        checkpoint, frozen["historical_fixed_checkpoint"]["sha256"], "historical checkpoint"
    )
    return {
        "role": "historical_fixed",
        "mode": "fixed",
        "path": str(path),
        "sha256": sha256(path),
        "experiment_id": receipt.get("experiment_id"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "curriculum": {key: value for key, value in curriculum.items() if key != "final_tace"},
    }


def audit_freeze_manifest(
    role: str,
    path: Path,
    *,
    training: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the immutable checkpoint bundle joining training to evaluation."""
    mode = ROLE_MODE[role]
    path = Path(path).resolve()
    if role == "historical_fixed":
        frozen = preregistration["frozen_inputs"]["historical_fixed_freeze_manifest"]
        _require_exact(path, Path(frozen["path"]), "historical freeze manifest path")
        _require_file_hash(path, frozen["sha256"], "historical freeze manifest")
        _require_read_only(path, "historical freeze manifest")
    manifest = load_json(path)
    for key, expected in {
        "kind": "lucid_frozen_training_checkpoint",
        "schema_version": 1,
        "state": "frozen_for_evaluation",
        "evaluation_only": True,
        "seed": EXPECTED_TRAINING_SEED,
        "mode": mode,
        "iterations": EXPECTED_TRAINING_ITERATIONS,
        "resume_forbidden": True,
    }.items():
        _require_exact(manifest.get(key), expected, f"{role}.freeze.{key}")
    _require_verified(manifest.get("verified"), f"{role}.freeze.verified")
    if role != "historical_fixed":
        code = manifest.get("code") or {}
        _require_exact(
            code.get("git_sha"), preregistration["git_sha"], f"{role}.freeze.code.git_sha"
        )
        _require_clean_status(code.get("git_status_short"), f"{role}.freeze.code.git_status_short")

    sections: dict[str, dict[str, Any]] = {}
    for name in ("checkpoint", "config", "curriculum", "final_capsule", "training_receipt"):
        section = manifest.get(name)
        _require(isinstance(section, dict), f"{role}.freeze.{name} is missing")
        raw_path = section.get("path")
        _require(isinstance(raw_path, str) and raw_path, f"{role}.freeze.{name}.path is missing")
        materialized = _materialized(raw_path)
        digest = _require_sha(section.get("sha256"), f"{role}.freeze.{name}.sha256")
        _require_file_hash(materialized, digest, f"{role}.freeze.{name}")
        if section.get("size_bytes") is not None:
            _require_exact(
                section.get("size_bytes"),
                materialized.stat().st_size,
                f"{role}.freeze.{name}.size_bytes",
            )
        sections[name] = {"path": str(materialized), "sha256": digest}

    _require(
        manifest["checkpoint"].get("read_only") is True,
        f"{role}.freeze checkpoint is not read-only",
    )
    _require_read_only(Path(sections["checkpoint"]["path"]), f"{role}.freeze checkpoint")
    _require_exact(
        Path(sections["training_receipt"]["path"]),
        Path(training["path"]),
        f"{role}.freeze/training receipt path",
    )
    _require_exact(
        sections["training_receipt"]["sha256"],
        training["sha256"],
        f"{role}.freeze/training receipt sha",
    )
    _require_exact(
        Path(sections["checkpoint"]["path"]),
        Path(training["checkpoint"]),
        f"{role}.freeze/training checkpoint path",
    )
    _require_exact(
        sections["checkpoint"]["sha256"],
        training["checkpoint_sha256"],
        f"{role}.freeze/training checkpoint sha",
    )
    _require_exact(
        Path(sections["curriculum"]["path"]),
        Path(training["curriculum"]["path"]),
        f"{role}.freeze/training curriculum path",
    )
    _require_exact(
        sections["curriculum"]["sha256"],
        training["curriculum"]["sha256"],
        f"{role}.freeze/training curriculum sha",
    )
    _require_exact(
        manifest["curriculum"].get("rows"),
        EXPECTED_TRAINING_ITERATIONS,
        f"{role}.freeze.curriculum.rows",
    )
    if role == "historical_fixed":
        frozen_inputs = preregistration["frozen_inputs"]
        for section, key in (
            ("checkpoint", "historical_fixed_checkpoint"),
            ("config", "historical_fixed_config"),
            ("training_receipt", "historical_fixed_training"),
        ):
            _require_exact(
                sections[section], frozen_inputs[key], f"historical freeze.{section} prereg binding"
            )
    return {"role": role, "path": str(path), "sha256": sha256(path), **sections}


def _audit_evaluation_config(
    role: str,
    protocol: Mapping[str, Any],
    checkpoint: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    config = protocol.get("resolved_training_config")
    _require(isinstance(config, dict), f"{role}: resolved_training_config is missing")
    source_raw = config.get("source")
    _require(
        isinstance(source_raw, str) and source_raw, f"{role}: training config source is missing"
    )
    source = _materialized(source_raw)
    source_sha = _require_sha(config.get("sha256"), f"{role}.training_config.sha256")
    _require_file_hash(source, source_sha, f"{role}.training_config.source")
    installed = config.get("installed")
    _require(
        isinstance(installed, list) and len(installed) == 1,
        f"{role}: expected one installed config",
    )
    installed_path = _materialized(installed[0])
    _require_exact(
        installed_path,
        (checkpoint.parent / "config.yaml").resolve(),
        f"{role}.training_config.installed path",
    )
    _require_file_hash(installed_path, source_sha, f"{role}.training_config.installed")
    _require_exact(source, Path(freeze["config"]["path"]), f"{role}.training_config/freeze path")
    _require_exact(source_sha, freeze["config"]["sha256"], f"{role}.training_config/freeze sha")
    return {"source": str(source), "sha256": source_sha, "installed": str(installed_path)}


def _audit_delay(delay: Mapping[str, Any], expected_steps: int, label: str) -> None:
    expected_lags = EXPECTED_EVALUATION_ENVS * 5
    for key, expected in {
        "action_delay_actuator_groups": 5,
        "action_delay_num_lags": expected_lags,
        "action_delay_min_steps": expected_steps,
        "action_delay_max_steps": expected_steps,
    }.items():
        _require_exact(delay.get(key), expected, f"{label}.{key}")
    _require_close(delay.get("action_delay_mean_steps"), float(expected_steps), f"{label}.mean")
    _require_close(
        delay.get("action_delay_nonzero_fraction"),
        0.0 if expected_steps == 0 else 1.0,
        f"{label}.nonzero_fraction",
    )
    histogram = delay.get("action_delay_histogram")
    expected_histogram = [0] * expected_steps + [expected_lags]
    _require_exact(histogram, expected_histogram, f"{label}.histogram")
    process_histogram = delay.get("action_delay_process_histogram")
    if process_histogram is not None:
        assignments = delay.get("action_delay_process_assignments")
        _require(
            isinstance(assignments, int) and not isinstance(assignments, bool) and assignments > 0,
            f"{label}.process_assignments is invalid",
        )
        _require_exact(
            process_histogram,
            [0] * expected_steps + [assignments],
            f"{label}.process_histogram",
        )
        _require_close(
            delay.get("action_delay_process_mean_steps"),
            float(expected_steps),
            f"{label}.process_mean",
        )


def _audit_metrics_file(
    role: str,
    run_id: str,
    preset: str,
    path: Path,
    *,
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file(), f"{role}.{preset}.metrics_path is missing: {path}")
    metrics = load_json(path)
    _require_exact(metrics.get("eval/protocol/preset_id"), preset, f"{role}.{preset}.raw preset")
    _require_exact(metrics.get("eval/protocol/branch_id"), run_id, f"{role}.{preset}.raw branch")
    _require_exact(
        set(metrics.get("eval/protocol/active_dr_terms") or []),
        EXPECTED_SCALABLE_TERMS,
        f"{role}.{preset}.raw active_dr_terms",
    )
    ranges = metrics.get("eval/protocol/dr_ranges")
    _require(isinstance(ranges, dict), f"{role}.{preset}.raw dr_ranges is missing")
    _require_exact(set(ranges), EXPECTED_SCALABLE_TERMS, f"{role}.{preset}.raw dr_ranges terms")

    expected_steps = 0 if preset in PHYSICS_LEVELS else LATENCY_STEPS[preset]
    fixed_report = metrics.get("eval/protocol/fixed_latency_report")
    _require(isinstance(fixed_report, dict), f"{role}.{preset}.raw fixed_latency_report is missing")
    _require_close(
        fixed_report.get("requested_steps"),
        float(expected_steps),
        f"{role}.{preset}.fixed requested",
    )
    _require_exact(
        fixed_report.get("pinned_terms"),
        ["randomize_action_delay"],
        f"{role}.{preset}.fixed pinned_terms",
    )
    _require_exact(
        metrics.get("eval/protocol/fixed_latency_steps"),
        expected_steps,
        f"{role}.{preset}.raw fixed_latency_steps",
    )
    scale = PHYSICS_LEVELS.get(preset)
    if scale is None:
        _require(
            metrics.get("eval/protocol/non_latency_dr_scale") is None,
            f"{role}.{preset}: latency cell has non-latency DR scale",
        )
        _require(
            metrics.get("eval/protocol/dr_scale_report") is None,
            f"{role}.{preset}: latency cell has a DR scale report",
        )
    else:
        _require_close(
            metrics.get("eval/protocol/non_latency_dr_scale"),
            scale,
            f"{role}.{preset}.raw non_latency_dr_scale",
        )
        report = metrics.get("eval/protocol/dr_scale_report")
        _require(isinstance(report, dict), f"{role}.{preset}.raw dr_scale_report is missing")
        _require_close(report.get("lambda_value"), scale, f"{role}.{preset}.raw scale report")
        _require_exact(
            set(report.get("scaled_terms") or []),
            EXPECTED_NON_LATENCY_TERMS,
            f"{role}.{preset}.raw scaled_terms",
        )
        _require_exact(report.get("num_scaled"), 5, f"{role}.{preset}.raw num_scaled")
        _require_exact(
            report.get("skipped_startup_terms"), [], f"{role}.{preset}.raw skipped_startup_terms"
        )
        _require_exact(
            report.get("skipped_unknown_params"), [], f"{role}.{preset}.raw skipped_unknown_params"
        )

    delay = {
        key.removeprefix("eval/delay/"): value
        for key, value in metrics.items()
        if key.startswith("eval/delay/")
    }
    _audit_delay(delay, expected_steps, f"{role}.{preset}.raw delay")

    bundle = metrics.get("eval/all_metrics_dict")
    _require(isinstance(bundle, dict), f"{role}.{preset}.raw all_metrics_dict is missing")
    motion_keys = bundle.get("motion_keys")
    terminated = bundle.get("terminated")
    progress = bundle.get("progress")
    for name, values in (
        ("motion_keys", motion_keys),
        ("terminated", terminated),
        ("progress", progress),
    ):
        _require(isinstance(values, list), f"{role}.{preset}.raw {name} is not a list")
        _require_exact(len(values), EXPECTED_EVALUATION_ENVS, f"{role}.{preset}.raw {name} length")
    _require(
        all(isinstance(key, str) and key for key in motion_keys),
        f"{role}.{preset}.raw motion_keys contains invalid entries",
    )
    _require_exact(
        len(set(motion_keys)), EXPECTED_EVALUATION_ENVS, f"{role}.{preset}.raw unique motion keys"
    )
    motion_digest = hashlib.sha256(("\n".join(sorted(motion_keys)) + "\n").encode()).hexdigest()
    _require_exact(motion_digest, panel["alias_keys_sha256"], f"{role}.{preset}.raw panel digest")
    _require(
        all(isinstance(value, bool) for value in terminated),
        f"{role}.{preset}.raw terminated contains non-booleans",
    )
    progress_values = [
        _require_rate(value, f"{role}.{preset}.raw progress[{index}]")
        for index, value in enumerate(progress)
    ]
    success = [0 if value else 1 for value in terminated]
    for index, (succeeded, value) in enumerate(zip(success, progress_values, strict=True)):
        _require(
            bool(succeeded) == (value >= 1.0),
            f"{role}.{preset}.raw episode {index} termination/progress disagree",
        )
    success_rate = sum(success) / EXPECTED_EVALUATION_ENVS
    progress_rate = sum(progress_values) / EXPECTED_EVALUATION_ENVS
    _require_close(
        metrics.get("eval/success/success_rate"), success_rate, f"{role}.{preset}.raw success rate"
    )
    _require_close(
        metrics.get("eval/success/progress_rate"),
        progress_rate,
        f"{role}.{preset}.raw progress rate",
    )
    failed_indices = [index for index, value in enumerate(terminated) if value]
    _require_exact(metrics.get("failed_idxes"), failed_indices, f"{role}.{preset}.raw failed_idxes")
    _require_exact(
        metrics.get("failed_keys"),
        [motion_keys[index] for index in failed_indices],
        f"{role}.{preset}.raw failed_keys",
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "success_rate": success_rate,
        "progress_rate": progress_rate,
        "failed_count": len(failed_indices),
        "active_dr_terms": sorted(EXPECTED_SCALABLE_TERMS),
        "dr_ranges": ranges,
        "delay": delay,
        "motion_keys_sha256": motion_digest,
    }


def _audit_live_dr_ladder(role: str, raw_by_preset: Mapping[str, Mapping[str, Any]]) -> None:
    baseline = raw_by_preset["phys_100"]["dr_ranges"]
    for preset, scale in PHYSICS_LEVELS.items():
        observed = raw_by_preset[preset]["dr_ranges"]
        for term in sorted(EXPECTED_NON_LATENCY_TERMS):
            expected = DS.scaled_term_params(baseline[term], scale, allow_extrapolation=True)
            expected, _ = DS.clamp_params_physical(expected)
            _require_nested_close(observed[term], expected, f"{role}.{preset}.live DR {term}")
        _require_nested_close(
            observed["randomize_action_delay"],
            {"delay_range": [0.0, 0.0]},
            f"{role}.{preset}.live DR randomize_action_delay",
        )
    nominal = raw_by_preset["phys_000"]["dr_ranges"]
    for preset, steps in LATENCY_STEPS.items():
        observed = raw_by_preset[preset]["dr_ranges"]
        for term in sorted(EXPECTED_NON_LATENCY_TERMS):
            _require_nested_close(observed[term], nominal[term], f"{role}.{preset}.clean DR {term}")
        _require_nested_close(
            observed["randomize_action_delay"],
            {"delay_range": [float(steps), float(steps)]},
            f"{role}.{preset}.live DR randomize_action_delay",
        )


def _canonical_json(value: Any) -> str:
    """Canonical JSON for strict cross-role instrument identity.

    Object key order is immaterial. Array order and JSON scalar spelling remain
    binding, so ``1`` and ``1.0`` are deliberately different: all four roles
    run the same evaluator/config and should materialize the same live state,
    not merely numerically close states.
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("live dr_ranges are not finite canonical JSON") from error


def _audit_cross_role_live_dr(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every preset to materialize identical DR state in every role."""
    reference_role = EVALUATION_ROLES[0]
    hashes: dict[str, str] = {}
    for preset in EXPECTED_PRESETS:
        reference = _canonical_json(evaluations[reference_role]["metrics"][preset]["dr_ranges"])
        hashes[preset] = hashlib.sha256(reference.encode()).hexdigest()
        for role in EVALUATION_ROLES[1:]:
            observed = _canonical_json(evaluations[role]["metrics"][preset]["dr_ranges"])
            _require(
                observed == reference,
                f"{role}.{preset}.dr_ranges differs from {reference_role} under canonical JSON",
            )
    return {
        "passed": True,
        "roles": list(EVALUATION_ROLES),
        "reference_role": reference_role,
        "presets": list(EXPECTED_PRESETS),
        "canonical_sha256_by_preset": hashes,
        "canonical_semantics": (
            "object keys sorted; array order, scalar JSON representation, values, and topology exact"
        ),
    }


def audit_evaluation(
    role: str,
    path: Path,
    *,
    preregistration: Mapping[str, Any],
    panel: Mapping[str, Any],
    training: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete, single-checkpoint 15-cell evaluation receipt."""
    _require(role in EVALUATION_ROLES, f"unknown evaluation role: {role}")
    mode = ROLE_MODE[role]
    path = Path(path).resolve()
    receipt = load_json(path)
    _require_exact(
        receipt.get("kind"), "lucid_frozen_checkpoint_robustness_evaluation", f"{role}.kind"
    )
    _require_exact(receipt.get("schema_version"), 1, f"{role}.schema_version")
    _require_exact(
        receipt.get("launcher_sha256"),
        preregistration["evaluator_sha256"],
        f"{role}.launcher_sha256",
    )
    _require_verified(receipt.get("verified"), f"{role}.verified")
    _require_exact(receipt.get("git_sha"), preregistration["git_sha"], f"{role}.git_sha")
    _require_clean_status(receipt.get("git_status_short"), f"{role}.git_status_short")

    protocol = receipt.get("protocol") or {}
    _require_exact(protocol.get("num_envs"), EXPECTED_EVALUATION_ENVS, f"{role}.protocol.num_envs")
    _require_exact(
        protocol.get("checkpoint_seeds"), [EXPECTED_TRAINING_SEED], f"{role}.checkpoint_seeds"
    )
    _require_exact(
        protocol.get("evaluation_seed_by_checkpoint_seed"),
        {str(EXPECTED_TRAINING_SEED): EXPECTED_EVALUATION_SEED},
        f"{role}.evaluation_seed mapping",
    )
    _require_exact(protocol.get("modes"), [mode], f"{role}.protocol.modes")
    _require_exact(protocol.get("presets"), _expected_preset_metadata(), f"{role}.protocol.presets")
    _require_exact(
        protocol.get("max_delay_capacity_steps"),
        EXPECTED_MAX_DELAY_STEPS,
        f"{role}.max_delay_capacity_steps",
    )
    _require_exact(
        protocol.get("physics_step_ms"), EXPECTED_PHYSICS_STEP_MS, f"{role}.physics_step_ms"
    )
    _require(protocol.get("no_learning") is True, f"{role}: evaluator was not frozen-policy")

    suite = protocol.get("suite") or {}
    _require_exact(
        suite.get("motion_count"), EXPECTED_EVALUATION_ENVS, f"{role}.suite.motion_count"
    )
    _require_exact(
        suite.get("motion_keys_sha256"),
        panel["alias_keys_sha256"],
        f"{role}.suite.motion_keys_sha256",
    )
    for key in ("pool_sha256", "split_sha256", "partition"):
        _require_exact(suite.get(key), panel[key], f"{role}.suite.{key}")
    _require_exact(suite.get("split_linkage"), "replicate-panel", f"{role}.suite.split_linkage")
    panel_link = suite.get("replicate_panel") or {}
    for key in ("motion_key", "source_clip_sha256", "replicates", "alias_keys_sha256"):
        _require_exact(panel_link.get(key), panel[key], f"{role}.suite.replicate_panel.{key}")
    linked_panel_raw = panel_link.get("receipt")
    _require(
        isinstance(linked_panel_raw, str) and linked_panel_raw,
        f"{role}: panel receipt link is missing",
    )
    linked_panel = _materialized(linked_panel_raw)
    _require(
        _same_file_or_content(linked_panel, Path(panel["path"])),
        f"{role}: evaluator references a different panel receipt: {linked_panel}",
    )

    runs = receipt.get("runs")
    _require(isinstance(runs, dict), f"{role}: runs must be an object")
    _require_exact(len(runs), len(EXPECTED_PRESETS), f"{role}.runs count")
    values: dict[tuple[str, str], float] = {}
    seen_presets: set[str] = set()
    checkpoint_paths: set[Path] = set()
    checkpoint_raw_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    metrics_paths: set[Path] = set()
    cells: set[tuple[str, int, int, str]] = set()
    raw_by_preset: dict[str, dict[str, Any]] = {}
    for run_id, run in runs.items():
        _require(isinstance(run, dict), f"{role}.runs.{run_id}: must be an object")
        _require_exact(
            run.get("checkpoint_seed"), EXPECTED_TRAINING_SEED, f"{role}.{run_id}.checkpoint_seed"
        )
        _require_exact(
            run.get("evaluation_seed"), EXPECTED_EVALUATION_SEED, f"{role}.{run_id}.evaluation_seed"
        )
        _require_exact(run.get("mode"), mode, f"{role}.{run_id}.mode")
        preset = run.get("preset")
        _require(preset in EXPECTED_PRESETS, f"{role}.{run_id}: unexpected preset {preset!r}")
        _require(preset not in seen_presets, f"{role}: duplicate preset {preset}")
        seen_presets.add(preset)
        cell = (role, EXPECTED_TRAINING_SEED, EXPECTED_EVALUATION_SEED, preset)
        _require(cell not in cells, f"{role}: duplicate cell {cell}")
        cells.add(cell)
        _require(run.get("complete") is True, f"{role}.{preset}: run is incomplete")
        runtime = run.get("runtime") or {}
        _require_exact(runtime.get("exit_code"), 0, f"{role}.{preset}.runtime.exit_code")
        summary = run.get("summary") or {}
        _require_exact(
            summary.get("motion_count"), EXPECTED_EVALUATION_ENVS, f"{role}.{preset}.motion_count"
        )
        for metric in METRICS:
            values[(metric, preset)] = _require_rate(
                summary.get(metric), f"{role}.{preset}.{metric}"
            )

        metrics_raw = run.get("metrics_path")
        _require(
            isinstance(metrics_raw, str) and metrics_raw,
            f"{role}.{preset}: metrics_path is missing",
        )
        metrics_path = _materialized(metrics_raw)
        _require(
            metrics_path not in metrics_paths, f"{role}: duplicate metrics_path {metrics_path}"
        )
        metrics_paths.add(metrics_path)
        raw = _audit_metrics_file(role, run_id, preset, metrics_path, panel=panel)
        raw_by_preset[preset] = raw
        for metric in METRICS:
            _require_close(
                summary.get(metric), raw[metric], f"{role}.{preset}.{metric} raw/summary"
            )
        _require_exact(
            summary.get("failed_count"), raw["failed_count"], f"{role}.{preset}.failed_count"
        )
        _require_exact(
            set(summary.get("active_dr_terms") or []),
            EXPECTED_SCALABLE_TERMS,
            f"{role}.{preset}.summary.active_dr_terms",
        )
        _require_nested_close(
            summary.get("dr_ranges"), raw["dr_ranges"], f"{role}.{preset}.summary.dr_ranges"
        )
        _require_nested_close(summary.get("delay"), raw["delay"], f"{role}.{preset}.summary.delay")

        checkpoint_raw = run.get("checkpoint")
        _require(
            isinstance(checkpoint_raw, str) and checkpoint_raw,
            f"{role}.{preset}: checkpoint is missing",
        )
        checkpoint = _materialized(checkpoint_raw)
        checkpoint_sha = _require_sha(
            run.get("checkpoint_sha256"), f"{role}.{preset}.checkpoint_sha256"
        )
        checkpoint_paths.add(checkpoint)
        checkpoint_raw_paths.add(checkpoint_raw)
        checkpoint_hashes.add(checkpoint_sha)

    _require_exact(seen_presets, set(EXPECTED_PRESETS), f"{role}.run presets")
    _require_exact(len(cells), len(EXPECTED_PRESETS), f"{role}.unique cells")
    _require_exact(len(checkpoint_paths), 1, f"{role}.checkpoint paths")
    _require_exact(len(checkpoint_raw_paths), 1, f"{role}.raw checkpoint paths")
    _require_exact(len(checkpoint_hashes), 1, f"{role}.checkpoint hashes")
    _require_exact(len(metrics_paths), len(EXPECTED_PRESETS), f"{role}.unique metrics paths")
    _audit_live_dr_ladder(role, raw_by_preset)
    checkpoint = next(iter(checkpoint_paths))
    checkpoint_raw = next(iter(checkpoint_raw_paths))
    checkpoint_sha = next(iter(checkpoint_hashes))
    _require_file_hash(checkpoint, checkpoint_sha, f"{role}.checkpoint")

    before = receipt.get("checkpoint_sha256_before")
    after = receipt.get("checkpoint_sha256_after")
    _require_exact(before, {checkpoint_raw: checkpoint_sha}, f"{role}.checkpoint_sha256_before")
    _require_exact(after, before, f"{role}.checkpoint_sha256_after")
    _require_exact(checkpoint, Path(freeze["checkpoint"]["path"]), f"{role}.checkpoint/freeze path")
    _require_exact(checkpoint_sha, freeze["checkpoint"]["sha256"], f"{role}.checkpoint/freeze sha")
    config_identity = _audit_evaluation_config(role, protocol, checkpoint, freeze)

    aggregate = receipt.get("mode_summary")
    _require(isinstance(aggregate, dict), f"{role}: mode_summary must be an object")
    _require_exact(set(aggregate), set(EXPECTED_PRESETS), f"{role}.mode_summary presets")
    for preset in EXPECTED_PRESETS:
        modes = aggregate[preset]
        _require(isinstance(modes, dict), f"{role}.mode_summary.{preset} must be an object")
        _require_exact(set(modes), {mode}, f"{role}.mode_summary.{preset} modes")
        block = modes[mode]
        _require_exact(block.get("num_runs"), 1, f"{role}.{preset}.aggregate.num_runs")
        metrics = block.get("metrics") or {}
        for metric in METRICS:
            metric_block = metrics.get(metric) or {}
            recorded = metric_block.get("per_checkpoint_seed")
            run_value = values[(metric, preset)]
            _require_exact(
                set(recorded or {}),
                {str(EXPECTED_TRAINING_SEED)},
                f"{role}.{preset}.{metric}.aggregate seeds",
            )
            aggregate_value = _require_rate(
                recorded[str(EXPECTED_TRAINING_SEED)], f"{role}.{preset}.{metric}.aggregate"
            )
            _require_close(aggregate_value, run_value, f"{role}.{preset}.{metric}.run/aggregate")
            _require_close(metric_block.get("mean"), run_value, f"{role}.{preset}.{metric}.mean")
            _require(
                metric_block.get("sample_std") is None,
                f"{role}.{preset}.{metric}: one-seed std must be null",
            )

    linked_training_raw = receipt.get("training_receipt")
    _require(
        isinstance(linked_training_raw, str) and linked_training_raw,
        f"{role}: training_receipt is missing",
    )
    linked_training = _materialized(linked_training_raw)
    _require_exact(
        linked_training,
        Path(training["path"]),
        f"{role}: evaluation is not linked to its role-specific training receipt",
    )
    _require_exact(sha256(linked_training), training["sha256"], f"{role}.training receipt sha")
    _require_exact(
        receipt.get("training_experiment_id"),
        training["experiment_id"],
        f"{role}.training_experiment_id",
    )
    _require_exact(
        checkpoint, Path(training["checkpoint"]), f"{role}.training/evaluation checkpoint"
    )
    _require_exact(
        checkpoint_sha,
        training["checkpoint_sha256"],
        f"{role}.training/evaluation checkpoint sha",
    )

    return {
        "role": role,
        "mode": mode,
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "training_config": config_identity,
        "metrics": {preset: raw_by_preset[preset] for preset in EXPECTED_PRESETS},
        "values": values,
        "cells": cells,
    }


def normalized_auc(
    values: Mapping[tuple[str, str], float], metric: str, grid: Sequence[tuple[str, float]]
) -> float:
    points = [(x, values[(metric, preset)]) for preset, x in grid]
    width = points[-1][0] - points[0][0]
    _require(width > 0.0, "AUC grid has no width")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / width


def profile(values: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        result[metric] = {
            "in_envelope_auc": normalized_auc(values, metric, IN_ENVELOPE_GRID),
            "frontier_auc": normalized_auc(values, metric, FRONTIER_GRID),
            "nominal_phys_000": values[(metric, "phys_000")],
            "lat_50ms": values[(metric, "lat_50ms")],
            "lat_60ms": values[(metric, "lat_60ms")],
        }
    result["mean_hard_success"] = sum(
        values[("success_rate", preset)] for preset in HARD_SUCCESS_PRESETS
    ) / len(HARD_SUCCESS_PRESETS)
    return result


def _delta(
    treatment: Mapping[str, Any], reference: Mapping[str, Any], metric: str, endpoint: str
) -> float:
    return float(treatment[metric][endpoint]) - float(reference[metric][endpoint])


def bridge_decision(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    historical = profiles["historical_fixed"]
    fresh = profiles["fresh_fixed"]
    deltas = {
        "frontier_success_auc": _delta(fresh, historical, "success_rate", "frontier_auc"),
        "frontier_progress_auc": _delta(fresh, historical, "progress_rate", "frontier_auc"),
        "in_envelope_success_auc": _delta(fresh, historical, "success_rate", "in_envelope_auc"),
        "in_envelope_progress_auc": _delta(fresh, historical, "progress_rate", "in_envelope_auc"),
        "lat_50ms_success": _delta(fresh, historical, "success_rate", "lat_50ms"),
    }
    limits = {
        "frontier_success_auc": BRIDGE_FRONTIER_TOLERANCE,
        "frontier_progress_auc": BRIDGE_FRONTIER_TOLERANCE,
        "in_envelope_success_auc": BRIDGE_IN_ENVELOPE_TOLERANCE,
        "in_envelope_progress_auc": BRIDGE_IN_ENVELOPE_TOLERANCE,
        "lat_50ms_success": BRIDGE_LAT50_TOLERANCE,
    }
    checks = {key: abs(deltas[key]) <= limit + FLOAT_TOLERANCE for key, limit in limits.items()}
    return {
        "fresh_minus_historical": deltas,
        "absolute_tolerances": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }


def candidate_decision(role: str, profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate = profiles[role]
    fresh = profiles["fresh_fixed"]
    deltas = {
        "frontier_success_auc": _delta(candidate, fresh, "success_rate", "frontier_auc"),
        "frontier_progress_auc": _delta(candidate, fresh, "progress_rate", "frontier_auc"),
        "in_envelope_success_auc": _delta(candidate, fresh, "success_rate", "in_envelope_auc"),
        "in_envelope_progress_auc": _delta(candidate, fresh, "progress_rate", "in_envelope_auc"),
        "mean_hard_success": float(candidate["mean_hard_success"])
        - float(fresh["mean_hard_success"]),
        "lat_60ms_success": _delta(candidate, fresh, "success_rate", "lat_60ms"),
    }
    checks = {
        "frontier_success_gain": deltas["frontier_success_auc"] + FLOAT_TOLERANCE
        >= CANDIDATE_FRONTIER_SUCCESS_GAIN,
        "frontier_progress_noninferiority": deltas["frontier_progress_auc"] + FLOAT_TOLERANCE
        >= CANDIDATE_FRONTIER_PROGRESS_FLOOR,
        "in_envelope_success_noninferiority": deltas["in_envelope_success_auc"] + FLOAT_TOLERANCE
        >= CANDIDATE_IN_ENVELOPE_FLOOR,
        "in_envelope_progress_noninferiority": deltas["in_envelope_progress_auc"] + FLOAT_TOLERANCE
        >= CANDIDATE_IN_ENVELOPE_FLOOR,
        "H_X1_mean_hard_success_gain": deltas["mean_hard_success"] + FLOAT_TOLERANCE
        >= H_X1_HARD_SUCCESS_GAIN,
    }
    return {
        "candidate": role,
        "candidate_minus_fresh_fixed": deltas,
        "binding_checks": checks,
        "lat_60ms_plus_0.05_nonbinding": deltas["lat_60ms_success"] + FLOAT_TOLERANCE
        >= LAT60_REPORTED_GAIN,
        "passed": all(checks.values()),
    }


def select_candidate(
    profiles: Mapping[str, Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    passing = [role for role in ("fixed_150", "fixed_u150") if candidates[role]["passed"]]
    if not passing:
        return {"selected": None, "reason": "no_candidate_passed", "passing_candidates": []}
    if len(passing) == 1:
        return {
            "selected": passing[0],
            "reason": "only_passing_candidate",
            "passing_candidates": passing,
        }

    frontier_delta = _delta(
        profiles["fixed_u150"], profiles["fixed_150"], "success_rate", "frontier_auc"
    )
    nominal_recovery = _delta(
        profiles["fixed_u150"], profiles["fixed_150"], "success_rate", "nominal_phys_000"
    )
    fixedu_frontier_within = frontier_delta + FLOAT_TOLERANCE >= -PREFERENCE_FRONTIER_GAIN
    preference_evidence = {
        "fixed_u150_minus_fixed_150_frontier_success_auc": frontier_delta,
        "fixed_u150_minus_fixed_150_phys_000_success": nominal_recovery,
        "fixed_u150_frontier_within_0.02": fixedu_frontier_within,
    }
    if frontier_delta > PREFERENCE_FRONTIER_GAIN + FLOAT_TOLERANCE:
        selected = "fixed_u150"
        reason = "fixed_u150_frontier_success_advantage_gt_0.02"
    elif nominal_recovery > PREFERENCE_NOMINAL_LOSS + FLOAT_TOLERANCE and fixedu_frontier_within:
        selected = "fixed_u150"
        reason = "fixed_u150_phys_000_gain_gt_0.01_with_frontier_within_0.02"
    else:
        selected = "fixed_150"
        reason = "default_preference_for_pure_support_extension"
    return {
        "selected": selected,
        "reason": reason,
        "passing_candidates": passing,
        "preference_evidence": preference_evidence,
    }


def analyze(
    *,
    historical_fixed: Path,
    fresh_fixed: Path,
    fixed_150: Path,
    fixed_u150: Path,
    fresh_fixed_training: Path,
    fixed_150_training: Path,
    fixed_u150_training: Path,
    preregistration: Path,
    expected_preregistration_sha: str,
    historical_fixed_freeze_manifest: Path,
    fresh_fixed_freeze_manifest: Path,
    fixed_150_freeze_manifest: Path,
    fixed_u150_freeze_manifest: Path,
) -> dict[str, Any]:
    """Audit all evidence, compute frozen endpoints, and apply the screen."""
    prereg = audit_preregistration(preregistration, expected_preregistration_sha)
    frozen_inputs = prereg["frozen_inputs"]
    panel = audit_panel(
        Path(frozen_inputs["panel_receipt"]["path"]),
        expected_path=Path(frozen_inputs["panel_receipt"]["path"]),
        expected_sha=frozen_inputs["panel_receipt"]["sha256"],
        expected_motion_path=Path(frozen_inputs["motion"]["path"]),
        expected_motion_sha=frozen_inputs["motion"]["sha256"],
    )
    h_r2 = audit_h_r2(
        Path(frozen_inputs["h_r2_analysis"]["path"]),
        expected_path=Path(frozen_inputs["h_r2_analysis"]["path"]),
        expected_sha=frozen_inputs["h_r2_analysis"]["sha256"],
    )
    training_paths = {
        "fresh_fixed": Path(fresh_fixed_training),
        "fixed_150": Path(fixed_150_training),
        "fixed_u150": Path(fixed_u150_training),
    }
    _require_exact(
        len({path.resolve() for path in training_paths.values()}),
        len(training_paths),
        "role-specific training receipt paths",
    )
    trainings = {
        role: audit_training(role, path, preregistration=prereg)
        for role, path in training_paths.items()
    }
    historical_training = audit_historical_training(
        Path(frozen_inputs["historical_fixed_training"]["path"]), preregistration=prereg
    )
    all_trainings = {"historical_fixed": historical_training, **trainings}
    freeze_paths = {
        "historical_fixed": Path(historical_fixed_freeze_manifest),
        "fresh_fixed": Path(fresh_fixed_freeze_manifest),
        "fixed_150": Path(fixed_150_freeze_manifest),
        "fixed_u150": Path(fixed_u150_freeze_manifest),
    }
    _require_exact(
        len({path.resolve() for path in freeze_paths.values()}),
        len(freeze_paths),
        "role-specific freeze manifest paths",
    )
    freezes = {
        role: audit_freeze_manifest(
            role, path, training=all_trainings[role], preregistration=prereg
        )
        for role, path in freeze_paths.items()
    }
    evaluation_paths = {
        "historical_fixed": Path(historical_fixed),
        "fresh_fixed": Path(fresh_fixed),
        "fixed_150": Path(fixed_150),
        "fixed_u150": Path(fixed_u150),
    }
    _require_exact(
        len({path.resolve() for path in evaluation_paths.values()}),
        len(evaluation_paths),
        "role-specific evaluation receipt paths",
    )
    evaluations = {
        role: audit_evaluation(
            role,
            path,
            preregistration=prereg,
            panel=panel,
            training=all_trainings[role],
            freeze=freezes[role],
        )
        for role, path in evaluation_paths.items()
    }
    _require_exact(
        len({evaluation["checkpoint"] for evaluation in evaluations.values()}),
        len(evaluations),
        "role-specific evaluation checkpoint paths",
    )
    _require_exact(
        len({evaluation["checkpoint_sha256"] for evaluation in evaluations.values()}),
        len(evaluations),
        "role-specific evaluation checkpoint hashes",
    )
    _require_exact(
        len({evaluation["sha256"] for evaluation in evaluations.values()}),
        len(evaluations),
        "role-specific evaluation receipt content hashes",
    )
    all_metrics_paths = [
        metric["path"]
        for evaluation in evaluations.values()
        for metric in evaluation["metrics"].values()
    ]
    _require_exact(
        len(set(all_metrics_paths)), len(all_metrics_paths), "combined unique raw metrics paths"
    )
    cross_role_live_dr = _audit_cross_role_live_dr(evaluations)
    all_cells = set().union(*(evaluation["cells"] for evaluation in evaluations.values()))
    _require_exact(len(all_cells), 60, "combined unique evaluation cells")

    profiles = {role: profile(evaluation["values"]) for role, evaluation in evaluations.items()}
    bridge = bridge_decision(profiles)
    candidates = {role: candidate_decision(role, profiles) for role in ("fixed_150", "fixed_u150")}
    selection = (
        select_candidate(profiles, candidates)
        if bridge["passed"]
        else {
            "selected": None,
            "reason": "fresh_fixed_historical_bridge_failed",
            "passing_candidates": [
                role for role, decision in candidates.items() if decision["passed"]
            ],
        }
    )
    status = (
        "invalid_bridge"
        if not bridge["passed"]
        else ("screen_pass" if selection["selected"] else "screen_fail")
    )

    return {
        "kind": "lucid_tier2_support_screen_analysis",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": _git(("rev-parse", "HEAD")),
        "git_status_short": _git(("status", "--short")),
        "claim_scope": {
            "status": "screening_only",
            "training_seeds": [EXPECTED_TRAINING_SEED],
            "evaluation_seeds": [EXPECTED_EVALUATION_SEED],
            "directional_claim_authorized": False,
            "superiority_claim_authorized": False,
            "interpretation": (
                "One-seed support screening only; a pass selects confirmatory work and "
                "does not establish superiority."
            ),
        },
        "inputs": {
            "preregistration": prereg,
            "panel_receipt": panel,
            "h_r2_analysis": h_r2,
            "training_receipts": all_trainings,
            "freeze_manifests": freezes,
            "evaluation_receipts": {
                role: {
                    key: value
                    for key, value in evaluation.items()
                    if key not in ("values", "cells")
                }
                for role, evaluation in evaluations.items()
            },
        },
        "instrument_audit": {
            "passed": True,
            "expected_evaluator_sha256": prereg["evaluator_sha256"],
            "expected_preregistration_sha256": prereg["sha256"],
            "evaluation_roles": list(EVALUATION_ROLES),
            "presets": list(EXPECTED_PRESETS),
            "cells_per_role": len(EXPECTED_PRESETS),
            "unique_cells": len(all_cells),
            "aliases_per_cell": EXPECTED_EVALUATION_ENVS,
            "checkpoint_seed": EXPECTED_TRAINING_SEED,
            "evaluation_seed": EXPECTED_EVALUATION_SEED,
            "cross_role_live_dr": cross_role_live_dr,
        },
        "frozen_contract": {
            "in_envelope_grid": dict(IN_ENVELOPE_GRID),
            "frontier_grid": dict(FRONTIER_GRID),
            "hard_success_presets": list(HARD_SUCCESS_PRESETS),
            "bridge": {
                "frontier_success_progress_absolute": BRIDGE_FRONTIER_TOLERANCE,
                "in_envelope_success_progress_absolute": BRIDGE_IN_ENVELOPE_TOLERANCE,
                "lat_50ms_success_absolute": BRIDGE_LAT50_TOLERANCE,
            },
            "candidate_vs_fresh_fixed": {
                "frontier_success_minimum_gain": CANDIDATE_FRONTIER_SUCCESS_GAIN,
                "frontier_progress_minimum_delta": CANDIDATE_FRONTIER_PROGRESS_FLOOR,
                "in_envelope_success_progress_minimum_delta": CANDIDATE_IN_ENVELOPE_FLOOR,
                "H_X1_mean_hard_success_minimum_gain": H_X1_HARD_SUCCESS_GAIN,
                "lat_60ms_success_minimum_gain_nonbinding": LAT60_REPORTED_GAIN,
            },
            "preference": {
                "default": "fixed_150",
                "fixed_u150_frontier_success_advantage_strictly_greater_than": PREFERENCE_FRONTIER_GAIN,
                "fixed_u150_minus_fixed_150_phys_000_success_strictly_greater_than": PREFERENCE_NOMINAL_LOSS,
                "fixed_u150_frontier_success_minimum_delta_vs_fixed_150": -PREFERENCE_FRONTIER_GAIN,
            },
        },
        "profiles": profiles,
        "historical_bridge": bridge,
        "candidate_screens": candidates,
        "decision": {
            "status": status,
            **selection,
            "screening_only": True,
            "directional_claim_authorized": False,
            "superiority_claim_authorized": False,
        },
        "verified": [
            "four role-specific receipts reconcile to 60 complete seed-8600/eval-8700 cells",
            "all 60 raw 512-episode arrays reconcile to run and aggregate outcomes",
            "all live DR ranges and pinned-delay histograms match their 15 preset contracts",
            "every preset's live DR ranges are canonically identical across all four roles",
            "three full 8000-row curricula independently establish the intended manipulation",
            "fixed_u150 carried exact recomputed lower-stratum parameters in 7990 TACE rows",
            "the SHA-pinned preregistration, H_R2 gate, panel, and four freeze bundles passed",
        ],
        "not_yet_verified": [
            "training-procedure variability beyond seed 8600",
            "a directional or superiority claim for either Tier-2 arm",
            "held-out-motion generalization, hardware transfer, or real-world safety",
        ],
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-fixed",
        "--historical-fixed-receipt",
        "--historical-fixed-eval",
        dest="historical_fixed",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-fixed",
        "--fresh-fixed-receipt",
        "--fresh-fixed-eval",
        dest="fresh_fixed",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fixed-150",
        "--fixed-150-receipt",
        "--fixed-150-eval",
        dest="fixed_150",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fixed-u150",
        "--fixed-u150-receipt",
        "--fixed-u150-eval",
        dest="fixed_u150",
        type=Path,
        required=True,
    )
    parser.add_argument("--fresh-fixed-training", type=Path, required=True)
    parser.add_argument("--fixed-150-training", type=Path, required=True)
    parser.add_argument("--fixed-u150-training", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha", required=True)
    parser.add_argument("--historical-fixed-freeze-manifest", type=Path, required=True)
    parser.add_argument("--fresh-fixed-freeze-manifest", type=Path, required=True)
    parser.add_argument("--fixed-150-freeze-manifest", type=Path, required=True)
    parser.add_argument("--fixed-u150-freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=MANIFESTS / "lucid_tier2_support_screen_analysis.json"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = analyze(
        historical_fixed=args.historical_fixed,
        fresh_fixed=args.fresh_fixed,
        fixed_150=args.fixed_150,
        fixed_u150=args.fixed_u150,
        fresh_fixed_training=args.fresh_fixed_training,
        fixed_150_training=args.fixed_150_training,
        fixed_u150_training=args.fixed_u150_training,
        preregistration=args.preregistration,
        expected_preregistration_sha=args.expected_preregistration_sha,
        historical_fixed_freeze_manifest=args.historical_fixed_freeze_manifest,
        fresh_fixed_freeze_manifest=args.fresh_fixed_freeze_manifest,
        fixed_150_freeze_manifest=args.fixed_150_freeze_manifest,
        fixed_u150_freeze_manifest=args.fixed_u150_freeze_manifest,
    )
    _write_exclusive(args.out, receipt)
    print(
        json.dumps(
            {
                "receipt": str(args.out.resolve()),
                "status": receipt["decision"]["status"],
                "selected": receipt["decision"]["selected"],
                "screening_only": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
