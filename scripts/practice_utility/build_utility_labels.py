#!/usr/bin/env python3
# ruff: noqa: I001  # repository isort and Ruff force-sort rules conflict
"""Assemble utility labels and run the preregistered Gate A/B analyses.

Claim-grade mode is deliberately fail closed.  It requires four artifacts that
were frozen independently of this analysis:

* quality-qualified branch evaluations (the deployment ``J_eff`` and measured
  harm channels, never training reward),
* one proxy-feature row for every exact ``(stage, seed, context_id)`` key,
* a dose-normalized noise floor measured with the same outcome estimand, units,
  horizon, and trailing window, and
* a preregistration that hashes the feature and noise-floor artifacts and fixes
  both gate decisions before labels are inspected.

The old training-log path remains available only through
``--exploratory-training-fallback``.  Its output is explicitly non-claim-grade
and can never authorize an estimator: training reward is not deployment
efficacy, and its missing harm channels are filled with zeros.

Canonical claim-grade artifact schemas are exercised in
``tests/practice_utility/test_build_utility_labels.py``.  In particular, the
preregistration has separate ``latent_proxy_audit`` and
``estimator_authorization`` sections because these answer opposite questions:
does the frozen latent proxy suffice for the paper claim, versus does failure
of every preregistered simple proxy authorize building an estimator?

The current ``UtilityRecord`` schema carries one final dose report, so any
future claim-grade utility assembly is restricted to the preregistered longest
horizon ``H_l``.  Shorter horizons must remain unavailable rather than be
normalized by the wrong final dose.  More importantly, the current shared
control does not measure passive per-context control dose, and the evaluation
artifact does not yet bind every result to its ``H_l`` policy/capsule, frozen
development suite, physics seeds, and per-evaluation receipt.  Claim-grade mode
therefore emits a blocked, non-claim receipt and does not assemble utility or
run Gate A/B.  Raw sign accuracy is also excluded for the nonnegative latent
gap; a preregistered leakage-free calibration remains required for Gate B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import proxy_audit as PA  # noqa: E402
from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from gear_sonic.research.practice_utility import utility_label as UL  # noqa: E402
from gear_sonic.research.practice_utility.schema import (  # noqa: E402
    ContextKey,
    DoseReport,
    sha256_of,
)

#: Training-side efficacy metric used only by the explicit exploratory path.
FALLBACK_EFFICACY = "Mean rewards"

#: Iterations averaged at a branch horizon.  The claim-grade value is also
#: fixed in the preregistration and evaluation/noise-floor metadata.
DEFAULT_EFFICACY_WINDOW = 4

LATENT_PROXY = "latent_gap_p90"
REQUIRED_HARM_CHANNELS = (
    "action_rate",
    "foot_slip",
    "contact_impulse",
    "torque_saturation",
)

PROBE_MANIFEST_KIND = "practice_utility_probe_manifest"
PREREGISTRATION_KIND = "practice_utility_gate_preregistration"
PROXY_FEATURE_KIND = "practice_utility_proxy_features"
BRANCH_EVALUATION_KIND = "practice_utility_branch_evaluations"
NOISE_FLOOR_KIND = "practice_utility_same_estimand_noise_floor"


def claim_blockers() -> list[dict[str, str]]:
    """Return the unresolved contracts that make label assembly non-claim-grade.

    These are implementation blockers, not fields that callers can self-attest
    in an input JSON.  Keeping them explicit prevents a superficially complete
    legacy artifact from becoming a Gate A/B input.
    """

    return [
        {
            "code": "ready_preflight_required",
            "scope": "preflight",
            "message": (
                "build_utility_labels has not verified a ready audit_probe_campaign "
                "preflight; the current campaign preflight remains blocked"
            ),
        },
        {
            "code": "shared_control_realized_dose_unimplemented",
            "scope": "dose",
            "message": (
                "the frozen shared control has no target kernels and therefore cannot "
                "provide passive per-context completed_kernel_steps for the "
                "realized-extra-dose denominator"
            ),
        },
        {
            "code": "branch_evaluation_h_l_policy_capsule_binding_unimplemented",
            "scope": "branch_evaluations",
            "message": (
                "each H_l evaluation is not yet hash-bound to the exact branch policy "
                "checkpoint and settled restart capsule"
            ),
        },
        {
            "code": "branch_evaluation_dev_suite_binding_unimplemented",
            "scope": "branch_evaluations",
            "message": (
                "each branch evaluation is not yet hash-bound to the frozen development "
                "suite used to compute deployment J_eff"
            ),
        },
        {
            "code": "branch_evaluation_physics_seed_binding_unimplemented",
            "scope": "branch_evaluations",
            "message": (
                "each branch evaluation does not yet carry and verify its deployment "
                "physics-seed set"
            ),
        },
        {
            "code": "per_evaluation_receipts_unimplemented",
            "scope": "branch_evaluations",
            "message": (
                "quality-qualified branch rows are not yet transitively bound to immutable "
                "per-evaluation receipts"
            ),
        },
        {
            "code": "latent_directional_calibration_unimplemented",
            "scope": "latent_direction",
            "message": (
                "the preregistered nested-CV univariate calibration has no immutable "
                "implementation, so raw nonnegative latent-gap signs cannot decide Gate B"
            ),
        },
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--campaign-dir", required=True, type=Path, help="directory holding branch dose reports"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/outputs"),
        help="training logs; read only in exploratory fallback mode",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        help="frozen claim-grade Gate A/B analysis plan",
    )
    parser.add_argument(
        "--proxy-features",
        type=Path,
        help="frozen per-(stage, seed, context_id) proxy-feature artifact",
    )
    parser.add_argument(
        "--branch-evaluations",
        type=Path,
        help="quality-qualified frozen-dev branch evaluations",
    )
    parser.add_argument(
        "--noise-floor",
        type=Path,
        help="same-estimand, dose-normalized utility noise-floor artifact",
    )
    parser.add_argument(
        "--exploratory-training-fallback",
        action="store_true",
        help=(
            "use training reward and zero-filled harm channels for a NON-CLAIM-GRADE "
            "diagnostic; this path cannot authorize an estimator"
        ),
    )
    parser.add_argument(
        "--efficacy-metric",
        default=None,
        help=f"training-log metric in exploratory mode (default: {FALLBACK_EFFICACY!r})",
    )
    parser.add_argument(
        "--efficacy-window",
        type=int,
        default=DEFAULT_EFFICACY_WINDOW,
        help="iterations averaged at each horizon; claim-grade mode must match preregistration",
    )
    parser.add_argument(
        "--shared-control",
        action="store_true",
        default=True,
        help="one control per (stage, seed), as screening uses",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def require_kind(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("kind") != expected:
        raise ValueError(f"{label} kind must be {expected!r}, got {payload.get('kind')!r}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def recompute_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Recompute the logical hash defined by :class:`ProbeManifest`.

    This intentionally mirrors ``ProbeManifest.manifest_sha256`` instead of
    trusting the self-reported value carried by the JSON file.
    """

    contexts = payload.get("contexts_per_stage")
    if not isinstance(contexts, Mapping):
        raise ValueError("probe manifest contexts_per_stage must be an object")
    frozen_contexts: dict[str, list[str]] = {}
    for stage, entries in sorted(contexts.items(), key=lambda item: str(item[0])):
        if not isinstance(entries, list):
            raise ValueError(f"manifest contexts for stage {stage!r} must be a list")
        frozen_contexts[str(stage)] = sorted(str(entry.get("context_id")) for entry in entries)
    return sha256_of(
        {
            "campaign_id": payload.get("campaign_id"),
            "contexts": frozen_contexts,
            "seeds": sorted(payload.get("seeds") or []),
            "epsilon": payload.get("epsilon"),
            "kernel_radius_bins": payload.get("kernel_radius_bins"),
            "horizons": payload.get("horizons") or {},
            "pool_sha256": payload.get("pool_sha256"),
            "split_sha256": payload.get("split_sha256"),
        }
    )


def validate_manifest(payload: Mapping[str, Any]) -> None:
    require_kind(payload, PROBE_MANIFEST_KIND, "probe manifest")
    for key in (
        "campaign_id",
        "manifest_sha256",
        "contexts_per_stage",
        "seeds",
        "horizons",
        "epsilon",
        "kernel_radius_bins",
        "pool_sha256",
        "split_sha256",
    ):
        if key not in payload:
            raise ValueError(f"probe manifest is missing {key!r}")
    if not isinstance(payload["contexts_per_stage"], dict) or not payload["contexts_per_stage"]:
        raise ValueError("probe manifest has no contexts_per_stage")
    if not isinstance(payload["seeds"], list) or not payload["seeds"]:
        raise ValueError("probe manifest has no seeds")
    if not isinstance(payload["horizons"], dict) or not payload["horizons"]:
        raise ValueError("probe manifest has no horizons")

    for field in ("manifest_sha256", "pool_sha256", "split_sha256"):
        value = payload[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"probe manifest {field} must be a lowercase SHA-256")

    for stage, entries in payload["contexts_per_stage"].items():
        if not isinstance(stage, str) or not stage:
            raise ValueError("probe manifest stage names must be non-empty strings")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"manifest contexts for stage {stage!r} must be a non-empty list")
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("context"), dict):
                raise ValueError(f"manifest context {stage!r}/{index} is malformed")
            try:
                computed_context_id = ContextKey.from_dict(entry["context"]).context_id
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"manifest context {stage!r}/{index} is invalid") from error
            if entry.get("context_id") != computed_context_id:
                raise ValueError(
                    f"manifest context_id mismatch at {stage!r}/{index}: "
                    f"{entry.get('context_id')!r} != {computed_context_id!r}"
                )
            if computed_context_id in seen:
                raise ValueError(
                    f"manifest stage {stage!r} repeats context_id {computed_context_id!r}"
                )
            seen.add(computed_context_id)

    computed_hash = recompute_manifest_sha256(payload)
    if payload["manifest_sha256"] != computed_hash:
        raise ValueError(
            "probe manifest manifest_sha256 does not match its recomputed logical hash: "
            f"{payload['manifest_sha256']} != {computed_hash}"
        )


FeatureKey = tuple[str, int, str]


def expected_feature_rows(manifest: Mapping[str, Any]) -> dict[FeatureKey, Mapping[str, Any]]:
    expected: dict[FeatureKey, Mapping[str, Any]] = {}
    for stage, contexts in manifest["contexts_per_stage"].items():
        if not isinstance(contexts, list):
            raise ValueError(f"manifest contexts for stage {stage!r} must be a list")
        for entry in contexts:
            context_id = entry.get("context_id")
            if not isinstance(context_id, str) or not context_id:
                raise ValueError(f"manifest stage {stage!r} has an entry without context_id")
            for seed_value in manifest["seeds"]:
                seed = int(seed_value)
                key = (str(stage), seed, context_id)
                if key in expected:
                    raise ValueError(f"manifest repeats feature key {key!r}")
                expected[key] = entry
    return expected


def validate_campaign_link(
    payload: Mapping[str, Any], manifest: Mapping[str, Any], label: str
) -> None:
    if payload.get("campaign_id") != manifest["campaign_id"]:
        raise ValueError(f"{label} campaign_id does not match the probe manifest")
    if payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError(f"{label} manifest_sha256 does not match the probe manifest")


def validate_preregistration(
    preregistration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    proxy_features_sha256: str,
    noise_floor_sha256: str,
    efficacy_window: int,
) -> dict[str, Any]:
    """Validate and normalize the claim-grade analysis plan.

    Thresholds are required to equal the implemented thresholds.  Silently
    accepting a different preregistered rule while running the hard-coded rule
    would make the receipt false even if all numerical results were correct.
    """
    require_kind(preregistration, PREREGISTRATION_KIND, "preregistration")
    validate_campaign_link(preregistration, manifest, "preregistration")
    if preregistration.get("analysis_mode") != "claim_grade":
        raise ValueError("preregistration analysis_mode must be 'claim_grade'")

    efficacy = preregistration.get("efficacy")
    gate_a = preregistration.get("gate_a")
    latent = preregistration.get("latent_proxy_audit")
    authorization = preregistration.get("estimator_authorization")
    for name, section in (
        ("efficacy", efficacy),
        ("gate_a", gate_a),
        ("latent_proxy_audit", latent),
        ("estimator_authorization", authorization),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"preregistration requires an object section {name!r}")

    assert isinstance(efficacy, dict)
    assert isinstance(gate_a, dict)
    assert isinstance(latent, dict)
    assert isinstance(authorization, dict)

    required_efficacy = {
        "name",
        "units",
        "utility_units",
        "window",
        "quality_qualified",
        "macro_average_group",
        "harm_channels",
    }
    missing = sorted(required_efficacy - set(efficacy))
    if missing:
        raise ValueError(f"preregistration efficacy is missing {missing}")
    if efficacy["name"] == FALLBACK_EFFICACY or "training" in str(efficacy["name"]).lower():
        raise ValueError("claim-grade efficacy cannot be a training metric")
    if efficacy["quality_qualified"] is not True:
        raise ValueError("claim-grade efficacy must be quality-qualified")
    if efficacy["macro_average_group"] != "motion_family":
        raise ValueError("claim-grade efficacy must be macro-averaged by motion_family")
    if set(efficacy["harm_channels"]) != set(REQUIRED_HARM_CHANNELS):
        raise ValueError(
            f"claim-grade efficacy must measure harm channels {REQUIRED_HARM_CHANNELS}"
        )
    if int(efficacy["window"]) != efficacy_window:
        raise ValueError(
            f"CLI efficacy window {efficacy_window} differs from preregistered "
            f"window {efficacy['window']}"
        )

    horizon_label = gate_a.get("horizon_label")
    if horizon_label not in manifest["horizons"]:
        raise ValueError(f"Gate A horizon {horizon_label!r} is absent from the manifest")
    longest_horizon = max(manifest["horizons"], key=lambda key: manifest["horizons"][key])
    if horizon_label != "H_l" or horizon_label != longest_horizon:
        raise ValueError(
            "claim-grade Gate A/B is restricted to the manifest's longest horizon H_l; "
            "shorter horizons need per-horizon dose accounting"
        )
    if int(gate_a.get("horizon_iterations", -1)) != int(manifest["horizons"][horizon_label]):
        raise ValueError("Gate A horizon_iterations does not match the probe manifest")
    if gate_a.get("noise_floor_sha256") != noise_floor_sha256:
        raise ValueError("preregistered noise_floor_sha256 does not match --noise-floor")
    if finite_float(gate_a.get("min_variance_ratio"), "Gate A min_variance_ratio") != 2.0:
        raise ValueError("Gate A min_variance_ratio must match the implemented value 2.0")
    if finite_float(gate_a.get("min_icc"), "Gate A min_icc") != 0.4:
        raise ValueError("Gate A min_icc must match the implemented value 0.4")
    origin_steps = gate_a.get("origin_global_step_by_stage")
    if not isinstance(origin_steps, dict) or set(origin_steps) != set(
        manifest["contexts_per_stage"]
    ):
        raise ValueError(
            "Gate A requires origin_global_step_by_stage with exact manifest-stage coverage"
        )
    normalized_origins = {}
    for stage, value in origin_steps.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Gate A origin step for stage {stage!r} must be a positive integer")
        normalized_origins[str(stage)] = value

    if latent.get("proxy") != LATENT_PROXY:
        raise ValueError(f"latent_proxy_audit.proxy must be {LATENT_PROXY!r}")
    if latent.get("horizon_label") != horizon_label:
        raise ValueError("latent-proxy and Gate A horizons must match")
    if latent.get("proxy_features_sha256") != proxy_features_sha256:
        raise ValueError("preregistered proxy_features_sha256 does not match --proxy-features")
    expected_rank_thresholds = {
        "min_abs_spearman": PA.SUFFICIENCY["min_abs_spearman"],
        "min_pairwise_accuracy": PA.SUFFICIENCY["min_pairwise_accuracy"],
    }
    if latent.get("rank_thresholds") != expected_rank_thresholds:
        raise ValueError("latent-proxy rank thresholds differ from the implemented diagnostics")
    if latent.get("directional_test") != "nested_cv_univariate_calibration":
        raise ValueError(
            "latent-proxy directional_test must preregister nested_cv_univariate_calibration"
        )
    if latent.get("raw_sign_accuracy_allowed") is not False:
        raise ValueError("raw sign accuracy must be excluded for nonnegative latent_gap_p90")

    if authorization.get("horizon_label") != horizon_label:
        raise ValueError("estimator-authorization and Gate A horizons must match")
    proxies = authorization.get("proxies")
    if not isinstance(proxies, list) or not proxies or len(set(proxies)) != len(proxies):
        raise ValueError("estimator_authorization.proxies must be a non-empty unique list")
    if LATENT_PROXY not in proxies:
        raise ValueError(f"estimator_authorization.proxies must include {LATENT_PROXY!r}")
    if authorization.get("inverse_decision") is not True:
        raise ValueError(
            "estimator_authorization.inverse_decision must be true: authorization means "
            "no preregistered simple proxy was sufficient"
        )

    return {
        "efficacy": dict(efficacy),
        "horizon_label": horizon_label,
        "min_variance_ratio": 2.0,
        "min_icc": 0.4,
        "origin_global_step_by_stage": normalized_origins,
        "grouping": latent.get("grouping"),
        "latent_rank_thresholds": expected_rank_thresholds,
        "latent_directional_test": latent["directional_test"],
        "authorization_proxies": list(proxies),
    }


def load_proxy_feature_index(
    artifact: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[FeatureKey, dict[str, float]]:
    """Validate a frozen feature table and require exact manifest coverage."""
    require_kind(artifact, PROXY_FEATURE_KIND, "proxy-feature artifact")
    validate_campaign_link(artifact, manifest, "proxy-feature artifact")
    if artifact.get("frozen_before_outcomes") is not True:
        raise ValueError("proxy-feature artifact must declare frozen_before_outcomes=true")
    encoder_sha256 = artifact.get("encoder_sha256")
    if not isinstance(encoder_sha256, str) or len(encoder_sha256) != 64:
        raise ValueError("proxy-feature artifact requires the frozen encoder_sha256")

    rows = artifact.get("records")
    if not isinstance(rows, list):
        raise ValueError("proxy-feature artifact records must be a list")
    index: dict[FeatureKey, dict[str, float]] = {}
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"proxy feature row {row_number} must be an object")
        try:
            key = (str(row["stage"]), int(row["seed"]), str(row["context_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"proxy feature row {row_number} requires exact stage, seed, context_id"
            ) from error
        if key in index:
            raise ValueError(f"duplicate proxy feature key {key!r}")
        features = row.get("proxy_features")
        if not isinstance(features, dict):
            raise ValueError(f"proxy feature row {key!r} has no proxy_features object")
        converted = {
            str(name): finite_float(value, f"proxy feature {key!r}/{name}")
            for name, value in features.items()
        }
        if LATENT_PROXY not in converted:
            raise ValueError(f"proxy feature row {key!r} is missing required {LATENT_PROXY!r}")
        if converted[LATENT_PROXY] < 0.0:
            raise ValueError(f"proxy feature row {key!r} has negative {LATENT_PROXY}")
        index[key] = converted

    expected = set(expected_feature_rows(manifest))
    actual = set(index)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "proxy-feature keys do not exactly cover the frozen manifest; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return index


def resolve_proxy_grouping(
    grouping: Any, manifest: Mapping[str, Any]
) -> tuple[str, dict[FeatureKey, str]]:
    """Resolve motion-family or an exact preregistered key-to-group table."""
    expected = expected_feature_rows(manifest)
    kind = grouping if isinstance(grouping, str) else None
    if isinstance(grouping, dict):
        kind = grouping.get("kind")

    if kind == "motion_family":
        groups: dict[FeatureKey, str] = {}
        for key, entry in expected.items():
            family = entry.get("family")
            if not isinstance(family, str) or not family:
                raise ValueError(f"manifest entry {key!r} has no motion family")
            groups[key] = family
        return "motion_family", groups

    if kind != "exact" or not isinstance(grouping, dict):
        raise ValueError(
            "latent_proxy_audit.grouping must be 'motion_family' or an exact grouping object"
        )
    assignments = grouping.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("exact proxy grouping requires an assignments list")
    groups = {}
    for row_number, row in enumerate(assignments):
        if not isinstance(row, dict):
            raise ValueError(f"grouping assignment {row_number} must be an object")
        try:
            key = (str(row["stage"]), int(row["seed"]), str(row["context_id"]))
            name = str(row["group"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"grouping assignment {row_number} requires stage, seed, context_id, group"
            ) from error
        if not name:
            raise ValueError(f"grouping assignment {key!r} has an empty group")
        if key in groups:
            raise ValueError(f"duplicate exact grouping key {key!r}")
        groups[key] = name
    if set(groups) != set(expected):
        missing = sorted(set(expected) - set(groups))
        extra = sorted(set(groups) - set(expected))
        raise ValueError(
            "exact preregistered grouping does not cover manifest keys; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return "exact", groups


def validate_branch_evaluations(
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    efficacy: Mapping[str, Any],
    horizon_label: str,
) -> dict[str, list[UL.BranchEvaluation]]:
    """Load measured deployment evaluations without synthesizing harm channels."""
    require_kind(artifact, BRANCH_EVALUATION_KIND, "branch-evaluation artifact")
    validate_campaign_link(artifact, manifest, "branch-evaluation artifact")
    measured = artifact.get("efficacy", artifact.get("estimand"))
    if not isinstance(measured, dict):
        raise ValueError("branch-evaluation artifact requires efficacy metadata")
    for field in (
        "name",
        "units",
        "utility_units",
        "window",
        "quality_qualified",
        "macro_average_group",
        "harm_channels",
    ):
        expected = efficacy[field]
        actual = measured.get(field)
        if field == "harm_channels":
            if not isinstance(actual, list) or set(actual) != set(expected):
                raise ValueError("branch-evaluation harm channels differ from preregistration")
        elif actual != expected:
            raise ValueError(
                f"branch-evaluation efficacy {field!r} differs from preregistration: "
                f"{actual!r} != {expected!r}"
            )
    if measured["quality_qualified"] is not True:
        raise ValueError("branch evaluations are not quality-qualified")
    if measured["macro_average_group"] != "motion_family":
        raise ValueError("branch evaluations are not macro-averaged by motion family")
    analysis_horizons = {horizon_label: manifest["horizons"][horizon_label]}
    if artifact.get("horizons") != analysis_horizons:
        raise ValueError(
            "branch-evaluation horizons must contain only the preregistered H_l; "
            "shorter horizons lack per-horizon dose accounting"
        )

    rows = artifact.get("records")
    if not isinstance(rows, list):
        raise ValueError("branch-evaluation records must be a list")
    by_branch: dict[str, list[UL.BranchEvaluation]] = {}
    seen: set[tuple[str, str]] = set()
    all_harms_zero = bool(rows)
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"branch evaluation row {row_number} must be an object")
        missing = sorted(
            {
                "branch_id",
                "role",
                "horizon_label",
                "j_eff",
                "clean_j_eff",
                *REQUIRED_HARM_CHANNELS,
            }
            - set(row)
        )
        if missing:
            raise ValueError(f"branch evaluation row {row_number} is missing {missing}")
        branch_id = str(row["branch_id"])
        role = str(row["role"])
        horizon_label = str(row["horizon_label"])
        key = (branch_id, horizon_label)
        if key in seen:
            raise ValueError(f"duplicate branch evaluation {key!r}")
        seen.add(key)
        if horizon_label not in analysis_horizons:
            raise ValueError(f"branch evaluation {key!r} has an unknown horizon")
        values = {
            name: finite_float(row[name], f"branch evaluation {key!r}/{name}")
            for name in ("j_eff", "clean_j_eff", *REQUIRED_HARM_CHANNELS)
        }
        if any(values[name] != 0.0 for name in REQUIRED_HARM_CHANNELS):
            all_harms_zero = False
        evaluation = UL.BranchEvaluation(
            branch_id=branch_id,
            role=role,
            horizon_label=horizon_label,
            j_eff=values["j_eff"],
            clean_j_eff=values["clean_j_eff"],
            action_rate=values["action_rate"],
            foot_slip=values["foot_slip"],
            contact_impulse=values["contact_impulse"],
            torque_saturation=values["torque_saturation"],
            extras={
                str(name): finite_float(value, f"branch evaluation {key!r}/extras/{name}")
                for name, value in (row.get("extras") or {}).items()
            },
        )
        by_branch.setdefault(branch_id, []).append(evaluation)

    if all_harms_zero:
        raise ValueError(
            "all branch-evaluation harm channels are zero; refusing a likely zero-filled "
            "claim-grade artifact"
        )

    expected_branches: dict[str, str] = {}
    campaign = manifest["campaign_id"]
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed_value in manifest["seeds"]:
            seed = int(seed_value)
            expected_branches[f"{campaign}_{stage}_s{seed}_control"] = "control"
            for index, _ in enumerate(contexts):
                expected_branches[f"{campaign}_{stage}_s{seed}_c{index}_intervention"] = (
                    "intervention"
                )
    if set(by_branch) != set(expected_branches):
        missing = sorted(set(expected_branches) - set(by_branch))
        extra = sorted(set(by_branch) - set(expected_branches))
        raise ValueError(
            "branch evaluations do not exactly cover the frozen campaign; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    expected_horizons = set(analysis_horizons)
    for branch_id, role in expected_branches.items():
        evaluations = by_branch[branch_id]
        if {entry.horizon_label for entry in evaluations} != expected_horizons:
            raise ValueError(f"branch {branch_id!r} lacks exact horizon coverage")
        if any(entry.role != role for entry in evaluations):
            raise ValueError(f"branch {branch_id!r} has the wrong role")
    return by_branch


def validate_noise_floor(
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    efficacy: Mapping[str, Any],
    horizon_label: str,
) -> list[float]:
    """Require Gate A samples in the exact units consumed by Gate A."""
    require_kind(artifact, NOISE_FLOOR_KIND, "noise-floor artifact")
    validate_campaign_link(artifact, manifest, "noise-floor artifact")
    estimand = artifact.get("estimand")
    if not isinstance(estimand, dict):
        raise ValueError("noise-floor artifact requires estimand metadata")
    required = {
        "name": efficacy["name"],
        "outcome_units": efficacy["units"],
        "utility_units": efficacy["utility_units"],
        "quantity": "dose_normalized_practice_utility",
        "normalization": "realized_extra_completed_kernel_steps",
        "horizon_label": horizon_label,
        "horizon_iterations": int(manifest["horizons"][horizon_label]),
        "window": int(efficacy["window"]),
    }
    for field, expected in required.items():
        if estimand.get(field) != expected:
            raise ValueError(
                f"noise-floor estimand {field!r} differs from the preregistered Gate A "
                f"estimand: {estimand.get(field)!r} != {expected!r}"
            )
    values = artifact.get("utility_deltas")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("noise-floor artifact requires at least two utility_deltas")
    return [
        finite_float(value, f"noise-floor utility_delta[{index}]")
        for index, value in enumerate(values)
    ]


def load_dose_report(
    campaign_dir: Path, branch_id: str, expected_global_step: int | None = None
) -> dict[str, Any] | None:
    reports = sorted((campaign_dir / branch_id).glob("dose_*.json"))
    if not reports:
        return None
    if expected_global_step is None:
        return load_json_object(reports[-1], f"dose report for {branch_id}")
    matches = []
    available_steps = []
    for path in reports:
        payload = load_json_object(path, f"dose report for {branch_id}")
        step = payload.get("global_step")
        available_steps.append(step)
        if step == expected_global_step:
            matches.append(payload)
    if len(matches) != 1:
        raise ValueError(
            f"branch {branch_id!r} requires exactly one dose report at origin + H_l = "
            f"step {expected_global_step}; found {len(matches)}, available={available_steps}"
        )
    return matches[0]


def load_branch(campaign_dir: Path, log_dir: Path, branch_id: str, metric: str):
    """Return (metric series, dose report) for the exploratory fallback."""
    series: dict[int, float] = {}
    for candidate in (log_dir / f"{branch_id}.log", campaign_dir / branch_id / "run.log"):
        if candidate.exists():
            series = RL.parse_run_log(candidate).series(metric)
            break
    return series, load_dose_report(campaign_dir, branch_id)


def dose_from_report(
    report: Mapping[str, Any] | None,
    role: str,
    branch_id: str,
    *,
    claim_grade: bool = False,
    expected_context_id: str | None = None,
) -> DoseReport:
    if report is None:
        if claim_grade:
            raise ValueError(f"claim-grade branch is missing dose report: {branch_id}")
        return DoseReport(branch_id=branch_id, context_id="unknown", role=role)  # type: ignore[arg-type]
    if claim_grade and report.get("branch_id", branch_id) != branch_id:
        raise ValueError(f"dose report branch_id does not match {branch_id}")
    if claim_grade and report.get("role") != role:
        raise ValueError(f"dose report for {branch_id} has the wrong role")
    if claim_grade and role == "intervention" and report.get("never_armed") is not False:
        raise ValueError(
            f"dose report for {branch_id} must record never_armed=false at claim horizon"
        )
    context_id = str(report.get("context_id", "unknown"))
    if claim_grade and expected_context_id is not None and context_id != expected_context_id:
        raise ValueError(
            f"dose report for {branch_id} has context_id {context_id!r}, "
            f"expected {expected_context_id!r}"
        )
    return DoseReport(
        branch_id=str(report.get("branch_id", branch_id)),
        context_id=context_id,
        role=role,  # type: ignore[arg-type]
        drawn_episodes=finite_float(report.get("drawn_episodes", 0.0), "drawn_episodes"),
        drawn_kernel_mass=finite_float(report.get("drawn_kernel_mass", 0.0), "drawn_kernel_mass"),
        completed_env_steps=finite_float(
            report.get("completed_env_steps", 0.0), "completed_env_steps"
        ),
        completed_kernel_steps=finite_float(
            report.get("completed_kernel_steps", 0.0), "completed_kernel_steps"
        ),
        early_terminations=int(report.get("early_terminations", 0)),
    )


def training_evaluations_for(
    series: Mapping[int, float],
    horizons: Mapping[str, int],
    role: str,
    branch_id: str,
    window: int = DEFAULT_EFFICACY_WINDOW,
) -> list[UL.BranchEvaluation]:
    """Build explicitly exploratory evaluations from a training metric."""
    out = []
    for label, horizon in horizons.items():
        usable = sorted(i for i in series if i <= horizon)
        if not usable:
            continue
        chosen = usable[-window:] if window > 1 else usable[-1:]
        value = sum(series[i] for i in chosen) / len(chosen)
        out.append(
            UL.BranchEvaluation(
                branch_id=branch_id,
                role=role,
                horizon_label=label,
                j_eff=value,
                clean_j_eff=value,
                action_rate=0.0,
                foot_slip=0.0,
                contact_impulse=0.0,
                torque_saturation=0.0,
                extras={
                    "iterations_averaged": float(len(chosen)),
                    "first_iteration": float(chosen[0]),
                    "last_iteration": float(chosen[-1]),
                },
            )
        )
    return out


# Backward-compatible name for callers that imported the old helper.  It is
# intentionally named as training-only at its definition and is unreachable in
# claim-grade mode.
evaluations_for = training_evaluations_for


def static_proxy_features(entry: Mapping[str, Any]) -> dict[str, float]:
    return {
        "native_failure_rate": finite_float(entry.get("failure_rate", 0.0), "failure_rate"),
        "sampling_probability": finite_float(
            entry.get("sampling_probability", 0.0), "sampling_probability"
        ),
        **{
            str(key): finite_float(value, f"manifest proxy feature {key}")
            for key, value in (entry.get("extras") or {}).items()
        },
    }


def merge_proxy_features(
    static: Mapping[str, float], frozen: Mapping[str, float], key: FeatureKey
) -> dict[str, float]:
    merged = dict(static)
    for name, value in frozen.items():
        if name in merged and merged[name] != value:
            raise ValueError(
                f"frozen proxy feature {key!r}/{name} conflicts with the manifest value"
            )
        merged[name] = value
    return merged


def assemble_claim_grade_records(
    manifest: Mapping[str, Any],
    campaign_dir: Path,
    evaluations: Mapping[str, Sequence[UL.BranchEvaluation]],
    proxy_features: Mapping[FeatureKey, Mapping[str, float]],
    horizon_label: str,
    origin_global_step_by_stage: Mapping[str, int],
) -> list[Any]:
    """Refuse the legacy shared-control dose normalization.

    The arguments document the intended future assembly seam, but no caller may
    opt around this blocker.  A replacement must first measure per-context
    control dose (or use independently paired controls) and bind every H_l
    evaluation to its complete lineage.
    """

    raise ValueError(
        "claim-grade assembly is blocked: shared-control realized dose and H_l "
        "evaluation lineage are unimplemented"
    )


def assemble_exploratory_records(args: argparse.Namespace, manifest: Mapping[str, Any]):
    metric = args.efficacy_metric or FALLBACK_EFFICACY
    records, skipped = [], []
    campaign = manifest["campaign_id"]
    horizons = manifest["horizons"]
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed_value in manifest["seeds"]:
            seed = int(seed_value)
            control_id = f"{campaign}_{stage}_s{seed}_control"
            control_series, control_report = load_branch(
                args.campaign_dir, args.log_dir, control_id, metric
            )
            if not control_series:
                skipped.append(f"{stage}/s{seed}: control log missing")
                continue
            for index, entry in enumerate(contexts):
                pair = f"{campaign}_{stage}_s{seed}_c{index}"
                branch_id = f"{pair}_intervention"
                series, intervention_report = load_branch(
                    args.campaign_dir, args.log_dir, branch_id, metric
                )
                if not series:
                    skipped.append(f"{branch_id}: log missing")
                    continue
                try:
                    record = UL.build_utility_record(
                        branch_pair_id=pair,
                        context=ContextKey.from_dict(entry["context"]),
                        policy_stage=str(stage),
                        seed=seed,
                        horizons=dict(horizons),
                        control_dose=dose_from_report(control_report, "control", control_id),
                        intervention_dose=dose_from_report(
                            intervention_report, "intervention", branch_id
                        ),
                        control_evaluations=training_evaluations_for(
                            control_series, horizons, "control", control_id, args.efficacy_window
                        ),
                        intervention_evaluations=training_evaluations_for(
                            series, horizons, "intervention", branch_id, args.efficacy_window
                        ),
                        epsilon=float(manifest["epsilon"]),
                        kernel_radius_bins=int(manifest["kernel_radius_bins"]),
                        base_distribution_sha256=str(manifest["manifest_sha256"]),
                        intervention_distribution_sha256=str(manifest["manifest_sha256"]),
                        proxy_features=static_proxy_features(entry),
                    )
                except ValueError as error:
                    skipped.append(f"{branch_id}: {error}")
                    continue
                records.append(record)
    return records, skipped, metric


def old_exploratory_noise_floor(path: Path | None, metric_name: str) -> list[float] | None:
    """Read the legacy training-metric floor only in explicit exploratory mode."""
    if path is None or not path.exists():
        return None
    report = load_json_object(path, "exploratory noise floor")
    metric = (report.get("metrics") or {}).get(metric_name)
    if not isinstance(metric, dict):
        return None
    values = metric.get("paired_deltas")
    if not isinstance(values, list):
        return None
    return [finite_float(value, "exploratory noise-floor delta") for value in values]


def group_function(groups: Mapping[FeatureKey, str]) -> Callable[[Any], str]:
    def group(record: Any) -> str:
        key = (record.policy_stage, int(record.seed), record.context.context_id)
        if key not in groups:
            raise ValueError(f"no preregistered proxy group for record {key!r}")
        return groups[key]

    return group


def require_proxy_coverage(records: Sequence[Any], proxies: Sequence[str]) -> None:
    """Refuse an inverse authorization decision over partially observed proxies.

    ``proxy_audit.audit_all_proxies`` intentionally skips unavailable proxies
    for general exploratory use.  That convenience is unsafe at the claim
    boundary: a missing preregistered comparator would otherwise make "no
    simple proxy suffices" easier to obtain.  Claim-grade authorization needs
    every named proxy on every record.
    """
    missing = [
        (record.policy_stage, int(record.seed), record.context.context_id, proxy)
        for record in records
        for proxy in proxies
        if proxy not in record.proxy_features
    ]
    if missing:
        raise ValueError(
            "preregistered estimator-authorization proxies lack complete feature coverage; "
            f"missing={missing[:8]}"
        )


def rank_only_proxy_diagnostic(result: PA.ProxyResult) -> dict[str, Any]:
    """Expose label-order diagnostics without the invalid raw-sign score.

    Difficulty and latent-gap proxies are nonnegative.  Comparing their raw
    signs to signed utility therefore measures only positive-label prevalence,
    not directional prediction.  A fitted calibration rule must be trained and
    tested without label leakage before any sign/sufficiency claim is valid.
    """
    return {
        "proxy": result.proxy,
        "horizon_label": result.horizon_label,
        "num_samples": result.num_samples,
        "num_groups": result.num_groups,
        "spearman": result.spearman,
        "pairwise_accuracy": result.pairwise_accuracy,
        "sign_flips_across_groups": result.sign_flips_across_groups,
        "per_group_spearman": result.per_group_spearman,
        "raw_sign_accuracy_excluded": True,
    }


def claim_grade_main(args: argparse.Namespace, manifest: Mapping[str, Any]) -> int:
    required_paths = {
        "--preregistration": args.preregistration,
        "--proxy-features": args.proxy_features,
        "--branch-evaluations": args.branch_evaluations,
        "--noise-floor": args.noise_floor,
    }
    missing = [flag for flag, path in required_paths.items() if path is None]
    if missing:
        raise ValueError(
            "claim-grade mode requires "
            + ", ".join(missing)
            + "; use --exploratory-training-fallback only for a non-claim diagnostic"
        )
    if args.efficacy_metric is not None:
        raise ValueError("--efficacy-metric is training-only and forbidden in claim-grade mode")
    assert args.preregistration is not None
    assert args.proxy_features is not None
    assert args.branch_evaluations is not None
    assert args.noise_floor is not None

    proxy_artifact = load_json_object(args.proxy_features, "proxy-feature artifact")
    noise_artifact = load_json_object(args.noise_floor, "noise-floor artifact")
    preregistration = load_json_object(args.preregistration, "preregistration")
    evaluation_artifact = load_json_object(args.branch_evaluations, "branch evaluations")
    proxy_hash = file_sha256(args.proxy_features)
    noise_hash = file_sha256(args.noise_floor)
    prereg_hash = file_sha256(args.preregistration)
    evaluation_hash = file_sha256(args.branch_evaluations)

    plan = validate_preregistration(
        preregistration,
        manifest,
        proxy_features_sha256=proxy_hash,
        noise_floor_sha256=noise_hash,
        efficacy_window=args.efficacy_window,
    )
    feature_index = load_proxy_feature_index(proxy_artifact, manifest)
    grouping_name, _ = resolve_proxy_grouping(plan["grouping"], manifest)
    evaluation_index = validate_branch_evaluations(
        evaluation_artifact, manifest, plan["efficacy"], plan["horizon_label"]
    )
    noise_values = validate_noise_floor(
        noise_artifact, manifest, plan["efficacy"], plan["horizon_label"]
    )
    horizon = plan["horizon_label"]
    blockers = claim_blockers()
    blocked_reason = (
        "claim-grade utility assembly and Gate A/B are not run while required provenance, "
        "realized-dose, and evaluation-lineage contracts remain unresolved"
    )
    payload = {
        "kind": "practice_utility_labels",
        "schema_version": 2,
        "analysis_mode": "claim_grade_blocked",
        "status": "blocked",
        "claim_grade": False,
        "usable_for_gate_a_b": False,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_sha256_recomputed": recompute_manifest_sha256(manifest),
        "manifest_sha256_verified": True,
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": prereg_hash,
        "source_artifacts": {
            "proxy_features": str(args.proxy_features.resolve()),
            "proxy_features_sha256": proxy_hash,
            "branch_evaluations": str(args.branch_evaluations.resolve()),
            "branch_evaluations_sha256": evaluation_hash,
            "noise_floor": str(args.noise_floor.resolve()),
            "noise_floor_sha256": noise_hash,
        },
        "efficacy_source": plan["efficacy"],
        "horizons": {horizon: manifest["horizons"][horizon]},
        "artifact_validation": {
            "status": "schema_and_direct_hash_links_only",
            "proxy_feature_rows": len(feature_index),
            "branch_ids": len(evaluation_index),
            "noise_floor_replicates": len(noise_values),
            "proxy_grouping": grouping_name,
            "caveat": (
                "these checks do not establish branch-policy/capsule, dev-suite, "
                "physics-seed, per-evaluation-receipt, or realized-control-dose lineage"
            ),
        },
        "blockers": blockers,
        "unavailable_horizons": {
            label: (
                "not emitted: UtilityRecord has one final DoseReport, so H_l dose cannot "
                "normalize a shorter-horizon utility"
            )
            for label in manifest["horizons"]
            if label != horizon
        },
        "summary": None,
        "reversals": {
            "available": False,
            "reason": "short-horizon utility is unavailable without per-horizon dose reports",
        },
        "gate_a_identifiability": {
            "status": "not_run",
            "valid_for_claim": False,
            "passes": False,
            "reason": blocked_reason,
        },
        "latent_proxy_predictiveness": {
            "decision_question": (
                "Does the frozen latent_gap_p90 proxy meet the preregistered "
                "predictiveness/sufficiency rule?"
            ),
            "grouping": grouping_name,
            "status": "not_run",
            "gate_a_prerequisite_passed": False,
            "directional_test": {
                "method": plan["latent_directional_test"],
                "implemented": False,
                "reason": (
                    "nested-CV univariate calibration is preregistered but not implemented; "
                    "raw sign accuracy is invalid for nonnegative latent_gap_p90"
                ),
            },
            "decision_complete": False,
            "supports_latent_proxy_claim": False,
            "rank_only_audit": None,
            "reason": blocked_reason,
        },
        "estimator_authorization_decision": {
            "decision_question": (
                "Did every preregistered simple proxy fail sufficiency, thereby authorizing "
                "an estimator, conditional on Gate A?"
            ),
            "inverse_of_proxy_sufficiency": True,
            "status": "not_run",
            "gate_a_prerequisite_passed": False,
            "valid_for_authorization": False,
            "authorizes_estimator": False,
            "reason": blocked_reason,
            "rank_only_proxy_audits": None,
        },
        "skipped": [],
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    print("claim-grade label assembly: BLOCKED")
    for blocker in blockers:
        print(f"  {blocker['code']}: {blocker['message']}")
    print("Gate A/B: NOT RUN; estimator authorization: False")
    print(f"wrote {args.output}")
    return 2


def exploratory_main(args: argparse.Namespace, manifest: Mapping[str, Any]) -> int:
    records, skipped, metric = assemble_exploratory_records(args, manifest)
    usable = [record for record in records if UL.is_usable(record)]
    print(
        f"pairs assembled: {len(records)}  usable labels: {len(usable)}  "
        f"skipped: {len(skipped)}"
    )
    for line in skipped[:8]:
        print(f"  skip {line}")
    if not usable:
        print("\nno usable exploratory labels; nothing to audit")
        return 1

    long_horizon = max(manifest["horizons"], key=lambda key: manifest["horizons"][key])
    short_horizon = min(manifest["horizons"], key=lambda key: manifest["horizons"][key])
    floor = old_exploratory_noise_floor(args.noise_floor, metric)
    gate_a = UL.assess_identifiability(usable, long_horizon, noise_floor=floor)
    inverse_report = PA.assess_sufficiency(usable, long_horizon, short_horizon=short_horizon)
    latent_result = None
    if all(LATENT_PROXY in record.proxy_features for record in usable):
        latent_result = PA.audit_proxy(usable, LATENT_PROXY, long_horizon)

    caveat = (
        "EXPLORATORY ONLY: training-side efficacy is not J_eff; clean efficacy is copied "
        "from training reward, all harm channels are zero-filled, and shorter-horizon "
        "utilities reuse the final dose report. No paper claim or estimator/allocator "
        "authorization may use this artifact."
    )
    payload = {
        "kind": "practice_utility_labels",
        "schema_version": 2,
        "analysis_mode": "exploratory_training_fallback",
        "claim_grade": False,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "efficacy_source": "training_side_mean_reward" if metric == FALLBACK_EFFICACY else metric,
        "efficacy_window": args.efficacy_window,
        "efficacy_caveat": caveat,
        "horizons": manifest["horizons"],
        "summary": UL.summarize_labels(usable, long_horizon),
        "reversals": PA.count_reversals(usable, short_horizon, long_horizon),
        "exploratory_gate_a_identifiability": gate_a.to_dict(),
        "latent_proxy_predictiveness": (
            {
                "valid_for_claim": False,
                "reason": caveat,
                "audit": latent_result.to_dict(),
            }
            if latent_result is not None
            else {
                "valid_for_claim": False,
                "reason": f"{caveat} Required {LATENT_PROXY!r} was not recorded.",
                "audit": None,
            }
        ),
        "estimator_authorization_decision": {
            "valid_for_authorization": False,
            "authorizes_estimator": False,
            "reason": caveat,
            "exploratory_proxy_insufficiency_audit": inverse_report.to_dict(),
        },
        "skipped": skipped,
        "records": [record.to_dict() for record in usable],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(caveat)
    print(f"wrote {args.output}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = load_json_object(args.manifest, "probe manifest")
    validate_manifest(manifest)
    if args.efficacy_window <= 0:
        raise ValueError("--efficacy-window must be positive")
    if args.exploratory_training_fallback:
        return exploratory_main(args, manifest)
    return claim_grade_main(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
