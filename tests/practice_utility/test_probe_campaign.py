"""Claim-grade, fail-closed Track-A preflight contracts."""

# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for mixed import/from-import blocks.
# ruff: noqa: I001

from __future__ import annotations

import json
from pathlib import Path

import torch

from gear_sonic.research.practice_utility import branch_capsule as BC
from gear_sonic.research.practice_utility import probe_campaign as PC
from gear_sonic.research.practice_utility import proxy_audit as PA
from gear_sonic.research.practice_utility.rng_capsule import RngState
from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    motion_hash,
    sha256_of,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def reference(path: Path, **extras) -> dict:
    return {"path": str(path), "sha256": PC.sha256_file(path), **extras}


def manifest_payload() -> dict:
    contexts = []
    for index, family in enumerate(("walk", "jump")):
        key = f"motion_{index}__A{index:03d}"
        context = ContextKey(
            key,
            motion_hash(key, 100, 30.0),
            index,
            50 * index,
            50 * (index + 1),
        )
        contexts.append(
            {
                "context": context.to_dict(),
                "context_id": context.context_id,
                "failure_rate": 0.2 + 0.5 * index,
                "sampling_probability": 0.1,
                "failure_quartile": 2 * index,
                "family": family,
                "contact_regime": "steady",
                "partition": "adaptation",
                "extras": {},
            }
        )
    payload = {
        "kind": "practice_utility_probe_manifest",
        "schema_version": 1,
        "campaign_id": "screen_test",
        "stages": ["late"],
        "seeds": [11, 12],
        "epsilon": 0.1,
        "kernel_radius_bins": 1,
        "horizons": {"H_s": 2, "H_l": 4},
        "pool_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "num_intervention_branches": 4,
        "num_control_branches": 2,
        "contexts_per_stage": {"late": contexts},
    }
    payload["manifest_sha256"] = PC.manifest_claim_sha256(payload)
    return payload


def full_origin(
    tmp_path: Path, manifest: dict, seed: int, dev_suite_sha256: str
) -> tuple[Path, Path, Path, dict]:
    step = 24
    capsule = tmp_path / f"seed_{seed}" / "origin.capsule.pt"
    provenance = BC.Provenance(
        resolved_config_sha256="1" * 64,
        motion_pool_manifest_sha256=manifest["pool_sha256"],
        dev_suite_sha256=dev_suite_sha256,
        source_commit="a" * 40,
        checkpoint_sha256="2" * 64,
    )
    BC.save_capsule(
        capsule,
        branch_id=f"origin_s{seed}",
        pair_id=f"origin_s{seed}",
        role="control",
        global_step=step,
        model_state={
            "policy_state_dict": {"w": torch.tensor([1.0, 2.0])},
            "value_state_dict": {"w": torch.tensor([3.0])},
            "combined_state_dict": {"w": torch.tensor([1.0])},
            "lr_scheduler_state_dict": {"last_epoch": step},
        },
        optimizer_state={"state": {0: {"step": step}}},
        trainer_state={"global_step": step, "trainer_state_obj": {"global_step": step}},
        env_state={"episode_count": 8},
        native_sampler_state={
            "adp_samp_num_episodes": torch.ones(2),
            "adp_samp_num_failures": torch.ones(2),
        },
        rng_state=RngState.capture(f"origin_s{seed}"),
        provenance=provenance,
    )
    checkpoint = tmp_path / f"seed_{seed}" / "origin_checkpoint.pt"
    BC.export_sonic_checkpoint(capsule, checkpoint)

    rows = []
    for global_bin, entry in enumerate(manifest["contexts_per_stage"]["late"]):
        rows.append(
            {
                **entry["context"],
                "context_id": entry["context_id"],
                "global_bin_id": global_bin,
                "sampling_probability": 0.5,
                "failure_rate": entry["failure_rate"],
                "num_episodes": 10.0,
                "num_failures": 2.0,
            }
        )
    snapshot = write_json(
        tmp_path / f"seed_{seed}" / "origin_snapshot.json",
        {
            "kind": "practice_utility_sampler_snapshot",
            "schema_version": 1,
            "global_step": step,
            "num_bins": 2,
            "num_active_bins": 2,
            "distribution_sha256": "3" * 64,
            "contexts": rows,
        },
    )
    context_ids = sorted(entry["context_id"] for entry in manifest["contexts_per_stage"]["late"])
    row = {
        "origin_step": step,
        "source_step": 12,
        "capsule": str(capsule),
        "capsule_sha256": PC.sha256_file(capsule),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": PC.sha256_file(checkpoint),
        "snapshot": str(snapshot),
        "snapshot_sha256": PC.sha256_file(snapshot),
        "resident_context_ids": context_ids,
        "num_resident_contexts": len(context_ids),
        "stability": {
            "reward": {
                "previous4": 10.0,
                "last4": 10.1,
                "relative_delta": 0.01,
                "absolute_relative_limit": 0.0666,
                "passes": True,
            },
            "length": {
                "previous4": 100.0,
                "last4": 101.0,
                "relative_delta": 0.01,
                "absolute_relative_limit": 0.0628,
                "passes": True,
            },
        },
        "settled": True,
        "seed": seed,
        "num_envs": 256,
        "blockers": [],
    }
    return capsule, checkpoint, snapshot, row


def valid_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = manifest_payload()
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    dev_suite = "d" * 64

    origins = {}
    snapshots = {}
    for seed in manifest["seeds"]:
        _, _, snapshot, row = full_origin(tmp_path, manifest, seed, dev_suite)
        origins[str(seed)] = row
        snapshots[seed] = snapshot
    common = sorted(entry["context_id"] for entry in manifest["contexts_per_stage"]["late"])
    origin_map = write_json(
        tmp_path / "origin_map.json",
        {
            "kind": "practice_utility_probe_origin_map",
            "schema_version": 1,
            "experiment_id": "origins_test",
            "source_checkpoint": "/frozen/source.pt",
            "source_checkpoint_sha256": "2" * 64,
            "source_step": 12,
            "origin_step": 24,
            "num_envs": 256,
            "seeds": manifest["seeds"],
            "origins": origins,
            "common_resident_context_ids": common,
            "num_common_resident_contexts": len(common),
            "usable_for_manifest_selection": True,
        },
    )

    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"frozen encoder")
    efficacy = {
        "name": "j_eff_macro_quality_success",
        "units": "success_fraction",
        "utility_units": "success_fraction_per_completed_kernel_step",
        "window": 4,
        "quality_qualified": True,
        "macro_average_group": "motion_family",
        "harm_channels": [
            "action_rate",
            "foot_slip",
            "contact_impulse",
            "torque_saturation",
        ],
    }
    estimand = {
        "name": efficacy["name"],
        "outcome_units": efficacy["units"],
        "utility_units": efficacy["utility_units"],
        "quantity": "dose_normalized_practice_utility",
        "normalization": "realized_extra_completed_kernel_steps",
        "horizon_label": "H_l",
        "horizon_iterations": 4,
        "window": 4,
    }
    efficacy_plan = write_json(
        tmp_path / "efficacy_plan.json",
        {
            "kind": "practice_utility_deployment_efficacy_plan",
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen": True,
            "frozen_before_campaign": True,
            "efficacy": efficacy,
            "estimand_sha256": sha256_of(estimand),
            "objective": {
                "name": "J_eff",
                "aggregation": "macro_mean_over_motion_families",
                "qualified_success_requires": [
                    "completion",
                    "mpjpe",
                    "foot_slip",
                    "high_frequency_action",
                    "undesired_contact",
                    "torque_saturation",
                ],
                "thresholds": {
                    "mpjpe": 0.1,
                    "foot_slip": 0.02,
                    "high_frequency_action": 0.1,
                    "undesired_contact": 0.05,
                    "torque_saturation": 0.02,
                },
                "thresholds_frozen": True,
                "threshold_source": "nominal baseline quantiles and safety limits",
            },
            "evaluation": {
                "policy_frozen": True,
                "matched_motion_panel": True,
                "matched_physics_seeds": True,
                "evaluation_receipt_per_branch_horizon": True,
                "quality_outcomes_reported_separately": True,
                "performer_and_content_splits_reported_separately": True,
                "learning_during_evaluation": False,
                "training_metric_fallback_allowed": False,
                "final_test_accessible": False,
                "dev_suite_sha256": dev_suite,
            },
        },
    )

    records = []
    for seed in manifest["seeds"]:
        for index, entry in enumerate(manifest["contexts_per_stage"]["late"]):
            records.append(
                {
                    "stage": "late",
                    "seed": seed,
                    "context_id": entry["context_id"],
                    "source_snapshot_sha256": PC.sha256_file(snapshots[seed]),
                    "proxy_features": {"latent_gap_p90": 0.1 + index},
                }
            )
    proxy = write_json(
        tmp_path / "proxy.json",
        {
            "kind": "practice_utility_proxy_features",
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen_before_outcomes": True,
            "encoder_sha256": PC.sha256_file(encoder),
            "records": records,
        },
    )
    noise = write_json(
        tmp_path / "noise.json",
        {
            "kind": "practice_utility_same_estimand_noise_floor",
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "estimand": estimand,
            "estimand_sha256": sha256_of(estimand),
            "efficacy_plan_sha256": PC.sha256_file(efficacy_plan),
            "epsilon": 0.0,
            "restart_protocol": "symmetric_fresh_restart",
            "origin_state": "settled",
            "replicate_design": "cross_seed_symmetric_restart",
            "randomness_contract": PC.STOCHASTIC_RANDOMNESS_CONTRACT,
            "utility_deltas": [-0.01, 0.0, 0.01],
        },
    )
    gates = write_json(
        tmp_path / "gates.json",
        {
            "kind": "practice_utility_gate_preregistration",
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "analysis_mode": "claim_grade",
            "estimand_sha256": sha256_of(estimand),
            "efficacy": efficacy,
            "gate_a": {
                "horizon_label": "H_l",
                "horizon_iterations": 4,
                "noise_floor_sha256": PC.sha256_file(noise),
                "min_variance_ratio": 2.0,
                "min_icc": 0.4,
                "origin_global_step_by_stage": {"late": 24},
            },
            "latent_proxy_audit": {
                "proxy": "latent_gap_p90",
                "horizon_label": "H_l",
                "grouping": "motion_family",
                "proxy_features_sha256": PC.sha256_file(proxy),
                "rank_thresholds": {
                    "min_abs_spearman": PA.SUFFICIENCY["min_abs_spearman"],
                    "min_pairwise_accuracy": PA.SUFFICIENCY["min_pairwise_accuracy"],
                },
                "directional_test": "nested_cv_univariate_calibration",
                "raw_sign_accuracy_allowed": False,
            },
            "estimator_authorization": {
                "horizon_label": "H_l",
                "proxies": ["native_failure_rate", "latent_gap_p90"],
                "inverse_decision": True,
            },
        },
    )
    preflight = write_json(
        tmp_path / "manifest.preflight.json",
        {
            "kind": "practice_utility_probe_preflight",
            "schema_version": 1,
            "preregistered": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": PC.sha256_file(manifest_path),
            "screening_control_strategy": PC.SHARED_CONTROL_STRATEGY,
            "randomness_contract": PC.STOCHASTIC_RANDOMNESS_CONTRACT,
            "origin_maps": {"late": reference(origin_map)},
            "encoder": reference(
                encoder,
                frozen=True,
                frozen_before_campaign=True,
                artifact_id="encoder_test",
            ),
            "proxy_features": reference(proxy),
            "efficacy_plan": reference(efficacy_plan),
            "noise_floor": reference(noise),
            "gate_preregistration": reference(gates),
        },
    )
    return manifest_path, preflight


def finding_codes(findings) -> set[str]:
    return {finding.code for finding in findings}


def test_missing_sidecar_fails_closed_with_actionable_blockers(tmp_path):
    manifest = write_json(tmp_path / "manifest.json", manifest_payload())
    report = PC.audit_probe_campaign(manifest)
    assert not report.ready
    assert {
        "preflight_bundle_missing",
        "origin_map_missing",
        "proxy_feature_coverage_missing",
        "same_estimand_noise_floor_missing",
        "gate_preregistration_missing",
        "branch_specs_unavailable",
    } <= finding_codes(report.blockers)


def test_complete_bundle_preserves_both_unimplemented_claim_blockers(tmp_path):
    manifest, preflight = valid_fixture(tmp_path)
    report = PC.audit_probe_campaign(manifest, preflight)
    assert not report.ready
    assert finding_codes(report.blockers) == {
        "shared_control_realized_dose_unimplemented",
        "latent_directional_calibration_unimplemented",
    }
    assert len(report.branch_specs) == 6
    assert sum(spec.role == "control" for spec in report.branch_specs) == 2
    assert sum(spec.role == "intervention" for spec in report.branch_specs) == 4
    assert all(spec.absolute_horizons == {"H_s": 26, "H_l": 28} for spec in report.branch_specs)
    assert all(spec.target_global_step == 28 for spec in report.branch_specs)
    assert "shared_controls_screening_only" in finding_codes(report.warnings)
    assert "gate_vocabulary_verified" in finding_codes(report.verified)


def test_manifest_tampering_is_detected_before_launch(tmp_path):
    manifest, preflight = valid_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["epsilon"] = 0.2
    write_json(manifest, payload)
    report = PC.audit_probe_campaign(manifest, preflight)
    assert not report.ready
    assert "manifest_hash_mismatch" in finding_codes(report.blockers)
    assert "preflight_manifest_file_hash_mismatch" in finding_codes(report.blockers)
