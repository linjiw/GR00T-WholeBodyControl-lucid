"""Fail-closed contracts for the claim-bearing utility-label analysis."""

# ruff: noqa: I001  # repository isort and Ruff force-sort rules conflict

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gear_sonic.research.practice_utility import directional_calibration as DC
from gear_sonic.research.practice_utility import proxy_audit as PA
from gear_sonic.research.practice_utility.schema import ContextKey, motion_hash
from scripts.practice_utility import build_utility_labels as B

HORIZONS = {"H_s": 2, "H_l": 4}
SEEDS = (11, 12)
CAMPAIGN = "screen_test"
STAGE = "late"


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def manifest_payload() -> dict:
    contexts = []
    for index, family in enumerate(("walk", "jump")):
        motion_key = f"motion_{index}__A{index:03d}"
        context = ContextKey(
            motion_key=motion_key,
            motion_hash=motion_hash(motion_key, 120, 30.0),
            bin_index=index,
            bin_start_frame=50 * index,
            bin_end_frame=50 * (index + 1),
        )
        contexts.append(
            {
                "context": context.to_dict(),
                "context_id": context.context_id,
                "failure_rate": 0.2 + 0.6 * index,
                "sampling_probability": 0.01 + 0.01 * index,
                "family": family,
                "extras": {"motion_root_speed_mean": 0.5 + index},
            }
        )
    payload = {
        "kind": B.PROBE_MANIFEST_KIND,
        "schema_version": 1,
        "campaign_id": CAMPAIGN,
        "stages": [STAGE],
        "contexts_per_stage": {STAGE: contexts},
        "seeds": list(SEEDS),
        "horizons": HORIZONS,
        "epsilon": 0.1,
        "kernel_radius_bins": 1,
        "pool_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "num_intervention_branches": len(contexts) * len(SEEDS),
        "num_control_branches": len(SEEDS),
    }
    payload["manifest_sha256"] = B.recompute_manifest_sha256(payload)
    return payload


def efficacy_metadata(**overrides) -> dict:
    payload = {
        "name": "j_eff_macro_quality_success",
        "units": "success_fraction",
        "utility_units": "success_fraction_per_completed_kernel_step",
        "window": 4,
        "quality_qualified": True,
        "macro_average_group": "motion_family",
        "harm_channels": list(B.REQUIRED_HARM_CHANNELS),
    }
    payload.update(overrides)
    return payload


def proxy_payload(manifest: dict) -> dict:
    records = []
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed in manifest["seeds"]:
            for index, entry in enumerate(contexts):
                records.append(
                    {
                        "stage": stage,
                        "seed": seed,
                        "context_id": entry["context_id"],
                        "proxy_features": {"latent_gap_p90": 0.1 + 0.8 * index},
                    }
                )
    return {
        "kind": B.PROXY_FEATURE_KIND,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_before_outcomes": True,
        "encoder_sha256": "e" * 64,
        "records": records,
    }


def noise_payload(manifest: dict) -> dict:
    return {
        "kind": B.NOISE_FLOOR_KIND,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "estimand": {
            "name": "j_eff_macro_quality_success",
            "outcome_units": "success_fraction",
            "utility_units": "success_fraction_per_completed_kernel_step",
            "quantity": "dose_normalized_practice_utility",
            "normalization": "realized_extra_completed_kernel_steps",
            "horizon_label": "H_l",
            "horizon_iterations": HORIZONS["H_l"],
            "window": 4,
        },
        "utility_deltas": [-1e-5, 0.0, 1e-5],
    }


def evaluation_payload(manifest: dict, **efficacy_overrides) -> dict:
    rows = []
    campaign = manifest["campaign_id"]
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed in manifest["seeds"]:
            control = f"{campaign}_{stage}_s{seed}_control"
            for horizon in ("H_l",):
                rows.append(
                    {
                        "branch_id": control,
                        "role": "control",
                        "horizon_label": horizon,
                        "j_eff": 0.50,
                        "clean_j_eff": 0.80,
                        "action_rate": 0.01,
                        "foot_slip": 0.02,
                        "contact_impulse": 10.0,
                        "torque_saturation": 0.01,
                    }
                )
            for index, _ in enumerate(contexts):
                intervention = f"{campaign}_{stage}_s{seed}_c{index}_intervention"
                for horizon in ("H_l",):
                    rows.append(
                        {
                            "branch_id": intervention,
                            "role": "intervention",
                            "horizon_label": horizon,
                            "j_eff": 0.55 if index == 0 else 0.45,
                            "clean_j_eff": 0.81,
                            "action_rate": 0.011,
                            "foot_slip": 0.021,
                            "contact_impulse": 10.5,
                            "torque_saturation": 0.011,
                        }
                    )
    return {
        "kind": B.BRANCH_EVALUATION_KIND,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "efficacy": efficacy_metadata(**efficacy_overrides),
        "horizons": {"H_l": HORIZONS["H_l"]},
        "records": rows,
    }


def preregistration_payload(
    manifest: dict, proxy_sha256: str, noise_sha256: str, *, grouping="motion_family"
) -> dict:
    return {
        "kind": B.PREREGISTRATION_KIND,
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "analysis_mode": "claim_grade",
        "efficacy": efficacy_metadata(),
        "gate_a": {
            "horizon_label": "H_l",
            "horizon_iterations": HORIZONS["H_l"],
            "noise_floor_sha256": noise_sha256,
            "min_variance_ratio": 2.0,
            "min_icc": 0.4,
            "origin_global_step_by_stage": {STAGE: 24},
        },
        "latent_proxy_audit": {
            "proxy": "latent_gap_p90",
            "horizon_label": "H_l",
            "grouping": grouping,
            "proxy_features_sha256": proxy_sha256,
            "rank_thresholds": {
                "min_abs_spearman": PA.SUFFICIENCY["min_abs_spearman"],
                "min_pairwise_accuracy": PA.SUFFICIENCY["min_pairwise_accuracy"],
            },
            "directional_test": "nested_cv_univariate_calibration",
            "directional_calibration": DC.default_algorithm_artifact(),
            "raw_sign_accuracy_allowed": False,
        },
        "estimator_authorization": {
            "horizon_label": "H_l",
            "proxies": ["latent_gap_p90"],
            "inverse_decision": True,
        },
    }


def write_doses(campaign_dir: Path, manifest: dict) -> None:
    campaign = manifest["campaign_id"]
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed in manifest["seeds"]:
            control = f"{campaign}_{stage}_s{seed}_control"
            write_json(
                campaign_dir / control / "dose_0004.json",
                {
                    "branch_id": control,
                    "context_id": "shared_control",
                    "role": "control",
                    "global_step": 28,
                    "never_armed": True,
                    "completed_env_steps": 1000,
                    "completed_kernel_steps": 100,
                },
            )
            for index, entry in enumerate(contexts):
                branch = f"{campaign}_{stage}_s{seed}_c{index}_intervention"
                write_json(
                    campaign_dir / branch / "dose_0004.json",
                    {
                        "branch_id": branch,
                        "context_id": entry["context_id"],
                        "role": "intervention",
                        "global_step": 28,
                        "never_armed": False,
                        "completed_env_steps": 1000,
                        "completed_kernel_steps": 150,
                    },
                )


def claim_fixture(tmp_path: Path) -> dict[str, Path | dict]:
    manifest = manifest_payload()
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    proxy_path = write_json(tmp_path / "proxy.json", proxy_payload(manifest))
    noise_path = write_json(tmp_path / "noise.json", noise_payload(manifest))
    evaluation_path = write_json(tmp_path / "evaluations.json", evaluation_payload(manifest))
    preregistration_path = write_json(
        tmp_path / "preregistration.json",
        preregistration_payload(manifest, B.file_sha256(proxy_path), B.file_sha256(noise_path)),
    )
    campaign_dir = tmp_path / "campaign"
    write_doses(campaign_dir, manifest)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "proxy_path": proxy_path,
        "noise_path": noise_path,
        "evaluation_path": evaluation_path,
        "preregistration_path": preregistration_path,
        "campaign_dir": campaign_dir,
        "output": tmp_path / "labels.json",
    }


def claim_argv(paths: dict[str, Path | dict]) -> list[str]:
    return [
        "--manifest",
        str(paths["manifest_path"]),
        "--campaign-dir",
        str(paths["campaign_dir"]),
        "--output",
        str(paths["output"]),
        "--proxy-features",
        str(paths["proxy_path"]),
        "--noise-floor",
        str(paths["noise_path"]),
        "--branch-evaluations",
        str(paths["evaluation_path"]),
        "--preregistration",
        str(paths["preregistration_path"]),
    ]


class TestClaimGradeBoundary:
    def test_default_refuses_to_fall_back_to_training_reward(self, tmp_path):
        manifest = write_json(tmp_path / "manifest.json", manifest_payload())
        with pytest.raises(ValueError, match="claim-grade mode requires"):
            B.main(
                [
                    "--manifest",
                    str(manifest),
                    "--campaign-dir",
                    str(tmp_path / "campaign"),
                    "--output",
                    str(tmp_path / "labels.json"),
                ]
            )

    def test_training_metric_flag_is_forbidden_in_claim_grade(self, tmp_path):
        paths = claim_fixture(tmp_path)
        with pytest.raises(ValueError, match="training-only"):
            B.main([*claim_argv(paths), "--efficacy-metric", "Mean rewards"])

    def test_claim_path_emits_blocked_non_claim_receipt_without_running_gates(
        self, tmp_path, monkeypatch
    ):
        paths = claim_fixture(tmp_path)

        def fail_if_gate_runs(*args, **kwargs):
            pytest.fail("Gate A/B must not run while claim lineage is blocked")

        monkeypatch.setattr(B.UL, "assess_identifiability", fail_if_gate_runs)
        monkeypatch.setattr(B.PA, "audit_proxy", fail_if_gate_runs)
        assert B.main(claim_argv(paths)) == 2
        payload = json.loads(Path(paths["output"]).read_text())
        assert payload["status"] == "blocked"
        assert payload["analysis_mode"] == "claim_grade_blocked"
        assert payload["claim_grade"] is False
        assert payload["usable_for_gate_a_b"] is False
        assert payload["manifest_sha256_verified"] is True
        assert payload["manifest_sha256_recomputed"] == payload["manifest_sha256"]
        assert payload["records"] == []
        assert payload["gate_a_identifiability"]["status"] == "not_run"
        assert "latent_proxy_predictiveness" in payload
        assert "estimator_authorization_decision" in payload
        assert "gate_b_sufficiency" not in payload
        assert payload["latent_proxy_predictiveness"]["grouping"] == "motion_family"
        assert payload["latent_proxy_predictiveness"]["decision_complete"] is False
        assert payload["latent_proxy_predictiveness"]["supports_latent_proxy_claim"] is False
        assert payload["latent_proxy_predictiveness"]["rank_only_audit"] is None
        assert payload["unavailable_horizons"] == {
            "H_s": (
                "not emitted: UtilityRecord has one final DoseReport, so H_l dose cannot "
                "normalize a shorter-horizon utility"
            )
        }
        blocker_codes = {blocker["code"] for blocker in payload["blockers"]}
        assert {
            "ready_preflight_required",
            "shared_control_realized_dose_unimplemented",
            "branch_evaluation_h_l_policy_capsule_binding_unimplemented",
            "branch_evaluation_dev_suite_binding_unimplemented",
            "branch_evaluation_physics_seed_binding_unimplemented",
            "per_evaluation_receipts_unimplemented",
        } == blocker_codes
        directional = payload["latent_proxy_predictiveness"]["directional_test"]
        assert directional["implemented"] is True
        assert (
            directional["algorithm_sha256"] == DC.default_algorithm_artifact()["algorithm_sha256"]
        )
        assert payload["estimator_authorization_decision"]["inverse_of_proxy_sufficiency"] is True
        assert payload["estimator_authorization_decision"]["valid_for_authorization"] is False
        assert payload["estimator_authorization_decision"]["authorizes_estimator"] is False

    def test_claim_grade_uses_only_exact_origin_plus_h_l_dose_report(self, tmp_path):
        paths = claim_fixture(tmp_path)
        dose_path = next(Path(paths["campaign_dir"]).glob("*_intervention/dose_*.json"))
        dose = json.loads(dose_path.read_text())
        dose["global_step"] = 27
        write_json(dose_path, dose)
        with pytest.raises(ValueError, match=r"origin \+ H_l"):
            B.load_dose_report(Path(paths["campaign_dir"]), dose["branch_id"], 28)

    def test_never_armed_intervention_is_not_claim_grade(self, tmp_path):
        paths = claim_fixture(tmp_path)
        dose_path = next(Path(paths["campaign_dir"]).glob("*_intervention/dose_*.json"))
        dose = json.loads(dose_path.read_text())
        dose["never_armed"] = True
        write_json(dose_path, dose)
        with pytest.raises(ValueError, match="never_armed=false"):
            B.dose_from_report(
                dose,
                "intervention",
                dose["branch_id"],
                claim_grade=True,
                expected_context_id=dose["context_id"],
            )

    def test_direct_claim_assembly_cannot_bypass_shared_control_blocker(self, tmp_path):
        with pytest.raises(ValueError, match="shared-control realized dose"):
            B.assemble_claim_grade_records(
                manifest_payload(),
                tmp_path,
                {},
                {},
                "H_l",
                {STAGE: 24},
            )


class TestManifestIntegrity:
    def test_recomputes_and_rejects_a_tampered_logical_manifest_hash(self):
        manifest = manifest_payload()
        B.validate_manifest(manifest)
        manifest["epsilon"] = 0.2
        with pytest.raises(ValueError, match="recomputed logical hash"):
            B.validate_manifest(manifest)

    def test_recomputes_context_id_from_the_canonical_context(self):
        manifest = manifest_payload()
        manifest["contexts_per_stage"][STAGE][0]["context"]["bin_end_frame"] += 1
        manifest["manifest_sha256"] = B.recompute_manifest_sha256(manifest)
        with pytest.raises(ValueError, match="context_id mismatch"):
            B.validate_manifest(manifest)


class TestFrozenProxyJoin:
    def test_requires_exact_stage_seed_context_coverage(self):
        manifest = manifest_payload()
        artifact = proxy_payload(manifest)
        artifact["records"].pop()
        with pytest.raises(ValueError, match="do not exactly cover"):
            B.load_proxy_feature_index(artifact, manifest)

    def test_rejects_duplicate_exact_key(self):
        manifest = manifest_payload()
        artifact = proxy_payload(manifest)
        artifact["records"].append(copy.deepcopy(artifact["records"][0]))
        with pytest.raises(ValueError, match="duplicate proxy feature key"):
            B.load_proxy_feature_index(artifact, manifest)

    def test_every_row_requires_latent_gap_p90(self):
        manifest = manifest_payload()
        artifact = proxy_payload(manifest)
        artifact["records"][0]["proxy_features"].clear()
        with pytest.raises(ValueError, match="latent_gap_p90"):
            B.load_proxy_feature_index(artifact, manifest)

    def test_motion_family_grouping_comes_from_manifest(self):
        manifest = manifest_payload()
        name, groups = B.resolve_proxy_grouping("motion_family", manifest)
        assert name == "motion_family"
        assert set(groups.values()) == {"walk", "jump"}

    def test_exact_preregistered_grouping_requires_exact_coverage(self):
        manifest = manifest_payload()
        assignments = [
            {"stage": stage, "seed": seed, "context_id": context_id, "group": "g"}
            for stage, seed, context_id in B.expected_feature_rows(manifest)
        ]
        assignments.pop()
        with pytest.raises(ValueError, match="does not cover"):
            B.resolve_proxy_grouping({"kind": "exact", "assignments": assignments}, manifest)

    def test_inverse_authorization_requires_every_preregistered_proxy(self):
        class Record:
            policy_stage = "late"
            seed = 11
            context = type("Context", (), {"context_id": "ctx"})()
            proxy_features = {"latent_gap_p90": 0.2}

        with pytest.raises(ValueError, match="complete feature coverage"):
            B.require_proxy_coverage([Record()], ["latent_gap_p90", "td_error"])


class TestPreregistrationAndNoiseFloor:
    def test_preregistration_must_hash_the_frozen_feature_artifact(self, tmp_path):
        paths = claim_fixture(tmp_path)
        preregistration = json.loads(Path(paths["preregistration_path"]).read_text())
        preregistration["latent_proxy_audit"]["proxy_features_sha256"] = "0" * 64
        write_json(Path(paths["preregistration_path"]), preregistration)
        with pytest.raises(ValueError, match="proxy_features_sha256"):
            B.main(claim_argv(paths))

    def test_nonnegative_latent_proxy_cannot_use_raw_sign_accuracy(self, tmp_path):
        paths = claim_fixture(tmp_path)
        preregistration = json.loads(Path(paths["preregistration_path"]).read_text())
        preregistration["latent_proxy_audit"]["raw_sign_accuracy_allowed"] = True
        write_json(Path(paths["preregistration_path"]), preregistration)
        with pytest.raises(ValueError, match="raw sign accuracy"):
            B.main(claim_argv(paths))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("outcome_units", "training_reward"),
            ("horizon_label", "H_s"),
            ("horizon_iterations", 999),
            ("window", 1),
            ("quantity", "raw_training_delta"),
        ],
    )
    def test_noise_floor_must_match_units_horizon_window_and_quantity(self, field, value):
        manifest = manifest_payload()
        artifact = noise_payload(manifest)
        artifact["estimand"][field] = value
        with pytest.raises(ValueError, match="noise-floor estimand"):
            B.validate_noise_floor(artifact, manifest, efficacy_metadata(), "H_l")

    def test_legacy_epsilon_zero_training_delta_report_is_rejected(self):
        manifest = manifest_payload()
        legacy = {
            "kind": "practice_utility_noise_floor",
            "schema_version": 1,
            "metrics": {"Mean rewards": {"paired_deltas": [0.0, 0.0]}},
        }
        with pytest.raises(ValueError, match="kind must be"):
            B.validate_noise_floor(legacy, manifest, efficacy_metadata(), "H_l")


class TestMeasuredEvaluations:
    def test_training_reward_estimand_is_rejected(self):
        manifest = manifest_payload()
        artifact = evaluation_payload(manifest, name="Mean rewards", quality_qualified=False)
        with pytest.raises(ValueError, match="differs from preregistration"):
            B.validate_branch_evaluations(artifact, manifest, efficacy_metadata(), "H_l")

    def test_zero_filled_harm_channels_are_rejected(self):
        manifest = manifest_payload()
        artifact = evaluation_payload(manifest)
        for row in artifact["records"]:
            for channel in B.REQUIRED_HARM_CHANNELS:
                row[channel] = 0.0
        with pytest.raises(ValueError, match="zero-filled"):
            B.validate_branch_evaluations(artifact, manifest, efficacy_metadata(), "H_l")

    def test_short_horizon_evaluation_is_refused_without_short_horizon_dose(self):
        manifest = manifest_payload()
        artifact = evaluation_payload(manifest)
        artifact["horizons"] = HORIZONS
        with pytest.raises(ValueError, match="only the preregistered H_l"):
            B.validate_branch_evaluations(artifact, manifest, efficacy_metadata(), "H_l")


def training_table(iteration: int, reward: float) -> str:
    return "\n".join(
        [
            f"│ Learning iteration {iteration} │",
            f"│ Mean rewards: {reward} │",
            "│ Mean length: 100.0 │",
        ]
    )


def test_explicit_exploratory_fallback_is_caveated_and_cannot_authorize(tmp_path):
    manifest = manifest_payload()
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    campaign_dir = tmp_path / "campaign"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    write_doses(campaign_dir, manifest)
    campaign = manifest["campaign_id"]
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed in manifest["seeds"]:
            control = f"{campaign}_{stage}_s{seed}_control"
            (log_dir / f"{control}.log").write_text(
                "\n".join(training_table(i, float(i)) for i in range(1, 5))
            )
            for index, _ in enumerate(contexts):
                branch = f"{campaign}_{stage}_s{seed}_c{index}_intervention"
                (log_dir / f"{branch}.log").write_text(
                    "\n".join(training_table(i, float(i) + 0.1 * (index + 1)) for i in range(1, 5))
                )
    output = tmp_path / "exploratory.json"
    assert (
        B.main(
            [
                "--manifest",
                str(manifest_path),
                "--campaign-dir",
                str(campaign_dir),
                "--log-dir",
                str(log_dir),
                "--output",
                str(output),
                "--exploratory-training-fallback",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["claim_grade"] is False
    assert "EXPLORATORY ONLY" in payload["efficacy_caveat"]
    assert payload["estimator_authorization_decision"]["authorizes_estimator"] is False
    assert payload["estimator_authorization_decision"]["valid_for_authorization"] is False
