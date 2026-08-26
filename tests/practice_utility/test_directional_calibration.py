"""CPU-only contracts for the directional latent-gap calibration."""

# ruff: noqa: I001  # repository isort and Ruff force-sort rules conflict

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from gear_sonic.research.practice_utility import directional_calibration as D
from gear_sonic.research.practice_utility.schema import (
    ContextKey,
    motion_hash,
    sha256_of,
)
from scripts.practice_utility import build_utility_labels as B
from scripts.practice_utility import freeze_directional_calibration as F
from tests.practice_utility.test_build_utility_labels import (
    manifest_payload,
    write_json,
)

FAMILY_SIZES = {
    "carry": 1,
    "crouch": 2,
    "dance": 2,
    "groom": 1,
    "idle": 1,
    "jump": 1,
    "other": 6,
    "run": 6,
    "search": 1,
    "walk": 3,
}


def design_rows() -> list[D.CalibrationDesignRow]:
    rows = []
    context_index = 0
    for family, count in FAMILY_SIZES.items():
        for _ in range(count):
            context_id = f"context_{context_index:02d}"
            for seed in (9300, 9301):
                rows.append(
                    D.CalibrationDesignRow(
                        sample_id=f"late|{seed}|{context_id}",
                        context_id=context_id,
                        motion_family=family,
                    )
                )
            context_index += 1
    return rows


def calibration_rows(*, inverse: bool = False, useless: bool = False):
    rows = []
    contexts = []
    for design in design_rows():
        if design.context_id not in {context_id for context_id, _ in contexts}:
            contexts.append((design.context_id, design.motion_family))
    for index, (context_id, family) in enumerate(contexts):
        proxy = float(index + 1)
        base = proxy - 12.5
        if inverse:
            base = -base
        if useless:
            base = float(((index * 11) % 17) - 8)
        for seed_index, seed in enumerate((9300, 9301)):
            rows.append(
                D.CalibrationRow(
                    sample_id=f"late|{seed}|{context_id}",
                    context_id=context_id,
                    motion_family=family,
                    latent_gap_p90=proxy,
                    utility=base + (-0.01 if seed_index == 0 else 0.01),
                )
            )
    return rows


def freezer_manifest_payload() -> dict:
    manifest = manifest_payload()
    contexts = []
    context_index = 0
    for family, count in FAMILY_SIZES.items():
        for _ in range(count):
            motion_key = f"motion_{context_index:02d}__A{context_index:03d}"
            context = ContextKey(
                motion_key=motion_key,
                motion_hash=motion_hash(motion_key, 200, 50.0),
                bin_index=context_index,
                bin_start_frame=50 * context_index,
                bin_end_frame=50 * (context_index + 1),
            )
            contexts.append(
                {
                    "context": context.to_dict(),
                    "context_id": context.context_id,
                    "failure_rate": (context_index + 1) / 25.0,
                    "sampling_probability": 1.0 / 24.0,
                    "family": family,
                    "extras": {},
                }
            )
            context_index += 1
    manifest["contexts_per_stage"]["late"] = contexts
    manifest["num_intervention_branches"] = len(contexts) * len(manifest["seeds"])
    manifest["manifest_sha256"] = B.recompute_manifest_sha256(manifest)
    B.validate_manifest(manifest)
    return manifest


class TestAlgorithmArtifact:
    def test_default_artifact_is_self_hashed_and_raw_sign_is_prohibited(self):
        artifact = D.default_algorithm_artifact()
        recorded = artifact.pop("algorithm_sha256")
        assert recorded == sha256_of(artifact)
        assert artifact["raw_proxy_sign_allowed"] is False

    def test_tampering_is_rejected_even_when_the_attacker_rehashes(self):
        artifact = copy.deepcopy(D.default_algorithm_artifact())
        artifact["raw_proxy_sign_allowed"] = True
        artifact.pop("algorithm_sha256")
        artifact["algorithm_sha256"] = sha256_of(artifact)
        with pytest.raises(ValueError, match="only implemented"):
            D.validate_algorithm_artifact(artifact)


class TestOutcomeBlindDesign:
    def test_24_context_10_family_screen_supports_the_frozen_nested_folds(self):
        result = D.validate_design_support(design_rows(), D.default_algorithm_artifact())
        assert result["status"] == "ready"
        assert result["outcomes_read"] is False
        assert result["support"] == {
            "num_records": 48,
            "num_contexts": 24,
            "num_motion_families": 10,
            "num_rankable_motion_families": 5,
            "contexts_per_motion_family": FAMILY_SIZES,
        }
        assert len(result["folds"]) == 5
        for fold in result["folds"]:
            assert set(fold["train_motion_families"]).isdisjoint(fold["test_motion_families"])
            assert set(fold["train_context_ids"]).isdisjoint(fold["test_context_ids"])
            assert fold["inner_folds_feasible"] is True

    def test_too_few_families_blocks_before_outcomes(self):
        rows = [D.CalibrationDesignRow(f"s{i}", f"c{i}", "only") for i in range(20)]
        result = D.validate_design_support(rows, D.default_algorithm_artifact())
        assert result["status"] == "blocked"
        assert result["outcomes_read"] is False
        assert "directional_motion_families_insufficient" in {
            blocker["code"] for blocker in result["blockers"]
        }


class TestDirectionalCalibration:
    def test_gate_a_failure_does_not_even_iterate_labels(self):
        class ExplodingRows:
            def __iter__(self) -> Iterator[D.CalibrationRow]:
                raise AssertionError("labels were read before Gate A")

        result = D.run_directional_test(
            ExplodingRows(),
            gate_a_passed=False,
            noise_floor_values=[],
            utility_units="success_fraction_per_completed_kernel_step",
            algorithm_artifact=D.default_algorithm_artifact(),
        )
        assert result.status == "blocked"
        assert result.gate_a_prerequisite_passed is False
        assert result.supports_latent_proxy_claim is False
        assert result.decision_complete is False

    def test_signed_affine_relationship_passes_with_deterministic_oof_predictions(self):
        kwargs = {
            "gate_a_passed": True,
            "noise_floor_values": [-0.1, 0.0, 0.1],
            "utility_units": "success_fraction_per_completed_kernel_step",
            "algorithm_artifact": D.default_algorithm_artifact(),
        }
        first = D.run_directional_test(calibration_rows(), **kwargs)
        second = D.run_directional_test(calibration_rows(), **kwargs)
        assert first.to_dict() == second.to_dict()
        assert first.status == "pass"
        assert first.supports_latent_proxy_claim is True
        assert first.decision_complete is True
        assert first.deadband == pytest.approx(0.1)
        assert first.diagnostics["outer_oof_sign_accuracy"] == pytest.approx(1.0)
        assert first.diagnostics["outer_oof_spearman"] == pytest.approx(1.0)
        assert first.diagnostics["outer_oof_pairwise_accuracy"] == pytest.approx(1.0)
        assert first.bootstrap["method"] == "hierarchical_motion_family_then_context_block"
        for fold in first.folds:
            assert set(fold["train_motion_families"]).isdisjoint(fold["test_motion_families"])

    def test_negative_raw_relationship_can_pass_only_after_signed_calibration(self):
        result = D.run_directional_test(
            calibration_rows(inverse=True),
            gate_a_passed=True,
            noise_floor_values=[-0.1, 0.0, 0.1],
            utility_units="utility_per_step",
            algorithm_artifact=D.default_algorithm_artifact(),
        )
        assert all(row.latent_gap_p90 >= 0.0 for row in calibration_rows(inverse=True))
        assert result.status == "pass"
        assert result.supports_latent_proxy_claim is True

    def test_inadequate_sign_support_blocks_instead_of_failing_the_proxy(self):
        rows = [
            D.CalibrationRow(
                row.sample_id,
                row.context_id,
                row.motion_family,
                row.latent_gap_p90,
                0.01,
            )
            for row in calibration_rows()
        ]
        result = D.run_directional_test(
            rows,
            gate_a_passed=True,
            noise_floor_values=[-0.1, 0.0, 0.1],
            utility_units="utility_per_step",
            algorithm_artifact=D.default_algorithm_artifact(),
        )
        assert result.status == "blocked"
        assert result.decision_complete is False
        assert "directional_sign_support_insufficient" in {
            blocker["code"] for blocker in result.blockers
        }

    def test_uninformative_relationship_does_not_pass_and_never_authorizes_estimator(self):
        result = D.run_directional_test(
            calibration_rows(useless=True),
            gate_a_passed=True,
            noise_floor_values=[-0.1, 0.0, 0.1],
            utility_units="utility_per_step",
            algorithm_artifact=D.default_algorithm_artifact(),
        )
        assert result.status == "fail"
        assert result.supports_latent_proxy_claim is False
        assert "authorizes_estimator" not in result.to_dict()


def test_freezer_writes_a_self_hashed_outcome_free_artifact(tmp_path):
    manifest = freezer_manifest_payload()
    path = write_json(tmp_path / "manifest.json", manifest)
    loaded, manifest_binding = F.load_bound_manifest(path)
    payload = F.build_artifact(
        loaded,
        manifest_binding=manifest_binding,
        git_identity={"sha": "a" * 40, "status_short": []},
        launcher=F.launcher_binding(),
    )
    recorded = payload.pop("artifact_sha256")
    assert recorded == sha256_of(payload)
    assert payload["contains_outcomes"] is False
    assert payload["design_support"]["status"] == "ready"
    assert payload["algorithm"] == D.default_algorithm_artifact()


def test_freezer_refuses_a_dirty_tree_before_writing(tmp_path, monkeypatch):
    manifest_path = write_json(tmp_path / "manifest.json", freezer_manifest_payload())
    output = tmp_path / "directional.json"
    monkeypatch.setattr(F, "_git_status", lambda: [" M claim_bearing.py"])

    def git_sha_must_not_run():
        raise AssertionError("dirty-tree refusal should precede git SHA collection")

    monkeypatch.setattr(F, "_git_sha", git_sha_must_not_run)
    with pytest.raises(RuntimeError, match="requires a clean committed tree"):
        F.main(["--manifest", str(manifest_path), "--output", str(output)])
    assert not output.exists()


def test_exclusive_atomic_publish_preserves_a_racing_winner(tmp_path, monkeypatch):
    output = tmp_path / "directional.json"
    real_link = F.os.link

    def racing_link(source, destination):
        destination.write_bytes(b"racing creator\n")
        real_link(source, destination)

    monkeypatch.setattr(F.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        F.write_json_exclusive_atomic(output, {"loser": True})
    assert output.read_bytes() == b"racing creator\n"
    assert list(tmp_path.glob("*.partial")) == []
    assert list(tmp_path.glob(".*.partial")) == []


def test_freezer_records_exact_git_launcher_and_manifest_provenance(tmp_path, monkeypatch):
    manifest_path = write_json(tmp_path / "manifest.json", freezer_manifest_payload())
    output = tmp_path / "directional.json"
    monkeypatch.setattr(F, "_git_status", lambda: [])
    monkeypatch.setattr(F, "_git_sha", lambda: "b" * 40)
    assert F.main(["--manifest", str(manifest_path), "--output", str(output)]) == 0

    payload = json.loads(output.read_text())
    artifact_sha256 = payload.pop("artifact_sha256")
    assert artifact_sha256 == sha256_of(payload)
    assert payload["git"] == {"sha": "b" * 40, "status_short": []}
    assert payload["launcher"] == {
        "path": str(Path(F.__file__).resolve()),
        "sha256": F.file_sha256(Path(F.__file__).resolve()),
    }
    assert payload["source_manifest"] == {
        "path": str(manifest_path.resolve()),
        "logical_sha256": payload["manifest_sha256"],
        "file_sha256": F.file_sha256(manifest_path),
    }
    assert payload["manifest_file_sha256"] == F.file_sha256(manifest_path)
