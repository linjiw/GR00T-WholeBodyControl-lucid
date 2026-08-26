"""Outcome-blind passive-dose plan and creation-CLI contracts."""

import json
from pathlib import Path

import pytest

from gear_sonic.research.practice_utility import dose_plan as DP
from gear_sonic.research.practice_utility.schema import ContextKey, motion_hash
from scripts.practice_utility import create_passive_dose_plan as CLI


def manifest_payload() -> dict:
    key = "motion_00"
    context = ContextKey(
        motion_key=key,
        motion_hash=motion_hash(key, 200, 50.0),
        bin_index=1,
        bin_start_frame=50,
        bin_end_frame=100,
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
        "contexts_per_stage": {
            "late": [{"context_id": context.context_id, "context": context.to_dict()}]
        },
    }
    payload["manifest_sha256"] = DP.probe_manifest_claim_sha256(payload)
    return payload


def write_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload(), indent=2) + "\n")
    return path


def plan_payload(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    launcher = Path(CLI.__file__).resolve()
    return DP.build_passive_dose_plan_payload(
        manifest,
        manifest_file_sha256=DP.file_sha256(manifest_path),
        sigma_frames=50.0,
        created_at="2026-08-26T00:00:00+00:00",
        source_manifest_path=str(manifest_path),
        source_commit="a" * 40,
        git_status_short=[],
        launcher_path=str(launcher),
        launcher_sha256=DP.file_sha256(launcher),
    )


def test_v2_plan_round_trip_is_bound_to_manifest_file_and_logical_hash(tmp_path):
    manifest_path = write_manifest(tmp_path)
    payload = plan_payload(manifest_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload, indent=2) + "\n")
    loaded = DP.load_passive_dose_plan(
        plan_path,
        expected_file_sha256=DP.file_sha256(plan_path),
        expected_campaign_id="screen_test",
        expected_manifest_sha256=manifest_payload()["manifest_sha256"],
        expected_manifest_file_sha256=DP.file_sha256(manifest_path),
    )
    assert loaded.logical_sha256 == payload["dose_plan_sha256"]
    assert [context.context_id for context in loaded.contexts_for("late")] == [
        payload["contexts_per_stage"]["late"][0]["context_id"]
    ]


def test_plan_rejects_a_tampered_kernel(tmp_path):
    manifest_path = write_manifest(tmp_path)
    payload = plan_payload(manifest_path)
    payload["kernel"]["sigma_frames"] = 25.0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must equal reference_bin_size_frames"):
        DP.load_passive_dose_plan(plan_path)


def test_manifest_derives_reference_bin_size_from_indexed_start_and_terminal_width():
    base = manifest_payload()["contexts_per_stage"]["late"][0]["context"]
    first = ContextKey.from_dict(base)
    terminal = ContextKey.from_dict(
        {
            **base,
            "bin_index": 3,
            "bin_start_frame": 150,
            "bin_end_frame": 170,
        }
    )
    assert DP.derive_reference_bin_size_frames([first, terminal]) == 50


def test_plan_builder_rejects_sigma_different_from_manifest_bin_size(tmp_path):
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    with pytest.raises(ValueError, match="manifest-derived reference bin size"):
        DP.build_passive_dose_plan_payload(
            manifest,
            manifest_file_sha256=DP.file_sha256(manifest_path),
            sigma_frames=25.0,
            created_at="now",
            source_manifest_path=str(manifest_path),
            source_commit="a" * 40,
            git_status_short=[],
            launcher_path=str(Path(CLI.__file__).resolve()),
            launcher_sha256=DP.file_sha256(Path(CLI.__file__).resolve()),
        )


def test_plan_builder_rejects_tampered_manifest(tmp_path):
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["epsilon"] = 0.2
    with pytest.raises(ValueError, match="logical hash"):
        DP.build_passive_dose_plan_payload(
            manifest,
            manifest_file_sha256=DP.file_sha256(manifest_path),
            sigma_frames=50.0,
            created_at="now",
            source_manifest_path=str(manifest_path),
            source_commit="a" * 40,
            git_status_short=[],
            launcher_path=str(Path(CLI.__file__).resolve()),
            launcher_sha256=DP.file_sha256(Path(CLI.__file__).resolve()),
        )


def test_cli_dry_run_writes_nothing(tmp_path, monkeypatch):
    manifest_path = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    monkeypatch.setattr(CLI, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(CLI, "_git_status", lambda: [])
    receipt = CLI.build(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--sigma-frames",
            "50",
        ]
    )
    assert receipt["execution_mode"] == "dry_run"
    assert receipt["outcome_blind"] is True
    assert not output.exists()
    assert not output.with_suffix(".creation_receipt.json").exists()


def test_cli_rejects_free_sigma_different_from_manifest_reference_size(tmp_path, monkeypatch):
    manifest_path = write_manifest(tmp_path)
    monkeypatch.setattr(CLI, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(CLI, "_git_status", lambda: [])
    with pytest.raises(ValueError, match="manifest-derived reference bin size"):
        CLI.build(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(tmp_path / "plan.json"),
                "--sigma-frames",
                "25",
            ]
        )


def test_cli_execute_atomically_writes_plan_and_receipt(tmp_path, monkeypatch):
    manifest_path = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    monkeypatch.setattr(CLI, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(CLI, "_git_status", lambda: [])
    result = CLI.build(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--sigma-frames",
            "50",
            "--execute",
        ]
    )
    receipt_path = output.with_suffix(".creation_receipt.json")
    assert output.is_file() and receipt_path.is_file()
    assert not list(tmp_path.glob("*.partial"))
    assert DP.file_sha256(output) == result["dose_plan"]["file_sha256"]
    assert json.loads(receipt_path.read_text())["result_artifacts_read"] == []
    loaded = DP.load_passive_dose_plan(output)
    assert loaded.manifest_file_sha256 == DP.file_sha256(manifest_path)
    assert loaded.source_commit == "a" * 40
    assert loaded.launcher_path == str(Path(CLI.__file__).resolve())
    assert loaded.launcher_sha256 == DP.file_sha256(Path(CLI.__file__).resolve())


def test_cli_refuses_to_overwrite_frozen_plan(tmp_path, monkeypatch):
    manifest_path = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    output.write_text("reserved")
    monkeypatch.setattr(CLI, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(CLI, "_git_status", lambda: [])
    with pytest.raises(FileExistsError):
        CLI.build(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--sigma-frames",
                "50",
                "--execute",
            ]
        )
    assert output.read_text() == "reserved"


def test_claim_execute_refuses_a_dirty_tree_before_publication(tmp_path, monkeypatch):
    manifest_path = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    monkeypatch.setattr(CLI, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(CLI, "_git_status", lambda: [" M claim_code.py"])
    with pytest.raises(RuntimeError, match="requires a clean committed Git tree"):
        CLI.build(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--sigma-frames",
                "50",
                "--execute",
            ]
        )
    assert not output.exists()


def test_claim_execute_refuses_git_identity_change(tmp_path, monkeypatch):
    manifest_path = write_manifest(tmp_path)
    output = tmp_path / "plan.json"
    commits = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(CLI, "_git_sha", lambda: next(commits))
    monkeypatch.setattr(CLI, "_git_status", lambda: [])
    with pytest.raises(RuntimeError, match="Git identity changed"):
        CLI.build(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--sigma-frames",
                "50",
                "--execute",
            ]
        )
    assert not output.exists()


def test_exclusive_atomic_publish_preserves_a_racing_winner(tmp_path, monkeypatch):
    output = tmp_path / "plan.json"
    real_link = CLI.os.link

    def racing_link(source, destination):
        destination.write_bytes(b"racing creator\n")
        real_link(source, destination)

    monkeypatch.setattr(CLI.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        CLI._publish_exclusive_atomic(output, b"losing writer\n")
    assert output.read_bytes() == b"racing creator\n"
    assert not list(tmp_path.glob(".*.partial"))


def test_cli_has_no_overwrite_escape_hatch():
    with pytest.raises(SystemExit):
        CLI.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--output",
                "plan.json",
                "--sigma-frames",
                "50",
                "--overwrite",
            ]
        )
