import json
from pathlib import Path

import pytest

from scripts.practice_utility import freeze_training_checkpoint as F


def receipt(tmp_path: Path, *, complete=True, rows=8000) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "final_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text("{}\n" * rows)
    capsule = tmp_path / "final.capsule.pt"
    capsule.write_bytes(b"capsule")
    config = tmp_path / "config.yaml"
    config.write_text("seed: 8600\n")
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "verified": ["complete"],
                "arms": {
                    "arm": {
                        "seed": 8600,
                        "mode": "lucid_ratchet_rg",
                        "complete": complete,
                        "checkpoint_exported": complete,
                        "iterations_parsed": rows,
                        "curriculum_rows": rows,
                        "checkpoint": str(checkpoint),
                        "capsule": str(capsule),
                        "curriculum_path": str(curriculum),
                    }
                },
            }
        )
    )
    return path, config, checkpoint


def test_manifest_pins_and_makes_checkpoint_read_only(tmp_path):
    training, config, checkpoint = receipt(tmp_path)
    manifest = F.build_manifest(
        training,
        config,
        8600,
        "lucid_ratchet_rg",
        8000,
        make_read_only=True,
    )
    assert manifest["state"] == "frozen_for_evaluation"
    assert manifest["checkpoint"]["sha256"] == F.sha256(checkpoint)
    assert manifest["checkpoint"]["read_only"] is True
    assert manifest["final_capsule"]["sha256"] == F.sha256(tmp_path / "final.capsule.pt")
    assert manifest["resume_forbidden"] is True
    assert checkpoint.stat().st_mode & 0o222 == 0


def test_incomplete_or_wrong_budget_is_rejected(tmp_path):
    training, config, _ = receipt(tmp_path, complete=False)
    with pytest.raises(ValueError, match="not complete"):
        F.build_manifest(training, config, 8600, "lucid_ratchet_rg", 8000, make_read_only=False)

    training, config, _ = receipt(tmp_path / "wrong", rows=7999)
    with pytest.raises(ValueError, match="iterations_parsed"):
        F.build_manifest(training, config, 8600, "lucid_ratchet_rg", 8000, make_read_only=False)


def test_output_is_exclusive(tmp_path):
    out = tmp_path / "freeze.json"
    F.write_exclusive(out, {"ok": True})
    with pytest.raises(FileExistsError):
        F.write_exclusive(out, {"ok": False})


def test_current_curriculum_row_count_is_verified(tmp_path):
    training, config, _ = receipt(tmp_path)
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text("{}\n" * 7999)
    with pytest.raises(ValueError, match="contains 7999"):
        F.build_manifest(training, config, 8600, "lucid_ratchet_rg", 8000, make_read_only=False)
