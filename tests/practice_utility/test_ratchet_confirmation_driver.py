from pathlib import Path
import subprocess

DRIVER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "practice_utility"
    / "run_ratchet_confirmation.sh"
)


def test_driver_is_valid_fail_closed_bash():
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    source = DRIVER.read_text()
    assert "set -euo pipefail" in source
    assert "automatic retry is forbidden" in source
    assert 'mkdir "${receipt_dir}/.started"' in source
    assert 'if [[ -e "${RAT_ANALYSIS}" ]]' in source
    assert "frozen checkpoint regained write bits" in source
    assert "freeze manifest is not bound" in source
    assert "prevents operational optional stopping" in source
    assert "existing analysis does not reproduce" in source
    assert ".protocol.resolved_training_config.source == $training_config_source" in source
    assert ".protocol.resolved_training_config.sha256 == $training_config_sha" in source
    assert "fixed_s8601_bridge.json" in source
    assert 'RAT_REPO="/home/linjiw/lucid-ratchet-confirm"' in source
    assert "confirmation worktree is not clean" in source
    assert 'if [[ "${1:-}" == "--preflight-only" ]]' in source
    assert "ratchet confirmation preflight passed" in source


def test_driver_freezes_exact_training_and_eval_matrix():
    source = DRIVER.read_text()
    for invocation in (
        "run_or_reuse_training 8600 lucid_ratchet_rg",
        "run_or_reuse_training 8602 fixed",
        "run_or_reuse_training 8602 lucid_ratchet_rg",
        "8600 lucid_ratchet_rg 8700",
        "8600 fixed 8700",
        "8602 fixed 8702",
        "8602 lucid_ratchet_rg 8702",
    ):
        assert invocation in source
    assert '--eval-seed-base "${eval_seed}"' in source
    assert ".instrument_audit.cell_count == 84" in source
    assert ".preregistered_decision.superiority_claim_authorized == false" in source
