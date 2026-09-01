"""Static and inertness contracts for the future Tier-2 support-screen supervisor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

DRIVER = (
    Path(__file__).resolve().parents[2] / "scripts" / "practice_utility" / "run_support_screen.sh"
)


def source() -> str:
    return DRIVER.read_text()


def main_body(text: str) -> str:
    return text[text.index("\nmain() {") : text.index("\n# Activation is checked")]


def function_body(text: str, name: str, next_name: str) -> str:
    return text[text.index(f"\n{name}() {{") : text.index(f"\n{next_name}() {{")]


def driver_library(tmp_path: Path) -> Path:
    library = tmp_path / "support_driver_library.sh"
    library.write_text(source().split("\n# Activation is checked", maxsplit=1)[0] + "\n")
    return library


def test_driver_is_valid_bash_and_inert_without_future_prereg_sha(tmp_path):
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    env = os.environ.copy()
    env.pop("LUCID_SUPPORT_SCREEN_PREREG_SHA256", None)
    env["LUCID_SUPPORT_SCREEN_ROOT"] = str(tmp_path / "must_not_exist")
    result = subprocess.run(
        ["bash", str(DRIVER), "--preflight-only"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "future frozen Tier-2 preregistration SHA-256" in result.stderr
    assert not (tmp_path / "must_not_exist").exists()


def test_activation_precedes_environment_cd_markers_and_gpu_access():
    text = source()
    activation = text.rindex(
        ': "${LUCID_SUPPORT_SCREEN_PREREG_SHA256:?set the future frozen Tier-2 preregistration SHA-256}"'
    )
    pre_source_check = text.index("assert_preregistered_state", activation)
    assert activation < pre_source_check < text.rindex('source "${SUP_ENV}"')
    assert "PATH=/usr/bin:/bin assert_preregistered_state" in text[activation:]
    assert activation < text.rindex('cd "${SUP_REPO}"')
    assert activation < text.rindex('main "$@"')
    assert 'mkdir "${receipt_dir}/.started"' in text
    assert "nvidia-smi --query-compute-apps=pid" in text


def test_invalid_preregistration_is_rejected_before_environment_is_sourced(tmp_path):
    sentinel = tmp_path / "environment_was_sourced"
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(f"touch {sentinel}\n")
    copied_driver = tmp_path / "run_support_screen.sh"
    copied_driver.write_text(
        source().replace(
            'readonly SUP_ENV="/home/linjiw/lucid/env/lucid_env.sh"',
            f'readonly SUP_ENV="{bootstrap}"',
        )
    )
    prereg = tmp_path / "invalid_prereg.json"
    prereg.write_text("{}\n")
    env = os.environ.copy()
    env["LUCID_SUPPORT_SCREEN_PREREG"] = str(prereg)
    env["LUCID_SUPPORT_SCREEN_PREREG_SHA256"] = hashlib.sha256(prereg.read_bytes()).hexdigest()
    result = subprocess.run(
        ["bash", str(copied_driver), "--preflight-only"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "preregistration does not satisfy" in result.stderr
    assert not sentinel.exists()


def test_future_preregistration_must_pin_clean_git_files_and_inputs():
    text = source()
    assert 'SUP_REPO="${LUCID_SUPPORT_SCREEN_REPO:-/home/linjiw/lucid-support-screen}"' in text
    assert ".code_state.worktree == $repo" in text
    assert ".code_state.clean_detached_worktree_required == true" in text
    assert 'git -C "${SUP_REPO}" rev-parse HEAD' in text
    assert 'git -C "${SUP_REPO}" symbolic-ref -q HEAD' in text
    assert "worktree must be detached" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "support-screen worktree is not clean" in text
    assert ".code_state.file_sha256 | to_entries[]" in text
    assert ".frozen_inputs | to_entries[]" in text
    assert 'readonly SUP_ENV="/home/linjiw/lucid/env/lucid_env.sh"' in text
    assert "LUCID_SUPPORT_SCREEN_ENV" not in text
    assert 'SUP_DRIVER_PATH="$(/usr/bin/readlink -f "${BASH_SOURCE[0]}")"' in text
    assert 'readonly SUP_ROOT="/home/linjiw/lucid-sonic/manifests/tier2_support_screen"' in text
    assert "LUCID_SUPPORT_SCREEN_ROOT" not in text
    assert ".execution.driver.path == $driver" in text
    assert ".environment.bootstrap.path == $environment" in text
    for path in (
        "scripts/practice_utility/run_curriculum_comparison.py",
        "scripts/practice_utility/train_with_delay.py",
        "scripts/practice_utility/run_curriculum_robustness_eval.py",
        "scripts/practice_utility/eval_with_delay.py",
        "scripts/practice_utility/freeze_training_checkpoint.py",
        "scripts/practice_utility/analyze_support_screen.py",
        "scripts/practice_utility/run_support_screen.sh",
        "gear_sonic/research/practice_utility/tace.py",
    ):
        assert path in text
    for key in (
        "panel_receipt",
        "h_r2_analysis",
        "motion",
        "encoder",
        "historical_fixed_training",
        "historical_fixed_freeze_manifest",
        "historical_fixed_checkpoint",
        "historical_fixed_config",
        "historical_fixed_launcher",
        "environment_bootstrap",
    ):
        assert key in text


def test_live_panel_and_single_motion_contract_accepts_exact_fixture_and_rejects_drift(tmp_path):
    motion_dir = tmp_path / "training" / "robot_filtered"
    motion_dir.mkdir(parents=True)
    motion = motion_dir / "motion.pkl"
    motion.write_bytes(b"frozen motion bytes")
    panel_dir = tmp_path / "panel" / "robot_filtered"
    panel_dir.mkdir(parents=True)
    stems = [f"alias_{index:03d}" for index in range(512)]
    for stem in stems:
        (panel_dir / f"{stem}.pkl").symlink_to(motion)
    motion_sha = hashlib.sha256(motion.read_bytes()).hexdigest()
    alias_sha = hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest()
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "kind": "lucid_replicate_panel",
                "schema_version": 1,
                "replicates": 512,
                "source_clip": str(motion),
                "source_clip_sha256": motion_sha,
                "motion_key": motion.stem,
                "motion_file": str(panel_dir),
                "alias_keys_sha256": alias_sha,
            }
        )
    )
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps({"frozen_inputs": {"motion": {"sha256": motion_sha}}}))
    library = driver_library(tmp_path)
    command = 'source "$1"; SUP_PANEL="$2"; SUP_MOTION_FILE="$3"; ' "validate_live_data_contract"
    env = os.environ.copy()
    env["LUCID_SUPPORT_SCREEN_PREREG"] = str(prereg)
    subprocess.run(
        ["bash", "-c", command, "driver-test", str(library), str(panel), str(motion)],
        env=env,
        check=True,
    )

    (panel_dir / "extra.txt").write_text("drift")
    result = subprocess.run(
        ["bash", "-c", command, "driver-test", str(library), str(panel), str(motion)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "expected 512" in result.stderr

    (panel_dir / "extra.txt").unlink()
    foreign = tmp_path / "foreign.pkl"
    foreign.write_bytes(b"foreign motion")
    attacked = panel_dir / f"{stems[0]}.pkl"
    attacked.unlink()
    attacked.symlink_to(foreign)
    result = subprocess.run(
        ["bash", "-c", command, "driver-test", str(library), str(panel), str(motion)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "single source clip" in result.stderr or "frozen training motion" in result.stderr

    attacked.unlink()
    attacked.symlink_to(motion)
    (motion_dir / "extra.pkl").write_bytes(b"second training motion")
    result = subprocess.run(
        ["bash", "-c", command, "driver-test", str(library), str(panel), str(motion)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact one-file directory" in result.stderr


def test_h_r2_is_a_binding_three_seed_pass_gate():
    text = source()
    for predicate in (
        ".instrument_audit.passed == true",
        '.claim_scope.status == "three_seed_decision"',
        '.preregistered_decision.status == "pass"',
        '.preregistered_decision.paired_training_seeds == ["8600", "8601", "8602"]',
        ".preregistered_decision.mechanism_pass == true",
        ".preregistered_decision.capability_components_pass == true",
        ".mechanism.summary.all_available_seeds_pass == true",
    ):
        assert predicate in text
    assert "Tier-2 remains gated" in text


def test_training_uses_three_serial_one_cell_from_scratch_boundaries():
    text = source()
    body = main_body(text)
    calls = (
        "run_or_reuse_training fresh_fixed fixed",
        "run_or_reuse_training fixed_150 fixed_150",
        "run_or_reuse_training fixed_u150 fixed_u150",
    )
    positions = [body.index(call) for call in calls]
    assert positions == sorted(positions)
    assert body.index("run_or_reuse_evaluation") > positions[-1]
    assert body.index("freeze_or_reuse_checkpoint") > positions[-1]
    assert "--from-scratch" in text
    assert "--num-envs 1024" in text
    assert "--iterations 8000" in text
    assert "--warmup-iterations 10" in text
    assert "--seeds 8600" in text
    assert "--max-delay 12" in text
    assert ".config.max_delay_steps == 12" in text
    assert 'expected_live_max="8"' in text
    assert 'expected_live_max="12"' in text
    assert ".live_delay_final.action_delay_num_lags == 5120" in text
    assert ".live_delay_final.action_delay_max_steps == $expected_live_max" in text


def test_fixedu150_training_mechanics_are_exact_and_not_only_named():
    text = source()
    assert "def sizes: [37, 37, 37, 37, 36, 36, 36, 768];" in text
    assert "def lambdas: [0.1875, 0.375, 0.5625, 0.75, 0.9375, 1.125, 1.3125, 1.5];" in text
    for predicate in (
        ".arm_spec.anchor_seed == $anchor_seed",
        ".arm_spec.stratum_sizes == sizes",
        ".arm_spec.stratum_lambdas == lambdas",
        ".arm_spec.top_fraction == 0.75",
        ".tace_final.num_focus == 1024",
        ".tace_final.num_strata == 8",
        ".tace_final.stratum_sizes == sizes",
        ".tace_final.stratum_lambdas == lambdas",
        ".expand_contract.passed == true",
    ):
        assert predicate in text
    assert "validate_fixedu150_curriculum" in text
    assert "post_warmup_tace_rows != 7990" in text
    assert "set(dispatch) != terms" in text
    assert "did not recompute all eight stratum parameters" in text
    assert "DS.scaled_term_params" in text
    assert "DS.clamp_params_physical" in text
    assert "anchor_params" in text
    assert "params do not recompute from anchor_params" in text
    assert 'dispatch["randomize_action_delay"]["stratum_params"]' in text
    assert "dispatcher anchor_params changed during the run" in text
    assert "training receipt tace_final does not equal the final raw curriculum row" in text
    assert 'tace.get("consolidating") is not False' in text
    assert "count < 0" in text
    assert 'counts[f"focus_s{index}"] <= 0' in text
    assert text.index("count < 0") < text.index('counts[f"focus_s{index}"] <= 0')


def test_every_new_fixed_arm_stream_audits_the_full_curriculum_before_freezing():
    text = source()
    body = main_body(text)
    assert "validate_fixed_curriculum" in text
    assert 'validate_fixed_curriculum "${curriculum}" "${role}"' in text
    assert "global_step is missing or non-contiguous" in text
    assert "expected 8000 total rows" in text
    assert "expected 10 warmup + 7990 post-warmup rows" in text
    assert "lambda is not fixed at" in text
    assert "a consolidation row replaced the frozen distribution" in text
    assert "post-warmup extrapolation state differs" in text
    assert "post-warmup physical clamp state differs" in text
    assert body.index("run_or_reuse_training fixed_u150") < body.index("freeze_or_reuse_checkpoint")


def test_started_markers_are_fail_closed_and_never_resume():
    text = source()
    assert text.count('mkdir "${receipt_dir}/.started"') == 2
    assert text.count('if [[ -e "${receipt_dir}/.started" ]]') == 2
    assert text.count("no resume or automatic retry is allowed") == 2
    assert "--resume" not in text
    assert "preserve it" in text
    assert "partial or ambiguous" in text


def test_each_gpu_marker_follows_wait_and_full_revalidation_immediately_precedes_launch():
    text = source()
    training = function_body(text, "run_or_reuse_training", "config_for_training_receipt")
    evaluation = function_body(text, "run_or_reuse_evaluation", "run_analyzer")
    for body, launcher in (
        (training, "python scripts/practice_utility/run_curriculum_comparison.py"),
        (evaluation, "python scripts/practice_utility/run_curriculum_robustness_eval.py"),
    ):
        wait = body.rindex("wait_for_idle_gpu")
        revalidate = body.index("full_live_revalidation", wait)
        marker = body.index('mkdir "${receipt_dir}/.started"', revalidate)
        launch = body.index(launcher, marker)
        assert wait < revalidate < marker < launch
        between = body[marker + len('mkdir "${receipt_dir}/.started"') : launch]
        assert not between.strip()


def test_adjacent_config_is_materialized_pre_marker_and_foreign_config_is_rejected(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "resolved.yaml"
    config.write_text("model: exact\n")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    library = driver_library(tmp_path)
    prepare = 'source "$1"; prepare_adjacent_config "$2" "$3" "$4"'
    subprocess.run(
        [
            "bash",
            "-c",
            prepare,
            "driver-test",
            str(library),
            str(checkpoint),
            str(config),
            digest,
        ],
        check=True,
    )
    adjacent = checkpoint_dir / "config.yaml"
    assert adjacent.is_symlink()
    assert adjacent.resolve() == config.resolve()

    adjacent.unlink()
    adjacent.write_text("model: foreign\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            prepare,
            "driver-test",
            str(library),
            str(checkpoint),
            str(config),
            digest,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not linked to the frozen config" in result.stderr


def test_all_training_finishes_and_four_bundles_freeze_before_evaluation():
    body = main_body(source())
    training_end = body.index(
        'run_or_reuse_training fixed_u150 fixed_u150 "${SUP_TRAIN_FIXEDU150}"'
    )
    freeze_start = body.index("historical_freeze=", training_end)
    evaluation_start = body.index("run_or_reuse_evaluation", freeze_start)
    assert training_end < freeze_start < evaluation_start
    for role in ("historical_fixed", "fresh_fixed", "fixed_150", "fixed_u150"):
        assert f"{role}.json" not in body  # paths are constructed centrally, not ad hoc
        assert role in body
    assert body.count("freeze_or_reuse_checkpoint") == 3
    assert 'historical_freeze="${SUP_HISTORICAL_FREEZE}"' in body
    assert body.count("validate_freeze_manifest") >= 5
    assert body.count("prepare_freeze_adjacent_config") == 4
    text = source()
    for section in ("checkpoint", "config", "curriculum", "final_capsule", "training_receipt"):
        assert f'"{section}":' in text
    assert "is not bound to the selected training arm" in text
    assert "manifest_lstat.st_nlink != 1" in text
    assert "assert_no_write_bits" in text


def test_freeze_manifest_rejects_self_consistent_foreign_lineage_and_hardlinks(tmp_path):
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    checkpoint.chmod(0o444)
    config = tmp_path / "config.yaml"
    config.write_text("seed: 8600\n")
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text("{}\n")
    capsule = tmp_path / "capsule.pt"
    capsule.write_bytes(b"final capsule")
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "arms": {
                    "branch": {
                        "seed": 8600,
                        "mode": "fixed",
                        "checkpoint": str(checkpoint),
                        "curriculum_path": str(curriculum),
                        "capsule": str(capsule),
                    }
                }
            }
        )
    )

    def section(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    git_sha = "a" * 40
    manifest_value = {
        "kind": "lucid_frozen_training_checkpoint",
        "schema_version": 1,
        "state": "frozen_for_evaluation",
        "evaluation_only": True,
        "seed": 8600,
        "mode": "fixed",
        "iterations": 8000,
        "resume_forbidden": True,
        "verified": ["fixture"],
        "code": {"git_sha": git_sha, "git_status_short": ""},
        "checkpoint": {
            **section(checkpoint),
            "read_only": True,
            "mode_octal": "0o444",
        },
        "config": section(config),
        "curriculum": {**section(curriculum), "rows": 8000},
        "final_capsule": section(capsule),
        "training_receipt": section(training),
    }
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps(manifest_value))
    manifest.chmod(0o444)
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps({"code_state": {"git_sha": git_sha}}))
    library = driver_library(tmp_path)
    command = 'source "$1"; validate_freeze_manifest "$2" "$3" "$4" fixed fresh_fixed'
    env = os.environ.copy()
    env["LUCID_SUPPORT_SCREEN_PREREG"] = str(prereg)
    subprocess.run(
        [
            "bash",
            "-c",
            command,
            "driver-test",
            str(library),
            str(manifest),
            str(training),
            str(config),
        ],
        env=env,
        check=True,
    )

    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"foreign checkpoint")
    foreign.chmod(0o444)
    forged_value = json.loads(json.dumps(manifest_value))
    forged_value["checkpoint"] = {
        **section(foreign),
        "read_only": True,
        "mode_octal": "0o444",
    }
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(forged_value))
    forged.chmod(0o444)
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "driver-test",
            str(library),
            str(forged),
            str(training),
            str(config),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not bound to the selected training arm" in result.stderr

    alias = tmp_path / "freeze-hardlink.json"
    os.link(manifest, alias)
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "driver-test",
            str(library),
            str(alias),
            str(training),
            str(config),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "hard links" in result.stderr


def test_evaluation_is_exact_four_by_fifteen_k512_matrix():
    text = source()
    body = main_body(text)
    assert body.count("run_or_reuse_evaluation") == 4
    for invocation in (
        'historical_fixed "${SUP_HISTORICAL_TRAINING}" "${SUP_HISTORICAL_CONFIG}" fixed',
        'fresh_fixed "${fresh_training}" "${fresh_config}" fixed',
        'fixed_150 "${fixed150_training}" "${fixed150_config}" fixed_150',
        'fixed_u150 "${fixedu150_training}" "${fixedu150_config}" fixed_u150',
    ):
        assert invocation in body
    assert "--num-envs 512" in text
    assert "--seeds 8600" in text
    assert "--eval-seed-base 8700" in text
    assert "((.runs | length) == 15)" in text
    assert '.protocol.evaluation_seed_by_checkpoint_seed == {"8600": 8700}' in text
    assert ".protocol.suite.motion_count == 512" in text
    assert ".protocol.suite.replicate_panel.replicates == 512" in text
    assert ".checkpoint_sha256_after == .checkpoint_sha256_before" in text
    presets = (
        "phys_000",
        "phys_025",
        "phys_050",
        "phys_075",
        "phys_100",
        "phys_125",
        "phys_150",
        "phys_175",
        "phys_200",
        "lat_10ms",
        "lat_20ms",
        "lat_30ms",
        "lat_40ms",
        "lat_50ms",
        "lat_60ms",
    )
    array = text[
        text.index("readonly -a SUP_PRESETS=(") : text.index(
            ")", text.index("readonly -a SUP_PRESETS=(")
        )
    ]
    assert all(preset in array for preset in presets)


def test_each_evaluation_cell_reconciles_raw_512_episode_and_delay_evidence():
    text = source()
    assert "validate_evaluation_metrics" in text
    for field in (
        '"eval/all_metrics_dict"',
        '"motion_keys"',
        '"terminated"',
        '"progress"',
        '"failed_idxes"',
        '"failed_keys"',
        '"eval/success/success_rate"',
        '"eval/success/progress_rate"',
        '"eval/protocol/active_dr_terms"',
        '"eval/delay/"',
    ):
        assert field in text
    assert "len(set(motion_keys)) != 512" in text
    assert "failed_idxes != expected_failed" in text
    assert "summary failed_count does not reconcile" in text
    assert '"action_delay_actuator_groups": 5' in text
    assert '"action_delay_num_lags": 2560' in text
    assert "expected_histogram[expected_delay] = 2560" in text
    assert "len(metrics_paths) != 15" in text
    assert "raw motion keys do not match the frozen live panel digest" in text
    assert "summary DR ranges do not reconcile with raw metrics" in text
    assert "live DR range for" in text
    assert "run/aggregate reconciliation failed" in text
    assert "branch id differs from" in text
    assert "metrics_path is outside its exact role/preset artifact directory" in text
    assert "log_path is outside its exact branch path or is linked" in text
    assert "++callbacks.practice_eval.output_dir=" in text
    assert ".git_status_short == []" in text
    assert ".launcher_sha256 == $evaluator_sha" in text


def test_historical_and_new_training_receipts_bind_provenance_inputs_and_commands():
    text = source()
    for predicate in (
        '.kind == "lucid_three_arm_training_comparison"',
        ".schema_version == 1",
        ".git_sha == $git_sha",
        ".git_status_short == $git_status",
        ".launcher_sha256 == $launcher_sha",
        ".config.motion_file == $motion_dir",
        '.config.smpl_motion_file == "dummy"',
    ):
        assert predicate in text
    assert "validate_training_command" in text
    assert "required argv entries are missing" in text
    assert "from-scratch command unexpectedly contains a checkpoint" in text
    assert "training Python executable differs from the pinned bootstrap environment" in text


def test_analyzer_receives_role_specific_inputs_and_frozen_lineage():
    text = source()
    flags = (
        "--historical-fixed",
        "--fresh-fixed",
        "--fixed-150",
        "--fixed-u150",
        "--fresh-fixed-training",
        "--fixed-150-training",
        "--fixed-u150-training",
        "--preregistration",
        "--expected-preregistration-sha",
        "--historical-fixed-freeze-manifest",
        "--fresh-fixed-freeze-manifest",
        "--fixed-150-freeze-manifest",
        "--fixed-u150-freeze-manifest",
    )
    analyzer = function_body(text, "run_analyzer", "validate_analysis")
    for flag in flags:
        assert flag in text
        assert analyzer.count(f"\n        {flag} ") == 1
    assert analyzer.strip().endswith('--out "${out}"\n}')
    assert ".instrument_audit.unique_cells == 60" in text
    assert ".instrument_audit.cross_role_live_dr.passed == true" in text
    assert ".instrument_audit.cross_role_live_dr.roles" in text
    assert '.claim_scope.status == "screening_only"' in text
    assert ".decision.directional_claim_authorized == false" in text
    assert ".decision.superiority_claim_authorized == false" in text


def test_analysis_is_immutable_and_recomputed_on_creation_or_reuse():
    text = source()
    assert "Immutable analysis/recompute parity is mandatory" in text
    assert "run_analyzer" in text
    assert "cmp -s" in text
    assert "del(.created_at)" in text
    assert "does not reproduce from exact frozen inputs" in text
    assert 'chmod a-w "${SUP_ANALYSIS}"' in text
    assert 'assert_no_write_bits "${SUP_ANALYSIS}"' in text
    create = text.index('if [[ -e "${SUP_ANALYSIS}" ]]')
    recompute = text.index("# Immutable analysis/recompute parity", create)
    assert recompute > create
    analysis = function_body(text, "run_analysis", "preflight")
    mktemp = analysis.index("mktemp -d")
    assert analysis.rindex("assert_preregistered_state", 0, mktemp) < mktemp
    assert analysis.rindex("validate_live_data_contract", 0, mktemp) < mktemp
    assert analysis.count("validate_evaluation_receipt") == 4


def test_preflight_only_never_calls_main():
    text = source()
    block = text[text.rindex('if [[ "${1:-}" == "--preflight-only" ]]') :]
    assert "preflight" in block
    assert "exit 0" in block
    assert block.index("exit 0") < block.index('main "$@"')
