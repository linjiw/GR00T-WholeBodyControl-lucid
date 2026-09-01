"""Static contracts for the dormant historical lucid_rg bridge supervisor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from gear_sonic.research.practice_utility import dr_scaling as DS

DRIVER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "practice_utility"
    / "run_ratchet_historical_bridge.sh"
)


def source() -> str:
    return DRIVER.read_text()


def main_body(text: str) -> str:
    return text[text.index("\nmain() {") : text.index("\n# Activation is checked")]


def driver_functions(tmp_path: Path) -> Path:
    """Materialize definitions only, without executing the dormant driver tail."""
    path = tmp_path / "driver_functions.sh"
    path.write_text(source().split("\n# Activation is checked", 1)[0] + "\n")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


HISTORICAL_PRESETS = (
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
)
SUMMARY_METRICS = (
    "success_rate",
    "progress_rate",
    "mpjpe_g",
    "mpjpe_l",
    "foot_slip_per_step_m",
    "undesired_contact_rate",
    "torque_saturation",
    "energy_proxy",
)
DR_TERMS = (
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
)


def baseline_ranges() -> dict:
    return {
        "add_joint_default_pos": {"pos_distribution_params": [-0.01, 0.01]},
        "base_com": {
            "com_range": {
                "x": [-0.025, 0.025],
                "y": [-0.05, 0.05],
                "z": [-0.05, 0.05],
            }
        },
        "physics_material": {
            "static_friction_range": [0.3, 1.6],
            "dynamic_friction_range": [0.3, 1.2],
            "restitution_range": [0.0, 0.5],
        },
        "push_robot": {
            "velocity_range": {
                "x": [-0.5, 0.5],
                "y": [-0.5, 0.5],
                "z": [-0.2, 0.2],
                "yaw": [-0.78, 0.78],
            }
        },
        "randomize_action_delay": {"delay_range": [0.0, 8.0]},
        "randomize_rigid_body_mass": {"mass_distribution_params": [0.8, 1.5]},
    }


def raw_ranges(preset: str) -> dict:
    physics_levels = {
        "phys_000": 0.00,
        "phys_025": 0.25,
        "phys_050": 0.50,
        "phys_075": 0.75,
        "phys_100": 1.00,
        "phys_125": 1.25,
        "phys_150": 1.50,
        "phys_175": 1.75,
        "phys_200": 2.00,
    }
    if preset not in physics_levels:
        latency_steps = {
            "lat_10ms": 2,
            "lat_20ms": 4,
            "lat_30ms": 6,
            "lat_40ms": 8,
            "lat_50ms": 10,
        }
        result = raw_ranges("phys_000")
        steps = float(latency_steps[preset])
        result["randomize_action_delay"] = {"delay_range": [steps, steps]}
        return result
    result = {}
    for term in set(DR_TERMS) - {"randomize_action_delay"}:
        value = DS.scaled_term_params(
            baseline_ranges()[term], physics_levels[preset], allow_extrapolation=True
        )
        value, _ = DS.clamp_params_physical(value)
        result[term] = value
    result["randomize_action_delay"] = {"delay_range": [0.0, 0.0]}
    return result


def write_evaluation_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, str, Path, Path, Path]:
    bridge = tmp_path / "historical_bridge.json"
    bridge.write_text(json.dumps({"experiment_id": None}))
    checkpoint = tmp_path / "final_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("mode: lucid_rg\n")
    repo = tmp_path / "historical-instrument"
    panel_motion = tmp_path / "panel-motion"
    panel_motion.mkdir()
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "motion_file": str(panel_motion),
                "motion_key": "walk_hands_on_back_loop_002__A066_M",
                "source_clip_sha256": (
                    "a7f10e7aa26e53cc4e346151d4ccd74e932e3aafa1cfaaac77dab8b8eec40929"
                ),
                "replicates": 512,
                "alias_keys_sha256": "placeholder-filled-below",
                "pool_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "partition": "adaptation",
            }
        )
    )
    artifact_root = tmp_path / "artifacts"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    experiment_id = "curriculum_robustness_ne512_20260901_000000"
    motion_keys = [f"alias_{index:04d}" for index in range(512)]
    alias_sha = hashlib.sha256(("\n".join(motion_keys) + "\n").encode()).hexdigest()
    panel_record = json.loads(panel.read_text())
    panel_record["alias_keys_sha256"] = alias_sha
    panel.write_text(json.dumps(panel_record))
    runs = {}
    commands = {}
    mode_summary = {}
    raw_names = {
        "mpjpe_g": "eval/all/mpjpe_g",
        "mpjpe_l": "eval/all/mpjpe_l",
        "foot_slip_per_step_m": "eval/quality/foot_slip_per_step_m",
        "undesired_contact_rate": "eval/quality/undesired_contact_rate",
        "torque_saturation": "eval/quality/torque_saturation",
        "energy_proxy": "eval/quality/energy_proxy",
    }
    physics_levels = {
        "phys_000": 0.00,
        "phys_025": 0.25,
        "phys_050": 0.50,
        "phys_075": 0.75,
        "phys_100": 1.00,
        "phys_125": 1.25,
        "phys_150": 1.50,
        "phys_175": 1.75,
        "phys_200": 2.00,
    }
    latency_steps = {
        "lat_10ms": 2,
        "lat_20ms": 4,
        "lat_30ms": 6,
        "lat_40ms": 8,
        "lat_50ms": 10,
    }
    for preset in HISTORICAL_PRESETS:
        run_id = f"{experiment_id}_s8600_lucid_rg_{preset}"
        metrics_path = (
            artifact_root / experiment_id / "seed_8600" / "lucid_rg" / preset / "metrics_eval.json"
        )
        metrics_path.parent.mkdir(parents=True)
        log_path = log_root / f"{run_id}.log"
        log_path.write_text("evaluation log\n")
        steps = 0 if preset in physics_levels else latency_steps[preset]
        event = (
            "tracking/lucid_curriculum" if preset in physics_levels else "tracking/lucid_eval_clean"
        )
        command = [
            sys.executable,
            str(repo / "scripts/practice_utility/eval_with_delay.py"),
            "--max-delay",
            "12",
            "--",
            f"checkpoint={checkpoint}",
            "+num_envs=512",
            "+headless=true",
            "+use_wandb=false",
            "+seed=8700",
            f"+manager_env/events={event}",
            "+use_encoder=g1",
            "+eval_callbacks=[practice_eval]",
            "+run_eval_loop=false",
            "++manager_env.config.train_only_events=[]",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={panel_motion}",
            "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
            (
                "++callbacks.practice_eval._target_="
                "gear_sonic.research.practice_utility.eval_callback."
                "PracticeRobustnessEvalCallback"
            ),
            "++callbacks.practice_eval.eval_frequency=1",
            "++callbacks.practice_eval.eval_only=true",
            f"++callbacks.practice_eval.output_dir={metrics_path.parent}",
            f"++callbacks.practice_eval.preset_id={preset}",
            f"++callbacks.practice_eval.branch_id={run_id}",
        ]
        if preset in physics_levels:
            command.extend(
                [
                    f"++callbacks.practice_eval.non_latency_dr_scale={physics_levels[preset]}",
                    "++callbacks.practice_eval.fixed_latency_steps=0",
                ]
            )
        else:
            command.append(f"++callbacks.practice_eval.fixed_latency_steps={steps}")
        commands[run_id] = command
        delay = {
            "action_delay_actuator_groups": 5,
            "action_delay_num_lags": 2560,
            "action_delay_min_steps": steps,
            "action_delay_max_steps": steps,
            "action_delay_mean_steps": float(steps),
            "action_delay_nonzero_fraction": 0.0 if steps == 0 else 1.0,
            "action_delay_histogram": [0] * steps + [2560],
        }
        ranges = raw_ranges(preset)
        raw = {
            "eval/protocol/preset_id": preset,
            "eval/protocol/branch_id": run_id,
            "eval/protocol/active_dr_terms": list(DR_TERMS),
            "eval/protocol/dr_ranges": ranges,
            "eval/protocol/fixed_latency_steps": steps,
            "eval/protocol/fixed_latency_report": {
                "requested_steps": float(steps),
                "pinned_terms": ["randomize_action_delay"],
            },
            "eval/protocol/non_latency_dr_scale": physics_levels.get(preset),
            "eval/protocol/dr_scale_report": (
                {
                    "lambda_value": physics_levels[preset],
                    "scaled_terms": sorted(set(DR_TERMS) - {"randomize_action_delay"}),
                    "num_scaled": 5,
                    "skipped_startup_terms": [],
                    "skipped_unknown_params": [],
                }
                if preset in physics_levels
                else None
            ),
            "eval/all_metrics_dict": {
                "motion_keys": motion_keys,
                "terminated": [False] * 512,
                "progress": [1.0] * 512,
            },
            "failed_idxes": [],
            "failed_keys": [],
            "eval/success/success_rate": 1.0,
            "eval/success/progress_rate": 1.0,
            **{f"eval/delay/{key}": value for key, value in delay.items()},
            **{name: 0.1 for name in raw_names.values()},
        }
        metrics_path.write_text(json.dumps(raw))
        summary = {
            "success_rate": 1.0,
            "progress_rate": 1.0,
            **{metric: 0.1 for metric in SUMMARY_METRICS[2:]},
            "motion_count": 512,
            "failed_count": 0,
            "active_dr_terms": list(DR_TERMS),
            "dr_ranges": ranges,
            "delay": delay,
        }
        runs[run_id] = {
            "checkpoint_seed": 8600,
            "evaluation_seed": 8700,
            "mode": "lucid_rg",
            "preset": preset,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "runtime": {"exit_code": 0},
            "summary": summary,
            "complete": True,
        }
        mode_summary[preset] = {
            "lucid_rg": {
                "num_runs": 1,
                "metrics": {
                    metric: {
                        "per_checkpoint_seed": {"8600": summary[metric]},
                        "mean": summary[metric],
                        "sample_std": None,
                    }
                    for metric in SUMMARY_METRICS
                },
            }
        }
    receipt = tmp_path / "evaluation.json"
    receipt.write_text(
        json.dumps(
            {
                "kind": "lucid_frozen_checkpoint_robustness_evaluation",
                "schema_version": 1,
                "experiment_id": experiment_id,
                "verified": ["fixture evaluation verified"],
                "launcher_sha256": (
                    "308e24150e4d4f03d0abf0dc6a427063ac662904bb3a7765488a9bff63cd94ca"
                ),
                "git_sha": "a" * 40,
                "git_status_short": [],
                "training_receipt": str(bridge),
                "training_experiment_id": None,
                "protocol": {
                    "num_envs": 512,
                    "checkpoint_seeds": [8600],
                    "evaluation_seed_by_checkpoint_seed": {"8600": 8700},
                    "modes": ["lucid_rg"],
                    "presets": {
                        "id_clean": "six channels collapsed to LUCID lambda=0 nominal",
                        "dr_full": "fresh draws from the complete six-channel training envelope",
                        "latency_60ms": (
                            "full five non-latency DR channels plus fixed 60 ms latency, "
                            "beyond the 0-40 ms train range"
                        ),
                    },
                    "max_delay_capacity_steps": 12,
                    "physics_step_ms": 5,
                    "no_learning": True,
                    "resolved_training_config": {
                        "source": str(config),
                        "sha256": digest(config),
                        "installed": [str(config)],
                    },
                    "suite": {
                        "motion_file": str(panel_motion),
                        "motion_count": 512,
                        "motion_keys_sha256": alias_sha,
                        "pool_sha256": panel_record["pool_sha256"],
                        "split_sha256": panel_record["split_sha256"],
                        "split_linkage": "replicate-panel",
                        "partition": panel_record["partition"],
                        "replicate_panel": {
                            "receipt": str(panel),
                            "motion_key": panel_record["motion_key"],
                            "source_clip_sha256": panel_record["source_clip_sha256"],
                            "replicates": 512,
                            "alias_keys_sha256": alias_sha,
                        },
                    },
                },
                "commands": commands,
                "runs": runs,
                "mode_summary": mode_summary,
            }
        )
    )
    return (
        receipt,
        bridge,
        checkpoint,
        config,
        panel,
        alias_sha,
        artifact_root,
        log_root,
        repo,
    )


def test_driver_is_valid_bash_and_inert_without_future_prereg_sha(tmp_path):
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    env = os.environ.copy()
    env.pop("LUCID_RATCHET_HISTORICAL_BRIDGE_PREREG_SHA256", None)
    env["LUCID_RATCHET_HISTORICAL_BRIDGE_ROOT"] = str(tmp_path / "must_not_exist")
    result = subprocess.run(
        ["bash", str(DRIVER), "--preflight-only"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "future frozen historical-bridge preregistration SHA-256" in result.stderr
    assert not (tmp_path / "must_not_exist").exists()


def test_activation_precedes_environment_cd_markers_writes_and_gpu():
    text = source()
    activation = text.rindex(
        ': "${LUCID_RATCHET_HISTORICAL_BRIDGE_PREREG_SHA256:'
        "?set the future frozen historical-bridge preregistration SHA-256}"
        '"'
    )
    pre_source_gate = text.rindex("assert_preregistered_state\nsource")
    assert activation < pre_source_gate
    assert activation < text.rindex('source "${HIST_ENV}"')
    assert activation < text.rindex('cd "${HIST_REPO}"')
    assert activation < text.rindex('main "$@"')
    assert 'mkdir "${receipt_dir}/.started"' in text
    assert "nvidia-smi --query-compute-apps=pid" in text
    body = main_body(text)
    assert body.index("preflight") < body.index("mkdir -p")


def test_exact_environment_bootstrap_is_preregistered_and_hashed_before_source():
    text = source()
    assert 'readonly HIST_ENV="/home/linjiw/lucid/env/lucid_env.sh"' in text
    assert (
        'EXPECTED_ENV_SHA256="aa1827d1b415cb21f8aadddc8a8985f62f3f1fc6807b96246ea2dab39d11d743"'
        in text
    )
    assert ".environment.path == $env_path" in text
    assert ".environment.sha256 == $env_sha" in text
    assert '.frozen_inputs.lucid_env == {"path": $env_path, "sha256": $env_sha}' in text
    assert "local required=(lucid_env panel_receipt motion h_r2_amendment h_r2_analysis)" in text
    assertion = text.index('assert_sha256 "${HIST_ENV}" "${EXPECTED_ENV_SHA256}"')
    source_env = text.rindex('source "${HIST_ENV}"')
    assert assertion < source_env


def test_future_prereg_pins_clean_detached_code_and_every_required_input():
    text = source()
    assert (
        'HIST_REPO="${LUCID_RATCHET_HISTORICAL_BRIDGE_REPO:-/home/linjiw/lucid-ratchet-historical-bridge}"'
        in text
    )
    assert ".code_state.worktree == $repo" in text
    assert ".code_state.clean_detached_worktree_required == true" in text
    assert 'git -C "${HIST_REPO}" symbolic-ref -q HEAD' in text
    assert "status --porcelain --untracked-files=all" in text
    assert "historical-bridge worktree is not clean" in text
    assert 'EXPECTED_INSTRUMENT_BASE_GIT_SHA="ca057e658acc59773e798057980b827d65988441"' in text
    assert ".code_state.instrument_base_git_sha == $instrument_base" in text
    assert ".code_state.allowed_changes_from_instrument_base == [" in text
    assert 'git -C "${HIST_REPO}" diff --name-status --no-renames' in text
    assert "historical instrument diff is not the exact four-file additive closure" in text
    assert '.staging.bundle_layout == "checkpoint_with_adjacent_true_config"' in text
    assert '.staging.allowed_copy_methods == ["copy", "reflink"]' in text
    assert ".staging.hardlinks_allowed == false" in text
    assert ".staging.source_artifact_mutation_allowed == false" in text
    assert 'running_driver="$(readlink -f "${BASH_SOURCE[0]}")"' in text
    assert "executed historical driver is outside the preregistered worktree" in text
    assert 'assert_sha256 "${running_driver}" "${expected_driver_sha}"' in text
    assert ".code_state.file_sha256 | to_entries[]" in text
    assert ".frozen_inputs | to_entries[]" in text
    for path in (
        "scripts/practice_utility/run_curriculum_robustness_eval.py",
        "scripts/practice_utility/analyze_ratchet.py",
        "scripts/practice_utility/analyze_ratchet_historical_bridge.py",
        "scripts/practice_utility/freeze_training_checkpoint.py",
        "scripts/practice_utility/run_ratchet_historical_bridge.sh",
        "tests/practice_utility/test_analyze_ratchet_historical_bridge.py",
        "tests/practice_utility/test_ratchet_historical_bridge_driver.py",
    ):
        assert path in text
    for key in ("panel_receipt", "motion", "h_r2_amendment", "h_r2_analysis"):
        assert key in text
    assert "h_r2_%s_s%s_freeze" in text
    assert "h_r2_%s_s%s_evaluation" in text
    assert "historical_lucid_s%s_%s" in text
    for kind in ("bridge", "checkpoint", "config", "curriculum"):
        assert kind in text


def test_instrument_closure_is_exactly_four_additive_files():
    text = source()
    expected = (
        "scripts/practice_utility/analyze_ratchet_historical_bridge.py",
        "scripts/practice_utility/run_ratchet_historical_bridge.sh",
        "tests/practice_utility/test_analyze_ratchet_historical_bridge.py",
        "tests/practice_utility/test_ratchet_historical_bridge_driver.py",
    )
    array = text[
        text.index("readonly -a EXPECTED_INSTRUMENT_ADDITIONS=(") : text.index(
            ")", text.index("readonly -a EXPECTED_INSTRUMENT_ADDITIONS=(")
        )
    ]
    assert all(path in array for path in expected)
    assert array.count("scripts/practice_utility/") == 2
    assert array.count("tests/practice_utility/") == 2
    assert "printf 'A\\t%s\\n'" in text
    assert 'git -C "${HIST_REPO}" ls-tree HEAD -- "${path}"' in text
    assert "expected_mode=100644" in text
    assert "expected_mode=100755" in text
    assert 'assert_regular_single_link "${HIST_REPO}/${path}"' in text


def test_h_r2_activation_accepts_capability_pass_or_fail_but_requires_every_h_r0():
    text = source()
    for predicate in (
        '.preregistered_decision.status == "pass"',
        '.preregistered_decision.status == "fail"',
        ".preregistered_decision.mechanism_complete == true",
        ".preregistered_decision.mechanism_pass == true",
        '.mechanism.summary.per_seed_all_gates_pass\n             == {"8600": true, "8601": true, "8602": true}',
        ".mechanism.summary.all_available_seeds_pass == true",
    ):
        assert predicate in text
    assert '.activation.h_r2_capability_status_allowed == ["pass", "fail"]' in text
    assert "bridge.audit_terminal_h_r2" in text
    assert "bridge.audit_h_r2_freeze_manifests" in text
    assert "bridge.audit_h_r2_amendment" in text
    assert "preregistered six-receipt H_R2 set differs" in text


def test_exact_historical_b8000_sources_are_hard_pinned():
    text = source()
    for digest in (
        "95aadf780c6bdf90e3d78e90b7ef14ee8a3b03a8362e776f39ea1408dc71fd2a",
        "e8ece9de91b5d73ea7ef920cc27047068ee1a25ea804d8c7001cf603fb31d70e",
        "aced3185ca7804d39e67d6223dd47f033808ea449500c1690b8f5d8f41613bf3",
        "4c0b49de050a4c09b687e339cdbed11e4f2a5a3b2130edd3e08649681ce369ff",
        "9997fe633cf33c319314a8fb28f239c8d70a15e9470209b828f1e591abce3568",
        "a3cd711fd0456fad745dc9a6b732a38461d63489818f4b5a22c754e9cfb9efb9",
        "e37dbdd0da02b42c81dac055d1f41e1a11911a84d062e6be11baeacd092413aa",
        "3e98983a34b8896fd45a8a72d032ad22048c4f517a7135f25018b0579b0b6e0d",
        "27d861498121a4b879d6cc47b1016f50e321bcd93db4a5458761e59a603d0537",
    ):
        assert digest in text
    assert ".config.num_envs == 1024" in text
    assert ".config.iterations == 8000" in text
    assert '.config.modes == ["lucid_rg"]' in text
    assert ".arm_spec.monotonic == false" in text
    assert ".sha256.final_capsule" in text
    assert 'HIST_STAGE_ROOT="${HIST_ROOT}/staged_historical_lucid_rg"' in text
    assert '"${expected_dir}/final_checkpoint.pt"' in text
    assert '"${expected_dir}/config.yaml"' in text
    assert "staged checkpoint must be a regular non-symlink file" in text
    assert "staged true config must be a regular non-symlink file" in text


def test_panel_alias_tree_is_rechecked_at_every_scientific_boundary():
    text = source()
    assert "validate_panel_alias_tree" in text
    assert "exactly the 512 .pkl aliases and no other entries" in text
    assert "every .pkl alias must be a symlink" in text
    assert "the aliases do not all resolve to the one canonical source clip" in text
    assert "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430" in text
    assert "a7f10e7aa26e53cc4e346151d4ccd74e932e3aafa1cfaaac77dab8b8eec40929" in text
    assert text.count("validate_panel_and_motion") >= 6
    evaluation = text[
        text.index("run_or_reuse_historical_evaluation()") : text.index(
            "validate_bridge_analysis()"
        )
    ]
    wait = evaluation.index("wait_for_idle_gpu")
    marker = evaluation.index('mkdir "${receipt_dir}/.started"')
    checks = [
        index
        for index in range(len(evaluation))
        if evaluation.startswith("validate_panel_and_motion", index)
    ]
    assert any(index < wait for index in checks)
    assert any(wait < index < marker for index in checks)
    analysis = text[text.index("run_analysis()") : text.index("preflight()")]
    assert "validate_analysis_inputs" in analysis
    analysis_gate = text[text.index("validate_analysis_inputs()") : text.index("publish_exclusive")]
    assert "validate_panel_and_motion" in analysis_gate


def test_panel_alias_tree_rejects_one_retargeted_alias(tmp_path):
    functions = driver_functions(tmp_path)
    motion = tmp_path / "source.pkl"
    motion.write_bytes(b"canonical-motion")
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    stems = [f"alias_{index:04d}" for index in range(512)]
    for stem in stems:
        (aliases / f"{stem}.pkl").symlink_to(motion)
    alias_sha = hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest()
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"source_clip": str(motion), "motion_file": str(aliases)}))

    command = 'source "$1"; validate_panel_alias_tree "$2" "$3" "$4" "$5"'
    good = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "_",
            str(functions),
            str(panel),
            str(motion),
            alias_sha,
            digest(motion),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert good.returncode == 0, good.stderr

    foreign = tmp_path / "foreign.pkl"
    foreign.write_bytes(b"foreign-motion")
    attacked = aliases / "alias_0256.pkl"
    attacked.unlink()
    attacked.symlink_to(foreign)
    bad = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "_",
            str(functions),
            str(panel),
            str(motion),
            alias_sha,
            digest(motion),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "one canonical source clip" in bad.stderr


def test_panel_alias_tree_rejects_extra_entry_and_uniform_same_byte_retarget(tmp_path):
    functions = driver_functions(tmp_path)
    motion = tmp_path / "source.pkl"
    motion.write_bytes(b"canonical-motion")
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    stems = [f"alias_{index:04d}" for index in range(512)]
    for stem in stems:
        (aliases / f"{stem}.pkl").symlink_to(motion)
    alias_sha = hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest()
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"source_clip": str(motion), "motion_file": str(aliases)}))
    command = 'source "$1"; validate_panel_alias_tree "$2" "$3" "$4" "$5"'
    args = [
        "bash",
        "-c",
        command,
        "_",
        str(functions),
        str(panel),
        str(motion),
        alias_sha,
        digest(motion),
    ]

    extra = aliases / "nested"
    extra.mkdir()
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "no other entries" in result.stderr
    extra.rmdir()

    foreign = tmp_path / "same-bytes-foreign.pkl"
    foreign.write_bytes(motion.read_bytes())
    for alias in aliases.glob("*.pkl"):
        alias.unlink()
        alias.symlink_to(foreign)
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "one canonical source clip" in result.stderr


def test_bad_original_adjacent_config_fails_before_marker(tmp_path):
    functions = driver_functions(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    checkpoint = original / "final_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    off_config = tmp_path / "off-config.yaml"
    off_config.write_text("arm: off\n")
    (original / "config.yaml").symlink_to(off_config)
    true_config = tmp_path / "true-config.yaml"
    true_config.write_text("arm: lucid_rg\n")
    marker = tmp_path / "must-not-be-created"
    command = (
        'source "$1"; ' 'validate_checkpoint_config_adjacency "$2" "$3" "$4" "$5"; ' 'mkdir "$6"'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "_",
            str(functions),
            str(checkpoint),
            str(true_config),
            digest(checkpoint),
            digest(true_config),
            str(marker),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not checkpoint.parent/config.yaml" in result.stderr
    assert not marker.exists()


def test_staged_adjacency_requires_regular_nonhardlinked_files(tmp_path):
    functions = driver_functions(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    checkpoint = staged / "final_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = staged / "config.yaml"
    config.write_text("arm: lucid_rg\n")
    command = 'source "$1"; validate_checkpoint_config_adjacency "$2" "$3" "$4" "$5"'
    args = [
        "bash",
        "-c",
        command,
        "_",
        str(functions),
        str(checkpoint),
        str(config),
        digest(checkpoint),
        digest(config),
    ]
    good = subprocess.run(args, text=True, capture_output=True, check=False)
    assert good.returncode == 0, good.stderr

    hardlink = tmp_path / "checkpoint-hardlink.pt"
    os.link(checkpoint, hardlink)
    bad = subprocess.run(args, text=True, capture_output=True, check=False)
    assert bad.returncode != 0
    assert "forbidden hardlink" in bad.stderr


def test_write_targets_reject_symlinks_hardlinks_and_symlinked_directories(tmp_path):
    functions = driver_functions(tmp_path)
    source_file = tmp_path / "source.json"
    source_file.write_text("{}\n")
    symlink_file = tmp_path / "symlink.json"
    symlink_file.symlink_to(source_file)
    command = 'source "$1"; assert_regular_single_link "$2"'
    result = subprocess.run(
        ["bash", "-c", command, "_", str(functions), str(symlink_file)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "regular non-symlink" in result.stderr

    hardlink = tmp_path / "hardlink.json"
    os.link(source_file, hardlink)
    result = subprocess.run(
        ["bash", "-c", command, "_", str(functions), str(source_file)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "forbidden hardlink" in result.stderr

    real_directory = tmp_path / "real-directory"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; assert_real_directory "$2"',
            "_",
            str(functions),
            str(linked_directory),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "non-symlink directory" in result.stderr


def test_all_three_historical_bundles_freeze_read_only_before_any_evaluation():
    text = source()
    body = main_body(text)
    freeze_calls = [body.index(f"freeze_or_reuse_historical {seed}") for seed in (8600, 8601, 8602)]
    eval_calls = [
        body.index(f"run_or_reuse_historical_evaluation {seed} {seed + 100}")
        for seed in (8600, 8601, 8602)
    ]
    assert freeze_calls == sorted(freeze_calls)
    assert eval_calls == sorted(eval_calls)
    assert max(freeze_calls) < min(eval_calls)
    assert "--make-read-only" in text
    assert "checkpoint config curriculum final_capsule training_receipt" in text
    assert "for section in checkpoint config; do" in text
    assert 'chmod a-w "${path}"' in text
    assert 'chmod a-w "$(staged_bundle_dir "${seed}")"' in text
    assert "staged bundle contains unexpected entries" in text
    assert "Original evidence is never mutated" in text
    assert "The hash-pinned historical bridge, curriculum, and capsule are original" in text
    assert "assert_no_write_bits" in text
    assert "bridge.audit_historical_bridges" in text


def test_evaluation_is_exact_serial_three_by_fourteen_k512_matrix():
    text = source()
    body = main_body(text)
    assert body.count("run_or_reuse_historical_evaluation") == 3
    for call in (
        "run_or_reuse_historical_evaluation 8600 8700",
        "run_or_reuse_historical_evaluation 8601 8701",
        "run_or_reuse_historical_evaluation 8602 8702",
    ):
        assert call in body
    assert "--num-envs 512" in text
    assert "--modes lucid_rg" in text
    assert '--eval-seed-base "${eval_seed}"' in text
    assert "((.runs | length) == 14)" in text
    array = text[
        text.index("readonly -a HIST_PRESETS=(") : text.index(
            ")", text.index("readonly -a HIST_PRESETS=(")
        )
    ]
    for preset in (
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
    ):
        assert preset in array


def test_evaluation_reuse_audit_rejects_lineage_aggregate_and_raw_attacks(tmp_path):
    functions = driver_functions(tmp_path)
    (
        receipt,
        bridge,
        checkpoint,
        config,
        panel,
        alias_sha,
        artifact_root,
        log_root,
        repo,
    ) = write_evaluation_fixture(tmp_path)
    command = (
        'source "$1"; validate_historical_evaluation_evidence '
        '"$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" '
        '"${12}" "${13}" "${14}" "${15}"'
    )
    args = [
        "bash",
        "-c",
        command,
        "_",
        str(functions),
        str(receipt),
        "8600",
        "8700",
        "a" * 40,
        str(bridge),
        str(checkpoint),
        digest(checkpoint),
        alias_sha,
        str(artifact_root),
        str(log_root),
        str(repo),
        str(panel),
        str(config),
        digest(config),
    ]

    def invoke():
        return subprocess.run(args, text=True, capture_output=True, check=False)

    baseline = json.loads(receipt.read_text())
    good = invoke()
    assert good.returncode == 0, good.stderr

    attacks = (
        ("kind", lambda value: value.update(kind="forged"), "kind differs"),
        (
            "launcher",
            lambda value: value.update(launcher_sha256="0" * 64),
            "launcher SHA differs",
        ),
        ("git", lambda value: value.update(git_sha="b" * 40), "git SHA differs"),
        (
            "dirty",
            lambda value: value.update(git_status_short=["?? rogue"]),
            "worktree was not clean",
        ),
        (
            "bridge",
            lambda value: value.update(training_receipt=str(tmp_path / "foreign.json")),
            "does not link this seed's historical bridge",
        ),
        (
            "training experiment",
            lambda value: value.update(training_experiment_id="substituted"),
            "training_experiment_id does not reconcile",
        ),
        (
            "panel suite",
            lambda value: value["protocol"]["suite"].update(pool_sha256="9" * 64),
            "suite does not exactly reconcile",
        ),
        (
            "config install",
            lambda value: value["protocol"]["resolved_training_config"]["installed"].append(
                str(tmp_path / "foreign-config.yaml")
            ),
            "resolved training-config lineage differs",
        ),
        (
            "command override",
            lambda value: next(iter(value["commands"].values())).append("+seed=9999"),
            "evaluator command differs",
        ),
        (
            "bounds",
            lambda value: next(iter(value["runs"].values()))["summary"].update(success_rate=1.1),
            "finite [0,1] rate",
        ),
        (
            "aggregate",
            lambda value: value["mode_summary"]["phys_000"]["lucid_rg"]["metrics"][
                "success_rate"
            ].update(mean=0.5),
            "aggregate mean differs",
        ),
    )
    for _, mutate, expected_error in attacks:
        attacked = json.loads(json.dumps(baseline))
        mutate(attacked)
        receipt.write_text(json.dumps(attacked))
        result = invoke()
        assert result.returncode != 0
        assert expected_error in result.stderr

    receipt.write_text(json.dumps(baseline))
    first_run = next(iter(baseline["runs"].values()))
    raw_path = Path(first_run["metrics_path"])
    raw_baseline = json.loads(raw_path.read_text())
    raw_attack = json.loads(json.dumps(raw_baseline))
    raw_attack["eval/success/success_rate"] = 0.5
    raw_path.write_text(json.dumps(raw_attack))
    result = invoke()
    assert result.returncode != 0
    assert "raw success does not reconcile" in result.stderr
    raw_path.write_text(json.dumps(raw_baseline))


def test_evaluation_reuse_audit_rejects_copy_delay_dr_and_array_attacks(tmp_path):
    functions = driver_functions(tmp_path)
    (
        receipt,
        bridge,
        checkpoint,
        config,
        panel,
        alias_sha,
        artifact_root,
        log_root,
        repo,
    ) = write_evaluation_fixture(tmp_path)
    command = (
        'source "$1"; validate_historical_evaluation_evidence '
        '"$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" '
        '"${12}" "${13}" "${14}" "${15}"'
    )
    args = [
        "bash",
        "-c",
        command,
        "_",
        str(functions),
        str(receipt),
        "8600",
        "8700",
        "a" * 40,
        str(bridge),
        str(checkpoint),
        digest(checkpoint),
        alias_sha,
        str(artifact_root),
        str(log_root),
        str(repo),
        str(panel),
        str(config),
        digest(config),
    ]

    def invoke():
        return subprocess.run(args, text=True, capture_output=True, check=False)

    baseline = json.loads(receipt.read_text())
    good = invoke()
    assert good.returncode == 0, good.stderr

    copied = json.loads(json.dumps(baseline))
    original_id, copied_run = next(iter(copied["runs"].items()))
    copied["runs"].pop(original_id)
    copied["runs"]["copied_h_r2_branch"] = copied_run
    copied["commands"]["copied_h_r2_branch"] = copied["commands"].pop(original_id)
    receipt.write_text(json.dumps(copied))
    result = invoke()
    assert result.returncode != 0
    assert "exact branch identity" in result.stderr

    receipt.write_text(json.dumps(baseline))
    lat50 = next(run for run in baseline["runs"].values() if run["preset"] == "lat_50ms")
    lat50_raw_path = Path(lat50["metrics_path"])
    lat50_raw = json.loads(lat50_raw_path.read_text())
    attacked_raw = json.loads(json.dumps(lat50_raw))
    attacked_raw["eval/delay/action_delay_histogram"] = [2560]
    lat50_raw_path.write_text(json.dumps(attacked_raw))
    attacked_receipt = json.loads(json.dumps(baseline))
    next(run for run in attacked_receipt["runs"].values() if run["preset"] == "lat_50ms")[
        "summary"
    ]["delay"]["action_delay_histogram"] = [2560]
    receipt.write_text(json.dumps(attacked_receipt))
    result = invoke()
    assert result.returncode != 0
    assert "live delay histogram" in result.stderr
    lat50_raw_path.write_text(json.dumps(lat50_raw))

    receipt.write_text(json.dumps(baseline))
    lat10 = next(run for run in baseline["runs"].values() if run["preset"] == "lat_10ms")
    lat10_raw_path = Path(lat10["metrics_path"])
    lat10_raw = json.loads(lat10_raw_path.read_text())
    attacked_raw = json.loads(json.dumps(lat10_raw))
    attacked_raw["eval/protocol/dr_ranges"]["add_joint_default_pos"]["pos_distribution_params"] = [
        -0.2,
        0.2,
    ]
    lat10_raw_path.write_text(json.dumps(attacked_raw))
    attacked_receipt = json.loads(json.dumps(baseline))
    next(run for run in attacked_receipt["runs"].values() if run["preset"] == "lat_10ms")[
        "summary"
    ]["dr_ranges"] = attacked_raw["eval/protocol/dr_ranges"]
    receipt.write_text(json.dumps(attacked_receipt))
    result = invoke()
    assert result.returncode != 0
    assert "non-latency DR drifted" in result.stderr
    lat10_raw_path.write_text(json.dumps(lat10_raw))

    receipt.write_text(json.dumps(baseline))
    phys175 = next(run for run in baseline["runs"].values() if run["preset"] == "phys_175")
    phys175_raw_path = Path(phys175["metrics_path"])
    phys175_raw = json.loads(phys175_raw_path.read_text())
    attacked_raw = json.loads(json.dumps(phys175_raw))
    attacked_raw["eval/all_metrics_dict"]["progress"][0] = 0.5
    phys175_raw_path.write_text(json.dumps(attacked_raw))
    result = invoke()
    assert result.returncode != 0
    assert "termination/progress arrays disagree" in result.stderr
    phys175_raw_path.write_text(json.dumps(phys175_raw))

    copied_path = json.loads(json.dumps(baseline))
    phys000_path = next(
        run["metrics_path"] for run in baseline["runs"].values() if run["preset"] == "phys_000"
    )
    next(run for run in copied_path["runs"].values() if run["preset"] == "phys_025")[
        "metrics_path"
    ] = phys000_path
    receipt.write_text(json.dumps(copied_path))
    result = invoke()
    assert result.returncode != 0
    assert "exact evaluator output cell" in result.stderr


def test_evaluation_validation_runs_before_reuse_or_next_seed():
    text = source()
    assert ".git_status_short == []" in text
    assert ".training_receipt == $bridge" in text
    assert "validate_historical_evaluation_evidence" in text
    evaluation = text[
        text.index("run_or_reuse_historical_evaluation()") : text.index(
            "validate_bridge_analysis()"
        )
    ]
    reuse = evaluation.index('echo "reusing complete historical evaluation receipt')
    assert (
        evaluation.index("validate_historical_evaluation", evaluation.index("if receipt=")) < reuse
    )
    launch = evaluation.index("python scripts/practice_utility/run_curriculum_robustness_eval.py")
    postlaunch = evaluation.index("validate_historical_evaluation", launch)
    assert launch < postlaunch


def test_started_markers_fail_closed_and_never_resume_or_retrain():
    text = source()
    assert text.count('mkdir "${receipt_dir}/.started"') == 1
    assert text.count('if [[ -e "${receipt_dir}/.started" ]]') == 1
    assert "preserve it, no resume or automatic retry is allowed" in text
    assert "--resume" not in text
    assert "run_curriculum_comparison.py" not in text
    assert "partial or ambiguous historical evaluation receipt set" in text
    evaluation = text[
        text.index("run_or_reuse_historical_evaluation()") : text.index(
            "validate_bridge_analysis()"
        )
    ]
    wait = evaluation.index("wait_for_idle_gpu")
    postwait_state = evaluation.index("assert_preregistered_state", wait)
    postwait_bundle = evaluation.index('validate_staged_bundle "${seed}" true', postwait_state)
    postwait_panel = evaluation.index("validate_panel_and_motion", postwait_bundle)
    marker = evaluation.index('mkdir "${receipt_dir}/.started"', postwait_panel)
    evaluator = evaluation.index(
        "python scripts/practice_utility/run_curriculum_robustness_eval.py", marker
    )
    assert wait < postwait_state < postwait_bundle < postwait_panel < marker < evaluator
    assert evaluation[marker:evaluator].strip() == 'mkdir "${receipt_dir}/.started"'


def test_analyzer_cli_matches_schema_and_binds_nine_receipts_six_freezes_three_bridges():
    text = source()
    for flag in (
        "--h-r2-analysis",
        "--h-r2-amendment",
        "--h-r2-freeze-manifest",
        "--historical-robustness-receipt",
        "--historical-training-bridge",
        "--out",
    ):
        assert flag in text
    assert "${@:1:3}" in text
    assert "${@:4:6}" in text
    assert "${@:10:3}" in text
    assert "exactly three historical receipts required" in text
    assert "exactly six H_R2 freezes required" in text
    assert "exactly three historical bridges required" in text
    assert ".instrument_audit.cell_count == 126" in text
    assert ".instrument_audit.h_r2_parent_audit.cell_count == 84" in text
    assert "nine evaluation receipts / 126 cells in total" in text


def test_analysis_is_o_excl_read_only_and_scientifically_recomputed():
    text = source()
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in text
    assert "0o444" in text
    assert 'chmod a-w "${destination}"' in text
    assert "Recompute after publication" in text
    assert "cmp -s" in text
    assert "del(.created_at)" in text
    assert "does not reproduce from exact frozen inputs" in text
    assert 'assert_no_write_bits "${HIST_ANALYSIS}"' in text
    assert "historical bridge analysis path is a symlink" in text
    assert 'assert_regular_single_link "${HIST_ANALYSIS}"' in text
    assert '.claim_scope.classification == "posthoc_descriptive"' in text
    assert ".claim_scope.binding == false" in text
    analysis = text[text.index("run_analysis()") : text.index("preflight()")]
    assert analysis.count("validate_analysis_inputs") >= 4
    candidate = analysis.index('candidate_dir="$(mktemp -d')
    assert analysis.index("validate_analysis_inputs") < candidate
    publish = analysis.index("publish_exclusive_read_only")
    assert analysis.index("validate_analysis_inputs", candidate) < publish


def test_analysis_publication_is_functionally_o_excl_and_does_not_follow_symlinks(tmp_path):
    functions = driver_functions(tmp_path)
    source_path = tmp_path / "candidate.json"
    source_path.write_text('{"value": 1}\n')
    destination = tmp_path / "analysis.json"
    command = 'source "$1"; publish_exclusive_read_only "$2" "$3"'
    args = [
        "bash",
        "-c",
        command,
        "_",
        str(functions),
        str(source_path),
        str(destination),
    ]
    first = subprocess.run(args, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert destination.read_text() == '{"value": 1}\n'
    assert destination.stat().st_mode & 0o222 == 0

    source_path.write_text('{"value": 2}\n')
    replay = subprocess.run(args, text=True, capture_output=True, check=False)
    assert replay.returncode != 0
    assert destination.read_text() == '{"value": 1}\n'

    victim = tmp_path / "victim.json"
    victim.write_text("preserve-me\n")
    symlink = tmp_path / "symlink-analysis.json"
    symlink.symlink_to(victim)
    attacked_args = [*args[:-1], str(symlink)]
    attacked = subprocess.run(attacked_args, text=True, capture_output=True, check=False)
    assert attacked.returncode != 0
    assert victim.read_text() == "preserve-me\n"


def test_preflight_only_exits_before_main():
    text = source()
    block = text[text.rindex('if [[ "${1:-}" == "--preflight-only" ]]') :]
    assert "preflight" in block
    assert "exit 0" in block
    assert block.index("exit 0") < block.index('main "$@"')
