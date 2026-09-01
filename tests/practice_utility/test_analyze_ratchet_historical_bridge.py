"""Strict receipt and arithmetic tests for the post-H_R2 historical bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.practice_utility import analyze_ratchet as R, analyze_ratchet_historical_bridge as A


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    path.chmod(0o444)
    return path


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def curriculum_rows(*, ratchet: bool, collapsed: bool = False):
    rows = []
    for step in range(1, 8001):
        if step <= 10:
            value = 0.0
        elif step < 100:
            value = step / 100.0
        else:
            value = 1.0
        if collapsed and step >= 7001:
            value = 0.10
        row = {"global_step": step, "lambda": value}
        if step <= 10:
            row["warmup_hold"] = True
        else:
            before = value
            if collapsed and step == 7001:
                before = 1.0
            row.update(
                {
                    "lambda_before": before,
                    "lambda_after": value,
                    "guard_tripped": False,
                }
            )
            if ratchet:
                row["latch_active"] = step == 1000
        rows.append(row)
    return rows


def write_curriculum(path: Path, *, ratchet: bool, collapsed: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = curriculum_rows(ratchet=ratchet, collapsed=collapsed)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def eval_receipt(
    *,
    mode: str,
    seed: int,
    value: tuple[float, float],
    checkpoint: Path,
    config: Path,
    panel: Path,
):
    panel_receipt = json.loads(panel.read_text())
    runs = {}
    mode_summary = {}
    checkpoint_hash = file_sha(checkpoint)
    config_hash = file_sha(config)
    for preset in R.ALL_PRESETS:
        branch = f"bridge_s{seed}_{mode}_{preset}"
        runs[branch] = {
            "checkpoint_seed": seed,
            "evaluation_seed": seed + 100,
            "mode": mode,
            "preset": preset,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "runtime": {"exit_code": 0},
            "summary": {
                "success_rate": value[0],
                "progress_rate": value[1],
                "motion_count": R.EXPECTED_NUM_ENVS,
            },
            "complete": True,
        }
        mode_summary[preset] = {
            mode: {
                "metrics": {
                    "success_rate": {"per_checkpoint_seed": {str(seed): value[0]}},
                    "progress_rate": {"per_checkpoint_seed": {str(seed): value[1]}},
                }
            }
        }
    return {
        "kind": "lucid_frozen_checkpoint_robustness_evaluation",
        "schema_version": 1,
        "launcher_sha256": R.EXPECTED_EVALUATOR_SHA256,
        "protocol": {
            "num_envs": R.EXPECTED_NUM_ENVS,
            "checkpoint_seeds": [seed],
            "evaluation_seed_by_checkpoint_seed": {str(seed): seed + 100},
            "modes": [mode],
            "max_delay_capacity_steps": R.EXPECTED_MAX_DELAY,
            "physics_step_ms": R.EXPECTED_PHYSICS_STEP_MS,
            "no_learning": True,
            "suite": {
                "motion_file": panel_receipt["motion_file"],
                "motion_count": R.EXPECTED_NUM_ENVS,
                "motion_keys_sha256": panel_receipt["alias_keys_sha256"],
                "pool_sha256": panel_receipt["pool_sha256"],
                "split_sha256": panel_receipt["split_sha256"],
                "split_linkage": "replicate-panel",
                "partition": panel_receipt["partition"],
                "replicate_panel": {
                    "receipt": str(panel),
                    "motion_key": panel_receipt["motion_key"],
                    "source_clip_sha256": panel_receipt["source_clip_sha256"],
                    "replicates": R.EXPECTED_NUM_ENVS,
                    "alias_keys_sha256": R.EXPECTED_PANEL_ALIAS_SHA256,
                },
            },
            "resolved_training_config": {
                "source": str(config),
                "sha256": config_hash,
                "installed": [str(config)],
            },
        },
        "runs": runs,
        "mode_summary": mode_summary,
        "checkpoint_sha256_before": {str(checkpoint): checkpoint_hash},
        "checkpoint_sha256_after": {str(checkpoint): checkpoint_hash},
        "verified": ["synthetic exact frozen instrument"],
    }


def write_eval(path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eval_receipt(**kwargs), indent=2, sort_keys=True) + "\n")
    return path


def write_ratchet_training(
    path: Path, seed: int, curriculum: Path, checkpoint: Path, config: Path
) -> Path:
    receipt = {
        "config": {
            "from_scratch": True,
            "num_envs": 1024,
            "iterations": 8000,
            "warmup_iterations": 10,
            "seeds": [seed],
            "modes": [R.RATCHET_MODE],
            "consolidation_fraction": 0,
            "max_delay_steps": 8,
        },
        "verified": ["synthetic ratchet source"],
        "arms": {
            f"ratchet_{seed}": {
                "seed": seed,
                "mode": R.RATCHET_MODE,
                "complete": True,
                "checkpoint_exported": True,
                "iterations_parsed": 8000,
                "curriculum_rows": 8000,
                "checkpoint": str(checkpoint),
                "curriculum_path": str(curriculum),
                "ratchet_bind_rows": 1,
                "arm_spec": {
                    "run_dir": str(Path("h_r2") / R.RATCHET_MODE / str(seed)),
                    "curriculum_mode": "lucid",
                    "anchor_ratio": 0.0,
                    "spread_strata": 1,
                    "return_guard": "relative",
                    "monotonic": True,
                    "fixed_lambda": 1.0,
                    "allow_extrapolation": False,
                    "margin": None,
                },
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def write_fixed_training(
    path: Path, seed: int, curriculum: Path, checkpoint: Path, config: Path
) -> Path:
    receipt = {
        "kind": "lucid_historical_training_cell_bridge",
        "schema_version": 1,
        "config": {
            "from_scratch": True,
            "num_envs": 1024,
            "iterations": 8000,
            "warmup_iterations": 10,
            "seeds": [seed],
            "modes": [R.FIXED_MODE],
            "consolidation_fraction": 0,
            "max_delay_steps": 8,
        },
        "arms": {
            f"fixed_{seed}": {
                "seed": seed,
                "mode": R.FIXED_MODE,
                "complete": True,
                "checkpoint_exported": True,
                "iterations_parsed": 8000,
                "curriculum_rows": 8000,
                "checkpoint": str(checkpoint),
                "curriculum_path": str(curriculum),
                "resolved_config": str(config),
                "arm_spec": {"curriculum_mode": "fixed"},
            }
        },
        "verified": ["synthetic fixed source"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def write_freeze_manifest(
    path: Path,
    *,
    mode: str,
    seed: int,
    checkpoint: Path,
    config: Path,
    curriculum: Path,
    training_receipt: Path,
) -> Path:
    receipt = {
        "kind": "lucid_frozen_training_checkpoint",
        "schema_version": 1,
        "state": "frozen_for_evaluation",
        "mode": mode,
        "seed": seed,
        "iterations": 8000,
        "evaluation_only": True,
        "resume_forbidden": True,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha(checkpoint),
            "read_only": True,
        },
        "config": {"path": str(config), "sha256": file_sha(config)},
        "curriculum": {
            "path": str(curriculum),
            "sha256": file_sha(curriculum),
            "rows": 8000,
        },
        "training_receipt": {
            "path": str(training_receipt),
            "sha256": file_sha(training_receipt),
        },
        "verified": ["synthetic immutable freeze"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def write_historical_bridge(
    path: Path, seed: int, checkpoint: Path, config: Path, curriculum: Path
) -> Path:
    receipt = {
        "kind": "lucid_historical_training_cell_bridge",
        "schema_version": 1,
        "config": {
            "from_scratch": True,
            "num_envs": 1024,
            "iterations": 8000,
            "warmup_iterations": 10,
            "seeds": [seed],
            "modes": [A.LUCID_MODE],
            "consolidation_fraction": 0,
            "max_delay_steps": 8,
        },
        "arms": {
            f"historical_{seed}": {
                "seed": seed,
                "mode": A.LUCID_MODE,
                "complete": True,
                "checkpoint_exported": True,
                "iterations_parsed": 8000,
                "curriculum_rows": 8000,
                "checkpoint": str(checkpoint),
                "curriculum_path": str(curriculum),
                "resolved_config": str(config),
                "arm_spec": {
                    "curriculum_mode": "lucid",
                    "anchor_ratio": 0.0,
                    "spread_strata": 1,
                    "return_guard": "relative",
                    "monotonic": False,
                    "allow_extrapolation": False,
                    "margin": None,
                },
            }
        },
        "sha256": {
            "checkpoint": file_sha(checkpoint),
            "resolved_config": file_sha(config),
            "curriculum": file_sha(curriculum),
        },
        "verified": ["synthetic exact historical bridge"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def write_amendment(
    path: Path, panel: Path, freezes: list[Path], evals: list[Path], trainings: list[Path]
) -> Path:
    by_key = {}
    for freeze_path in freezes:
        freeze = json.loads(freeze_path.read_text())
        by_key[(freeze["mode"], freeze["seed"])] = freeze

    def record(file_path: Path):
        return {"path": str(file_path), "sha256": file_sha(file_path)}

    fixed_8600 = by_key[(R.FIXED_MODE, 8600)]
    fixed_8601 = by_key[(R.FIXED_MODE, 8601)]
    ratchet_8601 = by_key[(R.RATCHET_MODE, 8601)]
    frozen_inputs = {
        "fixed_seed_8600_checkpoint_bundle": fixed_8600["checkpoint"],
        "fixed_seed_8600_config": fixed_8600["config"],
        "fixed_seed_8600_bridge": fixed_8600["training_receipt"],
        "fixed_seed_8601_checkpoint": fixed_8601["checkpoint"],
        "fixed_seed_8601_config": fixed_8601["config"],
        "fixed_seed_8601_bridge": fixed_8601["training_receipt"],
        "ratchet_seed_8601_checkpoint": ratchet_8601["checkpoint"],
        "ratchet_seed_8601_config": ratchet_8601["config"],
        "screen_training": ratchet_8601["training_receipt"],
        "screen_ratchet_evaluation": record(
            next(value for value in evals if value.name == f"{R.RATCHET_MODE}_8601.json")
        ),
        "screen_fixed_evaluation": record(
            next(value for value in evals if value.name == f"{R.FIXED_MODE}_8601.json")
        ),
        "panel_receipt": record(panel),
    }
    assert frozen_inputs["screen_training"]["sha256"] == file_sha(trainings[1])
    receipt = {
        "kind": "lucid_monotone_ratchet_confirmation_amendment",
        "schema_version": 1,
        "evaluation": {
            "evaluator_sha256": R.EXPECTED_EVALUATOR_SHA256,
            "panel_sha256": file_sha(panel),
            "num_envs": R.EXPECTED_NUM_ENVS,
            "evaluation_seed_by_training_seed": dict(R.EXPECTED_EVALUATION_SEED),
            "presets": list(R.ALL_PRESETS),
        },
        "frozen_inputs": frozen_inputs,
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("ratchet_historical_bridge")
    source_clip = root / "synthetic_motion.pkl"
    source_clip.write_bytes(b"synthetic motion bytes")
    panel_tree = root / "panel_motion_tree"
    panel_tree.mkdir()
    alias_names = [f"synthetic_motion__alias_{index:04d}" for index in range(512)]
    alias_hash = hashlib.sha256(("\n".join(sorted(alias_names)) + "\n").encode()).hexdigest()
    for alias_name in alias_names:
        (panel_tree / f"{alias_name}.pkl").symlink_to(source_clip)
    panel = root / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "kind": "lucid_replicate_panel",
                "schema_version": 1,
                "motion_key": "synthetic_motion",
                "source_clip": str(source_clip),
                "source_clip_sha256": file_sha(source_clip),
                "replicates": 512,
                "motion_file": str(panel_tree),
                "alias_keys_sha256": alias_hash,
                "pool_sha256": "2" * 64,
                "split_sha256": "3" * 64,
                "partition": "adaptation",
                "verified": ["synthetic exact panel"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    panel_hash = file_sha(panel)
    original_panel_hash = R.EXPECTED_PANEL_SHA256
    original_alias_hash = R.EXPECTED_PANEL_ALIAS_SHA256
    R.EXPECTED_PANEL_SHA256 = panel_hash
    R.EXPECTED_PANEL_ALIAS_SHA256 = alias_hash
    try:
        h_r2_evals = []
        h_r2_trainings = []
        h_r2_freezes = []
        h_r2_assets = {}
        historical_evals = []
        historical_bridges = []
        historical_hashes = {"checkpoint": {}, "config": {}, "curriculum": {}}
        ratchet_values = {8600: (0.60, 0.55), 8601: (0.80, 0.75), 8602: (0.70, 0.65)}
        fixed_values = {8600: (0.61, 0.56), 8601: (0.81, 0.76), 8602: (0.71, 0.66)}

        for mode, values in ((R.RATCHET_MODE, ratchet_values), (R.FIXED_MODE, fixed_values)):
            for seed in (8600, 8601, 8602):
                checkpoint = write_checkpoint(
                    root / "h_r2" / mode / str(seed) / "final_checkpoint.pt",
                    f"{mode}-{seed}",
                )
                config = write_config(
                    root / "h_r2" / mode / str(seed) / "config.yaml",
                    f"mode: {mode}\nseed: {seed}\n",
                )
                h_r2_assets[(mode, seed)] = (checkpoint, config)
                h_r2_evals.append(
                    write_eval(
                        root / "h_r2_eval" / f"{mode}_{seed}.json",
                        mode=mode,
                        seed=seed,
                        value=values[seed],
                        checkpoint=checkpoint,
                        config=config,
                        panel=panel,
                    )
                )

        for seed in (8600, 8601, 8602):
            checkpoint, config = h_r2_assets[(R.RATCHET_MODE, seed)]
            curriculum = write_curriculum(
                root / "h_r2_training" / str(seed) / "curriculum.jsonl", ratchet=True
            )
            training = write_ratchet_training(
                root / "h_r2_training" / str(seed) / "receipt.json",
                seed,
                curriculum,
                checkpoint,
                config,
            )
            h_r2_trainings.append(training)
            h_r2_freezes.append(
                write_freeze_manifest(
                    root / "h_r2_freezes" / f"{R.RATCHET_MODE}_{seed}.json",
                    mode=R.RATCHET_MODE,
                    seed=seed,
                    checkpoint=checkpoint,
                    config=config,
                    curriculum=curriculum,
                    training_receipt=training,
                )
            )

            fixed_checkpoint, fixed_config = h_r2_assets[(R.FIXED_MODE, seed)]
            fixed_curriculum = write_curriculum(
                root / "fixed_training" / str(seed) / "curriculum.jsonl", ratchet=False
            )
            fixed_training = write_fixed_training(
                root / "fixed_training" / str(seed) / "receipt.json",
                seed,
                fixed_curriculum,
                fixed_checkpoint,
                fixed_config,
            )
            h_r2_freezes.append(
                write_freeze_manifest(
                    root / "h_r2_freezes" / f"{R.FIXED_MODE}_{seed}.json",
                    mode=R.FIXED_MODE,
                    seed=seed,
                    checkpoint=fixed_checkpoint,
                    config=fixed_config,
                    curriculum=fixed_curriculum,
                    training_receipt=fixed_training,
                )
            )

        h_r2 = R.analyze(h_r2_evals, h_r2_trainings)
        assert h_r2["preregistered_decision"]["status"] in ("pass", "fail")
        h_r2_path = root / "h_r2_analysis.json"
        h_r2_path.write_text(json.dumps(h_r2, indent=2, sort_keys=True) + "\n")
        amendment = write_amendment(
            root / "h_r2_amendment.json",
            panel,
            h_r2_freezes,
            h_r2_evals,
            h_r2_trainings,
        )

        for seed in (8600, 8601, 8602):
            checkpoint = write_checkpoint(
                root / "historical" / str(seed) / "final_checkpoint.pt",
                f"historical-lucid-{seed}",
            )
            config = write_config(
                root / "historical" / str(seed) / "true_config.yaml",
                f"mode: lucid_rg\nseed: {seed}\n",
            )
            curriculum = write_curriculum(
                root / "historical" / str(seed) / "curriculum.jsonl",
                ratchet=False,
                collapsed=seed == 8601,
            )
            historical_hashes["checkpoint"][str(seed)] = file_sha(checkpoint)
            historical_hashes["config"][str(seed)] = file_sha(config)
            historical_hashes["curriculum"][str(seed)] = file_sha(curriculum)
            historical_evals.append(
                write_eval(
                    root / "historical_eval" / f"lucid_rg_{seed}.json",
                    mode=A.LUCID_MODE,
                    seed=seed,
                    value=(0.50, 0.45),
                    checkpoint=checkpoint,
                    config=config,
                    panel=panel,
                )
            )
            historical_bridges.append(
                write_historical_bridge(
                    root / "historical_bridge" / f"lucid_rg_{seed}.json",
                    seed,
                    checkpoint,
                    config,
                    curriculum,
                )
            )
    finally:
        R.EXPECTED_PANEL_SHA256 = original_panel_hash
        R.EXPECTED_PANEL_ALIAS_SHA256 = original_alias_hash

    return SimpleNamespace(
        root=root,
        panel=panel,
        panel_hash=panel_hash,
        alias_hash=alias_hash,
        h_r2=h_r2_path,
        amendment=amendment,
        amendment_hash=file_sha(amendment),
        h_r2_evals=h_r2_evals,
        h_r2_freezes=h_r2_freezes,
        historical_evals=historical_evals,
        historical_bridges=historical_bridges,
        historical_hashes=historical_hashes,
    )


@pytest.fixture
def exact_hashes(monkeypatch, bundle):
    monkeypatch.setattr(R, "EXPECTED_PANEL_SHA256", bundle.panel_hash)
    monkeypatch.setattr(R, "EXPECTED_PANEL_ALIAS_SHA256", bundle.alias_hash)
    monkeypatch.setattr(A, "EXPECTED_H_R2_AMENDMENT_SHA256", bundle.amendment_hash)
    monkeypatch.setattr(
        A, "EXPECTED_LUCID_CHECKPOINT_SHA256", bundle.historical_hashes["checkpoint"]
    )
    monkeypatch.setattr(A, "EXPECTED_LUCID_CONFIG_SHA256", bundle.historical_hashes["config"])
    monkeypatch.setattr(
        A, "EXPECTED_LUCID_CURRICULUM_SHA256", bundle.historical_hashes["curriculum"]
    )


def copy_json(source: Path, destination: Path) -> dict:
    value = json.loads(source.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def replace_path(paths, original: Path, replacement: Path):
    return [replacement if path == original else path for path in paths]


class TestExactBridge:
    def test_complete_analysis_is_descriptive_and_exact(self, bundle, exact_hashes):
        before = file_sha(bundle.h_r2)
        receipt = A.analyze(
            bundle.h_r2,
            bundle.amendment,
            bundle.h_r2_freezes,
            bundle.historical_evals,
            bundle.historical_bridges,
        )
        assert file_sha(bundle.h_r2) == before
        assert receipt["instrument_audit"]["cell_count"] == 126
        assert receipt["instrument_audit"]["per_mode_cell_count"] == {
            A.LUCID_MODE: 42,
            R.RATCHET_MODE: 42,
            R.FIXED_MODE: 42,
        }
        assert receipt["claim_scope"]["classification"] == "posthoc_descriptive"
        assert receipt["claim_scope"]["binding"] is False
        assert receipt["claim_scope"]["alters_H_R2"] is False
        assert receipt["claim_scope"]["noninferiority_claim_authorized"] is False
        assert receipt["claim_scope"]["superiority_claim_authorized"] is False
        assert receipt["activation"]["h_r2_unchanged"] is True

        success = receipt["arms"][A.LUCID_MODE]["success_rate"]
        assert success["in_envelope_auc"]["mean_auc"] == pytest.approx(0.50)
        assert success["frontier_auc"]["mean_auc"] == pytest.approx(0.50)
        assert success["legacy_phys_100_200_auc"]["mean_auc"] == pytest.approx(0.50)
        assert success[R.LATENCY_PRESET]["mean"] == pytest.approx(0.50)
        interaction = receipt["collapsed_seed_interaction"]["success_rate"]["frontier_auc"]
        assert interaction["ratchet_minus_lucid_per_seed"] == pytest.approx(
            {"8600": 0.10, "8601": 0.30, "8602": 0.20}
        )
        assert interaction["interaction"] == pytest.approx(0.15)
        assert interaction["binding"] is False
        assert interaction["inference"].startswith("none")

    def test_mechanism_table_preserves_predeclared_collapse(self, bundle, exact_hashes):
        receipt = A.analyze(
            bundle.h_r2,
            bundle.amendment,
            bundle.h_r2_freezes,
            bundle.historical_evals,
            bundle.historical_bridges,
        )
        table = receipt["mechanism_table"]
        assert table["8601"]["predeclared_historical_collapse"] is True
        assert table["8601"][A.LUCID_MODE]["final_lambda"] == pytest.approx(0.10)
        assert table["8601"][A.LUCID_MODE]["terminal_1000_high_lambda_iterations"] == 0
        assert table["8600"][A.LUCID_MODE]["final_lambda"] == pytest.approx(1.0)
        assert table["8602"][A.LUCID_MODE]["terminal_1000_high_lambda_iterations"] == 1000
        assert table["8601"][R.RATCHET_MODE]["blocked_pi_decrease_rows"] == 1
        assert table["8601"][R.FIXED_MODE]["evidence_kind"].endswith(
            "not_rederived_training_telemetry"
        )

    def test_cli_writes_nonbinding_receipt(self, bundle, exact_hashes, tmp_path):
        out = tmp_path / "analysis.json"
        assert (
            A.main(
                [
                    "--h-r2-analysis",
                    str(bundle.h_r2),
                    "--h-r2-amendment",
                    str(bundle.amendment),
                    "--h-r2-freeze-manifest",
                    *map(str, bundle.h_r2_freezes),
                    "--historical-robustness-receipt",
                    *map(str, bundle.historical_evals),
                    "--historical-training-bridge",
                    *map(str, bundle.historical_bridges),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        receipt = json.loads(out.read_text())
        assert receipt["kind"] == "lucid_ratchet_historical_bridge_analysis"
        assert receipt["frozen_descriptive_contract"]["cell_count"] == 126
        assert receipt["claim_scope"]["inference"] == "none"


class TestFailClosed:
    def test_rejects_nonterminal_h_r2(self, bundle, exact_hashes, tmp_path):
        path = tmp_path / "h_r2_nonterminal.json"
        value = copy_json(bundle.h_r2, path)
        value["preregistered_decision"]["status"] = "not_evaluable"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="not terminal pass/fail"):
            A.analyze(
                path,
                bundle.amendment,
                bundle.h_r2_freezes,
                bundle.historical_evals,
                bundle.historical_bridges,
            )

    def test_rejects_terminal_h_r2_with_failed_h_r0(self, bundle, exact_hashes, tmp_path):
        path = tmp_path / "h_r2_failed_h_r0.json"
        value = copy_json(bundle.h_r2, path)
        value["preregistered_decision"]["mechanism_pass"] = False
        value["mechanism"]["summary"]["all_available_seeds_pass"] = False
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="H_R0 mechanism gates did not all pass"):
            A.analyze(
                path,
                bundle.amendment,
                bundle.h_r2_freezes,
                bundle.historical_evals,
                bundle.historical_bridges,
            )

    def test_rejects_missing_historical_cell(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[0]
        path = tmp_path / "missing_cell.json"
        value = copy_json(original, path)
        branch = next(
            branch for branch, run in value["runs"].items() if run["preset"] == "phys_200"
        )
        del value["runs"][branch]
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="exactly 14 runs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_unpinned_h_r2_input(self, bundle, exact_hashes, tmp_path):
        path = tmp_path / "h_r2_unpinned.json"
        value = copy_json(bundle.h_r2, path)
        value["inputs"]["robustness_receipts"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="H_R2 robustness_receipts input hash differs"):
            A.analyze(
                path,
                bundle.amendment,
                bundle.h_r2_freezes,
                bundle.historical_evals,
                bundle.historical_bridges,
            )

    def test_rejects_wrong_evaluator_hash(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[0]
        path = tmp_path / "wrong_evaluator.json"
        value = copy_json(original, path)
        value["launcher_sha256"] = "e" * 64
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="evaluator hash differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_wrong_panel(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[0]
        wrong_panel = tmp_path / "wrong_panel.json"
        wrong_panel.write_text('{"kind":"different-panel"}\n')
        path = tmp_path / "wrong_panel_receipt.json"
        value = copy_json(original, path)
        value["protocol"]["suite"]["replicate_panel"]["receipt"] = str(wrong_panel)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="replicate-panel receipt hash differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_same_bytes_foreign_live_alias_target(self, bundle, exact_hashes, tmp_path):
        panel_record = json.loads(bundle.panel.read_text())
        source = Path(panel_record["source_clip"])
        foreign = tmp_path / source.name
        foreign.write_bytes(source.read_bytes())
        copied_tree = tmp_path / "retargeted_panel"
        copied_tree.mkdir()
        original_entries = sorted(Path(panel_record["motion_file"]).iterdir())
        for index, entry in enumerate(original_entries):
            target = foreign if index == 0 else source
            (copied_tree / entry.name).symlink_to(target)
        panel_record["motion_file"] = str(copied_tree)
        copied_panel = tmp_path / "retargeted_panel.json"
        copied_panel.write_text(json.dumps(panel_record, indent=2, sort_keys=True) + "\n")

        with pytest.raises(ValueError, match="frozen source clip"):
            A._audit_panel_identity(copied_panel)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("motion_keys_sha256", "9" * 64, "suite motion-key hash differs"),
            ("pool_sha256", "8" * 64, "suite pool_sha256 differs"),
            ("split_sha256", "7" * 64, "suite split_sha256 differs"),
        ],
    )
    def test_rejects_suite_identity_drift(
        self, bundle, exact_hashes, tmp_path, field, value, message
    ):
        original = bundle.historical_evals[0]
        path = tmp_path / f"wrong_{field}.json"
        receipt = copy_json(original, path)
        receipt["protocol"]["suite"][field] = value
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match=message):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_different_evaluated_motion_tree(self, bundle, exact_hashes, tmp_path):
        other_tree = tmp_path / "other_motion_tree"
        other_tree.mkdir()
        original = bundle.historical_evals[0]
        path = tmp_path / "wrong_motion_tree.json"
        receipt = copy_json(original, path)
        receipt["protocol"]["suite"]["motion_file"] = str(other_tree)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="evaluated motion tree differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_incomplete_per_run_motion_count(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[0]
        path = tmp_path / "wrong_motion_count.json"
        receipt = copy_json(original, path)
        next(iter(receipt["runs"].values()))["summary"]["motion_count"] = 511
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="did not score all 512 motions"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_h_r2_freeze_config_drift(self, bundle, exact_hashes, tmp_path):
        original = next(path for path in bundle.h_r2_freezes if "fixed_8602" in path.name)
        path = tmp_path / "wrong_freeze.json"
        receipt = copy_json(original, path)
        receipt["config"]["sha256"] = "0" * 64
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        freezes = replace_path(bundle.h_r2_freezes, original, path)
        with pytest.raises(ValueError, match="frozen config hash differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                freezes,
                bundle.historical_evals,
                bundle.historical_bridges,
            )

    def test_rejects_wrong_historical_evaluation_seed(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[2]
        path = tmp_path / "wrong_seed_map.json"
        value = copy_json(original, path)
        value["protocol"]["evaluation_seed_by_checkpoint_seed"]["8602"] = 8700
        for run in value["runs"].values():
            run["evaluation_seed"] = 8700
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="evaluation-seed mapping differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_config_provenance_drift(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[1]
        path = tmp_path / "wrong_config.json"
        value = copy_json(original, path)
        value["protocol"]["resolved_training_config"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="source config hash differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_checkpoint_before_after_drift(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_evals[0]
        path = tmp_path / "changed_checkpoint.json"
        value = copy_json(original, path)
        key = next(iter(value["checkpoint_sha256_after"]))
        value["checkpoint_sha256_after"][key] = "f" * 64
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_evals, original, path)
        with pytest.raises(ValueError, match="checkpoint before/after maps differ"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                paths,
                bundle.historical_bridges,
            )

    def test_rejects_bridge_with_wrong_true_config_pin(self, bundle, exact_hashes, tmp_path):
        original = bundle.historical_bridges[0]
        path = tmp_path / "wrong_bridge.json"
        value = copy_json(original, path)
        value["sha256"]["resolved_config"] = "a" * 64
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        paths = replace_path(bundle.historical_bridges, original, path)
        with pytest.raises(ValueError, match="true-config pin differs"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                bundle.historical_evals,
                paths,
            )

    def test_rejects_writable_historical_checkpoint(self, bundle, exact_hashes, tmp_path):
        original_eval = bundle.historical_evals[2]
        original_bridge = bundle.historical_bridges[2]
        eval_value = json.loads(original_eval.read_text())
        old_checkpoint = Path(next(iter(eval_value["checkpoint_sha256_before"])))
        writable = tmp_path / "writable_checkpoint.pt"
        writable.write_bytes(old_checkpoint.read_bytes())
        writable.chmod(0o644)

        eval_path = tmp_path / "writable_eval.json"
        for mapping in (
            eval_value["checkpoint_sha256_before"],
            eval_value["checkpoint_sha256_after"],
        ):
            digest = next(iter(mapping.values()))
            mapping.clear()
            mapping[str(writable)] = digest
        for run in eval_value["runs"].values():
            run["checkpoint"] = str(writable)
        eval_path.write_text(json.dumps(eval_value, indent=2, sort_keys=True) + "\n")

        bridge_path = tmp_path / "writable_bridge.json"
        bridge_value = copy_json(original_bridge, bridge_path)
        next(iter(bridge_value["arms"].values()))["checkpoint"] = str(writable)
        bridge_path.write_text(json.dumps(bridge_value, indent=2, sort_keys=True) + "\n")
        eval_paths = replace_path(bundle.historical_evals, original_eval, eval_path)
        bridge_paths = replace_path(bundle.historical_bridges, original_bridge, bridge_path)
        with pytest.raises(ValueError, match="not frozen read-only"):
            A.analyze(
                bundle.h_r2,
                bundle.amendment,
                bundle.h_r2_freezes,
                eval_paths,
                bridge_paths,
            )
