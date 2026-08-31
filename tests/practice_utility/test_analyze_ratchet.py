"""Deterministic arithmetic and receipt tests for the Tier-1 ratchet analyzer."""

import json

import pytest

from scripts.practice_utility import analyze_ratchet as A


def eval_receipt(rows):
    """Build ``{preset: {mode: {seed: (success, progress)}}}`` summary."""
    mode_summary = {}
    runs = {}
    checkpoints = {}
    for preset, modes in rows.items():
        for mode, seeds in modes.items():
            block = mode_summary.setdefault(preset, {}).setdefault(mode, {"metrics": {}})
            block["metrics"] = {
                metric: {
                    "per_checkpoint_seed": {str(seed): pair[index] for seed, pair in seeds.items()}
                }
                for index, metric in enumerate(A.METRICS)
            }
            for seed, pair in seeds.items():
                checkpoint = f"/test/{mode}_{seed}.pt"
                checkpoint_hash = f"{mode}-{seed}-sha"
                checkpoints[checkpoint] = checkpoint_hash
                branch = f"test_s{seed}_{mode}_{preset}"
                runs[branch] = {
                    "checkpoint_seed": seed,
                    "evaluation_seed": seed + 100,
                    "mode": mode,
                    "preset": preset,
                    "checkpoint_sha256": checkpoint_hash,
                    "runtime": {"exit_code": 0},
                    "summary": {"success_rate": pair[0], "progress_rate": pair[1]},
                    "complete": True,
                }
    seeds = sorted(
        {int(seed) for modes in rows.values() for values in modes.values() for seed in values}
    )
    return {
        "kind": "test_robustness",
        "launcher_sha256": A.EXPECTED_EVALUATOR_SHA256,
        "protocol": {
            "num_envs": A.EXPECTED_NUM_ENVS,
            "max_delay_capacity_steps": A.EXPECTED_MAX_DELAY,
            "physics_step_ms": A.EXPECTED_PHYSICS_STEP_MS,
            "no_learning": True,
            "suite": {
                "motion_count": A.EXPECTED_NUM_ENVS,
                "replicate_panel": {
                    "receipt": str(A.MANIFESTS / "replicate_panel_panel_hob002_k512.json"),
                    "replicates": A.EXPECTED_NUM_ENVS,
                    "alias_keys_sha256": A.EXPECTED_PANEL_ALIAS_SHA256,
                },
            },
        },
        "runs": runs,
        "mode_summary": mode_summary,
        "checkpoint_sha256_before": checkpoints,
        "checkpoint_sha256_after": checkpoints,
        "verified": ["synthetic complete matched instrument"],
    }


def full_grid_rows(seeds, ratchet=(0.80, 0.70), fixed=(0.81, 0.71)):
    rows = {}
    for preset, _ in (*A.IN_ENVELOPE_GRID, *A.FRONTIER_GRID):
        rows[preset] = {
            A.RATCHET_MODE: {seed: ratchet for seed in seeds},
            A.FIXED_MODE: {seed: fixed for seed in seeds},
        }
    rows[A.LATENCY_PRESET] = {
        A.RATCHET_MODE: {seed: ratchet for seed in seeds},
        A.FIXED_MODE: {seed: fixed for seed in seeds},
    }
    for preset in ("lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms"):
        rows[preset] = {
            A.RATCHET_MODE: {seed: ratchet for seed in seeds},
            A.FIXED_MODE: {seed: fixed for seed in seeds},
        }
    return rows


def write_training(tmp_path, seed=8600, rows=None, *, monotonic=True):
    rows = rows or []
    curriculum = tmp_path / f"curriculum_{seed}.jsonl"
    curriculum.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = tmp_path / f"training_{seed}.json"
    receipt.write_text(
        json.dumps(
            {
                "arms": {
                    f"ratchet_{seed}": {
                        "seed": seed,
                        "mode": A.RATCHET_MODE,
                        "curriculum_path": str(curriculum),
                        "ratchet_bind_rows": sum(bool(row.get("latch_active")) for row in rows),
                        "arm_spec": {"monotonic": monotonic},
                    }
                }
            }
        )
    )
    return receipt


def healthy_curriculum(num_rows=1200):
    rows = []
    for step in range(1, num_rows + 1):
        before = min(1.0, step / 100.0)
        after = before
        latch = step == 80
        if latch:
            before = after = 0.80
        rows.append(
            {
                "global_step": step,
                "lambda": after,
                "lambda_before": before,
                "lambda_after": after,
                "latch_active": latch,
                "guard_tripped": False,
            }
        )
    return rows


class TestAuc:
    def test_weights_are_normalized_trapezoids(self):
        assert A.trapezoid_weights(A.IN_ENVELOPE_GRID) == pytest.approx(
            {
                "phys_000": 0.125,
                "phys_025": 0.25,
                "phys_050": 0.25,
                "phys_075": 0.25,
                "phys_100": 0.125,
            }
        )
        assert A.trapezoid_weights(A.FRONTIER_GRID) == pytest.approx(
            {
                "phys_125": 1 / 6,
                "phys_150": 1 / 3,
                "phys_175": 1 / 3,
                "phys_200": 1 / 6,
            }
        )

    def test_profile_computes_success_and_drops_incomplete_seed(self, tmp_path):
        rows = full_grid_rows((8600, 8601))
        del rows["phys_200"][A.RATCHET_MODE][8601]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(eval_receipt(rows)))
        values = A.collect_robustness([path])
        result = A.profile(values, A.RATCHET_MODE, "success_rate", A.FRONTIER_GRID)
        assert result["per_seed"]["8600"]["auc"] == pytest.approx(0.80)
        assert result["incomplete_seeds"] == {"8601": ["phys_200"]}

    def test_split_receipts_union_seeds(self, tmp_path):
        paths = []
        for seed in (8600, 8601):
            path = tmp_path / f"eval_{seed}.json"
            path.write_text(json.dumps(eval_receipt(full_grid_rows((seed,)))))
            paths.append(path)
        values = A.collect_robustness(paths)
        result = A.profile(values, A.FIXED_MODE, "progress_rate", A.IN_ENVELOPE_GRID)
        assert result["complete_seeds"] == ["8600", "8601"]

    def test_conflicting_duplicate_cell_is_rejected(self, tmp_path):
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        first.write_text(json.dumps(eval_receipt(full_grid_rows((8600,)))))
        second.write_text(json.dumps(eval_receipt(full_grid_rows((8600,), ratchet=(0.10, 0.70)))))
        with pytest.raises(ValueError, match="conflicting robustness values"):
            A.collect_robustness([first, second])


class TestNoninferiority:
    @staticmethod
    def endpoint(values):
        return {
            "per_seed": {str(seed): {"auc": value} for seed, value in values.items()},
            "mean_auc": None,
        }

    def test_one_seed_is_screening_only(self):
        result = A.noninferiority(self.endpoint({8600: 0.91}), self.endpoint({8600: 0.92}), 0.02)
        assert result["within_margin_seeds"] == 1
        assert result["verdict"] == "screening_only"

    def test_two_of_three_within_margin_passes_without_superiority(self):
        ratchet = self.endpoint({8600: 0.89, 8601: 0.88, 8602: 0.80})
        fixed = self.endpoint({8600: 0.90, 8601: 0.90, 8602: 0.90})
        result = A.noninferiority(ratchet, fixed, 0.02)
        assert result["within_margin_seeds"] == 2
        assert result["strictly_favorable_seeds_descriptive"] == 0
        assert result["verdict"] == "pass"

    def test_only_one_of_three_within_margin_fails(self):
        ratchet = self.endpoint({8600: 0.89, 8601: 0.86, 8602: 0.80})
        fixed = self.endpoint({8600: 0.90, 8601: 0.90, 8602: 0.90})
        assert A.noninferiority(ratchet, fixed, 0.02)["verdict"] == "fail"


class TestMechanism:
    def test_healthy_ratchet_reaches_blocks_and_stays_high(self, tmp_path):
        receipt = write_training(tmp_path, rows=healthy_curriculum())
        per_seed, ignored = A.collect_mechanisms([receipt])
        block = per_seed["8600"]
        assert ignored == []
        assert block["reach_lambda_095_by_step_500"]["first_reach_step"] == 95
        assert block["reach_lambda_095_by_step_500"]["gate"] == "pass"
        assert block["pi_decrease_control"]["blocked_pi_decrease_rows"] == 1
        assert block["pi_decrease_control"]["guard_is_only_legal_decrease_gate"] == "pass"
        assert block["terminal_1000_high_lambda_exposure"]["high_lambda_fraction"] == 1.0
        assert block["terminal_1000_high_lambda_exposure"]["gate"] == "pass"
        assert A.mechanism_summary(per_seed)["all_available_seeds_pass"] is True

    def test_guarded_decrease_is_legal_but_unguarded_is_not(self, tmp_path):
        rows = healthy_curriculum()
        rows[500].update(
            {"lambda_before": 1.0, "lambda_after": 0.5, "lambda": 0.5, "guard_tripped": True}
        )
        rows[501].update(
            {"lambda_before": 0.5, "lambda_after": 0.4, "lambda": 0.4, "guard_tripped": False}
        )
        receipt = write_training(tmp_path, rows=rows)
        per_seed, _ = A.collect_mechanisms([receipt])
        control = per_seed["8600"]["pi_decrease_control"]
        assert control["actual_decrease_rows"] == 2
        assert control["guard_trip_rows"] == 1
        assert control["unguarded_decrease_rows"] == 1
        assert control["guard_is_only_legal_decrease_gate"] == "fail"

    def test_short_terminal_window_is_not_evaluable(self, tmp_path):
        receipt = write_training(tmp_path, rows=healthy_curriculum(600))
        per_seed, _ = A.collect_mechanisms([receipt])
        terminal = per_seed["8600"]["terminal_1000_high_lambda_exposure"]
        assert terminal["observed_iterations"] == 600
        assert terminal["gate"] == "not_evaluable"

    @pytest.mark.parametrize(("low_rows", "expected"), [(50, "pass"), (51, "fail")])
    def test_terminal_gate_uses_preregistered_95_percent_floor(self, tmp_path, low_rows, expected):
        rows = healthy_curriculum()
        for row in rows[-low_rows:]:
            row["lambda"] = 0.90
            row["lambda_before"] = 0.90
            row["lambda_after"] = 0.90
            row["guard_tripped"] = True
        receipt = write_training(tmp_path, rows=rows)
        per_seed, _ = A.collect_mechanisms([receipt])
        terminal = per_seed["8600"]["terminal_1000_high_lambda_exposure"]
        assert terminal["high_lambda_fraction"] == pytest.approx(1.0 - low_rows / 1000)
        assert terminal["gate"] == expected


class TestReceipt:
    def test_instrument_audit_rejects_missing_cell(self, tmp_path):
        rows = full_grid_rows((8601,))
        del rows["lat_40ms"][A.RATCHET_MODE][8601]
        evaluation = tmp_path / "evaluation.json"
        evaluation.write_text(json.dumps(eval_receipt(rows)))
        with pytest.raises(ValueError, match="preset set differs"):
            A.audit_instrument([evaluation])

    def test_cli_writes_screening_receipt_with_both_metrics(self, tmp_path):
        evaluation = tmp_path / "evaluation.json"
        evaluation.write_text(json.dumps(eval_receipt(full_grid_rows((8600,)))))
        training = write_training(tmp_path, rows=healthy_curriculum())
        out = tmp_path / "analysis.json"

        assert (
            A.main(
                [
                    "--robustness-receipt",
                    str(evaluation),
                    "--training-receipt",
                    str(training),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        receipt = json.loads(out.read_text())
        assert receipt["kind"] == "lucid_ratchet_analysis"
        assert receipt["instrument_audit"]["passed"] is True
        assert receipt["claim_scope"]["status"] == "screening_only"
        assert receipt["claim_scope"]["directional_claim_authorized"] is False
        assert receipt["preregistered_decision"]["status"] == "screen_pass"
        assert receipt["preregistered_decision"]["lat_50ms_is_secondary"] is True
        assert (
            receipt["ratchet_vs_fixed"]["success_rate"]["frontier_auc"]["verdict"]
            == "screening_only"
        )
        assert (
            receipt["ratchet_vs_fixed"]["progress_rate"]["in_envelope_auc"]["verdict"]
            == "screening_only"
        )
        assert receipt["arms"][A.RATCHET_MODE]["success_rate"]["frontier_auc"][
            "mean_auc"
        ] == pytest.approx(0.80)
        assert receipt["inputs"]["robustness_receipts"][0]["sha256"]
        assert "three-seed noninferiority verdict" in receipt["not_yet_verified"]

    def test_three_seed_analysis_uses_frozen_margins(self, tmp_path):
        evaluation = tmp_path / "evaluation.json"
        evaluation.write_text(json.dumps(eval_receipt(full_grid_rows((8600, 8601, 8602)))))
        training = write_training(tmp_path, rows=healthy_curriculum())
        receipt = A.analyze([evaluation], [training])
        assert receipt["frozen_contract"]["frontier_margin"] == 0.02
        assert receipt["frozen_contract"]["terminal_min_high_lambda_fraction"] == 0.95
        assert receipt["preregistered_decision"]["status"] == "not_evaluable"
        assert receipt["frozen_contract"]["in_envelope_margin"] == 0.01
        assert receipt["claim_scope"]["status"] == "three_seed_decision"
        assert receipt["joint_noninferiority"]["success_primary"]["verdict"] == "pass"
