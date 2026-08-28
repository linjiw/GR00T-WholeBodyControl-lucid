"""The LUCID-S analyzer, checked against receipts whose answers are known.

The analyzer runs once, at the end of a nine-hour campaign, against receipts
that do not exist yet. Every arithmetic and decision path is therefore
exercised here on synthetic receipts built to make each hypothesis pass or fail
by construction -- a wrong AUC or an inverted comparison found after the
campaign would cost the campaign.
"""

import json

import pytest

from scripts.practice_utility import analyze_lucid_s as A


def eval_receipt(rows):
    """rows: {preset: {mode: {seed: success_rate}}} -> a mode_summary receipt."""
    summary = {}
    for preset, modes in rows.items():
        for mode, seeds in modes.items():
            summary.setdefault(preset, {})[mode] = {
                "metrics": {
                    "success_rate": {
                        "per_checkpoint_seed": {str(s): v for s, v in seeds.items()}
                    }
                }
            }
    return {"mode_summary": summary}


def training_receipt(arms):
    """arms: {mode: {seed: {final_lambda, return_guard_trips, ...}}}."""
    out = {"arms": {}, "config": {"warmup_iterations": 10}}
    for mode, seeds in arms.items():
        for seed, fields in seeds.items():
            out["arms"][f"{mode}_s{seed}"] = {
                "seed": seed,
                "mode": mode,
                "curriculum_path": "",
                "arm_spec": {"spread_strata": 1, "return_guard": "absolute"},
                **fields,
            }
    return out


class TestProfileAuc:
    def test_flat_profile_equals_its_own_success_rate(self):
        summary = eval_receipt(
            {p: {"m": {8600: 0.5}} for p in ("id_clean", "dr_050", "dr_full", "dr_125")}
        )["mode_summary"]
        auc = A.profile_auc(summary, "m")
        assert auc["available"]
        assert auc["mean"] == pytest.approx(50.0)

    def test_a_decaying_profile_is_below_its_clean_cell(self):
        summary = eval_receipt(
            {
                "id_clean": {"m": {8600: 0.9}},
                "dr_050": {"m": {8600: 0.6}},
                "dr_full": {"m": {8600: 0.4}},
                "dr_125": {"m": {8600: 0.2}},
            }
        )["mode_summary"]
        auc = A.profile_auc(summary, "m")
        # Trapezoid over s in {0, .5, 1, 1.25}, normalised by the 1.25 width.
        expected = 100.0 * (0.375 + 0.25 + 0.075) / 1.25
        assert auc["mean"] == pytest.approx(expected)
        assert auc["mean"] < 90.0

    def test_a_missing_cell_makes_the_profile_unavailable(self):
        summary = eval_receipt(
            {p: {"m": {8600: 0.5}} for p in ("id_clean", "dr_050", "dr_full")}
        )["mode_summary"]
        auc = A.profile_auc(summary, "m")
        assert auc["available"] is False
        assert auc["missing_preset"] == "dr_125"

    def test_extrapolation_cell_actually_moves_the_score(self):
        # If dr_125 were being ignored, these two would score the same.
        base = {p: {"m": {8600: 0.6}} for p in ("id_clean", "dr_050", "dr_full")}
        strong = A.profile_auc(
            eval_receipt({**base, "dr_125": {"m": {8600: 0.6}}})["mode_summary"], "m"
        )
        weak = A.profile_auc(
            eval_receipt({**base, "dr_125": {"m": {8600: 0.0}}})["mode_summary"], "m"
        )
        assert strong["mean"] > weak["mean"]


class TestPaired:
    def test_deltas_are_in_success_points_and_signed(self):
        out = A.paired({"8600": 0.60, "8601": 0.50}, {"8600": 0.55, "8601": 0.55})
        assert out["per_seed_pts"]["8600"] == pytest.approx(5.0)
        assert out["per_seed_pts"]["8601"] == pytest.approx(-5.0)
        assert out["mean_pts"] == pytest.approx(0.0)
        assert out["favorable_seeds"] == 1

    def test_only_seeds_present_in_both_arms_are_compared(self):
        out = A.paired({"8600": 0.6, "8602": 0.6}, {"8600": 0.5})
        assert out["num_seeds"] == 1


class TestControllerSummary:
    def test_reads_terminal_lambda_and_guard_trips(self):
        training = training_receipt(
            {"lucid_rg": {8600: {"final_lambda": 0.95, "return_guard_trips": 1},
                          8601: {"final_lambda": 0.99, "return_guard_trips": 0}}}
        )
        out = A.controller_summary(training, "lucid_rg")
        assert out["mean_terminal_lambda"] == pytest.approx(0.97)
        assert out["max_guard_trips"] == 1

    def test_an_unknown_arm_is_empty_not_an_error(self):
        out = A.controller_summary(training_receipt({}), "nothing")
        assert out["per_seed"] == {}
        assert out["mean_terminal_lambda"] is None


class TestEndToEnd:
    """Drive main() on receipts constructed so every verdict is known."""

    def _write(self, tmp_path, *, treatment_wins: bool):
        seeds = (8600, 8601, 8602)
        if treatment_wins:
            full = {"id_clean": 0.80, "dr_050": 0.70, "dr_full": 0.60, "dr_125": 0.50}
            s4 = {"id_clean": 0.70, "dr_050": 0.62, "dr_full": 0.55, "dr_125": 0.45}
        else:
            full = {"id_clean": 0.40, "dr_050": 0.35, "dr_full": 0.30, "dr_125": 0.20}
            s4 = {"id_clean": 0.40, "dr_050": 0.35, "dr_full": 0.30, "dr_125": 0.20}
        lucid = {"id_clean": 0.60, "dr_050": 0.52, "dr_full": 0.45, "dr_125": 0.35}
        fixed = {"id_clean": 0.55, "dr_050": 0.58, "dr_full": 0.59, "dr_125": 0.40}
        origin = {"id_clean": 0.85, "dr_050": 0.40, "dr_full": 0.20, "dr_125": 0.10}
        rows = {}
        for preset in ("id_clean", "dr_050", "dr_full", "dr_125", "latency_60ms"):
            rows[preset] = {}
            for mode, table in (
                ("ta_lucid_50_s4_rg", full), ("lucid_s4", s4), ("lucid", lucid),
                ("fixed", fixed), ("origin", origin), ("lucid_rg", lucid),
            ):
                value = table.get(preset, 0.3)
                rows[preset][mode] = {s: value for s in seeds}
        s7_eval = eval_receipt({p: {m: v for m, v in modes.items()
                                    if m in ("lucid", "fixed")}
                                for p, modes in rows.items()})
        s8_eval = eval_receipt({p: {m: v for m, v in modes.items()
                                    if m in ("ta_lucid_50_s4_rg", "lucid_s4", "lucid_rg")}
                                for p, modes in rows.items()})
        origin_eval = eval_receipt({p: {"origin": modes["origin"]}
                                    for p, modes in rows.items()})
        s7_train = training_receipt(
            {"lucid": {s: {"final_lambda": 0.5, "return_guard_trips": 5} for s in seeds},
             "fixed": {s: {"final_lambda": 1.0, "return_guard_trips": 0} for s in seeds}}
        )
        s8_train = training_receipt(
            {"lucid_rg": {s: {"final_lambda": 0.97, "return_guard_trips": 1} for s in seeds},
             "lucid_s4": {s: {"final_lambda": 0.8, "return_guard_trips": 3} for s in seeds},
             "ta_lucid_50_s4_rg": {s: {"final_lambda": 0.95, "return_guard_trips": 1}
                                   for s in seeds}}
        )
        prereg = {
            "hypotheses": {f"H_S{i}": f"claim {i}" for i in range(1, 6)},
            "decision_rules": {"tie": "< 2 pts", "if_every_arm_still_collapses": "report it"},
            "logical_sha256": "deadbeef",
        }
        paths = {}
        for name, doc in (
            ("prereg", prereg), ("s7t", s7_train), ("s8t", s8_train),
            ("s7e", s7_eval), ("s8e", s8_eval), ("oe", origin_eval),
        ):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(doc))
            paths[name] = path
        return paths

    def _run(self, tmp_path, paths):
        assert A.main([
            "--preregistration", str(paths["prereg"]),
            "--stage7-training", str(paths["s7t"]),
            "--stage8-training", str(paths["s8t"]),
            "--stage7-eval", str(paths["s7e"]),
            "--stage8-eval", str(paths["s8e"]),
            "--origin-eval", str(paths["oe"]),
            "--receipt-dir", str(tmp_path), "--bootstrap-samples", "0",
        ]) == 0
        receipt = sorted(tmp_path.glob("lucid_s_analysis_*.json"))[-1]
        return json.loads(receipt.read_text())

    def test_every_hypothesis_passes_when_built_to(self, tmp_path):
        out = self._run(tmp_path, self._write(tmp_path, treatment_wins=True))
        verdicts = {k: v["verdict"] for k, v in out["hypotheses"].items()}
        assert verdicts == {f"H_S{i}": "pass" for i in range(1, 6)}

    def test_every_hypothesis_fails_when_built_to(self, tmp_path):
        out = self._run(tmp_path, self._write(tmp_path, treatment_wins=False))
        verdicts = {k: v["verdict"] for k, v in out["hypotheses"].items()}
        # H_S2 is about the controller, which is unchanged between the two
        # fixtures, so it stays a pass; the outcome hypotheses must all flip.
        assert verdicts["H_S2"] == "pass"
        assert all(verdicts[k] == "fail" for k in ("H_S1", "H_S3", "H_S4", "H_S5"))

    def test_missing_origin_leaves_retention_unevaluable_not_wrong(self, tmp_path):
        paths = self._write(tmp_path, treatment_wins=True)
        assert A.main([
            "--preregistration", str(paths["prereg"]),
            "--stage7-training", str(paths["s7t"]),
            "--stage8-training", str(paths["s8t"]),
            "--stage7-eval", str(paths["s7e"]),
            "--stage8-eval", str(paths["s8e"]),
            "--receipt-dir", str(tmp_path), "--bootstrap-samples", "0",
        ]) == 0
        out = json.loads(sorted(tmp_path.glob("lucid_s_analysis_*.json"))[-1].read_text())
        assert out["hypotheses"]["H_S4"]["verdict"] == "not evaluable"

    def test_the_receipt_records_which_preregistration_it_scored(self, tmp_path):
        out = self._run(tmp_path, self._write(tmp_path, treatment_wins=True))
        assert out["preregistration"]["logical_sha256"] == "deadbeef"
        assert out["kind"] == "lucid_support_expansion_analysis"


class TestPairedSection:
    def test_disabled_when_samples_are_zero(self):
        assert A.paired_section([], 0) == {"enabled": False}

    def test_unavailable_rather_than_wrong_when_no_metrics_exist(self):
        out = A.paired_section([{"runs": {}}], 100)
        assert out["enabled"] is True and out["available"] is False

    def test_reports_intervals_for_the_preregistered_differences(self, tmp_path):
        import json as _json
        from gear_sonic.research.practice_utility import motion_paired as MP

        def metrics(name, failed):
            path = tmp_path / f"{name}.json"
            path.write_text(_json.dumps({
                "eval/success/success_rate": (102 - len(failed)) / 102,
                "eval/failed_metrics_dict": {
                    "failed_idxes": list(failed),
                    "failed_keys": [f"m{i}" for i in failed],
                },
            }))
            return str(path)

        runs = {}
        for mode, n in (("ta_lucid_50_s4_rg", 10), ("fixed", 40), ("lucid", 35),
                        ("lucid_s4", 20), ("origin", 5)):
            for preset in ("id_clean", "dr_050", "dr_full", "dr_125"):
                for seed in (8600, 8601, 8602):
                    runs[f"{mode}_{preset}_{seed}"] = {
                        "mode": mode, "preset": preset, "checkpoint_seed": seed,
                        "metrics_path": metrics(f"{mode}{preset}{seed}", range(n)),
                    }
        out = A.paired_section([{"runs": runs}], 500)
        assert out["available"] is True
        assert out["auc_weights"]["id_clean"] == pytest.approx(0.2)
        diffs = out["preregistered_differences"]
        assert diffs["H_S1"]["delta_pts"] == pytest.approx(100.0 * 15 / 102)
        assert diffs["H_S3_auc"]["delta_pts"] == pytest.approx(100.0 * 30 / 102)
        assert diffs["H_S4_fixed_vs_origin"]["delta_pts"] == pytest.approx(-100.0 * 35 / 102)
        assert diffs["H_S3_auc"]["excludes_zero"] is True
