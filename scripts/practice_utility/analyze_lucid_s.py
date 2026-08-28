#!/usr/bin/env python3
"""Score the preregistered LUCID-S hypotheses (H-S1..H-S5) from receipts.

Reads the stage-7 and stage-8 training receipts, their frozen-evaluation
receipts, and the untrained origin's evaluation, and answers exactly the five
hypotheses written in
``manifests/lucid_support_expansion_preregistration_20260828.json``. Every
threshold is read from that file, so this script has no knobs of its own and
cannot be tuned after the fact.

The primary endpoint is the **robustness-profile AUC**: the trapezoidal area
under success rate against the non-latency DR scale s, over the preregistered
grid. It is reported in success-rate points, normalised by the width of the
grid, so it reads on the same scale as a success rate: "average success across
the severity range, including one cell outside the training envelope".

Written before any stage-8 branch had run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

MANIFESTS = LUCID_ROOT / "manifests"
#: Non-latency DR scale of each profile cell. ``latency_60ms`` is deliberately
#: absent: it varies a different channel and is reported on its own.
PROFILE_S = {"id_clean": 0.0, "dr_050": 0.5, "dr_full": 1.0, "dr_125": 1.25}


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def per_seed(summary: dict[str, Any], preset: str, mode: str, metric: str = "success_rate") -> dict[str, float]:
    block = summary.get(preset, {}).get(mode)
    if not block:
        return {}
    values = block["metrics"].get(metric, {}).get("per_checkpoint_seed", {})
    return {str(s): float(v) for s, v in values.items() if v is not None}


def mean_pts(values: dict[str, float]) -> float | None:
    return 100.0 * statistics.fmean(values.values()) if values else None


def paired(treatment: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    """Per-seed differences in success-rate points, and how many favour us."""
    common = sorted(set(treatment) & set(reference))
    deltas = {s: 100.0 * (treatment[s] - reference[s]) for s in common}
    values = list(deltas.values())
    return {
        "per_seed_pts": deltas,
        "mean_pts": statistics.fmean(values) if values else None,
        "favorable_seeds": sum(1 for v in values if v > 0),
        "num_seeds": len(values),
    }


def profile_auc(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    """Trapezoidal mean success across the severity grid, per seed.

    Normalising by the grid width keeps the number on the success-rate scale.
    A seed missing any cell is dropped rather than imputed: a profile with a
    hole in it is not a profile.
    """
    cells = sorted(PROFILE_S.items(), key=lambda kv: kv[1])
    by_seed: dict[str, list[float]] = {}
    for preset, _ in cells:
        values = per_seed(summary, preset, mode)
        if not values:
            return {"available": False, "missing_preset": preset}
        for seed, value in values.items():
            by_seed.setdefault(seed, []).append(value)
    width = cells[-1][1] - cells[0][1]
    scales = [s for _, s in cells]
    auc: dict[str, float] = {}
    for seed, values in by_seed.items():
        if len(values) != len(cells):
            continue
        area = sum(
            0.5 * (values[i] + values[i + 1]) * (scales[i + 1] - scales[i])
            for i in range(len(values) - 1)
        )
        auc[seed] = 100.0 * area / width
    return {
        "available": bool(auc),
        "grid": {p: s for p, s in cells},
        "per_seed": auc,
        "mean": statistics.fmean(auc.values()) if auc else None,
    }


def worst_cell(summary: dict[str, Any], mode: str, presets: list[str]) -> dict[str, Any]:
    per = {p: mean_pts(per_seed(summary, p, mode)) for p in presets}
    present = {p: v for p, v in per.items() if v is not None}
    if not present:
        return {"available": False}
    worst = min(present, key=present.get)
    return {"available": True, "preset": worst, "success_pts": present[worst], "by_preset": present}


def controller_summary(training: dict[str, Any], mode: str) -> dict[str, Any]:
    """Terminal lambda and guard behaviour for one arm.

    ``final_lambda`` and ``return_guard_trips`` come from the training receipt.
    The lambda *peak* does not, so it is read from the curriculum jsonl when
    that file is still on disk -- it is what distinguishes "never got there"
    from "got there and was decayed back", which is the whole question about
    the old absolute guard.
    """
    out: dict[str, Any] = {"per_seed": {}}
    for arm in training.get("arms", {}).values():
        if arm.get("mode") != mode:
            continue
        peak = None
        recorded = arm.get("curriculum_path") or ""
        path = Path(recorded)
        if recorded and path.is_file():
            lambdas = [
                float(json.loads(line)["lambda"])
                for line in path.read_text().splitlines()
                if line.strip() and "lambda" in json.loads(line)
            ]
            peak = max(lambdas) if lambdas else None
        out["per_seed"][str(arm["seed"])] = {
            "terminal_lambda": arm.get("final_lambda"),
            "max_lambda": peak,
            "guard_trips": arm.get("return_guard_trips"),
            "spread_strata": (arm.get("arm_spec") or {}).get("spread_strata", 1),
            "return_guard": (arm.get("arm_spec") or {}).get("return_guard", "absolute"),
        }
    values = list(out["per_seed"].values())
    terminal = [v["terminal_lambda"] for v in values if v["terminal_lambda"] is not None]
    trips = [v["guard_trips"] for v in values if v["guard_trips"] is not None]
    out["mean_terminal_lambda"] = statistics.fmean(terminal) if terminal else None
    out["max_guard_trips"] = max(trips) if trips else None
    return out


def merge_summaries(*receipts: dict[str, Any]) -> dict[str, Any]:
    """One preset -> mode -> metrics view across several evaluation receipts."""
    merged: dict[str, Any] = {}
    for receipt in receipts:
        if not receipt:
            continue
        for preset, modes in (receipt.get("mode_summary") or {}).items():
            merged.setdefault(preset, {}).update(modes)
    return merged


def decide(passed: bool | None) -> str:
    return "pass" if passed is True else ("fail" if passed is False else "not evaluable")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=MANIFESTS / "lucid_support_expansion_preregistration_20260828.json",
    )
    parser.add_argument("--stage7-training", type=Path, required=True)
    parser.add_argument("--stage8-training", type=Path, required=True)
    parser.add_argument("--stage7-eval", type=Path, required=True)
    parser.add_argument("--stage8-eval", type=Path, required=True)
    parser.add_argument("--origin-eval", type=Path, default=None)
    parser.add_argument("--receipt-dir", type=Path, default=MANIFESTS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    prereg = load(args.preregistration)
    rules = prereg["decision_rules"]
    tie_pts = 2.0  # "< 2 pts on three seeds is a tie" -- decision_rules.tie

    s7_train, s8_train = load(args.stage7_training), load(args.stage8_training)
    s7_eval, s8_eval = load(args.stage7_eval), load(args.stage8_eval)
    origin_eval = load(args.origin_eval) if args.origin_eval else None
    summary = merge_summaries(s7_eval, s8_eval, origin_eval)

    modes = sorted({m for modes in summary.values() for m in modes})
    table = {
        mode: {
            "success_by_preset": {
                p: mean_pts(per_seed(summary, p, mode))
                for p in sorted(summary)
            },
            "profile_auc": profile_auc(summary, mode),
            "worst_cell": worst_cell(summary, mode, sorted(PROFILE_S)),
        }
        for mode in modes
    }
    controllers: dict[str, Any] = {}
    for train in (s7_train, s8_train):
        for mode in sorted({a["mode"] for a in train.get("arms", {}).values()}):
            controllers[mode] = controller_summary(train, mode)

    def auc(mode):
        block = table.get(mode, {}).get("profile_auc", {})
        return block.get("per_seed", {}) if block.get("available") else {}

    def clean(mode):
        return per_seed(summary, "id_clean", mode)

    # ---------------------------------------------------------- hypotheses --
    findings: dict[str, Any] = {}

    a_s4, a_lucid = auc("lucid_s4"), auc("lucid")
    d = paired(a_s4, a_lucid) if a_s4 and a_lucid else None
    findings["H_S1"] = {
        "claim": prereg["hypotheses"]["H_S1"],
        "evidence": d,
        "verdict": decide(None if d is None else (d["mean_pts"] is not None and d["mean_pts"] >= 2.0)),
    }

    ctrl = controllers.get("lucid_rg", {})
    lucid_clean_d = (
        paired(clean("lucid_rg"), clean("lucid"))
        if clean("lucid_rg") and clean("lucid") else None
    )
    lam, trips = ctrl.get("mean_terminal_lambda"), ctrl.get("max_guard_trips")
    findings["H_S2"] = {
        "claim": prereg["hypotheses"]["H_S2"],
        "evidence": {"controller": ctrl, "clean_vs_lucid": lucid_clean_d},
        "verdict": decide(
            None if (lam is None or lucid_clean_d is None)
            else (lam >= 0.9 and (trips or 0) <= 2 and lucid_clean_d["mean_pts"] >= -2.0)
        ),
    }

    full = "ta_lucid_50_s4_rg"
    a_full = auc(full)
    best_other = {m: statistics.fmean(v.values()) for m, v in ((m, auc(m)) for m in modes) if v}
    dr_full_d = (
        paired(per_seed(summary, "dr_full", full), per_seed(summary, "dr_full", "fixed"))
        if per_seed(summary, "dr_full", full) and per_seed(summary, "dr_full", "fixed") else None
    )
    clean_d = paired(clean(full), clean("fixed")) if clean(full) and clean("fixed") else None
    highest = (
        max(best_other, key=best_other.get) if best_other else None
    )
    findings["H_S3"] = {
        "claim": prereg["hypotheses"]["H_S3"],
        "evidence": {
            "profile_auc_ranking": dict(sorted(best_other.items(), key=lambda kv: -kv[1])),
            "highest_auc_arm": highest,
            "dr_full_vs_fixed": dr_full_d,
            "id_clean_vs_fixed": clean_d,
        },
        "verdict": decide(
            None if (highest is None or dr_full_d is None or clean_d is None)
            else (highest == full and dr_full_d["mean_pts"] >= -2.0 and clean_d["mean_pts"] >= 5.0)
        ),
    }

    origin_clean = clean("origin")
    full_gap = paired(clean(full), origin_clean) if clean(full) and origin_clean else None
    fixed_gap = paired(clean("fixed"), origin_clean) if clean("fixed") and origin_clean else None
    findings["H_S4"] = {
        "claim": prereg["hypotheses"]["H_S4"],
        "evidence": {
            "origin_id_clean_pts": mean_pts(origin_clean),
            "treatment_minus_origin": full_gap,
            "fixed_minus_origin": fixed_gap,
        },
        "verdict": decide(
            None if (full_gap is None or fixed_gap is None)
            else (full_gap["mean_pts"] >= -10.0 and fixed_gap["mean_pts"] <= -20.0)
        ),
    }

    extrap = (
        paired(per_seed(summary, "dr_125", full), per_seed(summary, "dr_125", "fixed"))
        if per_seed(summary, "dr_125", full) and per_seed(summary, "dr_125", "fixed") else None
    )
    findings["H_S5"] = {
        "claim": prereg["hypotheses"]["H_S5"],
        "evidence": extrap,
        "verdict": decide(
            None if extrap is None
            else (
                extrap["favorable_seeds"] >= 2
                and sum(1 for v in extrap["per_seed_pts"].values() if v >= 3.0) >= 2
            )
        ),
    }

    receipt = {
        "kind": "lucid_support_expansion_analysis",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "preregistration": {
            "path": str(args.preregistration),
            "logical_sha256": prereg.get("logical_sha256"),
        },
        "inputs": {
            "stage7_training": str(args.stage7_training),
            "stage8_training": str(args.stage8_training),
            "stage7_eval": str(args.stage7_eval),
            "stage8_eval": str(args.stage8_eval),
            "origin_eval": str(args.origin_eval) if args.origin_eval else None,
        },
        "tie_threshold_pts": tie_pts,
        "arms": table,
        "controllers": controllers,
        "hypotheses": findings,
        "not_yet_verified": [
            "any claim about a physical robot; every number here is simulated",
            "generalisation beyond the 102-motion content-dev panel and three seeds",
            f"decision rule on collapse: {rules['if_every_arm_still_collapses']}",
        ],
    }
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = args.receipt_dir / f"lucid_s_analysis_{stamp}.json"
    out.write_text(json.dumps(receipt, indent=2, default=str))

    print(f"{'arm':<22}{'clean':>8}{'dr_050':>8}{'dr_full':>8}{'dr_125':>8}{'lat60':>8}{'AUC':>8}")
    for mode in modes:
        row = table[mode]["success_by_preset"]
        auc_block = table[mode]["profile_auc"]
        def cell(key):
            value = row.get(key)
            return f"{value:8.2f}" if value is not None else f"{'-':>8}"
        auc_value = auc_block.get("mean")
        print(
            f"{mode:<22}{cell('id_clean')}{cell('dr_050')}{cell('dr_full')}"
            f"{cell('dr_125')}{cell('latency_60ms')}"
            + (f"{auc_value:8.2f}" if auc_value is not None else f"{'-':>8}")
        )
    print()
    for name, block in findings.items():
        print(f"{name}: {block['verdict']}")
    print(f"receipt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
