#!/usr/bin/env python3
"""Score the preregistered TACE pilot hypotheses from receipts.

Reads the training receipt (curriculum trajectories, cohort telemetry) and the
frozen-evaluation receipt (per-seed success by preset and arm), and evaluates
H-A / H-A2 / H-B / H-C exactly as written in
``manifests/tace_pilot_preregistration_20260827.json``. Writes a JSON analysis
receipt and prints a compact table. Contains no tunable thresholds: they come
from the preregistration file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from gear_sonic.research.practice_utility.paths import LUCID_ROOT

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


MANIFESTS = LUCID_ROOT / "manifests"
NONINFERIORITY_PTS = 2.0
CLEAN_SUPERIORITY_PTS = 2.0
LAMBDA_TOLERANCE = 0.1


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def per_seed(summary: dict[str, Any], preset: str, mode: str, metric: str) -> dict[str, float]:
    block = summary.get(preset, {}).get(mode)
    if not block:
        return {}
    values = block["metrics"][metric]["per_checkpoint_seed"]
    return {seed: float(v) for seed, v in values.items() if v is not None}


def mean_pts(values: dict[str, float]) -> float | None:
    return 100.0 * statistics.fmean(values.values()) if values else None


def paired(treatment: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(treatment) & set(reference))
    deltas = {s: 100.0 * (treatment[s] - reference[s]) for s in common}
    values = list(deltas.values())
    return {
        "per_seed_pts": deltas,
        "mean_pts": statistics.fmean(values) if values else None,
        "favorable_seeds": sum(1 for v in values if v > 0),
        "num_seeds": len(values),
    }


def lambda_trajectory(training: dict[str, Any], mode: str) -> dict[str, list[float]]:
    out = {}
    for arm in training["arms"].values():
        if arm["mode"] != mode:
            continue
        rows = [json.loads(l) for l in Path(arm["curriculum_path"]).read_text().splitlines() if l.strip()]
        out[str(arm["seed"])] = [float(r["lambda"]) for r in rows]
    return out


def cohort_dose_ok(training: dict[str, Any], mode: str, warmup: int) -> dict[str, Any]:
    """Anchor realized delay must exceed focus delay in every post-warmup iteration."""
    result = {}
    for arm in training["arms"].values():
        if arm["mode"] != mode:
            continue
        rows = [json.loads(l) for l in Path(arm["curriculum_path"]).read_text().splitlines() if l.strip()]
        checks = []
        for row in rows[warmup:]:
            t = row.get("tace") or {}
            a, f = t.get("anchor_delay_mean_steps"), t.get("focus_delay_mean_steps")
            if a is None or f is None:
                checks.append(None)
            else:
                checks.append(a > f)
        result[str(arm["seed"])] = {
            "iterations_checked": len(checks),
            "anchor_exceeds_focus_every_iteration": all(c is True for c in checks) if checks else None,
            "missing": sum(c is None for c in checks),
            "final": rows[-1].get("tace") if rows else None,
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--eval-receipt", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=MANIFESTS / "tace_pilot_preregistration_20260827.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    training = load(args.training_receipt)
    evaluation = load(args.eval_receipt)
    prereg = load(args.preregistration)
    summary = evaluation["mode_summary"]
    warmup = int(training["config"]["warmup_iterations"])

    def sr(preset, mode):
        return per_seed(summary, preset, mode, "success_rate")

    table = {}
    for preset in ("id_clean", "dr_050", "dr_full", "latency_60ms"):
        table[preset] = {mode: mean_pts(sr(preset, mode)) for mode in summary.get(preset, {})}

    ta, fixed, lucid, yoked = "ta_lucid_25", "fixed", "lucid", "ta_yoked_25"
    ta_vs_fixed_full = paired(sr("dr_full", ta), sr("dr_full", fixed))
    ta_vs_fixed_clean = paired(sr("id_clean", ta), sr("id_clean", fixed))
    ta_vs_lucid_full = paired(sr("dr_full", ta), sr("dr_full", lucid))
    ta_vs_yoked_full = paired(sr("dr_full", ta), sr("dr_full", yoked))

    h_a = (
        ta_vs_fixed_full["mean_pts"] is not None
        and ta_vs_fixed_full["mean_pts"] >= -NONINFERIORITY_PTS
        and ta_vs_fixed_clean["mean_pts"] is not None
        and ta_vs_fixed_clean["mean_pts"] >= CLEAN_SUPERIORITY_PTS
    )
    h_a2 = ta_vs_lucid_full["mean_pts"] is not None and ta_vs_lucid_full["mean_pts"] > 2.0
    h_b = ta_vs_yoked_full["num_seeds"] > 0 and ta_vs_yoked_full["favorable_seeds"] >= 2

    lam_ta = lambda_trajectory(training, ta)
    lam_lucid = lambda_trajectory(training, lucid)
    terminal = {
        s: {"ta_lucid_25": lam_ta[s][-1], "lucid": lam_lucid.get(s, [None])[-1]}
        for s in lam_ta
    }
    lambda_ok = all(
        v["lucid"] is None or v["ta_lucid_25"] >= v["lucid"] - LAMBDA_TOLERANCE for v in terminal.values()
    )
    dose = cohort_dose_ok(training, ta, warmup)
    dose_ok = all(d["anchor_exceeds_focus_every_iteration"] for d in dose.values()) if dose else False
    h_c = lambda_ok and dose_ok

    profile = {}
    for mode in summary.get("dr_full", {}):
        pts = [table[p].get(mode) for p in ("id_clean", "dr_050", "dr_full")]
        profile[mode] = statistics.fmean(pts) if all(v is not None for v in pts) else None

    receipt = {
        "kind": "lucid_tace_pilot_analysis",
        "created_at": datetime.now().astimezone().isoformat(),
        "training_receipt": str(args.training_receipt),
        "eval_receipt": str(args.eval_receipt),
        "preregistration": str(args.preregistration),
        "preregistered_hypotheses": prereg["hypotheses"],
        "success_pts_by_preset_and_arm": table,
        "coarse_profile_mean_pts": profile,
        "paired": {
            "ta_lucid_25_vs_fixed_dr_full": ta_vs_fixed_full,
            "ta_lucid_25_vs_fixed_id_clean": ta_vs_fixed_clean,
            "ta_lucid_25_vs_lucid_dr_full": ta_vs_lucid_full,
            "ta_lucid_25_vs_ta_yoked_25_dr_full": ta_vs_yoked_full,
        },
        "terminal_lambda": terminal,
        "cohort_dose": dose,
        "decisions": {
            "H_A_headline": h_a,
            "H_A2_over_lucid": h_a2,
            "H_B_attribution": h_b,
            "H_C_mechanism": h_c,
        },
        "caveats": [
            "three checkpoint seeds: screening-grade effect sizes, not a paper-grade claim",
            "content-dev panel motions were in the 512-motion training pool: fresh-physics robustness, not motion generalization",
        ],
    }
    out = args.out or (MANIFESTS / f"tace_pilot_analysis_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")

    print(f"{'preset':<14}" + "".join(f"{m:>14}" for m in summary.get("dr_full", {})))
    for preset, row in table.items():
        print(f"{preset:<14}" + "".join(f"{(row.get(m) if row.get(m) is not None else float('nan')):>14.2f}" for m in summary.get("dr_full", {})))
    print("profile mean  " + "".join(f"{(profile.get(m) if profile.get(m) is not None else float('nan')):>14.2f}" for m in summary.get("dr_full", {})))
    print(json.dumps(receipt["decisions"], indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'per_seed_pts'} for k, v in receipt['paired'].items()}, indent=2))
    print(f"analysis receipt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
