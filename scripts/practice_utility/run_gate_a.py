#!/usr/bin/env python3
"""Run Gate A over evaluation receipts and write its decision as a receipt.

Gate A asks the question that has to come before any curriculum comparison: is
there a hard bin that equal-budget direct mixed training does *not* already
learn? If there is not, a curriculum is unnecessary here and that is the result.

The bin is chosen by a frozen rule from the *reference* arm only (see
``learnability_gate.select_hard_bin``), so the choice cannot be steered by which
treatment happens to look good. Saturated bins and the 60 ms latency cells are
excluded, the latter by name.

Usage::

    run_gate_a.py --eval receipts/a.json --eval receipts/b.json \
        --curriculum-arm lucid --curriculum-arm ta_lucid_50_s4_rg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import learnability_gate as G  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", dest="evals", type=Path, action="append", required=True)
    parser.add_argument("--reference-arm", default="origin")
    parser.add_argument("--direct-mixed-arm", default="fixed")
    parser.add_argument("--curriculum-arm", dest="curricula", action="append", default=[])
    parser.add_argument("--learned-margin-pts", type=float, default=5.0)
    parser.add_argument("--curriculum-margin-pts", type=float, default=5.0)
    parser.add_argument("--tie-pts", type=float, default=2.0)
    parser.add_argument("--grade", default="screening", choices=("screening", "confirmatory"))
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipts = [json.loads(p.read_text()) for p in args.evals]
    table = G.per_preset_by_mode(receipts)
    chosen, candidates = G.select_hard_bin(table, reference_mode=args.reference_arm)
    thresholds = G.GateAThresholds(
        learned_margin_pts=args.learned_margin_pts,
        curriculum_margin_pts=args.curriculum_margin_pts,
        tie_pts=args.tie_pts,
    )
    result = (
        G.score_gate_a(
            table, chosen.preset,
            direct_mixed_mode=args.direct_mixed_arm,
            reference_mode=args.reference_arm,
            curriculum_modes=tuple(args.curricula),
            thresholds=thresholds,
        )
        if chosen is not None
        else None
    )

    out = {
        "kind": "lucid_gate_a_learnability",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "evidence_grade": args.grade,
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in args.evals
        ],
        "selection_rule": (
            "the hardest RANKABLE preset by the reference arm's success there; "
            "saturated presets excluded by measurement, 60 ms cells excluded by name"
        ),
        "banned_ranking_presets": sorted(G.BANNED_RANKING_PRESETS),
        "candidates": [c.to_dict() for c in candidates],
        "selected_bin": None if chosen is None else chosen.to_dict(),
        "gate_a": None if result is None else result.to_dict(),
        "verdict": "no_rankable_bin" if result is None else result.verdict,
        "not_yet_verified": [
            "anything about a physical robot",
            "motion generalization -- the panel is an in-pool partition, so this "
            "is fresh-physics robustness",
        ],
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = args.receipt_dir / f"gate_a_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str))

    print("candidate bins (reference arm, spread, rankable):")
    for c in candidates:
        mark = "*" if chosen is not None and c.preset == chosen.preset else " "
        print(f" {mark} {c.preset:<16}{c.reference_pts:7.2f}{c.spread_pts:8.2f}  {c.reason}")
    if result is None:
        print("\nno rankable bin: the evaluation grid cannot separate policies anywhere.")
    else:
        print(f"\nselected bin  {result.preset}")
        print(f"origin        {result.origin_pts:.2f}")
        print(f"direct mixed  {result.direct_mixed_pts:.2f}  ({result.direct_mixed_minus_origin_pts:+.2f})")
        if result.best_curriculum_arm:
            print(f"best curric.  {result.best_curriculum_arm} {result.best_curriculum_pts:.2f} "
                  f"({result.curriculum_minus_direct_pts:+.2f} vs direct)")
        print(f"verdict       {result.verdict}")
        print(f"              {result.rationale}")
    print(f"receipt       {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
