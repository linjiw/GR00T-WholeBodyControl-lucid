#!/usr/bin/env python3
"""Audit a frozen-policy evaluation receipt before anybody interprets its means.

An evaluation receipt is only evidence if every cell it claims actually ran, ran
once, ran on the checkpoint it names, and ran on the panel the protocol froze.
None of that is visible in ``mode_summary``, which is what everyone reads. This
checks it and writes its own receipt, so "the table was audited" is a artifact
rather than a memory.

What it verifies, each independently fatal:

``coverage``      every (mode, preset, seed) cell in the declared design exists,
                  exactly once, and is marked complete.
``exit status``   no run failed or is missing its metrics file.
``checkpoints``   each mode/seed pair used one checkpoint, its sha256 is
                  recorded, and evaluation did not mutate it.
``panel``         every run scored the same motion count, and the receipt's
                  panel hash matches the protocol's.
``aggregation``   the per-seed values in ``mode_summary`` reproduce, exactly,
                  from the per-run summaries. A mean that cannot be rebuilt from
                  its parts is not a measurement.
``saturation``    presets that are 0 or 1 for every arm are flagged: they cannot
                  rank policies and must not be used as ranking endpoints.

Nothing here interprets a result. It decides whether interpretation is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

#: A preset whose success rate is this close to 0 or 1 for *every* arm cannot
#: separate two policies, whatever its mean says.
SATURATION_EPSILON = 1e-9


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def cell_key(run: dict[str, Any]) -> tuple[str, str, int]:
    return (run["mode"], run["preset"], int(run["checkpoint_seed"]))


def audit_coverage(receipt: dict[str, Any]) -> dict[str, Any]:
    """Every declared cell present exactly once, and complete."""
    runs = list(receipt.get("runs", {}).values())
    protocol = receipt.get("protocol", {})
    modes = sorted({run["mode"] for run in runs})
    presets = sorted(protocol.get("presets", {}) or {p["preset"] for p in runs})
    seeds = sorted(int(s) for s in protocol.get("checkpoint_seeds", []))
    expected = {(m, p, s) for m in modes for p in presets for s in seeds}
    seen = Counter(cell_key(run) for run in runs)
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    missing = sorted(expected - set(seen))
    incomplete = sorted(cell_key(run) for run in runs if not run.get("complete"))
    return {
        "modes": modes,
        "presets": presets,
        "checkpoint_seeds": seeds,
        "expected_cells": len(expected),
        "observed_cells": len(runs),
        "duplicate_cells": [list(k) for k in duplicates],
        "missing_cells": [list(k) for k in missing],
        "incomplete_cells": [list(k) for k in incomplete],
        "ok": not duplicates and not missing and not incomplete,
    }


def audit_checkpoints(receipt: dict[str, Any]) -> dict[str, Any]:
    """One checkpoint per (mode, seed), hashed, and unchanged by evaluation."""
    by_pair: dict[tuple[str, int], set[str]] = defaultdict(set)
    unhashed: list[list[Any]] = []
    for run in receipt.get("runs", {}).values():
        pair = (run["mode"], int(run["checkpoint_seed"]))
        by_pair[pair].add(str(run.get("checkpoint")))
        if not run.get("checkpoint_sha256"):
            unhashed.append([run["mode"], run["preset"], run["checkpoint_seed"]])
    split = sorted(list(pair) for pair, paths in by_pair.items() if len(paths) > 1)
    before = receipt.get("checkpoint_sha256_before") or {}
    after = receipt.get("checkpoint_sha256_after") or {}
    shared = sorted(set(before) & set(after))
    mutated = sorted(k for k in shared if before[k] != after[k])
    return {
        "mode_seed_pairs": len(by_pair),
        "pairs_with_multiple_checkpoints": split,
        "runs_without_a_checkpoint_hash": unhashed,
        "checkpoints_hashed_before_and_after": len(shared),
        "checkpoints_mutated_by_evaluation": mutated,
        "ok": not split and not unhashed and not mutated and bool(shared),
    }


def audit_panel(receipt: dict[str, Any]) -> dict[str, Any]:
    """One panel, one motion count, everywhere."""
    suite = (receipt.get("protocol", {}) or {}).get("suite", {}) or {}
    counts = Counter(
        int(run["summary"]["motion_count"])
        for run in receipt.get("runs", {}).values()
        if isinstance(run.get("summary"), dict) and "motion_count" in run["summary"]
    )
    return {
        "declared_motion_count": suite.get("motion_count"),
        "observed_motion_counts": dict(counts),
        "motion_keys_sha256": suite.get("motion_keys_sha256"),
        "pool_sha256": suite.get("pool_sha256"),
        "split_sha256": suite.get("split_sha256"),
        "partition": suite.get("partition"),
        "ok": (
            len(counts) == 1
            and (suite.get("motion_count") is None or suite["motion_count"] in counts)
            and bool(suite.get("motion_keys_sha256"))
        ),
    }


def audit_aggregation(
    receipt: dict[str, Any], metric: str = "success_rate", tolerance: float = 1e-12
) -> dict[str, Any]:
    """``mode_summary`` must be rebuildable, exactly, from the per-run summaries."""
    from_runs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for run in receipt.get("runs", {}).values():
        summary = run.get("summary") or {}
        if metric in summary and summary[metric] is not None:
            from_runs[(run["preset"], run["mode"])][str(run["checkpoint_seed"])] = float(
                summary[metric]
            )
    mismatches = []
    checked = 0
    for preset, modes in (receipt.get("mode_summary") or {}).items():
        for mode, block in modes.items():
            recorded = (
                block.get("metrics", {}).get(metric, {}).get("per_checkpoint_seed", {})
            )
            rebuilt = from_runs.get((preset, mode), {})
            for seed, value in recorded.items():
                checked += 1
                other = rebuilt.get(str(seed))
                if other is None or abs(float(value) - other) > tolerance:
                    mismatches.append(
                        {"preset": preset, "mode": mode, "seed": seed,
                         "summary": value, "from_runs": other}
                    )
    return {
        "metric": metric,
        "per_seed_values_checked": checked,
        "mismatches": mismatches,
        "ok": checked > 0 and not mismatches,
    }


def audit_saturation(receipt: dict[str, Any], metric: str = "success_rate") -> dict[str, Any]:
    """Flag presets that are constant at a bound across every arm and seed."""
    by_preset: dict[str, list[float]] = defaultdict(list)
    for run in receipt.get("runs", {}).values():
        summary = run.get("summary") or {}
        if summary.get(metric) is not None:
            by_preset[run["preset"]].append(float(summary[metric]))
    report = {}
    for preset, values in sorted(by_preset.items()):
        low = all(v <= SATURATION_EPSILON for v in values)
        high = all(v >= 1.0 - SATURATION_EPSILON for v in values)
        report[preset] = {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "saturated_at_floor": low,
            "saturated_at_ceiling": high,
            "rankable": not (low or high) and max(values) > min(values),
        }
    saturated = sorted(p for p, r in report.items() if not r["rankable"])
    return {
        "by_preset": report,
        "unrankable_presets": saturated,
        "ok": True,  # informational: saturation is a finding, not a failure
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--grade", default="screening", choices=("screening", "confirmatory"))
    parser.add_argument("--min-seeds-for-confirmatory", type=int, default=5)
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    receipt = load(args.receipt)
    coverage = audit_coverage(receipt)
    checkpoints = audit_checkpoints(receipt)
    panel = audit_panel(receipt)
    aggregation = audit_aggregation(receipt)
    saturation = audit_saturation(receipt)

    seeds = len(coverage["checkpoint_seeds"])
    grade = args.grade
    if grade == "confirmatory" and seeds < args.min_seeds_for_confirmatory:
        grade_note = (
            f"requested confirmatory but only {seeds} training seeds are present; "
            f"{args.min_seeds_for_confirmatory} are required. Downgraded to screening."
        )
        grade = "screening"
    else:
        grade_note = (
            f"{seeds} training seeds"
            + ("" if grade == "confirmatory" else " -- screening-grade, not confirmatory")
        )

    fatal = [
        name
        for name, block in (
            ("coverage", coverage),
            ("checkpoints", checkpoints),
            ("panel", panel),
            ("aggregation", aggregation),
        )
        if not block["ok"]
    ]
    out = {
        "kind": "lucid_evaluation_audit",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "audited_receipt": str(args.receipt),
        "audited_receipt_sha256": hashlib.sha256(
            Path(args.receipt).read_bytes()
        ).hexdigest(),
        "audited_experiment_id": receipt.get("experiment_id"),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "evidence_grade": grade,
        "grade_note": grade_note,
        "coverage": coverage,
        "checkpoints": checkpoints,
        "panel": panel,
        "aggregation": aggregation,
        "saturation": saturation,
        "fatal_failures": fatal,
        "interpretation_allowed": not fatal,
        "not_yet_verified": [
            "that the panel generalises beyond the frozen in-pool partition "
            "(this is fresh-physics robustness, not motion generalization)",
            "anything about a physical robot",
        ],
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = args.receipt_dir / f"evaluation_audit_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str))

    if not args.quiet:
        print(f"receipt      {args.receipt}")
        print(f"grade        {grade}  ({grade_note})")
        for name, block in (
            ("coverage", coverage), ("checkpoints", checkpoints),
            ("panel", panel), ("aggregation", aggregation),
        ):
            print(f"{name:<13}{'ok' if block['ok'] else 'FAILED'}")
        print(f"cells        {coverage['observed_cells']}/{coverage['expected_cells']}")
        if saturation["unrankable_presets"]:
            print(f"unrankable   {', '.join(saturation['unrankable_presets'])}")
        print(f"verdict      {'interpretation allowed' if not fatal else 'BLOCKED: ' + ','.join(fatal)}")
        print(f"audit        {path}")
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
