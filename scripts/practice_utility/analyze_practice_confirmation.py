#!/usr/bin/env python3
"""Apply the frozen multi-seed practice-allocation confirmation rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.research.practice_utility import practice_confirmation as PC

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DEFAULT_PREREG = (
    WORKSPACE
    / "receipts/manifests/lucid_practice_allocation_confirmation_preregistration_20260904.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", nargs="+", type=Path, required=True)
    parser.add_argument("--training", nargs="+", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    preregistration = json.loads(args.preregistration.read_text())
    cells = PC.load_evaluations(args.evaluation)
    exposures = PC.load_exposures(args.training)
    result = PC.analyze(cells, exposures, preregistration)
    result["preregistration"] = str(args.preregistration.resolve())
    result["evaluation_inputs"] = [str(path.resolve()) for path in args.evaluation]
    result["training_inputs"] = [str(path.resolve()) for path in args.training]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    audit = result["instrument_audit"]
    print(
        f"cells {audit['observed_expected_cells']}/{audit['expected_cells']}; "
        f"audit={'PASS' if audit['complete'] else 'INCOMPLETE'}"
    )
    for name, decision in result["decisions"].items():
        print(f"{name}: {decision['verdict']}")
    print(f"receipt -> {args.out}")
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
