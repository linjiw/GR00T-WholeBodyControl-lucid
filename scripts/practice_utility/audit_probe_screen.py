#!/usr/bin/env python3
"""Audit a frozen Track-A probe screen; never launch it.

The default points at the historical v1 manifest.  Because that manifest has
no claim-grade preflight sidecar, the default invocation is expected to exit 2
and enumerate the missing evidence rather than infer it from old filenames.
"""

# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for mixed import/from-import blocks.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import probe_campaign as PC  # noqa: E402
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402

DEFAULT_MANIFEST = LUCID_ROOT / "manifests/probe_screen_v1_late.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--preflight",
        type=Path,
        help="hashed preregistration bundle (default: <manifest-stem>.preflight.json)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = PC.audit_probe_campaign(args.manifest, args.preflight)
    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_suffix(args.output.suffix + ".partial")
        staging.write_text(json.dumps(payload, indent=2) + "\n")
        staging.replace(args.output)
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
