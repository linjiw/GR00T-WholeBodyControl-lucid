#!/usr/bin/env python3
"""Canonicalize a box-gate trace for an exposure-matched open-loop control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.research.practice_utility import vector_yoke as VY


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-box-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-strata", type=int, default=8)
    parser.add_argument("--expected-channel", action="append", dest="expected_channels")
    parser.add_argument("--max-intensity", type=float, default=VY.DEFAULT_MAX_INTENSITY)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    schedule = VY.canonicalize_box_trace(
        args.source_box_trace,
        expected_channels=args.expected_channels,
        expected_strata=args.expected_strata,
        max_intensity=args.max_intensity,
    )
    schedule.write(args.out)
    print(
        json.dumps(
            {
                "schedule": str(args.out.resolve()),
                "canonical_sha256": schedule.canonical_sha256,
                "source_sha256": schedule.source_sha256,
                "records": len(schedule.records),
                "channels": list(schedule.channels),
                "stratum_sizes": list(schedule.stratum_sizes),
                "warmup_backfilled_records": schedule.warmup_backfilled_records,
                "probe_transition_records": schedule.probe_transition_records,
                "frontier_final": schedule.records[-1].frontier_vector,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
