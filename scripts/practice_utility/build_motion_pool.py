#!/usr/bin/env python3
"""Scan a motion pool, build group-disjoint splits, and freeze them to disk.

Run once per pool. The outputs are the anchor for everything downstream --
context keys, interventions, dose accounting, utility labels -- so they are
content-hashed and, once written, treated as immutable.

Two splits are written, because BONES-SEED cannot support one that closes both
leakage channels at once (see ``split.build_split``):

``performer``  unseen performers, actions may recur  -> test-repetition
``content``    unseen actions, performers may recur  -> test-content

Reporting them separately is deliberate. A single blended "OOD" number would
hide which generalization was actually measured.

Example
-------
    source $LUCID_ROOT/lucid_env.sh
    python scripts/practice_utility/build_motion_pool.py \\
        --pool-dir $LUCID_ROOT/pools/debug512/robot_filtered \\
        --pool-id debug512 \\
        --output-dir $LUCID_ROOT/manifests
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.research.practice_utility import motion_pool as M  # noqa: E402
from gear_sonic.research.practice_utility import split as S  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", required=True, type=Path,
                        help="directory of robot_filtered *.pkl clips")
    parser.add_argument("--pool-id", required=True,
                        help="short identifier, e.g. debug512")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--linkages", nargs="+", default=["performer", "content"],
                        choices=["performer", "content", "performer_and_content"])
    parser.add_argument("--ratios", type=json.loads, default=None,
                        help='JSON, e.g. \'{"adaptation":0.6,"dev":0.2,"test":0.2}\'')
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="retain exact-duplicate trajectories (default: drop)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="permit replacing existing frozen manifests")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"scanning {args.pool_dir} ...")
    scan = M.scan_pool(args.pool_dir, limit=args.limit)
    print(f"  {scan.num_motions} clips, {len(scan.performer_counts())} performers")
    if scan.unparsed:
        print(f"  WARNING: {len(scan.unparsed)} keys could not be parsed and were skipped")
    if scan.duplicate_groups:
        print(f"  {len(scan.duplicate_groups)} exact-duplicate trajectory groups")
        if not args.keep_duplicates:
            scan = M.drop_exact_duplicates(scan)
            print(f"  deduplicated -> {scan.num_motions} clips")

    pool_sha = M.pool_sha256(scan)
    summary = scan.summary()
    other = summary["family_counts"].get(M.FALLBACK_FAMILY, 0)
    share = other / scan.num_motions if scan.num_motions else 0.0
    print(f"  pool_sha256 {pool_sha[:16]}  unclassified family share {share:.1%}")
    if share > 0.15:
        print("  WARNING: a large unclassified share weakens family stratification; "
              "consider extending FAMILY_RULES")

    outputs: dict[str, Path] = {}
    pool_path = args.output_dir / f"pool_{args.pool_id}.json"
    _write(pool_path, {
        "kind": "practice_utility_motion_pool",
        "schema_version": 1,
        "pool_id": args.pool_id,
        "pool_sha256": pool_sha,
        "source_root": scan.source_root,
        "deduplicated": not args.keep_duplicates,
        "summary": summary,
        "duplicate_groups": scan.duplicate_groups,
        "unparsed": scan.unparsed,
        "motions": [r.to_dict() for r in scan.records],
    }, args.overwrite)
    outputs["pool"] = pool_path

    for linkage in args.linkages:
        try:
            result = S.build_split(scan, pool_sha, linkage=linkage,
                                   ratios=args.ratios, seed=args.seed)
        except S.SplitError as error:
            print(f"  split[{linkage}] REFUSED: {error}")
            continue
        path = args.output_dir / f"split_{args.pool_id}_{linkage}.json"
        _write(path, result.to_dict(), args.overwrite)
        outputs[f"split_{linkage}"] = path
        stats = result.stats
        print(f"  split[{linkage}] {stats['total_groups']} groups, "
              f"largest {stats['largest_group_share']:.2%} -> {path.name}")
        for name, part in stats["partitions"].items():
            print(f"      {name:11s} {part['motions']:5d} clips "
                  f"({part['share']:.3f} vs {part['target_share']}) "
                  f"{part['groups']:4d} groups  {part['duration_seconds'] / 60:7.1f} min")

    print("\nwrote:")
    for label, path in outputs.items():
        print(f"  {label:20s} {path}")
    return 0


def _write(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"{path} already exists. These manifests are frozen inputs to the "
            "measurement; pass --overwrite only if you intend to invalidate every "
            "label and branch derived from them."
        )
    staging = path.with_suffix(".json.partial")
    staging.write_text(json.dumps(payload, indent=2, sort_keys=False))
    staging.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
