#!/usr/bin/env python3
"""Materialise a seeded training subset of a frozen split partition.

Training SONIC from scratch is a different experiment from fine-tuning it, and
it needs a training set sized to the compute available rather than to what the
released model was trained on. What it must *not* change is the instrument: the
evaluation panel stays the frozen ``dev`` partition of the same pool, so every
number produced from here is directly comparable to everything measured before.

So the subset is drawn only from the ``adaptation`` partition, which the content
split already separated from ``dev`` and ``test`` by motion-content linkage.
Sub-sampling inside one partition cannot leak into another, and the receipt
records the exact keys, the seed, and the sha256 of the sorted key list, so a
campaign can prove which motions it trained on.

Symlinks only: no clip is copied, and every target is verified to exist before
the subset is declared usable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility.paths import LUCID_ROOT, relocate  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-manifest", type=Path, default=LUCID_ROOT / "manifests/pool_debug512.json"
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=LUCID_ROOT / "manifests/split_debug512_content.json"
    )
    parser.add_argument(
        "--partition",
        default="adaptation",
        help="the split partition to draw from; never 'dev' or 'test'",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="how many motions to keep; 0 or >= partition size keeps the whole partition",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--name", required=True, help="subset id, e.g. train064")
    parser.add_argument("--root", type=Path, default=LUCID_ROOT / "pools/subsets")
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    return parser.parse_args(argv)


def sha256_of_keys(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.partition in ("dev", "test"):
        raise SystemExit(
            f"refusing to build a training subset from {args.partition!r}: that partition is "
            "the evaluation instrument"
        )
    pool = json.loads(args.pool_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    if split["pool_sha256"] != pool["pool_sha256"]:
        raise SystemExit("pool and split manifests do not match")

    available = sorted(k for k, v in split["assignment"].items() if v == args.partition)
    if not available:
        raise SystemExit(f"partition {args.partition!r} is empty")
    size = len(available) if args.size <= 0 else min(args.size, len(available))
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(available, size))

    motion_by_key = {row["motion_key"]: row for row in pool["motions"]}
    missing = [k for k in selected if k not in motion_by_key]
    if missing:
        raise SystemExit(f"split keys missing from pool: {missing[:3]}")

    motion_dir = args.root / args.name / "robot_filtered"
    motion_dir.mkdir(parents=True, exist_ok=True)
    for key in selected:
        source = relocate(motion_by_key[key]["path"]).resolve()
        if not source.is_file():
            raise SystemExit(f"clip missing on disk: {source}")
        link = motion_dir / f"{key}.pkl"
        if link.is_symlink():
            if link.resolve() != source:
                raise SystemExit(f"existing link points elsewhere: {link}")
        elif link.exists():
            raise SystemExit(f"subset path is not a symlink: {link}")
        else:
            link.symlink_to(source)
    present = {p.stem for p in motion_dir.glob("*.pkl")}
    extra = sorted(present - set(selected))
    if extra:
        raise SystemExit(f"subset directory holds motions outside the selection: {extra[:3]}")

    receipt: dict[str, Any] = {
        "kind": "lucid_training_subset",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "name": args.name,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "config": {
            "pool_manifest": str(args.pool_manifest),
            "split_manifest": str(args.split_manifest),
            "partition": args.partition,
            "requested_size": args.size,
            "seed": args.seed,
        },
        "pool_sha256": pool["pool_sha256"],
        "split_sha256": split["split_sha256"],
        "split_linkage": split["linkage"],
        "motion_file": str(motion_dir.resolve()),
        "motion_count": len(selected),
        "partition_size": len(available),
        "motion_keys_sha256": sha256_of_keys(selected),
        "motion_keys": selected,
        "verified": [
            f"every one of {len(selected)} clips resolves to an existing file",
            "the subset directory contains nothing outside the selection",
            f"drawn only from the {args.partition!r} partition, so dev and test are untouched",
        ],
        "not_yet_verified": [
            "that this subset is large enough to train a useful policy -- that is the experiment",
        ],
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    out = args.receipt_dir / f"training_subset_{args.name}.json"
    out.write_text(json.dumps(receipt, indent=2))
    print(f"{args.name}: {len(selected)} motions from {args.partition} "
          f"({len(available)} available)  keys_sha256={receipt['motion_keys_sha256'][:16]}")
    print(f"motion_file {motion_dir.resolve()}")
    print(f"receipt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
