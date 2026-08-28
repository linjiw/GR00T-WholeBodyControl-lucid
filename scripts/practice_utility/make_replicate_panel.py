#!/usr/bin/env python3
"""Build a K-alias replicate panel so a one-motion policy can be evaluated at all.

SONIC's evaluation callback slices every scored quantity to
``[:num_unique_motions]``. On a one-motion panel that is **environment 0 alone**:
one Bernoulli trial per (arm, seed, preset), and worse, the eval loop's bound is
driven by env 0, so when env 0 dies the loop ends and every other environment is
frozen mid-episode and recorded as ``terminated=False`` with ``progress < 1``.
Those episodes are censored *and* mislabelled, biased toward overstating success,
and the bias is worst in exactly the hard cells a robustness ladder exists to
measure.

The fix is to give the evaluator K distinct motion *keys* that all resolve to the
same clip. Then ``num_unique_motions == K``, the bound becomes K, every
environment is scored, every episode runs to its own termination or to the end of
the clip, and ``success_rate`` / ``progress`` become means over K genuine
episodes. Nothing about the physics or the policy changes -- only how many
independent episodes the instrument is willing to look at.

The aliases are symlinks to one file, so the panel costs no disk and every
replicate is provably the same clip: the receipt records the shared target's
sha256 and the identity is checkable after the fact.
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

from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-receipt",
        type=Path,
        required=True,
        help="a lucid_training_subset receipt naming exactly one motion",
    )
    parser.add_argument("--replicates", type=int, default=512)
    parser.add_argument("--name", required=True, help="panel id, e.g. panel_hob002_k512")
    parser.add_argument("--root", type=Path, default=LUCID_ROOT / "pools/panels")
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")
    subset = json.loads(args.subset_receipt.read_text())
    keys = subset["motion_keys"]
    if len(keys) != 1:
        raise SystemExit(
            f"a replicate panel aliases ONE clip; {args.subset_receipt} names {len(keys)}"
        )
    key = keys[0]
    source = (Path(subset["motion_file"]) / f"{key}.pkl").resolve()
    if not source.is_file():
        raise SystemExit(f"clip missing on disk: {source}")

    panel = args.root / args.name / "robot_filtered"
    panel.mkdir(parents=True, exist_ok=True)
    alias_keys = [f"{key}__rep{i:04d}" for i in range(args.replicates)]
    for alias in alias_keys:
        link = panel / f"{alias}.pkl"
        if link.is_symlink():
            if link.resolve() != source:
                raise SystemExit(f"existing alias points elsewhere: {link}")
        elif link.exists():
            raise SystemExit(f"panel path is not a symlink: {link}")
        else:
            link.symlink_to(source)

    present = {p.stem for p in panel.glob("*.pkl")}
    extra = sorted(present - set(alias_keys))
    if extra:
        raise SystemExit(f"panel holds files outside the alias set: {extra[:3]}")
    resolved = {p.resolve() for p in panel.glob("*.pkl")}
    if resolved != {source}:
        raise SystemExit("aliases do not all resolve to the one source clip")

    receipt = {
        "kind": "lucid_replicate_panel",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "name": args.name,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "why": (
            "SONIC's eval callback slices scored quantities to [:num_unique_motions]. With one "
            "motion that is env 0 alone, and the eval loop's bound is driven by env 0, so every "
            "other env is frozen mid-episode and recorded as not-terminated with progress < 1 -- "
            "censored and mislabelled, biased toward overstating success. K distinct keys "
            "resolving to one clip make num_unique_motions = K, so every env is scored and every "
            "episode completes."
        ),
        "source_subset_receipt": str(args.subset_receipt),
        "motion_key": key,
        "source_clip": str(source),
        "source_clip_sha256": sha256_file(source),
        "replicates": args.replicates,
        "motion_file": str(panel.resolve()),
        "alias_keys_sha256": hashlib.sha256(
            ("\n".join(sorted(alias_keys)) + "\n").encode()
        ).hexdigest(),
        "pool_sha256": subset.get("pool_sha256"),
        "split_sha256": subset.get("split_sha256"),
        "partition": subset.get("config", {}).get("partition"),
        "verified": [
            f"{args.replicates} aliases created, every one a symlink",
            "every alias resolves to the single source clip, checked by resolved-path set equality",
            "the panel directory holds nothing outside the alias set",
        ],
        "not_yet_verified": [
            "that the evaluator actually reports motion_count == replicates and zero censoring; "
            "that is the instrument-validity gate and must be measured, not assumed",
        ],
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    out = args.receipt_dir / f"replicate_panel_{args.name}.json"
    out.write_text(json.dumps(receipt, indent=2))
    print(f"{args.name}: {args.replicates} aliases of {key}")
    print(f"motion_file {panel.resolve()}")
    print(f"receipt {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
