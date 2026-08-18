#!/usr/bin/env python3
"""Build a frozen probe campaign from a live sampler snapshot.

Contexts cannot be invented from the stored motion files: SONIC decides bin
boundaries at load time from each clip's resampled frame count, and which bins
are *resident* depends on the current motion batch. So a campaign is designed
from a snapshot taken inside a real run (written by ``PracticeContextCallback``
with ``snapshot_path`` set), one snapshot per policy stage.

Each resident context is enriched with offline motion-structure features, which
supply the contact-regime stratum and the descriptors the audit later uses to
ask whether two contexts of equal difficulty differ for structural reasons.

Example
-------
    python scripts/practice_utility/create_probe_manifest.py \\
        --snapshot early=.../snapshot_early.json \\
        --snapshot late=.../snapshot_late.json \\
        --pool-manifest  .../pool_debug512.json \\
        --split-manifest .../split_debug512_performer.json \\
        --contexts-per-stage 24 --seeds 0 1 \\
        --output .../probe_oracle_screen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.research.practice_utility import probe_manifest as PM  # noqa: E402
from gear_sonic.research.practice_utility import proxy_features as PF  # noqa: E402
from gear_sonic.research.practice_utility.schema import ContextKey  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", action="append", required=True, metavar="STAGE=PATH",
                        help="sampler snapshot for one policy stage; repeatable")
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--campaign-id", default="oracle_screen_v1")
    parser.add_argument("--contexts-per-stage", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--kernel-radius", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs=3, default=[8, 32, 128],
                        metavar=("H_S", "H_M", "H_L"))
    parser.add_argument("--selection-seed", type=int, default=20260818)
    parser.add_argument("--train-partition", default="adaptation")
    parser.add_argument("--skip-motion-features", action="store_true",
                        help="skip offline feature extraction (faster, coarser strata)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_clip_index(pool: dict) -> dict[str, dict]:
    return {m["motion_key"]: m for m in pool["motions"]}


def build_candidates(snapshot, clips, assignment, partition, with_features):
    """Turn resident contexts into stratifiable candidates."""
    import joblib

    cache: dict[str, dict] = {}
    candidates, skipped_partition, skipped_missing = [], 0, 0

    for entry in snapshot["contexts"]:
        motion_key = entry["motion_key"]
        if assignment.get(motion_key) != partition:
            skipped_partition += 1
            continue
        record = clips.get(motion_key)
        if record is None:
            skipped_missing += 1
            continue

        regime, extras = "unknown", {}
        if with_features:
            if motion_key not in cache:
                cache[motion_key] = joblib.load(record["path"])[motion_key]
            clip = cache[motion_key]
            frames = clip["dof"].shape[0]
            # Snapshot bins index SONIC's resampled timeline; clamp into the
            # stored clip so features describe a real slice of this motion.
            start = min(entry["bin_start_frame"], max(0, frames - 2))
            end = min(max(entry["bin_end_frame"], start + 2), frames)
            try:
                features = PF.features_for_bin(clip, start, end)
                regime = PF.contact_regime_proxy(features)
                extras = features.as_proxy_features()
            except ValueError:
                regime = "unknown"

        candidates.append(
            PM.ContextCandidate(
                context=ContextKey.from_dict(entry),
                failure_rate=float(entry.get("failure_rate", 0.0)),
                sampling_probability=float(entry.get("sampling_probability", 0.0)),
                family=record.get("family", "other"),
                contact_regime=regime,
                partition=partition,
                extras=extras,
            )
        )
    return candidates, skipped_partition, skipped_missing


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.output} exists. A probe manifest is frozen before the campaign; "
            "replacing it after seeing results would turn a measurement into a search."
        )

    pool = json.loads(args.pool_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    clips = load_clip_index(pool)
    assignment = split["assignment"]

    candidates_per_stage = {}
    for item in args.snapshot:
        if "=" not in item:
            raise SystemExit(f"--snapshot expects STAGE=PATH, got {item!r}")
        stage, path = item.split("=", 1)
        snapshot = json.loads(Path(path).read_text())
        candidates, skipped_partition, skipped_missing = build_candidates(
            snapshot, clips, assignment, args.train_partition, not args.skip_motion_features
        )
        print(f"stage {stage!r}: {len(snapshot['contexts'])} resident contexts -> "
              f"{len(candidates)} candidates "
              f"({skipped_partition} outside {args.train_partition}, "
              f"{skipped_missing} not in pool)")
        if not candidates:
            raise SystemExit(
                f"stage {stage!r} has no candidate contexts in the "
                f"{args.train_partition!r} partition; check the snapshot and split"
            )
        regimes: dict[str, int] = {}
        for candidate in candidates:
            regimes[candidate.contact_regime] = regimes.get(candidate.contact_regime, 0) + 1
        print(f"           contact regimes: {dict(sorted(regimes.items()))}")
        candidates_per_stage[stage] = candidates

    horizons = dict(zip(("H_s", "H_m", "H_l"), args.horizons))
    manifest = PM.build_probe_manifest(
        campaign_id=args.campaign_id,
        candidates_per_stage=candidates_per_stage,
        num_contexts=args.contexts_per_stage,
        seeds=args.seeds,
        horizons=horizons,
        pool_sha256=pool["pool_sha256"],
        split_sha256=split["split_sha256"],
        epsilon=args.epsilon,
        kernel_radius_bins=args.kernel_radius,
        selection_seed=args.selection_seed,
        notes=f"built from snapshots: {', '.join(args.snapshot)}",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(".json.partial")
    staging.write_text(json.dumps(manifest.to_dict(), indent=2))
    staging.replace(args.output)

    print(f"\ncampaign {manifest.campaign_id} [{manifest.manifest_sha256[:16]}]")
    print(f"  {manifest.num_branches} intervention branches + "
          f"{manifest.num_control_branches} shared controls")
    for stage, coverage in manifest.coverage().items():
        print(f"  {stage:8s} quartiles={coverage['failure_quartiles']} "
              f"families={len(coverage['families'])} regimes={coverage['contact_regimes']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
