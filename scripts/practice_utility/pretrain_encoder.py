#!/usr/bin/env python3
"""Pre-train and freeze the LUCID temporal VAE used to measure the latent gap.

The encoder is a **measuring instrument**, so three properties matter more than
its reconstruction score:

*Trained only on the adaptation split.* Reference motion from the dev or test
partitions must never reach it. The latent gap is a feature used to predict
utility and to compare methods on held-out motions; an encoder that had seen
those motions would make held-out numbers partly in-sample.

*Resampled to the control rate.* Clips are stored at 30 fps but the policy runs
at 50 Hz, and the gap is computed on 50 Hz command/execution windows. Training
on 30 fps windows would give the encoder different temporal statistics than it
meets at measurement time.

*Frozen and fingerprinted.* An instrument that drifts during the experiment
measures nothing. The saved artifact carries the weight hash, the pool and split
hashes, and the window spec, so every gap measurement can name the instrument
that produced it.

Example
-------
    source $LUCID_ROOT/lucid_env.sh
    python scripts/practice_utility/pretrain_encoder.py \\
        --pool-manifest  $LUCID_ROOT/manifests/pool_debug512.json \\
        --split-manifest $LUCID_ROOT/manifests/split_debug512_performer.json \\
        --output $LUCID_ROOT/artifacts/lucid_encoder_debug512.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.research.practice_utility import latent_gap_probe as L  # noqa: E402
from gear_sonic.research.practice_utility.paths import relocate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-partition", default="adaptation",
                        help="partition the encoder may see (default: adaptation)")
    parser.add_argument("--control-hz", type=float, default=50.0,
                        help="resample reference clips to this rate")
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=2,
                        help="keep every Nth window when building the corpus")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--holdout-fraction", type=float, default=0.1,
                        help="windows withheld from training to report generalization")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resample(sequence: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Linearly resample a ``(T, J)`` trajectory to ``target_fps``."""
    if abs(source_fps - target_fps) < 1e-6:
        return sequence
    frames = sequence.shape[0]
    duration = frames / source_fps
    target_frames = max(2, int(round(duration * target_fps)))
    source_t = np.linspace(0.0, duration, frames)
    target_t = np.linspace(0.0, duration, target_frames)
    return np.stack(
        [np.interp(target_t, source_t, sequence[:, j]) for j in range(sequence.shape[1])],
        axis=1,
    )


def build_corpus(args, motions, allowed) -> tuple[torch.Tensor, dict]:
    """Windows from the permitted partition only, at the control rate."""
    import joblib

    spec = L.WindowSpec(length=args.window_length, stride=args.window_stride)
    windows, used, skipped, dofs = [], 0, 0, None
    for record in motions:
        if record["motion_key"] not in allowed:
            continue
        clip = joblib.load(relocate(record["path"]))[record["motion_key"]]
        dof = np.asarray(clip["dof"], dtype=np.float32)
        dof = resample(dof, float(clip.get("fps", 30)), args.control_hz).astype(np.float32)
        dofs = dofs or dof.shape[1]
        if dof.shape[1] != dofs:
            raise SystemExit(f"inconsistent dof width: {dof.shape[1]} vs {dofs}")
        clip_windows = L.build_windows(torch.from_numpy(dof), spec)
        if clip_windows.shape[0] == 0:
            skipped += 1
            continue
        windows.append(clip_windows[:: args.sample_stride])
        used += 1

    if not windows:
        raise SystemExit("no training windows; check the split and window length")
    corpus = torch.cat(windows, dim=0)
    return corpus, {"clips_used": used, "clips_too_short": skipped, "num_dofs": dofs}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.output} exists. The encoder is a frozen instrument; replacing it "
            "invalidates every latent-gap measurement taken with it. Pass --overwrite "
            "only if you intend that."
        )

    pool = json.loads(args.pool_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    assignment = split["assignment"]

    allowed = {k for k, p in assignment.items() if p == args.train_partition}
    withheld = sorted({p for p in assignment.values()} - {args.train_partition})
    print(f"training partition {args.train_partition!r}: {len(allowed)} clips")
    print(f"withheld from the encoder entirely: {withheld}")

    corpus, stats = build_corpus(args, pool["motions"], allowed)
    print(f"corpus: {tuple(corpus.shape)} windows from {stats['clips_used']} clips "
          f"({stats['clips_too_short']} too short), {stats['num_dofs']} dofs")

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(corpus.shape[0], generator=generator)
    holdout_size = int(args.holdout_fraction * corpus.shape[0])
    holdout = corpus[order[:holdout_size]]
    train = corpus[order[holdout_size:]]
    print(f"train {train.shape[0]} / holdout {holdout.shape[0]} windows")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    started = time.time()
    model, history = L.train_encoder(
        train,
        num_joints=stats["num_dofs"],
        spec=L.WindowSpec(args.window_length, args.window_stride),
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        device=device,
        log_every=max(1, args.epochs // 10),
    )
    elapsed = time.time() - started

    with torch.no_grad():
        batch = holdout[: min(4096, holdout.shape[0])].to(device)
        reconstruction, mu, logvar = model(batch)
        holdout_recon = float((reconstruction - batch).pow(2).sum(dim=(1, 2)).mean())
        noise_recon = float(
            (model(torch.randn_like(batch))[0] - torch.randn_like(batch))
            .pow(2).sum(dim=(1, 2)).mean()
        )

    fingerprint = L.encoder_fingerprint(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "lucid_temporal_vae_encoder",
        "schema_version": 1,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "encoder_fingerprint": fingerprint,
        "num_joints": stats["num_dofs"],
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "latent_dim": args.latent_dim,
        "hidden_channels": args.hidden_channels,
        "control_hz": args.control_hz,
        "train_partition": args.train_partition,
        "withheld_partitions": withheld,
        "pool_sha256": pool["pool_sha256"],
        "split_sha256": split["split_sha256"],
        "split_linkage": split["linkage"],
        "num_train_windows": int(train.shape[0]),
        "num_holdout_windows": int(holdout.shape[0]),
        "clips_used": stats["clips_used"],
        "seed": args.seed,
        "epochs": args.epochs,
        "train_seconds": round(elapsed, 1),
        "final_train_recon": history[-1]["recon"],
        "first_train_recon": history[0]["recon"],
        "holdout_recon": holdout_recon,
        "noise_recon_control": noise_recon,
        "history": history,
    }
    staging = args.output.with_suffix(".partial")
    torch.save(payload, staging)
    staging.replace(args.output)

    receipt = args.output.with_suffix(".json")
    receipt.write_text(json.dumps(
        {k: v for k, v in payload.items() if k not in ("state_dict", "history")}, indent=2
    ))

    print(f"\nrecon  {history[0]['recon']:.4f} -> {history[-1]['recon']:.4f} "
          f"(holdout {holdout_recon:.4f}, noise control {noise_recon:.4f})")
    if holdout_recon >= noise_recon:
        print("  WARNING: the encoder reconstructs held-out motion no better than noise; "
              "it has not learned motion structure and must not be used as an instrument")
    print(f"fingerprint {fingerprint[:16]}  trained in {elapsed:.1f}s")
    print(f"wrote {args.output}\n      {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
