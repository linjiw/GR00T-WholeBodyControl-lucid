#!/usr/bin/env python3
"""Write an immutable provenance marker for one completed training checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--make-read-only",
        action="store_true",
        help="remove write bits from the checkpoint after validating it",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def matching_arm(receipt: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    matches = [
        arm
        for arm in (receipt.get("arms") or {}).values()
        if int(arm.get("seed", -1)) == seed and arm.get("mode") == mode
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one seed={seed} mode={mode} arm, found {len(matches)}")
    return matches[0]


def count_jsonl_objects(path: Path) -> int:
    rows = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"curriculum row {line_number} is not an object: {path}")
        rows += 1
    return rows


def _git(args: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def build_manifest(
    training_receipt: Path,
    config: Path,
    seed: int,
    mode: str,
    iterations: int,
    *,
    make_read_only: bool,
) -> dict[str, Any]:
    training_receipt = training_receipt.resolve()
    config = config.resolve()
    receipt = load_object(training_receipt)
    arm = matching_arm(receipt, seed, mode)
    verified = receipt.get("verified")
    if not isinstance(verified, list) or not verified:
        raise ValueError("training receipt is not verified")
    if arm.get("complete") is not True or arm.get("checkpoint_exported") is not True:
        raise ValueError("training arm is not complete with an exported checkpoint")
    if int(arm.get("iterations_parsed", -1)) != iterations:
        raise ValueError(f"iterations_parsed is not {iterations}")
    if int(arm.get("curriculum_rows", -1)) != iterations:
        raise ValueError(f"curriculum_rows is not {iterations}")

    checkpoint = Path(arm["checkpoint"]).resolve()
    curriculum = Path(arm["curriculum_path"]).resolve()
    capsule = Path(arm["capsule"]).resolve()
    for label, path in (
        ("checkpoint", checkpoint),
        ("curriculum", curriculum),
        ("final capsule", capsule),
        ("config", config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    curriculum_rows = count_jsonl_objects(curriculum)
    if curriculum_rows != iterations:
        raise ValueError(
            f"curriculum contains {curriculum_rows} JSON-object rows, expected {iterations}"
        )

    if make_read_only:
        checkpoint.chmod(stat.S_IMODE(checkpoint.stat().st_mode) & ~0o222)
    checkpoint_mode = stat.S_IMODE(checkpoint.stat().st_mode)
    if make_read_only and checkpoint_mode & 0o222:
        raise ValueError(f"checkpoint is still writable: {checkpoint}")

    repo = Path(__file__).resolve().parents[2]
    return {
        "kind": "lucid_frozen_training_checkpoint",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "state": "frozen_for_evaluation",
        "seed": seed,
        "mode": mode,
        "iterations": iterations,
        "resume_forbidden": True,
        "evaluation_only": True,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "mode_octal": oct(checkpoint_mode),
            "read_only": not bool(checkpoint_mode & 0o222),
        },
        "config": {
            "path": str(config),
            "sha256": sha256(config),
            "size_bytes": config.stat().st_size,
        },
        "curriculum": {
            "path": str(curriculum),
            "sha256": sha256(curriculum),
            "size_bytes": curriculum.stat().st_size,
            "rows": curriculum_rows,
        },
        "final_capsule": {
            "path": str(capsule),
            "sha256": sha256(capsule),
            "size_bytes": capsule.stat().st_size,
        },
        "training_receipt": {
            "path": str(training_receipt),
            "sha256": sha256(training_receipt),
            "size_bytes": training_receipt.stat().st_size,
        },
        "code": {
            "git_sha": _git(("rev-parse", "HEAD"), repo),
            "git_status_short": _git(("status", "--short"), repo),
        },
        "verified": [
            "training receipt is verified",
            "exactly one requested arm completed the equal iteration budget",
            "checkpoint, resolved config, curriculum, and training receipt are SHA-pinned",
            "checkpoint is reserved for frozen evaluation and may not be resumed",
        ],
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        args.training_receipt,
        args.config,
        args.seed,
        args.mode,
        args.iterations,
        make_read_only=args.make_read_only,
    )
    write_exclusive(args.out, manifest)
    print(f"frozen checkpoint manifest {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
