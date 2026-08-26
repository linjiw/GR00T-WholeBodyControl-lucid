#!/usr/bin/env python3
"""Freeze an outcome-blind v2 passive-dose projection plan.

The source is only a frozen probe manifest. No branch output, utility label,
or evaluation result is accepted by this CLI. By default it performs a dry run
and prints the hashes that would be frozen. ``--execute`` atomically writes the
plan and a separate creation receipt.

Example::

    python scripts/practice_utility/create_passive_dose_plan.py \
      --manifest /data/.../probe_screen_v2_late_20260826.json \
      --output /data/.../probe_screen_v2_late_20260826.passive_dose_plan_v2.json \
      --sigma-frames 50 --execute
"""

# Ruff's import order conflicts with the repository's isort profile; E402 is
# intentional for the repository-root bootstrap below.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import dose_plan as DP


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="creation receipt (default: OUTPUT with .creation_receipt.json suffix)",
    )
    parser.add_argument(
        "--sigma-frames",
        required=True,
        type=float,
        help=(
            "Gaussian-kernel sigma; must equal the reference bin size derived from "
            "the manifest ContextKeys"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="atomically write the plan and creation receipt (default is dry-run)",
    )
    return parser.parse_args(argv)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _publish_exclusive_atomic(path: Path, content: bytes) -> None:
    """Publish complete bytes without overwriting an existing or racing file."""

    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    return output.splitlines()


def _git_identity() -> dict[str, Any]:
    sha = _git_sha()
    status = _git_status()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise RuntimeError(f"git rev-parse returned an invalid commit SHA: {sha!r}")
    return {"sha": sha, "status_short": status}


def build(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    git_identity = _git_identity()
    if args.execute and git_identity["status_short"]:
        raise RuntimeError(
            "claim --execute requires a clean committed Git tree; dirty entries: "
            f"{git_identity['status_short']}"
        )
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None
        else output_path.with_suffix(".creation_receipt.json")
    )
    if len({manifest_path, output_path, receipt_path}) != 3:
        raise ValueError("manifest, output, and receipt paths must be distinct")
    if not manifest_path.is_file():
        raise ValueError(f"manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("manifest must contain a JSON object")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest_file_sha256 = DP.file_sha256(manifest_path)
    launcher_path = Path(__file__).resolve()
    launcher_sha256 = DP.file_sha256(launcher_path)
    plan = DP.build_passive_dose_plan_payload(
        manifest,
        manifest_file_sha256=manifest_file_sha256,
        sigma_frames=args.sigma_frames,
        created_at=created_at,
        source_manifest_path=str(manifest_path),
        source_commit=git_identity["sha"],
        git_status_short=git_identity["status_short"],
        launcher_path=str(launcher_path),
        launcher_sha256=launcher_sha256,
    )
    plan_bytes = _json_bytes(plan)
    plan_file_sha256 = _bytes_sha256(plan_bytes)
    receipt = {
        "kind": "practice_utility_passive_dose_plan_creation_receipt",
        "schema_version": 1,
        "status": "complete" if not git_identity["status_short"] else "blocked",
        "created_at": created_at,
        "execution_mode": "write" if args.execute else "dry_run",
        "outcome_blind": True,
        "accepted_input_kinds": ["practice_utility_probe_manifest"],
        "result_artifacts_read": [],
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha256,
            "logical_sha256": plan["manifest_sha256"],
        },
        "dose_plan": {
            "path": str(output_path),
            "file_sha256": plan_file_sha256,
            "logical_sha256": plan["dose_plan_sha256"],
            "schema_version": plan["schema_version"],
            "num_contexts": sum(len(rows) for rows in plan["contexts_per_stage"].values()),
            "reference_bin_size_frames": plan["kernel"]["reference_bin_size_frames"],
            "sigma_frames": plan["kernel"]["sigma_frames"],
        },
        "implementation": {
            "source_commit": git_identity["sha"],
            "source_tree_clean": not git_identity["status_short"],
            "source_tree_status": git_identity["status_short"],
            "launcher_path": str(launcher_path),
            "launcher_sha256": launcher_sha256,
        },
        "warnings": ([] if not git_identity["status_short"] else ["source_tree_dirty_at_dry_run"]),
    }

    if args.execute:
        if _git_identity() != git_identity:
            raise RuntimeError("Git identity changed while freezing the passive-dose plan")
        if DP.file_sha256(launcher_path) != launcher_sha256:
            raise RuntimeError("launcher bytes changed while freezing the passive-dose plan")
        _publish_exclusive_atomic(output_path, plan_bytes)
        _publish_exclusive_atomic(receipt_path, _json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    try:
        result = build(argv)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
