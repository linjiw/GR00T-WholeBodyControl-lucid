#!/usr/bin/env python3
"""Freeze the outcome-blind LUCID directional-calibration algorithm and folds."""

# ruff: noqa: I001  # repository isort and Ruff force-sort rules conflict

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import (  # noqa: E402
    directional_calibration as DC,
)
from gear_sonic.research.practice_utility.schema import sha256_of  # noqa: E402
from scripts.practice_utility import build_utility_labels as B  # noqa: E402

ARTIFACT_KIND = "practice_utility_latent_directional_calibration_preregistration"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    return output.splitlines()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def clean_git_identity() -> dict[str, Any]:
    status = _git_status()
    if status:
        raise RuntimeError(
            "claim-grade directional-calibration freezing requires a clean committed "
            f"tree; dirty entries: {status}"
        )
    sha = _git_sha()
    if not _is_lower_hex(sha, 40):
        raise RuntimeError(f"git rev-parse returned an invalid commit SHA: {sha!r}")
    return {"sha": sha, "status_short": status}


def load_bound_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read once, validate, and bind both logical and byte-level manifest hashes."""
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"probe manifest is not valid JSON: {resolved}") from error
    if not isinstance(manifest, dict):
        raise ValueError("probe manifest must contain one JSON object")
    B.validate_manifest(manifest)
    return manifest, {
        "path": str(resolved),
        "logical_sha256": manifest["manifest_sha256"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def launcher_binding(path: Path = Path(__file__)) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically while refusing a racing destination.

    A hard link from a fully flushed same-directory staging inode gives both
    properties that an ``exists`` check followed by ``write_text`` cannot:
    readers never see a partial file, and a racing creator wins rather than
    being overwritten.
    """
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
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


def design_rows(manifest: dict) -> list[DC.CalibrationDesignRow]:
    rows = []
    for stage, contexts in sorted(manifest["contexts_per_stage"].items()):
        for seed_value in manifest["seeds"]:
            seed = int(seed_value)
            for entry in contexts:
                context_id = str(entry["context_id"])
                family = entry.get("family")
                if not isinstance(family, str) or not family:
                    raise ValueError(f"manifest context {context_id!r} has no motion family")
                rows.append(
                    DC.CalibrationDesignRow(
                        sample_id=f"{stage}|{seed}|{context_id}",
                        context_id=context_id,
                        motion_family=family,
                    )
                )
    return rows


def build_artifact(
    manifest: dict,
    *,
    manifest_binding: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    launcher: Mapping[str, Any],
) -> dict:
    """Build a deterministic, outcome-free preregistration artifact."""
    if git_identity.get("status_short") != [] or not _is_lower_hex(git_identity.get("sha"), 40):
        raise ValueError("artifact provenance requires a clean 40-character Git identity")
    if manifest_binding.get("logical_sha256") != manifest["manifest_sha256"]:
        raise ValueError("manifest provenance logical hash differs from the validated manifest")
    for label, binding, hash_field in (
        ("manifest", manifest_binding, "file_sha256"),
        ("launcher", launcher, "sha256"),
    ):
        if not isinstance(binding.get("path"), str) or not _is_lower_hex(
            binding.get(hash_field), 64
        ):
            raise ValueError(f"{label} provenance requires an exact path and SHA-256")
    algorithm = DC.default_algorithm_artifact()
    support = DC.validate_design_support(design_rows(manifest), algorithm)
    payload = {
        "kind": ARTIFACT_KIND,
        "schema_version": 1,
        "frozen_before_outcomes": True,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_binding["file_sha256"],
        "source_manifest": dict(manifest_binding),
        "git": dict(git_identity),
        "launcher": dict(launcher),
        "algorithm": algorithm,
        "design_support": support,
        "contains_outcomes": False,
    }
    payload["artifact_sha256"] = sha256_of(payload)
    return payload


def main(argv=None) -> int:
    args = parse_args(argv)
    git_identity = clean_git_identity()
    manifest, manifest_binding = load_bound_manifest(args.manifest)
    launcher = launcher_binding()
    if clean_git_identity() != git_identity:
        raise RuntimeError("Git identity changed while freezing directional calibration")
    payload = build_artifact(
        manifest,
        manifest_binding=manifest_binding,
        git_identity=git_identity,
        launcher=launcher,
    )
    write_json_exclusive_atomic(args.output, payload)
    print(
        f"directional calibration design: {payload['design_support']['status']} "
        f"[{payload['algorithm']['algorithm_sha256'][:16]}]"
    )
    print(f"wrote immutable preregistration {args.output}")
    return 0 if payload["design_support"]["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
