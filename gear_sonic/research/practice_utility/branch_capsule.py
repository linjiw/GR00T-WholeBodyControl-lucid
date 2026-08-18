"""Save and restore everything a paired continuation needs.

SONIC's ``ModelSaveCallback`` persists the policy, value model, optimizer, and
``env.get_env_state_dict()``. For ordinary training that is enough. For a paired
branch-and-continue experiment it is not, because two things are missing:

1. **Random state.** Without it, resuming starts a fresh stream, so a control
   and an intervention branch launched from "the same" checkpoint do not begin
   from the same randomness.
2. **A binding between the sampler counters and the motion pool.**
   ``adp_samp_num_episodes`` / ``adp_samp_num_failures`` are indexed by global
   bin id, which is only meaningful relative to a specific motion pool. Restored
   against a different pool they are silently wrong.

A capsule therefore stores model state, optimizer state, trainer state, env
state, the native sampler counters, the full RNG state, and the hashes that pin
config, motion pool, evaluation suite, source commit, and source checkpoint.

Restore order matters and is enforced by :func:`load_capsule`: **RNG first**,
then env/sampler, then model and optimizer. Any randomness consumed while
rebuilding modules is then drawn from the same stream position as in the
original run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gear_sonic.research.practice_utility.rng_capsule import RngState
from gear_sonic.research.practice_utility.schema import SCHEMA_VERSION, sha256_of

#: Keys every capsule payload must carry.
REQUIRED_KEYS = (
    "schema_version",
    "branch_id",
    "pair_id",
    "role",
    "global_step",
    "model_state",
    "optimizer_state",
    "trainer_state",
    "env_state",
    "native_sampler_state",
    "rng",
    "provenance",
)

#: Sampler counters that must travel with the capsule.
SAMPLER_KEYS = ("adp_samp_num_episodes", "adp_samp_num_failures")


class CapsuleIntegrityError(RuntimeError):
    """Raised when a capsule is malformed or resumed against a mismatched setup."""


@dataclass
class Provenance:
    """Hashes that make a branch traceable back to its inputs."""

    resolved_config_sha256: str
    motion_pool_manifest_sha256: str
    dev_suite_sha256: str
    source_commit: str
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "resolved_config_sha256": self.resolved_config_sha256,
            "motion_pool_manifest_sha256": self.motion_pool_manifest_sha256,
            "dev_suite_sha256": self.dev_suite_sha256,
            "source_commit": self.source_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @staticmethod
    def from_dict(payload: dict[str, str]) -> Provenance:
        return Provenance(**{k: payload[k] for k in Provenance.__dataclass_fields__})

    def mismatches(self, other: Provenance, ignore: tuple[str, ...] = ()) -> list[str]:
        """Fields on which two provenances disagree."""
        return [
            field
            for field in Provenance.__dataclass_fields__
            if field not in ignore and getattr(self, field) != getattr(other, field)
        ]


def save_capsule(
    path: str | os.PathLike[str],
    *,
    branch_id: str,
    pair_id: str,
    role: str,
    global_step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    trainer_state: dict[str, Any],
    env_state: dict[str, Any],
    native_sampler_state: dict[str, Any],
    rng_state: RngState,
    provenance: Provenance,
) -> str:
    """Write a branch capsule and return its content hash.

    The returned hash covers identity and provenance, not the weight tensors, so
    it is cheap to compute and stable across equivalent serializations.
    """
    if role not in ("control", "intervention"):
        raise ValueError(f"unknown branch role: {role!r}")
    missing = [k for k in SAMPLER_KEYS if k not in native_sampler_state]
    if missing:
        raise CapsuleIntegrityError(
            f"native_sampler_state is missing {missing}; a capsule without the "
            "adaptive-sampling counters cannot reproduce the curriculum"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "branch_id": branch_id,
        "pair_id": pair_id,
        "role": role,
        "global_step": int(global_step),
        "model_state": _to_cpu(model_state),
        "optimizer_state": _to_cpu(optimizer_state),
        "trainer_state": _to_cpu(trainer_state),
        "env_state": _to_cpu(env_state),
        "native_sampler_state": _to_cpu(native_sampler_state),
        "rng": rng_state.to_payload(),
        "provenance": provenance.to_dict(),
    }
    payload["capsule_sha256"] = _capsule_hash(payload)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted save never leaves a half-written
    # capsule that a later resume would silently trust.
    staging = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, staging)
    staging.replace(path)
    return payload["capsule_sha256"]


def load_capsule(
    path: str | os.PathLike[str],
    *,
    expected_provenance: Provenance | None = None,
    ignore_provenance_fields: tuple[str, ...] = (),
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a capsule, verify it, and (by default) restore the RNG first.

    Args:
        expected_provenance: if given, the capsule must agree with it. This is
            what stops a branch from being continued against a different motion
            pool or config, which would invalidate the paired comparison.
        ignore_provenance_fields: fields allowed to differ. ``checkpoint_sha256``
            legitimately differs once branches have trained past their shared
            origin.
        restore_rng: restore the captured random streams immediately. Leave on
            unless the caller intends to restore them itself, in order.
    """
    # map_location keeps an archive readable on a busy or GPU-less machine.
    payload = torch.load(Path(path), weights_only=False, map_location="cpu")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise CapsuleIntegrityError(f"capsule at {path} is missing keys: {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CapsuleIntegrityError(
            f"capsule schema {payload['schema_version']} != expected {SCHEMA_VERSION}"
        )
    recorded = payload.get("capsule_sha256")
    if recorded is not None and recorded != _capsule_hash(payload):
        raise CapsuleIntegrityError(
            f"capsule at {path} failed its integrity check; it was modified after saving"
        )

    provenance = Provenance.from_dict(payload["provenance"])
    if expected_provenance is not None:
        bad = provenance.mismatches(expected_provenance, ignore=ignore_provenance_fields)
        if bad:
            raise CapsuleIntegrityError(
                f"capsule provenance disagrees with the run on {bad}; continuing "
                "would make the paired comparison invalid"
            )

    rng_state = RngState.from_payload(payload["rng"])
    if restore_rng:
        rng_state.restore()

    payload["rng_state"] = rng_state
    payload["provenance_obj"] = provenance
    return payload


def fork_pair(
    capsule_path: str | os.PathLike[str],
    pair_id: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, str]:
    """Write control and intervention capsules from one source checkpoint.

    Both branches are byte-identical at fork time apart from ``branch_id`` and
    ``role``. That is the whole point: any later difference is attributable to
    the intervention rather than to initialization.
    """
    payload = load_capsule(capsule_path, restore_rng=False)
    output_dir = Path(output_dir)
    written: dict[str, str] = {}
    for role in ("control", "intervention"):
        branch_id = f"{pair_id}_{role}"
        forked = dict(payload)
        for key in ("rng_state", "provenance_obj", "capsule_sha256"):
            forked.pop(key, None)
        forked["branch_id"] = branch_id
        forked["pair_id"] = pair_id
        forked["role"] = role
        forked["capsule_sha256"] = _capsule_hash(forked)

        path = output_dir / f"{branch_id}.capsule.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".partial")
        torch.save(forked, staging)
        staging.replace(path)
        written[role] = str(path)
    return written


def assert_fork_identical(control_path: str, intervention_path: str) -> None:
    """Verify two forked capsules differ only in identity fields.

    Run this before every paired campaign. A silent difference in model or
    sampler state at fork time would masquerade as a treatment effect.
    """
    control = torch.load(Path(control_path), weights_only=False, map_location="cpu")
    treated = torch.load(Path(intervention_path), weights_only=False, map_location="cpu")

    if control["pair_id"] != treated["pair_id"]:
        raise CapsuleIntegrityError("capsules belong to different pairs")
    if {control["role"], treated["role"]} != {"control", "intervention"}:
        raise CapsuleIntegrityError("expected exactly one control and one intervention capsule")
    if control["global_step"] != treated["global_step"]:
        raise CapsuleIntegrityError("forked capsules start from different global steps")

    for key in ("model_state", "optimizer_state", "native_sampler_state", "env_state"):
        if not _states_equal(control[key], treated[key]):
            raise CapsuleIntegrityError(f"forked capsules differ in {key}")
    if control["rng"]["torch_cpu_state"].ne(treated["rng"]["torch_cpu_state"]).any():
        raise CapsuleIntegrityError("forked capsules differ in torch CPU RNG state")
    if control["provenance"] != treated["provenance"]:
        raise CapsuleIntegrityError("forked capsules differ in provenance")


def export_sonic_checkpoint(
    capsule_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> str:
    """Write a capsule out as a checkpoint SONIC's own loader accepts.

    This is what makes *stage-conditioned* probing possible: a branch can start
    from a mid-adaptation checkpoint rather than always from the released model.
    Without it every branch shares one policy stage, and the programme cannot ask
    whether practice utility depends on how far training has progressed -- which
    is one of its three headline questions.

    Raises if the capsule predates the split policy/value layout, rather than
    silently writing a checkpoint that would load the wrong weights.
    """
    payload = torch.load(Path(capsule_path), weights_only=False, map_location="cpu")
    model_state = payload.get("model_state", {})
    if not isinstance(model_state, dict) or "policy_state_dict" not in model_state:
        raise CapsuleIntegrityError(
            f"{capsule_path} stores a combined model state and cannot be split into "
            "policy and value dicts. Re-save with a current PracticeCapsuleCallback."
        )
    policy = model_state.get("policy_state_dict") or {}
    if not policy:
        raise CapsuleIntegrityError(
            f"{capsule_path} has an empty policy_state_dict; it cannot seed a branch"
        )

    checkpoint = {
        "policy_state_dict": policy,
        "value_state_dict": model_state.get("value_state_dict") or None,
        "optimizer_state_dict": payload.get("optimizer_state") or None,
        "lr_scheduler_state_dict": model_state.get("lr_scheduler_state_dict") or None,
        "state": payload.get("trainer_state", {}).get("trainer_state_obj"),
        "env_state_dict": payload.get("env_state") or {},
        # Provenance travels with the checkpoint so a branch can be traced back
        # to the capsule and campaign it came from.
        "practice_utility": {
            "source_capsule": str(capsule_path),
            "capsule_sha256": payload.get("capsule_sha256"),
            "branch_id": payload.get("branch_id"),
            "pair_id": payload.get("pair_id"),
            "global_step": payload.get("global_step"),
            "provenance": payload.get("provenance"),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".partial")
    torch.save(checkpoint, staging)
    staging.replace(output)
    return str(output)


def _capsule_hash(payload: dict[str, Any]) -> str:
    return sha256_of(
        {
            "schema_version": payload["schema_version"],
            "branch_id": payload["branch_id"],
            "pair_id": payload["pair_id"],
            "role": payload["role"],
            "global_step": payload["global_step"],
            "provenance": payload["provenance"],
        }
    )


def _to_cpu(value: Any) -> Any:
    """Recursively move every tensor to CPU.

    A capsule holding CUDA tensors can only be opened on a machine with a free
    GPU -- loading one during a training run fails with an out-of-memory error,
    which is exactly when a campaign wants to inspect or export it. Capsules are
    archives and must be readable anywhere.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _states_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and bool(torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)
