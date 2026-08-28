"""Filesystem roots for the LUCID practice-utility program.

The data root holds everything that is deliberately *not* in Git: ``manifests/``
(JSON receipts), ``artifacts/`` (encoders, capsules, curriculum logs),
``outputs/`` (run logs), ``pools/`` (motion pools), and ``tmp/`` (``TMPDIR``;
``/tmp/isaaclab`` may be owned by another user).

It defaults to the original host's absolute path so that existing receipts,
drivers and documented commands keep resolving byte-identically there. Set
``LUCID_ROOT`` to relocate it on a host that cannot provide ``/data`` --
``env/lucid_env.sh`` exports it. Resolution happens at import time, so the
environment must be set before Python starts; the env script guarantees that.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Data root on the host the program was developed on.
DEFAULT_LUCID_ROOT = Path("/data/robotixx/lucid-sonic")


def lucid_root() -> Path:
    """Return the data root, honouring the ``LUCID_ROOT`` environment variable."""
    value = os.environ.get("LUCID_ROOT")
    return Path(value) if value else DEFAULT_LUCID_ROOT


LUCID_ROOT = lucid_root()
MANIFESTS = LUCID_ROOT / "manifests"
ARTIFACTS = LUCID_ROOT / "artifacts"
OUTPUTS = LUCID_ROOT / "outputs"
POOLS = LUCID_ROOT / "pools"
TMP = LUCID_ROOT / "tmp"


def relocate(recorded: str | Path) -> Path:
    """Re-root a path recorded on another host onto this host's data root.

    Frozen manifests store absolute paths under the data root of the host that
    wrote them. The path itself is not part of any content hash -- clip hashes
    cover the trajectory bytes -- so re-rooting is safe and changes no identity.

    Returns the recorded path unchanged when it exists, or when it does not sit
    under a known data root; only a path that is both missing and re-rootable is
    rewritten, and only if the rewrite actually resolves. That keeps a genuinely
    missing file reported as missing at its original location.
    """
    path = Path(recorded)
    if path.exists() or LUCID_ROOT == DEFAULT_LUCID_ROOT:
        return path
    try:
        moved = LUCID_ROOT / path.relative_to(DEFAULT_LUCID_ROOT)
    except ValueError:
        return path
    return moved if moved.exists() else path


__all__ = [
    "ARTIFACTS",
    "DEFAULT_LUCID_ROOT",
    "LUCID_ROOT",
    "MANIFESTS",
    "OUTPUTS",
    "POOLS",
    "TMP",
    "lucid_root",
    "relocate",
]
