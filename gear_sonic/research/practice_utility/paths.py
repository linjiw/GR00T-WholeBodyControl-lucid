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

__all__ = [
    "ARTIFACTS",
    "DEFAULT_LUCID_ROOT",
    "LUCID_ROOT",
    "MANIFESTS",
    "OUTPUTS",
    "POOLS",
    "TMP",
    "lucid_root",
]
