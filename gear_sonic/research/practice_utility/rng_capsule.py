"""Random-state capture and counter-based stream primitives for paired branches.

Why a plain seed is not enough
------------------------------
Setting ``seed=0`` on both branches does not make them comparable. The moment
the intervention branch draws a different motion bin, the two runs consume the
global RNG differently, and from then on they diverge in push timing, sampled
physics parameters, action noise, and minibatch order. The measured difference
would then contain both the intervention effect and unmatched trajectory noise.

Counter-based streams can fix this. Every non-context random channel derives its
seed from a *content key* rather than from stream position::

    seed = H(pair_id, env_id, episode_index, channel_name)

When every production random channel is wired through these primitives, control
and intervention draw the *same* friction, push time, and action noise for the
same (env, episode), while only the context selector differs. The helpers in
this module do not themselves install that wiring. Until a live channel audit
proves those call sites, capsules must record ``counter_rng_enabled=False``.

What this module does not claim
-------------------------------
GPU physics is not bitwise reproducible: kernel scheduling and atomics make
exact trajectory equality unattainable. So this is a *receipt*, not a proof.
The empirical noise floor comes from the epsilon=0 paired branches, and no
report may describe "same seed" as "identical trajectory".
"""

# Ruff's force-sort-within-sections setting conflicts with the repository's
# authoritative isort profile for mixed import/from-import blocks.
# ruff: noqa: I001

from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gear_sonic.research.practice_utility.schema import RngReceipt, sha256_of

#: Random channels that must be matched across a paired branch.
MATCHED_CHANNELS = (
    "friction",
    "push_time",
    "push_impulse",
    "base_com",
    "body_mass",
    "joint_default_pos",
    "action_noise",
    "observation_noise",
    "minibatch_shuffle",
    "episode_init",
)

#: The one channel that is *meant* to differ between branches.
TREATMENT_CHANNEL = "context_selection"

_UINT32 = 1 << 32


def derive_seed(pair_id: str, env_id: int, episode_index: int, channel: str) -> int:
    """Derive a reproducible 32-bit seed from a content key.

    Deterministic across processes, machines, and restarts: it depends only on
    the key, never on how much randomness has already been consumed.
    """
    if channel not in MATCHED_CHANNELS and channel != TREATMENT_CHANNEL:
        raise ValueError(
            f"unknown random channel {channel!r}; declare it in MATCHED_CHANNELS "
            "so paired branches stay matched"
        )
    key = f"{pair_id}|{int(env_id)}|{int(episode_index)}|{channel}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % _UINT32


def channel_generator(
    pair_id: str, env_id: int, episode_index: int, channel: str, device: str = "cpu"
) -> torch.Generator:
    """A torch generator seeded from the content key for one channel."""
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_seed(pair_id, env_id, episode_index, channel))
    return generator


@dataclass
class RngState:
    """A complete snapshot of every random stream we can capture."""

    python_state: Any
    numpy_state: Any
    torch_cpu_state: torch.Tensor
    torch_cuda_states: list[torch.Tensor]
    counter_rng_enabled: bool
    pair_id: str
    deterministic_flags: dict[str, Any]

    @staticmethod
    def capture(pair_id: str, counter_rng_enabled: bool = False) -> RngState:
        """Capture the current global random state of the process.

        ``counter_rng_enabled`` is evidence about production integration, not
        about whether :func:`derive_seed` exists. It therefore defaults to
        false and may only be set true by a caller that has audited every
        declared matched channel end to end.
        """
        cuda_states: list[torch.Tensor] = []
        if torch.cuda.is_available():
            cuda_states = [s.clone() for s in torch.cuda.get_rng_state_all()]
        return RngState(
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            torch_cpu_state=torch.get_rng_state().clone(),
            torch_cuda_states=cuda_states,
            counter_rng_enabled=counter_rng_enabled,
            pair_id=pair_id,
            deterministic_flags=current_determinism_flags(),
        )

    def restore(self) -> None:
        """Restore every captured stream.

        Must run *before* model, optimizer, and sampler state are restored, so
        that any randomness consumed during their construction is drawn from the
        same position as in the original run.
        """
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_states and torch.cuda.is_available():
            available = torch.cuda.device_count()
            if len(self.torch_cuda_states) != available:
                raise RuntimeError(
                    f"capsule holds {len(self.torch_cuda_states)} CUDA RNG states but "
                    f"{available} devices are visible; resume on matching hardware "
                    "or the paired comparison is invalid"
                )
            torch.cuda.set_rng_state_all(self.torch_cuda_states)

    def receipt(self) -> RngReceipt:
        """Hashable evidence of this state, safe to store in a manifest."""
        return RngReceipt(
            python_state_sha256=_hash_state(self.python_state),
            numpy_state_sha256=_hash_state(self.numpy_state),
            torch_cpu_state_sha256=_hash_tensor(self.torch_cpu_state),
            torch_cuda_state_sha256=[_hash_tensor(s) for s in self.torch_cuda_states],
            context_stream_key=sha256_of({"pair_id": self.pair_id, "channel": TREATMENT_CHANNEL}),
            counter_rng_enabled=self.counter_rng_enabled,
            deterministic_flags=self.deterministic_flags,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serializable form for ``torch.save``."""
        return {
            "python_state": self.python_state,
            "numpy_state": self.numpy_state,
            "torch_cpu_state": self.torch_cpu_state,
            "torch_cuda_states": self.torch_cuda_states,
            "counter_rng_enabled": self.counter_rng_enabled,
            "pair_id": self.pair_id,
            "deterministic_flags": self.deterministic_flags,
        }

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> RngState:
        return RngState(
            python_state=payload["python_state"],
            numpy_state=payload["numpy_state"],
            torch_cpu_state=payload["torch_cpu_state"],
            torch_cuda_states=list(payload.get("torch_cuda_states", [])),
            counter_rng_enabled=bool(payload.get("counter_rng_enabled", False)),
            pair_id=str(payload.get("pair_id", "unknown")),
            deterministic_flags=dict(payload.get("deterministic_flags", {})),
        )


def current_determinism_flags() -> dict[str, Any]:
    """Record the determinism-related settings in force."""
    flags: dict[str, Any] = {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    try:
        flags["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:  # pragma: no cover - older torch
        flags["deterministic_algorithms"] = None
    if torch.cuda.is_available():
        flags["device_count"] = torch.cuda.device_count()
        flags["device_name"] = torch.cuda.get_device_name(0)
    return flags


def enable_determinism(warn_only: bool = True) -> dict[str, Any]:
    """Turn on the determinism settings that are available, and report them.

    ``warn_only`` keeps operations without deterministic kernels working instead
    of raising -- important because SONIC's physics and several PPO kernels have
    no deterministic implementation. The point is to remove every avoidable
    source of divergence, not to pretend the remainder is gone.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:  # pragma: no cover - platform dependent
        pass
    return current_determinism_flags()


def _hash_state(state: Any) -> str:
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def _hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
