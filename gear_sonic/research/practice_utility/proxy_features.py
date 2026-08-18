"""Offline motion-structure features for the difficulty/utility audit.

The audit's sharpest question is not "does failure rate predict utility" but
"do two contexts with the *same* failure rate behave differently, and if so,
what distinguishes them?". Answering it needs descriptors of the motion itself,
computed from the reference trajectory rather than from the policy's behaviour,
so they are available before any branch runs and cannot be contaminated by the
outcome being predicted.

What is computed here, and what is not
--------------------------------------
Everything below is derived from quantities present in the retargeted G1 clip:
``root_trans_offset`` (T, 3), ``root_rot`` (T, 4) quaternions, and ``dof``
(T, 29) joint angles.

**Contact features are deliberately absent.** True foot contact, single-support
fraction, and contact-transition counts require forward kinematics on the robot
model: the retargeted clips carry zeroed ``smpl_joints``, and the SMPL pool
stores pose *parameters* (24x3 axis-angle), not 3D joint positions. Inventing
contact from a root-height heuristic would produce a confident-looking feature
that is not measuring contact, so those features are left to the online path,
where the simulator reports real contact state and
:mod:`quality_metrics` already consumes it.

What *is* available is a physically grounded flight proxy: during ballistic
flight the root's vertical acceleration is approximately -g regardless of pose,
so :func:`ballistic_fraction` measures something real rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: Gravitational acceleration used by the flight detector (m/s^2).
GRAVITY = 9.81

#: Vertical acceleration within this tolerance of -g counts as ballistic.
BALLISTIC_TOLERANCE = 3.0

#: Spectral energy above this fraction of Nyquist counts as "high frequency".
HF_BAND_START = 0.25


@dataclass(frozen=True)
class MotionFeatures:
    """Structure descriptors for one reference clip or bin."""

    num_frames: int
    fps: float
    duration_seconds: float

    root_speed_mean: float
    root_speed_max: float
    root_vertical_range: float
    root_vertical_speed_rms: float
    root_angular_speed_mean: float
    root_angular_speed_max: float

    joint_speed_rms: float
    joint_speed_q90: float
    joint_acceleration_rms: float
    joint_jerk_rms: float
    joint_range_mean: float
    joint_range_max: float

    spectral_complexity: float
    high_frequency_ratio: float
    ballistic_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def as_proxy_features(self, prefix: str = "motion_") -> dict[str, float]:
        """Namespaced form, ready to merge into ``UtilityRecord.proxy_features``."""
        return {f"{prefix}{k}": v for k, v in self.to_dict().items()}


def compute_motion_features(
    dof: np.ndarray,
    root_trans: np.ndarray | None = None,
    root_rot: np.ndarray | None = None,
    fps: float = 30.0,
) -> MotionFeatures:
    """Describe the structure of one reference trajectory.

    Args:
        dof: ``(T, J)`` joint angles in radians.
        root_trans: ``(T, 3)`` root translation in metres, if available.
        root_rot: ``(T, 4)`` root orientation quaternions, if available.
        fps: sampling rate of the clip.

    Root features fall back to zero when the corresponding array is absent,
    rather than raising: a clip missing root data still yields usable joint
    descriptors, and a silently-zero feature is visible in the audit as a
    constant column.
    """
    dof = np.asarray(dof, dtype=np.float64)
    if dof.ndim != 2:
        raise ValueError(f"dof must be (T, J), got shape {dof.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    num_frames = int(dof.shape[0])
    if num_frames == 0:
        raise ValueError("dof has no frames")
    dt = 1.0 / fps

    joint_speed = _difference(dof, dt)
    joint_acceleration = _difference(joint_speed, dt)
    joint_jerk = _difference(joint_acceleration, dt)
    joint_range = dof.max(axis=0) - dof.min(axis=0) if num_frames > 1 else np.zeros(dof.shape[1])

    root_speed = np.zeros(1)
    root_vertical_range = 0.0
    root_vertical_speed = np.zeros(1)
    ballistic = 0.0
    if root_trans is not None:
        root_trans = np.asarray(root_trans, dtype=np.float64)
        if root_trans.ndim != 2 or root_trans.shape[0] != num_frames:
            raise ValueError(
                f"root_trans must be (T, 3) matching dof's {num_frames} frames, "
                f"got {root_trans.shape}"
            )
        velocity = _difference(root_trans, dt)
        root_speed = np.linalg.norm(velocity, axis=-1) if velocity.size else np.zeros(1)
        root_vertical_range = float(root_trans[:, 2].max() - root_trans[:, 2].min())
        root_vertical_speed = velocity[:, 2] if velocity.size else np.zeros(1)
        ballistic = ballistic_fraction(root_trans[:, 2], fps)

    root_angular_speed = np.zeros(1)
    if root_rot is not None:
        root_rot = np.asarray(root_rot, dtype=np.float64)
        if root_rot.ndim != 2 or root_rot.shape[1] != 4:
            raise ValueError(f"root_rot must be (T, 4) quaternions, got {root_rot.shape}")
        root_angular_speed = quaternion_angular_speed(root_rot, fps)

    return MotionFeatures(
        num_frames=num_frames,
        fps=float(fps),
        duration_seconds=num_frames / fps,
        root_speed_mean=_mean(root_speed),
        root_speed_max=_max(root_speed),
        root_vertical_range=root_vertical_range,
        root_vertical_speed_rms=_rms(root_vertical_speed),
        root_angular_speed_mean=_mean(root_angular_speed),
        root_angular_speed_max=_max(root_angular_speed),
        joint_speed_rms=_rms(joint_speed),
        joint_speed_q90=_quantile(np.abs(joint_speed), 0.90),
        joint_acceleration_rms=_rms(joint_acceleration),
        joint_jerk_rms=_rms(joint_jerk),
        joint_range_mean=float(joint_range.mean()) if joint_range.size else 0.0,
        joint_range_max=float(joint_range.max()) if joint_range.size else 0.0,
        spectral_complexity=spectral_complexity(dof),
        high_frequency_ratio=high_frequency_ratio(dof),
        ballistic_fraction=ballistic,
    )


def quaternion_angular_speed(quaternions: np.ndarray, fps: float) -> np.ndarray:
    """Angular speed in rad/s between consecutive orientations.

    Uses the geodesic angle on the unit quaternion sphere, with the double-cover
    sign ambiguity resolved by taking ``|dot|`` -- otherwise a quaternion and its
    negation, which represent the same rotation, would register as a 180-degree
    jump.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.shape[0] < 2:
        return np.zeros(1)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    unit = quaternions / np.maximum(norms, 1e-12)
    dots = np.abs((unit[1:] * unit[:-1]).sum(axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dots) * fps


def ballistic_fraction(
    height: np.ndarray, fps: float, tolerance: float = BALLISTIC_TOLERANCE
) -> float:
    """Share of frames whose vertical acceleration is consistent with free fall.

    A physically grounded flight proxy: during flight the root accelerates at
    approximately -g whatever the pose is doing. It is a proxy, not a contact
    detector -- a fast downward squat can momentarily look ballistic -- so it is
    reported as a fraction to be audited, never used as ground truth.
    """
    height = np.asarray(height, dtype=np.float64).reshape(-1)
    if height.shape[0] < 3:
        return 0.0
    acceleration = np.diff(height, n=2) * (fps**2)
    return float(np.mean(np.abs(acceleration + GRAVITY) < tolerance))


def spectral_complexity(signal: np.ndarray) -> float:
    """Spectral entropy of the joint trajectory, normalized to ``[0, 1]``.

    Near 0 means the motion concentrates in a few frequencies (a simple, cyclic
    behaviour); near 1 means energy is spread broadly (a complex or noisy one).
    The mean is removed first so a static pose does not read as complex.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 4:
        return 0.0
    centred = signal - signal.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centred, axis=0)) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    distribution = (power.sum(axis=1) / power.sum()) if power.ndim == 2 else power / total
    distribution = distribution[distribution > 0]
    if distribution.size <= 1:
        return 0.0
    entropy = float(-(distribution * np.log(distribution)).sum())
    return entropy / math.log(distribution.size)


def high_frequency_ratio(signal: np.ndarray, band_start: float = HF_BAND_START) -> float:
    """Share of spectral energy above ``band_start`` of Nyquist."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 4:
        return 0.0
    if not 0.0 < band_start < 1.0:
        raise ValueError(f"band_start must lie in (0, 1), got {band_start}")
    centred = signal - signal.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centred, axis=0)) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    cutoff = int(band_start * power.shape[0])
    return float(power[cutoff:].sum() / total)


def features_for_bin(
    clip: dict[str, Any], start_frame: int, end_frame: int, fps: float | None = None
) -> MotionFeatures:
    """Features for one bin, sliced out of a full clip.

    Bin-level features are what the audit needs: a clip-level average would blur
    a difficult transition into the calm seconds around it, which is precisely
    the distinction the study is trying to resolve.
    """
    dof = np.asarray(clip["dof"])
    if not 0 <= start_frame < end_frame <= dof.shape[0]:
        raise ValueError(
            f"bin [{start_frame}, {end_frame}) is out of range for a "
            f"{dof.shape[0]}-frame clip"
        )
    window = slice(start_frame, end_frame)
    root_trans = clip.get("root_trans_offset")
    root_rot = clip.get("root_rot")
    return compute_motion_features(
        dof=dof[window],
        root_trans=np.asarray(root_trans)[window] if root_trans is not None else None,
        root_rot=np.asarray(root_rot)[window] if root_rot is not None else None,
        fps=float(fps if fps is not None else clip.get("fps", 30)),
    )


def contact_regime_proxy(features: MotionFeatures) -> str:
    """Coarse regime label for stratification: ``aerial`` / ``dynamic`` / ``steady``.

    A *proxy* built from measurable root dynamics, not a contact classification.
    Its only job is to keep a probe campaign from filling up with motions of one
    dynamical character; the audit never treats it as ground truth.
    """
    if features.ballistic_fraction > 0.15:
        return "aerial"
    if features.root_speed_mean > 0.8 or features.joint_speed_rms > 2.0:
        return "dynamic"
    return "steady"


def _difference(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros((1,) + values.shape[1:])
    return np.diff(values, axis=0) / dt


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(values.mean()) if values.size else 0.0


def _max(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(values.max()) if values.size else 0.0


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values)
    return float(np.quantile(values, q)) if values.size else 0.0
