#!/usr/bin/env python3
"""Evaluate frozen LUCID curriculum checkpoints under matched deployment DR."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility.paths import (  # noqa: E402
    LUCID_ROOT,
    relocate,
)
from scripts.practice_utility import (  # noqa: E402
    run_latency_ab as LA,
    run_throughput_probe as TP,
)

MODES = (
    "lucid",
    "fixed",
    "off",
    "origin",
    "fixed_nolat",
    "fixed_latonly",
    "ta_lucid_25",
    "ta_lucid_50",
    "ta_yoked_25",
    "ta_yoked_50",
    "ta_yoked_25x",
    "ta_yoked_50x",
    "lucid_s4",
    "lucid_rg",
    "lucid_s4_rg",
    "ta_lucid_50_s4_rg",
    "lucid_latcap_s4_rg",
    "ta_lucid_50_latcap_s4_rg",
    "lucid_latonly_s4_rg",
    "ta_lucid_50_latonly_s4_rg",
    "lucid_s4_rg_h6000",
    "lucid_margin_s4_rg",
    "lucid_ratchet_rg",
    "fixed_150",
    "fixed_u",
    "fixed_u150",
    # Monotone support-expansion arms (scalar gate / ramp, per-channel box)
    # and the asymmetric-support arms from the channel attribution.
    "gate_150",
    "ramp_150",
    "box_150",
    "box_asym",
    "ramp_asym",
    "fixed_asym",
    "gate_300",
    "fixed_300",
    "box_fast_300",
    "gate_300_ng",
    "box_fast_300_ng",
    # Practice-allocation arms: the lambda = 1 envelope with a fixed 25% share
    # of the same environments reallocated to one practised condition. They ask
    # where extra training is productive, before any scheduler exists.
    "prac_null",
    "prac_easy",
    "prac_push",
    "prac_fric",
    "prac_pushfric",
)
#: Deployment-latency ladder: nominal physics on every other channel, with
#: actuation latency pinned at a fixed level. The stock ``latency_60ms`` cell
#: stacks 60 ms on top of the full envelope and reads 0.00% for every arm ever
#: measured, the untrained origin included, so it cannot rank policies. These
#: cells isolate the one axis a real deployment actually varies.
PRESET_FIXED_LATENCY_STEPS = {
    "lat_10ms": 2,
    "lat_20ms": 4,
    "lat_30ms": 6,
    "lat_40ms": 8,
    "lat_50ms": 10,
    "lat_60ms": 12,
    # Held-out rungs past the 60 ms ceiling of the training envelope. Each one
    # needs a delay buffer at least this deep (``--max-delay``); the default 12
    # would silently clamp all three onto the 60 ms rung.
    "lat_80ms": 16,
    "lat_100ms": 20,
    "lat_120ms": 24,
}
#: Physics-only ladder: the five non-latency channels scaled, actuation latency
#: pinned to ZERO. The `dr_*` cells all carry the full 0-40 ms latency envelope
#: while `id_clean` carries none, so a `dr_*` ladder moves two factors at once.
#: That was survivable when every arm had been fine-tuned from a policy trained
#: with latency; a from-scratch no-DR arm has never seen latency at all, and a
#: ladder that floors the control cannot rank anything.
PRESET_PHYSICS_ONLY = {
    "phys_000": 0.0,
    "phys_025": 0.25,
    "phys_050": 0.5,
    "phys_075": 0.75,
    "phys_100": 1.0,
    "phys_125": 1.25,
    "phys_150": 1.5,
    "phys_175": 1.75,
    "phys_200": 2.0,
    # The wide corner, for arms whose ceiling is 3.0: outside the support of
    # every arm capped at 2.0 or below, inside for the 3.0 arms on the
    # channels they actually reached (label from the arm's realized frontier).
    "phys_250": 2.5,
    "phys_300": 3.0,
}
#: Single-channel attribution cells: ONE event term widened past its training
#: envelope while the other four physics channels sit at their full (1.0)
#: envelope and actuation latency is pinned to zero. The scalar ladder above
#: moves all five channels together, so a drop at phys_150 cannot say which
#: physics broke the policy -- and the friction floor clamps at lambda ~1.385,
#: so past that the scalar ladder is silently a mass/CoM/push ladder. These
#: cells are the per-channel marginals: the same affine intensity, applied to
#: one term. Friction is stepped finely because its physical clamp makes 1.5
#: and 2.0 differ only in the (benign) high bound.
CHANNEL_TERMS = {
    "fric": "physics_material",
    "mass": "randomize_rigid_body_mass",
    "com": "base_com",
    "joint": "add_joint_default_pos",
    "push": "push_robot",
}
PRESET_CHANNEL: dict[str, dict[str, float]] = {
    "ch_fric_125": {"physics_material": 1.25},
    "ch_fric_150": {"physics_material": 1.5},
    "ch_fric_200": {"physics_material": 2.0},
    "ch_mass_200": {"randomize_rigid_body_mass": 2.0},
    "ch_mass_300": {"randomize_rigid_body_mass": 3.0},
    "ch_com_200": {"base_com": 2.0},
    "ch_com_300": {"base_com": 3.0},
    "ch_joint_200": {"add_joint_default_pos": 2.0},
    "ch_joint_300": {"add_joint_default_pos": 3.0},
    "ch_push_200": {"push_robot": 2.0},
    "ch_push_300": {"push_robot": 3.0},
    # Above every level any arm practises, so it stays a held-out cell for the
    # push-practice arm as well as for its controls.
    "ch_push_350": {"push_robot": 3.5},
}
#: Pairwise cells: TWO terms widened together, the rest at 1.0. The scalar
#: ladder moves five channels at once and the marginals move one, so neither can
#: say whether a pair costs more than its parts. These are the smallest cells
#: that can: the contact-mechanics account of the measured interaction residual
#: predicts push and friction specifically, and predicts a loss larger than the
#: sum of the two marginals at the same intensities.
PRESET_PAIR: dict[str, dict[str, float]] = {
    "ch_push_fric_200_150": {"push_robot": 2.0, "physics_material": 1.5},
    "ch_push_fric_300_150": {"push_robot": 3.0, "physics_material": 1.5},
    # Above the corner the combination arm practises, so it stays held out for it too.
    "ch_push_fric_350_150": {"push_robot": 3.5, "physics_material": 1.5},
}
#: Actuator-side cells. These do NOT scale an event-term range around a nominal in
#: the same sense as the physics channels: the four actuator terms live only in the
#: ``tracking/lucid_actuator`` event preset, so a cell naming one of them also
#: selects that preset. Scaling a channel to 0.0 collapses its range to a point at
#: the nominal, which is how ``act_off`` turns every actuator channel off while
#: keeping the same preset, and is the within-preset baseline these cells are read
#: against. The six inherited channels sit at their envelope throughout, exactly as
#: they do for the physics cells.
ACTUATOR_TERMS = (
    "randomize_joint_effort_limit",
    "randomize_joint_friction",
    "randomize_joint_armature",
    "randomize_joint_velocity_limit",
)
PRESET_ACTUATOR: dict[str, dict[str, float]] = {
    # Every actuator channel collapsed to its nominal: the reference for the rest.
    "act_off": {name: 0.0 for name in ACTUATOR_TERMS},
    # Peak torque. At scale s the range is [1 - 0.5s, 1] of the rating, so 1.5
    # reaches a quarter of peak on the worst-drawn environment.
    **{f"act_effort_{int(s * 100):03d}": {**{n: 0.0 for n in ACTUATOR_TERMS},
                                          "randomize_joint_effort_limit": s}
       for s in (0.5, 1.0, 1.5)},
    # Gearbox friction, in N.m added. The asset declares none, so this is the
    # channel that adds physics rather than perturbing it.
    **{f"act_friction_{int(s * 100):03d}": {**{n: 0.0 for n in ACTUATOR_TERMS},
                                            "randomize_joint_friction": s}
       for s in (0.5, 1.0, 2.0, 3.0)},
    # Reflected inertia; expected to behave like the smooth channels.
    **{f"act_armature_{int(s * 100):03d}": {**{n: 0.0 for n in ACTUATOR_TERMS},
                                            "randomize_joint_armature": s}
       for s in (1.0, 2.0)},
    # Speed ceiling. Read with care: below what the clip demands this makes the
    # reference untrackable rather than hard, which is not a barrier.
    **{f"act_velocity_{int(s * 100):03d}": {**{n: 0.0 for n in ACTUATOR_TERMS},
                                            "randomize_joint_velocity_limit": s}
       for s in (0.5, 1.0, 1.5)},
}

#: Every cell that scales named channels, marginal, pairwise or actuator.
PRESET_SCALED: dict[str, dict[str, float]] = {**PRESET_CHANNEL, **PRESET_PAIR, **PRESET_ACTUATOR}
PRESETS = {
    "id_clean": "tracking/lucid_eval_clean",
    "dr_full": "tracking/lucid_curriculum",
    "latency_60ms": "tracking/lucid_eval_latency_60ms",
    # Robustness-profile cells: full latency envelope with the five non-latency
    # channels scaled to a fraction of their training maximum.
    "dr_025": "tracking/lucid_curriculum",
    "dr_050": "tracking/lucid_curriculum",
    "dr_075": "tracking/lucid_curriculum",
    # Past the training envelope. A deployment claim is about conditions the
    # randomization did not anticipate, so the profile must not stop at 1.
    "dr_125": "tracking/lucid_curriculum",
    "dr_150": "tracking/lucid_curriculum",
    **{name: "tracking/lucid_eval_clean" for name in PRESET_FIXED_LATENCY_STEPS},
    **{name: "tracking/lucid_curriculum" for name in PRESET_PHYSICS_ONLY},
    **{name: "tracking/lucid_curriculum" for name in PRESET_SCALED},
    # The actuator terms exist only in this preset, so a cell that names one must
    # select it. Every other cell keeps the preset it always had.
    **{name: "tracking/lucid_actuator" for name in PRESET_ACTUATOR},
}
PRESET_DR_SCALE = {
    "dr_025": 0.25,
    "dr_050": 0.5,
    "dr_075": 0.75,
    "dr_125": 1.25,
    "dr_150": 1.5,
}


def requested_preset_metadata(presets: list[str]) -> dict[str, dict[str, Any]]:
    """Describe exactly the preset overrides requested for this evaluation."""
    metadata: dict[str, dict[str, Any]] = {}
    for preset in presets:
        if preset not in PRESETS:
            raise ValueError(f"unsupported preset {preset!r}")
        row: dict[str, Any] = {"event_preset": PRESETS[preset]}
        if preset in PRESET_PHYSICS_ONLY:
            row.update(
                {
                    "non_latency_dr_scale": PRESET_PHYSICS_ONLY[preset],
                    "fixed_latency_steps": 0,
                }
            )
        elif preset in PRESET_SCALED:
            row.update(
                {
                    "non_latency_dr_scale": 1.0,
                    "channel_dr_scales": dict(PRESET_SCALED[preset]),
                    "fixed_latency_steps": 0,
                }
            )
        elif preset in PRESET_DR_SCALE:
            row["non_latency_dr_scale"] = PRESET_DR_SCALE[preset]
        elif preset in PRESET_FIXED_LATENCY_STEPS:
            row["fixed_latency_steps"] = PRESET_FIXED_LATENCY_STEPS[preset]
        metadata[preset] = row
    return metadata


#: Steps pinned by the stock ``latency_60ms`` event preset (see its YAML).
LATENCY_60MS_STEPS = 12


def requested_latency_steps(preset: str) -> int | None:
    """Physics steps a latency cell pins its live lag to; None for non-latency cells."""
    if preset in PRESET_FIXED_LATENCY_STEPS:
        return PRESET_FIXED_LATENCY_STEPS[preset]
    if preset == "latency_60ms":
        return LATENCY_60MS_STEPS
    return None


def assert_latency_within_capacity(presets: list[str], max_delay: int) -> None:
    """Refuse any latency cell whose requested lag exceeds the delay-buffer capacity.

    ``--max-delay`` sizes the actuator delay buffers, and ``events_reset_safe``
    clamps every requested lag to that capacity with ``min(high, capacity)``
    without raising. A ``lat_120ms`` cell (24 steps) run at the default
    capacity of 12 would therefore be measured as a 60 ms cell, and
    ``lat_80ms``/``lat_100ms``/``lat_120ms`` would collapse onto one
    measurement with no error. Fail closed before anything is launched.
    """
    truncated = [
        (preset, steps)
        for preset in presets
        if (steps := requested_latency_steps(preset)) is not None and steps > max_delay
    ]
    if truncated:
        cells = ", ".join(f"{preset} ({steps} steps)" for preset, steps in truncated)
        needed = max(steps for _, steps in truncated)
        raise ValueError(
            f"latency cell(s) exceed the delay-buffer capacity and would be silently "
            f"truncated to {max_delay} steps ({max_delay * 5} ms): {cells}; "
            f"--max-delay is {max_delay}. Raise --max-delay to at least {needed}."
        )


CALLBACK = "gear_sonic.research.practice_utility.eval_callback.PracticeRobustnessEvalCallback"
SUMMARY_METRICS = (
    "success_rate",
    "progress_rate",
    "mpjpe_g",
    "mpjpe_l",
    "foot_slip_per_step_m",
    "undesired_contact_rate",
    "torque_saturation",
    "energy_proxy",
)
HIGHER_IS_BETTER = {"success_rate", "progress_rate"}
EXPECTED_DR_TERMS = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
#: The actuator preset declares four more schedulable terms. The mechanism gate
#: asserts an EXACT term set, so without this an actuator cell would report ten
#: live terms against an expected six, the equality would fail, and `verified`
#: would empty for the WHOLE receipt including every unrelated cell in it.
#:
#: Keeping the gate exact rather than relaxing it is the point: on the actuator
#: preset it now proves the four channels were actually present and schedulable
#: in the run, so a preset that failed to load them reports six terms and fails
#: the gate, which is precisely the evidence a new channel needs.
ACTUATOR_DR_TERMS = {
    "randomize_joint_effort_limit",
    "randomize_joint_friction",
    "randomize_joint_armature",
    "randomize_joint_velocity_limit",
}


def expected_dr_terms(preset: str) -> set[str]:
    """The exact set of schedulable terms a cell on this preset must report."""
    if preset in PRESET_ACTUATOR:
        return EXPECTED_DR_TERMS | ACTUATOR_DR_TERMS
    return EXPECTED_DR_TERMS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-receipt",
        type=Path,
        default=LUCID_ROOT / "manifests/curriculum_comparison_ne128_20260820_143058.json",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8600, 8601, 8602])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=None,
        help="defaults to the arms present in the training receipt",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=tuple(PRESETS),
        default=["id_clean", "dr_full", "latency_60ms"],
    )
    parser.add_argument("--eval-seed-base", type=int, default=8700)
    parser.add_argument("--max-delay", type=int, default=12)
    parser.add_argument(
        "--training-config",
        type=Path,
        help="resolved SONIC config.yaml; defaults to the source checkpoint's config",
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=LUCID_ROOT / "manifests/pool_debug512.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=LUCID_ROOT / "manifests/split_debug512_content.json",
    )
    parser.add_argument("--partition", default="dev")
    parser.add_argument(
        "--panel-receipt",
        type=Path,
        default=None,
        help=(
            "evaluate on a lucid_replicate_panel instead of a split partition. Needed for a "
            "single-motion policy: the frozen 102-motion dev panel is not a meaningful test "
            "of one, and a literal one-motion panel is scored on environment 0 alone."
        ),
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=LUCID_ROOT / "pools/debug512/content_dev",
    )
    parser.add_argument(
        "--smpl-motion-file",
        default="data/motion_lib_bones_seed/smpl_filtered",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/curriculum_robustness_eval",
    )
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def materialize_suite(
    pool_manifest: Path,
    split_manifest: Path,
    partition: str,
    suite_root: Path,
) -> dict[str, Any]:
    """Create a stable symlink-only motion panel and verify every target."""
    pool = load_json(pool_manifest)
    split = load_json(split_manifest)
    if split["pool_sha256"] != pool["pool_sha256"]:
        raise ValueError("pool and split manifests do not match")
    selected = {key for key, assigned in split["assignment"].items() if assigned == partition}
    motion_by_key = {row["motion_key"]: row for row in pool["motions"]}
    missing = sorted(selected - motion_by_key.keys())
    if missing:
        raise ValueError(f"split keys missing from pool: {missing[:3]}")

    motion_dir = suite_root / "robot_filtered"
    motion_dir.mkdir(parents=True, exist_ok=True)
    for key in sorted(selected):
        source = relocate(motion_by_key[key]["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        link = motion_dir / f"{key}.pkl"
        if link.is_symlink():
            if link.resolve() != source:
                raise ValueError(f"existing suite link points elsewhere: {link}")
        elif link.exists():
            raise ValueError(f"suite path is not a symlink: {link}")
        else:
            link.symlink_to(source)

    actual = {path.stem for path in motion_dir.glob("*.pkl")}
    extras = sorted(actual - selected)
    if extras:
        raise ValueError(f"suite contains motions outside frozen partition: {extras[:3]}")
    return {
        "motion_file": str(motion_dir.resolve()),
        "motion_count": len(selected),
        "motion_keys_sha256": hashlib.sha256(
            ("\n".join(sorted(selected)) + "\n").encode()
        ).hexdigest(),
        "pool_sha256": pool["pool_sha256"],
        "split_sha256": split["split_sha256"],
        "split_linkage": split["linkage"],
        "partition": partition,
    }


def panel_suite(panel_receipt: Path) -> dict[str, Any]:
    """Use a replicate panel as the evaluation suite, verifying it first."""
    panel = json.loads(Path(panel_receipt).read_text())
    if panel.get("kind") != "lucid_replicate_panel":
        raise ValueError(f"{panel_receipt} is not a lucid_replicate_panel receipt")
    motion_dir = Path(panel["motion_file"])
    present = sorted(p.stem for p in motion_dir.glob("*.pkl"))
    if len(present) != panel["replicates"]:
        raise ValueError(
            f"panel declares {panel['replicates']} replicates but {len(present)} are on disk"
        )
    targets = {p.resolve() for p in motion_dir.glob("*.pkl")}
    if len(targets) != 1:
        raise ValueError(f"panel aliases resolve to {len(targets)} distinct clips, expected 1")
    return {
        "motion_file": str(motion_dir.resolve()),
        "motion_count": len(present),
        "motion_keys_sha256": hashlib.sha256(("\n".join(present) + "\n").encode()).hexdigest(),
        "pool_sha256": panel.get("pool_sha256"),
        "split_sha256": panel.get("split_sha256"),
        "split_linkage": "replicate-panel",
        "partition": panel.get("partition"),
        "replicate_panel": {
            "receipt": str(panel_receipt),
            "motion_key": panel["motion_key"],
            "source_clip_sha256": panel["source_clip_sha256"],
            "replicates": panel["replicates"],
            "alias_keys_sha256": panel["alias_keys_sha256"],
        },
    }


def checkpoint_index(training_receipt: dict[str, Any]) -> dict[tuple[int, str], Path]:
    index = {}
    for arm in training_receipt["arms"].values():
        index[(int(arm["seed"]), arm["mode"])] = Path(arm["checkpoint"]).resolve()
    return index


def ensure_checkpoint_configs(
    checkpoints: list[Path], training_receipt: dict[str, Any], explicit: Path | None = None
) -> dict[str, Any]:
    """Expose the resolved architecture config beside each exported checkpoint."""
    source = explicit
    if source is None:
        source_checkpoint = Path(training_receipt["config"]["checkpoint"]).resolve()
        source = source_checkpoint.parent / "config.yaml"
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"resolved training config not found: {source}")
    source_hash = file_sha256(source)
    installed = []
    for checkpoint in sorted(set(checkpoints)):
        destination = checkpoint.parent / "config.yaml"
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or file_sha256(destination) != source_hash:
                raise ValueError(f"checkpoint has a different config.yaml: {destination}")
        else:
            destination.symlink_to(source)
        installed.append(str(destination))
    return {"source": str(source), "sha256": source_hash, "installed": installed}


def rotated(items: list[str], offset: int) -> list[str]:
    if not items:
        return []
    offset %= len(items)
    return items[offset:] + items[:offset]


def channel_override(preset: str) -> str:
    """Hydra dict override for a scaled cell, e.g. ``{physics_material:1.5}``.

    Marginal cells name one term and pairwise cells name two; the override is
    written the same way for both.
    """
    scales = PRESET_SCALED[preset]
    body = ",".join(f"{name}:{value}" for name, value in sorted(scales.items()))
    return f"++callbacks.practice_eval.channel_dr_scales={{{body}}}"


def build_command(
    args: argparse.Namespace,
    checkpoint: Path,
    mode: str,
    preset: str,
    eval_seed: int,
    branch_id: str,
    output_dir: Path,
    motion_file: str,
) -> list[str]:
    assert_latency_within_capacity([preset], args.max_delay)
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "eval_with_delay.py"),
        "--max-delay",
        str(args.max_delay),
        "--",
        f"checkpoint={checkpoint}",
        f"+num_envs={args.num_envs}",
        "+headless=true",
        "+use_wandb=false",
        f"+seed={eval_seed}",
        f"+manager_env/events={PRESETS[preset]}",
        "+use_encoder=g1",
        "+eval_callbacks=[practice_eval]",
        "+run_eval_loop=false",
        "++manager_env.config.train_only_events=[]",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={args.smpl_motion_file}",
        f"++callbacks.practice_eval._target_={CALLBACK}",
        "++callbacks.practice_eval.eval_frequency=1",
        "++callbacks.practice_eval.eval_only=true",
        f"++callbacks.practice_eval.output_dir={output_dir}",
        f"++callbacks.practice_eval.preset_id={preset}",
        f"++callbacks.practice_eval.branch_id={branch_id}",
        *(
            [
                f"++callbacks.practice_eval.non_latency_dr_scale={PRESET_PHYSICS_ONLY[preset]}",
                "++callbacks.practice_eval.fixed_latency_steps=0",
            ]
            if preset in PRESET_PHYSICS_ONLY
            else [
                "++callbacks.practice_eval.non_latency_dr_scale=1.0",
                channel_override(preset),
                "++callbacks.practice_eval.fixed_latency_steps=0",
            ]
            if preset in PRESET_SCALED
            else (
                [f"++callbacks.practice_eval.non_latency_dr_scale={PRESET_DR_SCALE[preset]}"]
                if preset in PRESET_DR_SCALE
                else (
                    [
                        "++callbacks.practice_eval.fixed_latency_steps="
                        f"{PRESET_FIXED_LATENCY_STEPS[preset]}"
                    ]
                    if preset in PRESET_FIXED_LATENCY_STEPS
                    else []
                )
            )
        ),
    ]


def scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    all_dict = metrics.get("eval/all_metrics_dict", {})
    return {
        "success_rate": scalar(metrics.get("eval/success/success_rate")),
        "progress_rate": scalar(metrics.get("eval/success/progress_rate")),
        "mpjpe_g": scalar(metrics.get("eval/all/mpjpe_g")),
        "mpjpe_l": scalar(metrics.get("eval/all/mpjpe_l")),
        "foot_slip_per_step_m": scalar(metrics.get("eval/quality/foot_slip_per_step_m")),
        "undesired_contact_rate": scalar(metrics.get("eval/quality/undesired_contact_rate")),
        "torque_saturation": scalar(metrics.get("eval/quality/torque_saturation")),
        "energy_proxy": scalar(metrics.get("eval/quality/energy_proxy")),
        "channel_dr_scales": metrics.get("eval/protocol/channel_dr_scales"),
        "quality_missing_signals": metrics.get("eval/quality/missing_signals", []),
        "active_dr_terms": metrics.get("eval/protocol/active_dr_terms", []),
        "dr_ranges": metrics.get("eval/protocol/dr_ranges", {}),
        "delay": {
            key.removeprefix("eval/delay/"): value
            for key, value in metrics.items()
            if key.startswith("eval/delay/")
        },
        "motion_count": len(all_dict.get("motion_keys", [])),
        "failed_count": len(metrics.get("failed_keys", [])),
    }


def aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for preset in PRESETS:
        preset_runs = [run for run in runs.values() if run["preset"] == preset and run["complete"]]
        if not preset_runs:
            continue
        grouped[preset] = {}
        for mode in MODES:
            members = [run for run in preset_runs if run["mode"] == mode]
            if not members:
                continue
            metric_summary = {}
            for metric in SUMMARY_METRICS:
                per_seed = {
                    str(run["checkpoint_seed"]): run["summary"].get(metric) for run in members
                }
                values = [float(value) for value in per_seed.values() if value is not None]
                metric_summary[metric] = {
                    "per_checkpoint_seed": per_seed,
                    "mean": statistics.fmean(values) if values else None,
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                }
            grouped[preset][mode] = {"num_runs": len(members), "metrics": metric_summary}
    return grouped


def paired_comparisons(summary: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for preset, modes in summary.items():
        comparisons[preset] = {}
        pairs = [
            (treatment, other)
            for treatment in modes
            for other in ("fixed", "off", "lucid")
            if other in modes and treatment != other and treatment not in ("fixed", "off")
        ]
        for treatment, other in pairs:
            metrics = {}
            for metric in SUMMARY_METRICS:
                lucid = modes[treatment]["metrics"][metric]["per_checkpoint_seed"]
                reference = modes[other]["metrics"][metric]["per_checkpoint_seed"]
                common = sorted(set(lucid) & set(reference))
                deltas = {
                    seed: float(lucid[seed]) - float(reference[seed])
                    for seed in common
                    if lucid[seed] is not None and reference[seed] is not None
                }
                values = list(deltas.values())
                metrics[metric] = {
                    "treatment_minus_reference_per_seed": deltas,
                    "lucid_minus_reference_per_seed": deltas,
                    "mean_delta": statistics.fmean(values) if values else None,
                    "favorable_direction": "positive" if metric in HIGHER_IS_BETTER else "negative",
                }
            comparisons[preset][f"{treatment}_vs_{other}"] = metrics
    return comparisons


def delay_matches(preset: str, summary: dict[str, Any]) -> bool:
    delay = summary.get("delay", {})
    if delay.get("action_delay_actuator_groups") != 5:
        return False
    if preset == "id_clean":
        return delay.get("action_delay_max_steps") == 0
    if preset == "dr_full" or preset in PRESET_DR_SCALE:
        return (
            delay.get("action_delay_min_steps", -1) >= 0
            and delay.get("action_delay_max_steps") == 8
            and delay.get("action_delay_nonzero_fraction", 0) > 0
        )
    if preset in PRESET_PHYSICS_ONLY or preset in PRESET_SCALED:
        # Latency is pinned to zero, so every live lag must read zero.
        return delay.get("action_delay_max_steps") == 0
    if preset in PRESET_FIXED_LATENCY_STEPS:
        # A ladder cell is only a measurement of that latency if every live lag
        # really sat at it. A zero-step rung is legitimately all-zero.
        steps = PRESET_FIXED_LATENCY_STEPS[preset]
        return (
            delay.get("action_delay_min_steps") == steps
            and delay.get("action_delay_max_steps") == steps
        )
    return delay.get("action_delay_min_steps") == 12 and delay.get("action_delay_max_steps") == 12


def main(argv=None) -> int:
    args = parse_args(argv)
    presets = list(dict.fromkeys(args.presets))
    assert_latency_within_capacity(presets, args.max_delay)
    training_receipt = load_json(args.training_receipt)
    suite = (
        panel_suite(args.panel_receipt)
        if args.panel_receipt
        else materialize_suite(
            args.pool_manifest, args.split_manifest, args.partition, args.suite_root
        )
    )
    checkpoints = checkpoint_index(training_receipt)
    receipt_modes = list(dict.fromkeys(arm["mode"] for arm in training_receipt["arms"].values()))
    modes = list(dict.fromkeys(args.modes)) if args.modes else receipt_modes
    specs = []
    for seed_index, checkpoint_seed in enumerate(args.seeds):
        eval_seed = args.eval_seed_base + seed_index
        for preset in rotated(presets, seed_index):
            for mode in rotated(modes, seed_index):
                checkpoint = checkpoints.get((checkpoint_seed, mode))
                if checkpoint is None or not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"checkpoint missing for seed={checkpoint_seed} mode={mode}"
                    )
                specs.append((checkpoint_seed, eval_seed, mode, preset, checkpoint))
    training_config = ensure_checkpoint_configs(
        [row[4] for row in specs], training_receipt, args.training_config
    )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"curriculum_robustness_ne{args.num_envs}_{stamp}"
    experiment_root = args.artifact_root / experiment_id
    commands = {}
    output_dirs = {}
    for checkpoint_seed, eval_seed, mode, preset, checkpoint in specs:
        branch_id = f"{experiment_id}_s{checkpoint_seed}_{mode}_{preset}"
        output_dir = experiment_root / f"seed_{checkpoint_seed}" / mode / preset
        output_dirs[branch_id] = output_dir
        commands[branch_id] = build_command(
            args,
            checkpoint,
            mode,
            preset,
            eval_seed,
            branch_id,
            output_dir,
            suite["motion_file"],
        )
        print(f"[{branch_id}]\n" + "\n".join(commands[branch_id]))
    if not args.execute:
        print(json.dumps({"suite": suite, "num_runs": len(specs)}, indent=2))
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes_before = {
        str(checkpoint): file_sha256(checkpoint)
        for checkpoint in sorted(set(row[4] for row in specs))
    }
    runs: dict[str, dict[str, Any]] = {}
    receipt_path = args.receipt_dir / f"{experiment_id}.json"

    def make_receipt() -> dict[str, Any]:
        summary = aggregate(runs)
        complete = len(runs) == len(specs) and all(run["complete"] for run in runs.values())
        mechanisms = complete and all(
            delay_matches(run["preset"], run["summary"])
            and set(run["summary"].get("active_dr_terms", [])) == expected_dr_terms(run["preset"])
            for run in runs.values()
        )
        hashes_after = {path: file_sha256(Path(path)) for path in checkpoint_hashes_before}
        frozen = checkpoint_hashes_before == hashes_after
        return {
            "kind": "lucid_frozen_checkpoint_robustness_evaluation",
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment_id": experiment_id,
            "git_sha": TP.git_sha(),
            "git_status_short": TP.git_status(),
            "launcher_sha256": file_sha256(Path(__file__)),
            "training_receipt": str(args.training_receipt.resolve()),
            "training_experiment_id": training_receipt.get("experiment_id"),
            "protocol": {
                "estimand": "frozen-policy robustness; training reward is excluded",
                "num_envs": args.num_envs,
                "checkpoint_seeds": args.seeds,
                "evaluation_seed_by_checkpoint_seed": {
                    str(seed): args.eval_seed_base + index for index, seed in enumerate(args.seeds)
                },
                "modes": modes,
                "presets": requested_preset_metadata(presets),
                "max_delay_capacity_steps": args.max_delay,
                "physics_step_ms": 5,
                "suite": suite,
                "resolved_training_config": training_config,
                "motion_generalization_claim": (
                    "none: the frozen dev panel was included in the 512-motion training pool; "
                    "this is a fresh-physics and latency robustness evaluation"
                ),
                "primary_outcomes": ["success_rate", "progress_rate", "mpjpe_g", "mpjpe_l"],
                "secondary_batch_diagnostics": [
                    "foot_slip_per_step_m",
                    "undesired_contact_rate",
                    "torque_saturation",
                    "energy_proxy",
                ],
                "no_learning": True,
            },
            "commands": commands,
            "runs": runs,
            "mode_summary": summary,
            "paired_comparisons": paired_comparisons(summary),
            "checkpoint_sha256_before": checkpoint_hashes_before,
            "checkpoint_sha256_after": hashes_after,
            "verified": (
                [
                    "all frozen checkpoints completed the matched motion panel",
                    "all six DR terms remained active in evaluation",
                    "all runs used five delayed-actuator groups with the prescribed live lag",
                    "checkpoint hashes were unchanged by evaluation",
                    "mode comparisons use matched checkpoint and evaluation seeds",
                ]
                if mechanisms and frozen
                else []
            ),
            "not_yet_verified": [
                *([] if complete else ["the full requested evaluation matrix"]),
                "unseen-motion generalization",
                (
                    "episode-masked physical-quality comparison; current batch diagnostics "
                    "include auto-reset environments after their scored motion terminates"
                ),
                "hardware transfer or real-world safety",
                "statistical significance beyond three checkpoint seeds",
            ],
        }

    try:
        for checkpoint_seed, eval_seed, mode, preset, checkpoint in specs:
            branch_id = f"{experiment_id}_s{checkpoint_seed}_{mode}_{preset}"
            output_dir = output_dirs[branch_id]
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.log_dir / f"{branch_id}.log"
            runtime = LA.run_arm(commands[branch_id], log_path, args.min_free_mib)
            metrics_path = output_dir / "metrics_eval.json"
            metrics = load_json(metrics_path) if metrics_path.is_file() else {}
            summary = summarize_metrics(metrics) if metrics else {}
            run_complete = (
                runtime["exit_code"] == 0
                and metrics_path.is_file()
                and summary.get("motion_count") == suite["motion_count"]
            )
            runs[branch_id] = {
                "checkpoint_seed": checkpoint_seed,
                "evaluation_seed": eval_seed,
                "mode": mode,
                "preset": preset,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hashes_before[str(checkpoint)],
                "metrics_path": str(metrics_path),
                "log_path": str(log_path),
                "runtime": runtime,
                "summary": summary,
                "complete": run_complete,
            }
            receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")
            print(json.dumps({"branch_id": branch_id, "summary": summary}, indent=2), flush=True)
    finally:
        receipt_path.write_text(json.dumps(make_receipt(), indent=2) + "\n")
        # Announce the path from the finally block too. The receipt is written
        # on every arm and again here, but this print used to sit *after* the
        # try/finally, so a crash produced a complete receipt that no driver
        # could find -- every driver locates it by grepping stdout for exactly
        # this line. 54 cells of a preregistered evaluation were nearly
        # re-run because of that gap.
        print(f"receipt {receipt_path}", flush=True)

    receipt = make_receipt()
    print(json.dumps(receipt["mode_summary"], indent=2))
    return 0 if receipt["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
