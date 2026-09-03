#!/usr/bin/env python3
"""Train and compare SONIC LUCID, fixed-DR, and no-DR branches."""

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

from gear_sonic.research.practice_utility import (  # noqa: E402
    branch_capsule as BC,
    dr_scaling as DS,
)
from gear_sonic.research.practice_utility.paths import LUCID_ROOT  # noqa: E402
from scripts.practice_utility import (  # noqa: E402
    run_latency_ab as LA,
    run_throughput_probe as TP,
)

OBSERVER = "gear_sonic.research.practice_utility.observer.PracticeObserverCallback"
CURRICULUM = "gear_sonic.research.practice_utility.dr_curriculum.LucidCurriculumCallback"
CAPSULE = "gear_sonic.research.practice_utility.callbacks.PracticeCapsuleCallback"
#: Arm name -> (curriculum mode, anchor ratio, yoked source arm). The TACE arms
#: pin a fixed cohort of envs to the full envelope (see tace.py); the yoked arm
#: replays its source arm's lambda trajectory for the same seed with no feedback.
ARMS: dict[str, tuple[str, float, str | None]] = {
    "lucid": ("lucid", 0.0, None),
    "fixed": ("fixed", 0.0, None),
    "off": ("off", 0.0, None),
    "ta_lucid_25": ("lucid", 0.25, None),
    "ta_lucid_50": ("lucid", 0.50, None),
    "ta_yoked_25": ("yoked", 0.25, "ta_lucid_25"),
    "ta_yoked_50": ("yoked", 0.50, "ta_lucid_50"),
    # Cross-seed yoking: seed s replays the schedule learned on the *next* seed.
    # Same-seed yoking is bit-identical to its source (deterministic simulator),
    # so it cannot test online feedback; this can.
    "ta_yoked_25x": ("yoked", 0.25, "ta_lucid_25"),
    "ta_yoked_50x": ("yoked", 0.50, "ta_lucid_50"),
}
CROSS_SEED_ARMS = {"ta_yoked_25x", "ta_yoked_50x"}
#: Channel-attribution arms: fixed intensity with per-term overrides.
NON_LATENCY_TERMS = (
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_rigid_body_mass",
)
ARM_TERM_OVERRIDES: dict[str, dict[str, float]] = {
    "fixed_nolat": {"randomize_action_delay": 0.0},
    "fixed_latonly": {term: 0.0 for term in NON_LATENCY_TERMS},
    # Latency-only *curriculum* arms. The untrained origin is already robust to
    # the five non-latency channels (60.5% at the full envelope, 56.2% at 1.25x
    # it) and has zero margin at a pinned 60 ms; training the full envelope
    # destroys it, and channel attribution says latency carries 89% of that
    # damage. So the only axis with headroom is the one that must be approached
    # gently -- which is what these arms are: a gap-gated, stratified,
    # relatively-guarded curriculum on latency alone.
    "lucid_latonly_s4_rg": {term: 0.0 for term in NON_LATENCY_TERMS},
    "ta_lucid_50_latonly_s4_rg": {term: 0.0 for term in NON_LATENCY_TERMS},
}
ARMS.update({"fixed_nolat": ("fixed", 0.0, None), "fixed_latonly": ("fixed", 0.0, None)})
ARMS.update(
    {
        "lucid_latonly_s4_rg": ("lucid", 0.0, None),
        "ta_lucid_50_latonly_s4_rg": ("lucid", 0.50, None),
    }
)

#: LUCID-S arms. ``spread_strata = K`` splits the focus cohort into K intensity
#: strata so the training mixture spans ``(0, lambda]`` rather than the single
#: point ``lambda``; ``return_guard = "relative"`` replaces the absolute return
#: floor, which the 128-iteration horizon study showed is not scale-stable.
#: The two are separate arms as well as a combined one, because a combined-only
#: result cannot say which change did the work.
ARMS.update(
    {
        "lucid_s4": ("lucid", 0.0, None),
        "lucid_rg": ("lucid", 0.0, None),
        "lucid_s4_rg": ("lucid", 0.0, None),
        "ta_lucid_50_s4_rg": ("lucid", 0.50, None),
    }
)
ARM_SPREAD_STRATA: dict[str, int] = {
    "lucid_s4": 4,
    "lucid_s4_rg": 4,
    "ta_lucid_50_s4_rg": 4,
}
#: Per-channel *ceilings*. Unlike ARM_TERM_OVERRIDES, which pins a channel at a
#: constant, a cap lets the curriculum still schedule the channel up to its own
#: limit -- the one thing a scalar lambda cannot express. These arms exist for
#: the case where channel attribution names a single destructive channel; the
#: cap value is set at launch by --latency-cap and recorded in the receipt.
CAP_ARMS = ("lucid_latcap_s4_rg", "ta_lucid_50_latcap_s4_rg")
ARMS.update(
    {
        "lucid_latcap_s4_rg": ("lucid", 0.0, None),
        "ta_lucid_50_latcap_s4_rg": ("lucid", 0.50, None),
    }
)
ARM_RETURN_GUARD: dict[str, str] = {
    "lucid_rg": "relative",
    "lucid_s4_rg": "relative",
    "ta_lucid_50_s4_rg": "relative",
    "lucid_latcap_s4_rg": "relative",
    "ta_lucid_50_latcap_s4_rg": "relative",
}
ARM_SPREAD_STRATA.update({arm: 4 for arm in CAP_ARMS})
#: The margin-signal arm: same strata and relative guard as lucid_s4_rg, but
#: the controller reads the termination margin from every env as a ratio
#: against a 64-env yardstick cohort held at lambda = 0, with a dead band.
MARGIN_ARMS = ("lucid_margin_s4_rg",)
ARMS.update({"lucid_margin_s4_rg": ("lucid", 0.0, None)})
ARM_SPREAD_STRATA.update({arm: 4 for arm in MARGIN_ARMS})
ARM_RETURN_GUARD.update({arm: "relative" for arm in MARGIN_ARMS})
#: Support-extension arms: fixed DR trained PAST the lambda = 1 envelope -- the
#: frontier-exposure lever the capability ladder identified (every arm orders by
#: time spent at the hardest physics it ever saw). Training-side physical clamps
#: apply (friction floor etc.), and the delay buffer must be sized for the
#: extended latency range or it silently clamps; build_command enforces that.
ARM_FIXED_LAMBDA: dict[str, float] = {"fixed_150": 1.5}
ARMS.update({"fixed_150": ("fixed", 0.0, None)})
MARGIN_OBSERVER = "gear_sonic.research.practice_utility.margin_observer.MarginObserverCallback"
LATONLY_ARMS = ("lucid_latonly_s4_rg", "ta_lucid_50_latonly_s4_rg")
ARM_SPREAD_STRATA.update({arm: 4 for arm in LATONLY_ARMS})
ARM_RETURN_GUARD.update({arm: "relative" for arm in LATONLY_ARMS})
#: Expand-don't-replace support arms: open-loop fixed-mode per-env lambda
#: mixtures. The top stratum pins ``ARM_TOP_FRACTION`` of the focus cohort at
#: ``fixed_lambda`` -- the frontier, where the capability ladder says all the
#: resolution lives -- while the remaining envs spread over the lower doses
#: ``fixed_lambda * (k+1)/K`` as the retention tail. ``fixed_u`` keeps the
#: frontier at the lambda = 1 envelope (the pure mixture control);
#: ``fixed_u150`` puts it at 1.5x with training-side physical clamps, fusing
#: the fixed_150 support lever with the mixture. The controller is inert
#: (fixed mode), so no signal can evacuate the frontier.
EXPAND_ARMS = ("fixed_u", "fixed_u150")
ARMS.update({arm: ("fixed", 0.0, None) for arm in EXPAND_ARMS})
ARM_SPREAD_STRATA.update({arm: 8 for arm in EXPAND_ARMS})
ARM_FIXED_LAMBDA.update({"fixed_u150": 1.5})
ARM_TOP_FRACTION: dict[str, float] = {arm: 0.75 for arm in EXPAND_ARMS}
EXPAND_EXPECTED_CLAMPS: dict[str, list[str]] = {
    "fixed_u": [],
    "fixed_u150": ["physics_material"],
}
EXPECTED_SCALABLE_TERMS = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
#: Stage-A anti-collapse arm: the same unstratified latent-gap curriculum and
#: relative return guard as ``lucid_rg``, but PI-law decreases are projected
#: away. The return guard remains the only path that can lower lambda.
RATCHET_ARMS = ("lucid_ratchet_rg",)
ARMS.update({arm: ("lucid", 0.0, None) for arm in RATCHET_ARMS})
ARM_RETURN_GUARD.update({arm: "relative" for arm in RATCHET_ARMS})
#: Monotone support-expansion arms, the pair the ratchet result asked for.
#: ``gate_150`` widens the frontier when a probe stratum held one step ABOVE it
#: survives; ``ramp_150`` widens it on a fixed linear schedule and reads
#: nothing. They share stratum count, stratum sizes, probe placement and
#: terminal support, so the only thing that differs is the decision rule -- the
#: comparison that the ratchet arm, being distributionally identical to fixed
#: DR, could not make. Both are monotone by construction, so neither can
#: evacuate difficulty the way the unconstrained controller did in 2 of 6 cells.
#: ``box_150`` is the gate with a VECTOR frontier: one entry per randomization
#: channel, each raised on its own probe evidence, the single probe stratum
#: visiting the channels in rotation (box_gate.py). Same strata, same probe
#: size, same per-channel ceiling as gate_150, so it can only differ in which
#: channels widen and when -- never by training on more support.
EXPANSION_ARMS = ("gate_150", "ramp_150", "box_150")
SURVIVAL_OBSERVER = (
    "gear_sonic.research.practice_utility.survival_observer.SurvivalObserverCallback"
)
ARMS.update(
    {
        "gate_150": ("gate", 0.0, None),
        "ramp_150": ("ramp", 0.0, None),
        "box_150": ("box", 0.0, None),
    }
)
ARM_SPREAD_STRATA.update({arm: 8 for arm in EXPANSION_ARMS})
ARM_RETURN_GUARD.update({arm: "relative" for arm in EXPANSION_ARMS})
#: Frontier ceiling per expansion arm. Kept separate from ARM_FIXED_LAMBDA
#: because these arms do not train at a fixed lambda: they *end* there.
ARM_FRONTIER_MAX: dict[str, float] = {arm: 1.5 for arm in EXPANSION_ARMS}
#: Cohort shares: a 12.5% probe above the frontier, 62.5% at the frontier, and
#: a 25% retained tail -- the same tail share as the fixed_u150 mixture, so the
#: retention question and the scheduling question stay separable.
EXPANSION_PROBE_FRACTION = 0.125
EXPANSION_FRONTIER_FRACTION = 0.625
#: The probe sits one expansion step above the frontier, so a probe draw is
#: always the next support level rather than the current one.
EXPANSION_STEP = 0.125


def _max_frontier_drop(curriculum: list[dict[str, Any]]) -> float:
    """Largest downward movement of the applied frontier, over a whole run.

    Zero is the claim. It is computed from the written trajectory rather than
    from the controller's own state so that the receipt checks the arm rather
    than trusting it.
    """
    worst = 0.0
    previous: float | None = None
    for row in curriculum:
        value = row.get("frontier_lambda", row.get("lambda"))
        if not isinstance(value, (int, float)):
            continue
        if previous is not None and value < previous:
            worst = max(worst, previous - float(value))
        previous = float(value)
    return worst


def expansion_stratum_sizes(
    num_focus: int,
    num_strata: int,
    frontier_fraction: float = EXPANSION_FRONTIER_FRACTION,
    probe_fraction: float = EXPANSION_PROBE_FRACTION,
) -> list[int]:
    """Stratum sizes for an expansion arm: tail, then frontier, then probe.

    Order matters: the callback reads the last stratum as the probe and the
    second to last as the frontier, so the sizes are returned in that order.
    """
    if not 0.0 < probe_fraction < 1.0:
        raise ValueError(f"probe_fraction must be in (0, 1), got {probe_fraction}")
    if not 0.0 < frontier_fraction < 1.0:
        raise ValueError(f"frontier_fraction must be in (0, 1), got {frontier_fraction}")
    if frontier_fraction + probe_fraction >= 1.0:
        raise ValueError(
            "frontier and probe shares must leave a retention tail, got "
            f"{frontier_fraction} + {probe_fraction}"
        )
    if num_strata < 3:
        raise ValueError(f"expansion arms need >= 3 strata, got {num_strata}")
    probe = int(round(probe_fraction * num_focus))
    frontier = int(round(frontier_fraction * num_focus))
    tail_total = num_focus - probe - frontier
    lower = num_strata - 2
    if tail_total < lower:
        raise ValueError(
            f"{num_focus} focus envs leave a {tail_total}-env tail, too thin for "
            f"{lower} lower strata"
        )
    base, extra = divmod(tail_total, lower)
    sizes = [base + (1 if index < extra else 0) for index in range(lower)]
    return sizes + [frontier, probe]


#: Asymmetric-support arms, from the single-channel attribution sweep of
#: 2026-09-02: for trained policies mass, CoM and joint offsets are nearly free
#: to three times their range while push disturbance is the binding channel
#: (0.71-0.77 success at 3x) and friction is clamped past ~1.385. A uniform
#: 1.5 ceiling therefore withholds width where it is free and spends it where
#: it is not. These arms let the cheap channels reach 2.0 and hold push,
#: friction and latency at 1.5:
#:   box_asym    the box gate with per-channel ceilings (discovers the order online)
#:   ramp_asym   the open-loop control: scalar frontier 1.0 -> 2.0 with the three
#:               binding channels capped at 1.5 (M5: same terminal support)
#:   fixed_asym  the width control: the asymmetric box from iteration 0
#: Their terminal support exceeds the 1.5 arms' on three channels, so the
#: scalar phys_175/phys_200 cells are IN SUPPORT for them on those channels
#: and must be labelled; the per-channel 3x cells stay outside every ceiling.
ASYM_CEILINGS: dict[str, float] = {
    "randomize_rigid_body_mass": 2.0,
    "base_com": 2.0,
    "add_joint_default_pos": 2.0,
    "physics_material": 1.5,
    "push_robot": 1.5,
    "randomize_action_delay": 1.5,
}
ASYM_MAX = max(ASYM_CEILINGS.values())
ASYM_ARMS = ("box_asym", "ramp_asym", "fixed_asym")
ARMS.update(
    {"box_asym": ("box", 0.0, None), "ramp_asym": ("ramp", 0.0, None), "fixed_asym": ("fixed", 0.0, None)}
)
EXPANSION_ARMS = (*EXPANSION_ARMS, "box_asym", "ramp_asym")
ARM_SPREAD_STRATA.update({"box_asym": 8, "ramp_asym": 8})
ARM_RETURN_GUARD.update({arm: "relative" for arm in ASYM_ARMS})
ARM_FRONTIER_MAX.update({"box_asym": ASYM_MAX, "ramp_asym": ASYM_MAX})
ARM_FIXED_LAMBDA.update({"fixed_asym": ASYM_MAX})
#: Caps that turn a scalar frontier/fixed lambda into the asymmetric box: the
#: scalar reaches ASYM_MAX and each capped channel stops at its own ceiling.
ASYM_CAPS: dict[str, float] = {name: cap for name, cap in ASYM_CEILINGS.items() if cap < ASYM_MAX}

#: Beyond-the-safe-width arms (prototype batch 4, 2026-09-02). At a 1.5
#: ceiling the scalar gate merely matches fixed width, because 1.5 is safe.
#: Feedback can only earn its place where an open-loop ceiling is NOT safe:
#: these arms set the ceiling at 3.0, where the attribution sweep says push
#: breaks trained policies (0.71-0.77 at 3x). fixed_300 trains there blind;
#: gate_300 must stop where its probe fails; box_fast_300 may stop push and
#: keep widening the channels that are still free. Latency is held at 1.5 on
#: all three so the 60 ms delay buffer (--max-delay 12) stays exact.
#: ``gate_300_ng`` is gate_300 with the relative-return guard effectively
#: disabled (a 99% drop tolerance). It exists because gate_300 stopped at 1.5
#: for the WRONG reason: from a warm start the guard's reference is the return
#: earned at low difficulty, so once difficulty rises the trailing mean can
#: never recover to 75% of that best and the guard latches permanently -- it
#: froze expansion on 1,509 of 1,990 iterations while the probe was still
#: clearing its threshold 67% of the time. Without the guard, the survival
#: probe alone decides where to stop, which is the claim the method actually
#: rests on.
#: ``box_fast_300_ng`` is the per-channel companion to gate_300_ng. At a fast
#: cadence and a 3.0 ceiling the guarded box still converged to a nearly
#: uniform frontier (1.5 on four channels, 1.375 on two) because the guard
#: froze every channel at once and no channel's probe ever fell below its
#: threshold: at lambda <= 1.5 all six channels look equally survivable, so
#: the box has nothing to discriminate on. Per-channel asymmetry can only
#: appear where the channels actually differ, which the attribution sweep
#: places near 2.5-3.0 and which the guard prevented any arm from reaching.
WIDE_ARMS = ("gate_300", "fixed_300", "box_fast_300", "gate_300_ng", "box_fast_300_ng")
WIDE_MAX = 3.0
WIDE_CAPS: dict[str, float] = {"randomize_action_delay": 1.5}
WIDE_BOX_CEILINGS: dict[str, float] = {
    "randomize_rigid_body_mass": WIDE_MAX,
    "base_com": WIDE_MAX,
    "add_joint_default_pos": WIDE_MAX,
    "physics_material": WIDE_MAX,
    "push_robot": WIDE_MAX,
    "randomize_action_delay": 1.5,
}
ARMS.update(
    {
        "gate_300": ("gate", 0.0, None),
        "fixed_300": ("fixed", 0.0, None),
        "box_fast_300": ("box", 0.0, None),
        "gate_300_ng": ("gate", 0.0, None),
        "box_fast_300_ng": ("box", 0.0, None),
    }
)
EXPANSION_ARMS = (*EXPANSION_ARMS, "gate_300", "box_fast_300", "gate_300_ng", "box_fast_300_ng")
ARM_SPREAD_STRATA.update({"gate_300": 8, "box_fast_300": 8, "gate_300_ng": 8, "box_fast_300_ng": 8})
#: Per-arm relative-return-guard tolerance, overriding --return-relative-drop.
#: 0.99 means "only a 99% collapse counts as harm", i.e. the guard is inert.
ARM_RETURN_DROP: dict[str, float] = {"gate_300_ng": 0.99, "box_fast_300_ng": 0.99}
ARM_RETURN_GUARD.update({arm: "relative" for arm in WIDE_ARMS})
ARM_FRONTIER_MAX.update(
    {"gate_300": WIDE_MAX, "box_fast_300": WIDE_MAX, "gate_300_ng": WIDE_MAX, "box_fast_300_ng": WIDE_MAX}
)
ARM_FIXED_LAMBDA.update({"fixed_300": WIDE_MAX})
#: Per-arm channel caps (scalar frontier/fixed lambda clamped per term) and
#: per-arm box ceilings (vector frontier bounded per term).
ARM_TERM_CAPS: dict[str, dict[str, float]] = {
    "ramp_asym": ASYM_CAPS,
    "fixed_asym": ASYM_CAPS,
    "gate_300": WIDE_CAPS,
    "fixed_300": WIDE_CAPS,
    "gate_300_ng": WIDE_CAPS,
}
ARM_BOX_CEILINGS: dict[str, dict[str, float]] = {
    "box_asym": ASYM_CEILINGS,
    "box_fast_300": WIDE_BOX_CEILINGS,
    "box_fast_300_ng": WIDE_BOX_CEILINGS,
}
GATE_ARMS = ("gate_150", "box_150", "box_asym", "gate_300", "box_fast_300", "gate_300_ng", "box_fast_300_ng")
BOX_ARMS = ("box_150", "box_asym", "box_fast_300", "box_fast_300_ng")

#: ---------------------------------------------------------------------------
#: Practice-allocation arms (screen, 2026-09-02). These answer a question that
#: comes BEFORE any scheduler: where is extra training actually productive?
#:
#: Every arm keeps the lambda = 1 envelope, the architecture, the reward, the
#: motion, the origin checkpoint and the iteration budget. The only difference
#: is what a fixed 25% share of the SAME 1,024 environments practises. Nothing
#: is added: the share is taken from the lambda = 1 cohort, so a targeted arm
#: trains on fewer standard-mixture episodes, not on more episodes.
#:
#:   prac_null      the matched control: the practice share trains at lambda 1
#:                  like everything else, so the dispatcher is active and only
#:                  the practice CONTENT differs from the arms below
#:   prac_easy      the placebo: the share practises the three channels the
#:                  attribution sweep found nearly free (mass, CoM, joint at
#:                  3x, where the origin already scores 0.949/0.988/0.990)
#:   prac_push      the bottleneck: the share practises push at 3x, where the
#:                  origin scores 0.746 -- difficult, and far from the floor
#:   prac_fric      friction at 1.5x alone, the second factor on its own
#:   prac_pushfric  both factors together, at the SAME levels the single-factor
#:                  arms use, so the four arms form a 2x2 on {push practice,
#:                  friction practice} and the interaction is estimable rather
#:                  than confounded with a change of dose
#:
#: The levels are read off the measured single-channel sweep rather than chosen,
#: so "difficult" means a measured success level, not an intuition. The plain
#: ``fixed`` arm at one stratum is the second control and answers how much comes
#: from simply continuing to train.
#: Amended 2026-09-03 (A1): prac_pushfric practised push at 2.0 while prac_push practised it
#: at 3.0, so their difference mixed "add friction" with "lower the push dose" and answered a
#: question about recipes. Both now practise push at 3.0, prac_fric supplies friction alone,
#: and the four arms {null, push, fric, pushfric} form a 2x2 whose interaction term is
#: estimable. Amendment made before any arm was trained.
PRACTICE_ARMS = ("prac_null", "prac_easy", "prac_push", "prac_fric", "prac_pushfric")
#: Share of the focus cohort reallocated to the practice stratum. Identical for
#: every arm, so the arms differ only in what that share practises.
PRACTICE_FRACTION = 0.25
PRACTICE_CHANNELS: dict[str, dict[str, float]] = {
    "prac_null": {},
    "prac_easy": {
        "randomize_rigid_body_mass": 3.0,
        "base_com": 3.0,
        "add_joint_default_pos": 3.0,
    },
    "prac_push": {"push_robot": 3.0},
    "prac_fric": {"physics_material": 1.5},
    "prac_pushfric": {"push_robot": 3.0, "physics_material": 1.5},
}
ARMS.update({arm: ("fixed", 0.0, None) for arm in PRACTICE_ARMS})
ARM_SPREAD_STRATA.update({arm: 2 for arm in PRACTICE_ARMS})
ARM_TOP_FRACTION.update({arm: PRACTICE_FRACTION for arm in PRACTICE_ARMS})
#: The arm's maximum applied intensity on ANY channel; drives the extrapolation
#: flag and the delay-buffer check.
ARM_PRACTICE_MAX: dict[str, float] = {
    arm: max([1.0, *PRACTICE_CHANNELS[arm].values()]) for arm in PRACTICE_ARMS
}


def practice_vectors(mode: str) -> list[dict[str, float]]:
    """Per-stratum channel intensities for a practice arm, low stratum first.

    Stratum 0 is the retained lambda = 1 mixture and carries no entries, so
    every channel there trains exactly where the control arm has it. Stratum 1
    is the practice share and names only the channels it practises.
    """
    if mode not in PRACTICE_ARMS:
        raise ValueError(f"{mode!r} is not a practice-allocation arm")
    return [{}, dict(PRACTICE_CHANNELS[mode])]


#: ---------------------------------------------------------------------------
#: Actuator-barrier arms (2026-09-03). These exist to test one structural claim.
#:
#: Fixed randomization here is ALREADY a curriculum. dr_scaling.scale_range
#: shrinks every range toward its nominal, so the support at any intensity is
#: strictly nested inside the support at the maximum, and every term is reset
#: mode, so all 1024 environments redraw independently every episode. A batch at
#: full intensity therefore contains near-nominal episodes, and staging cannot
#: withhold what fixed randomization keeps supplying. That is the simplest
#: explanation for every curriculum result in this project tying fixed DR, and
#: the scale function's own docstring says as much.
#:
#: The prediction is that a curriculum can only help when the target is
#: CONCENTRATED, so that easy episodes are genuinely absent. These arms test it by
#: changing one thing: whether the target actuator range is a range or a point.
#:
#:   act_off      the actuator channels at nominal; what the budget buys with none
#:   act_range    the target as a RANGE, which is self-curricularizing
#:   act_point    the same target as a POINT: every environment, every episode
#:   act_ramp     an open-loop schedule from nominal to that point
#:   act_gate     the probe-gated curriculum expanding toward that point
#:
#: act_point versus act_range is the decisive contrast and it is a one-line
#: configuration difference, which is what makes the claim cheap to falsify.
#: act_ramp and act_gate say whether staging reaches a point that direct training
#: cannot, and whether feedback adds anything over the schedule.
#:
#: The channel and the target value are launch arguments, not constants, because
#: which channel is worth training on is decided by the frozen-policy screen.
ACTUATOR_ARMS = ("act_off", "act_range", "act_point", "act_ramp", "act_gate")
ARMS.update({
    "act_off": ("fixed", 0.0, None),
    "act_range": ("fixed", 0.0, None),
    "act_point": ("fixed", 0.0, None),
    "act_ramp": ("ramp", 0.0, None),
    "act_gate": ("gate", 0.0, None),
})
ARM_SPREAD_STRATA.update({"act_ramp": 8, "act_gate": 8})
ARM_RETURN_GUARD.update({arm: "relative" for arm in ACTUATOR_ARMS})
#: Every actuator arm needs the preset that declares the four actuator terms.
ARM_EVENT_PRESET: dict[str, str] = {arm: "tracking/lucid_actuator" for arm in ACTUATOR_ARMS}
#: The channel each arm varies, and the parameter that carries its range.
ACTUATOR_RANGE_PARAM = {
    "effort_limit": ("randomize_joint_effort_limit", "effort_limit_scale_range", 1.0),
    "joint_friction": ("randomize_joint_friction", "joint_friction_range", 0.0),
    "armature": ("randomize_joint_armature", "armature_scale_range", 1.0),
    "velocity_limit": ("randomize_joint_velocity_limit", "velocity_limit_scale_range", 1.0),
}


def actuator_overrides(mode: str, channel: str, target: float) -> list[str]:
    """Hydra overrides that give one actuator arm its target distribution.

    Every actuator channel except the one under test is collapsed to a point at
    its nominal, so a difference between arms is attributable to the one channel.
    The channel under test gets a range for ``act_range`` and a point for the
    rest; for the scheduled arms the point is where the schedule ENDS, and the
    curriculum moves it there from the nominal.
    """
    if mode not in ACTUATOR_ARMS:
        raise ValueError(f"{mode!r} is not an actuator-barrier arm")
    if channel not in ACTUATOR_RANGE_PARAM:
        raise ValueError(f"unknown actuator channel {channel!r}")
    overrides = []
    for name, (term, param, nominal) in ACTUATOR_RANGE_PARAM.items():
        if name == channel and mode != "act_off":
            low, high = (min(nominal, target), max(nominal, target)) if mode == "act_range" else (target, target)
        else:
            low, high = nominal, nominal  # collapsed to its nominal: inert
        overrides.append(
            f"++manager_env.events.{term}.params.{param}=[{low},{high}]")
    return overrides


#: Every arm whose applied lambda can exceed the lambda = 1 envelope, and the
#: ceiling it can reach. The delay-buffer capacity check reads THIS, not the
#: fixed-lambda table: an expansion arm launched at the default --max-delay
#: would silently train latency at 1.0x while its telemetry claimed 1.5x.
ARM_LAMBDA_CEILING: dict[str, float] = {**ARM_FIXED_LAMBDA, **ARM_FRONTIER_MAX, **ARM_PRACTICE_MAX}
#: Per-arm ceiling on the LATENCY channel specifically, for the delay-buffer
#: check: an asymmetric arm reaches 2.0 on mass but holds latency at 1.5.
ARM_DELAY_CEILING: dict[str, float] = {
    **{arm: ASYM_CEILINGS["randomize_action_delay"] for arm in ASYM_ARMS},
    **{arm: WIDE_CAPS["randomize_action_delay"] for arm in WIDE_ARMS},
    # No practice arm widens latency, so the standard buffer is enough however
    # far the practised physics channels reach.
    **{arm: 1.0 for arm in PRACTICE_ARMS},
}
MODES = tuple(ARMS)


def expand_stratum_sizes(num_focus: int, num_strata: int, top_fraction: float) -> list[int]:
    """Final stratum sizes for an expand arm: thin near-equal tail, fat top."""
    if not 0.0 < top_fraction < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}")
    if num_strata < 2:
        raise ValueError(f"expand arms need >= 2 strata, got {num_strata}")
    top = int(round(top_fraction * num_focus))
    tail = num_focus - top
    lower = num_strata - 1
    if tail < lower:
        raise ValueError(
            f"{num_focus} focus envs leave a {tail}-env tail, too thin for " f"{lower} lower strata"
        )
    base, extra = divmod(tail, lower)
    sizes = [base + (1 if index < extra else 0) for index in range(lower)]
    return sizes + [top]


def expand_arm_contract(mode: str, num_envs: int) -> dict[str, Any]:
    """Return the frozen manipulation contract for an expand-support arm."""
    if mode not in EXPAND_ARMS:
        raise ValueError(f"{mode!r} is not an expand-support arm")
    num_strata = 8
    fixed_lambda = 1.5 if mode == "fixed_u150" else 1.0
    stratum_lambdas = [fixed_lambda * (index + 1) / num_strata for index in range(num_strata)]
    return {
        "curriculum_mode": "fixed",
        "anchor_ratio": 0.0,
        "num_anchor": 0,
        "num_focus": num_envs,
        "spread_strata": num_strata,
        "stratum_sizes": expand_stratum_sizes(num_envs, num_strata, 0.75),
        "stratum_lambdas": stratum_lambdas,
        "fixed_lambda": fixed_lambda,
        "allow_extrapolation": mode == "fixed_u150",
        "physical_clamp": EXPAND_EXPECTED_CLAMPS[mode],
        "max_delay_steps": int(fixed_lambda * 8),
        "delay_stratum_ranges": [[0.0, 8.0 * value] for value in stratum_lambdas[:-1]] + [None],
        "dispatch_terms": sorted(EXPECTED_SCALABLE_TERMS),
    }


def _float_sequence_matches(observed: Any, expected: list[float]) -> bool:
    return (
        isinstance(observed, list)
        and len(observed) == len(expected)
        and all(
            isinstance(value, (int, float)) and abs(float(value) - target) <= 1e-9
            for value, target in zip(observed, expected)
        )
    )


def _nested_values_match(observed: Any, expected: Any) -> bool:
    """Compare nested dispatcher parameters with a tight numeric tolerance."""
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(observed) - float(expected)) <= 1e-9
    if isinstance(expected, dict) and isinstance(observed, dict):
        return observed.keys() == expected.keys() and all(
            _nested_values_match(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
        return len(observed) == len(expected) and all(
            _nested_values_match(left, right) for left, right in zip(observed, expected)
        )
    return observed == expected


def validate_expand_arm_contract(
    mode: str,
    arm: dict[str, Any],
    curriculum: list[dict[str, Any]],
    *,
    num_envs: int,
    max_delay: int,
) -> dict[str, Any]:
    """Reconcile an expand arm's self-description against live TACE telemetry.

    Expand arms deliberately have ``anchor_ratio == 0``. The legacy mechanics
    gate only inspected TACE when that ratio was positive, so a run could omit
    the mixture dispatcher entirely and still be marked verified. This check is
    independent of the anchor gate and pins every manipulation-bearing field.
    """
    expected = expand_arm_contract(mode, num_envs)
    spec = arm.get("arm_spec") or {}
    tace = arm.get("tace_final") or {}
    errors: list[str] = []

    def exact(label: str, observed: Any, target: Any) -> None:
        if observed != target:
            errors.append(f"{label}: expected {target!r}, observed {observed!r}")

    exact("arm_spec.curriculum_mode", spec.get("curriculum_mode"), "fixed")
    exact("arm_spec.anchor_ratio", spec.get("anchor_ratio"), 0.0)
    exact("arm_spec.anchor_seed", spec.get("anchor_seed"), arm.get("seed"))
    exact("arm_spec.spread_strata", spec.get("spread_strata"), expected["spread_strata"])
    exact("arm_spec.stratum_sizes", spec.get("stratum_sizes"), expected["stratum_sizes"])
    if not _float_sequence_matches(spec.get("stratum_lambdas"), expected["stratum_lambdas"]):
        errors.append(
            "arm_spec.stratum_lambdas: expected "
            f"{expected['stratum_lambdas']!r}, observed {spec.get('stratum_lambdas')!r}"
        )
    exact("arm_spec.top_fraction", spec.get("top_fraction"), 0.75)
    exact("arm_spec.fixed_lambda", spec.get("fixed_lambda"), expected["fixed_lambda"])
    exact(
        "arm_spec.allow_extrapolation",
        spec.get("allow_extrapolation"),
        expected["allow_extrapolation"],
    )
    exact("arm_spec.physical_clamp", spec.get("physical_clamp") or [], expected["physical_clamp"])
    exact("arm_spec.max_delay_steps", spec.get("max_delay_steps"), expected["max_delay_steps"])
    exact("max_delay_steps", max_delay, expected["max_delay_steps"])

    required_stratum_keys = {f"focus_s{index}" for index in range(expected["spread_strata"])}

    def reconcile_tace(label: str, telemetry: Any, *, require_counts: bool) -> bool:
        before = len(errors)
        if not isinstance(telemetry, dict):
            errors.append(f"{label}: expected a mapping, observed {telemetry!r}")
            return False
        exact(f"{label}.num_anchor", telemetry.get("num_anchor"), expected["num_anchor"])
        exact(f"{label}.num_focus", telemetry.get("num_focus"), expected["num_focus"])
        exact(f"{label}.anchor_ratio", telemetry.get("anchor_ratio"), expected["anchor_ratio"])
        exact(f"{label}.num_strata", telemetry.get("num_strata"), expected["spread_strata"])
        exact(f"{label}.stratum_sizes", telemetry.get("stratum_sizes"), expected["stratum_sizes"])
        if not _float_sequence_matches(
            telemetry.get("stratum_lambdas"), expected["stratum_lambdas"]
        ):
            errors.append(
                f"{label}.stratum_lambdas: expected {expected['stratum_lambdas']!r}, "
                f"observed {telemetry.get('stratum_lambdas')!r}"
            )

        dispatch = telemetry.get("dispatch")
        if not isinstance(dispatch, dict):
            errors.append(f"{label}.dispatch: expected a mapping, observed {dispatch!r}")
            dispatch = {}
        exact(f"{label}.dispatch terms", sorted(dispatch), expected["dispatch_terms"])
        for term in expected["dispatch_terms"]:
            term_telemetry = dispatch.get(term)
            if not isinstance(term_telemetry, dict):
                errors.append(f"{label}.dispatch.{term}: missing dispatcher telemetry")
                continue
            exact(f"{label}.dispatch.{term}.term", term_telemetry.get("term"), term)
            exact(
                f"{label}.dispatch.{term}.num_strata",
                term_telemetry.get("num_strata"),
                expected["spread_strata"],
            )
            params = term_telemetry.get("stratum_params")
            baseline = term_telemetry.get("anchor_params")
            if not isinstance(params, list) or len(params) != expected["spread_strata"]:
                errors.append(
                    f"{label}.dispatch.{term}.stratum_params: expected "
                    f"{expected['spread_strata']} entries, observed {params!r}"
                )
            elif not isinstance(baseline, dict):
                errors.append(f"{label}.dispatch.{term}.anchor_params: missing baseline mapping")
            else:
                for index, dose in enumerate(expected["stratum_lambdas"][:-1]):
                    target = DS.scaled_term_params(baseline, dose, expected["allow_extrapolation"])
                    if expected["allow_extrapolation"]:
                        target, _ = DS.clamp_params_physical(target)
                    if not _nested_values_match(params[index], target):
                        errors.append(
                            f"{label}.dispatch.{term}.stratum_params[{index}]: "
                            f"expected {target!r}, observed {params[index]!r}"
                        )
                        break
                if params[-1] is not None:
                    errors.append(
                        f"{label}.dispatch.{term}.stratum_params[-1]: expected "
                        f"frontier passthrough, observed {params[-1]!r}"
                    )
            if require_counts:
                counts = term_telemetry.get("env_counts")
                if not isinstance(counts, dict) or not required_stratum_keys.issubset(counts):
                    errors.append(
                        f"{label}.dispatch.{term}.env_counts: missing one or more focus strata"
                    )
        return len(errors) == before

    reconcile_tace("tace", tace, require_counts=True)

    instrumented_rows = [row for row in curriculum if isinstance(row.get("tace"), dict)]
    if not instrumented_rows:
        errors.append("curriculum: no rows contain TACE telemetry")
    for index, row in enumerate(instrumented_rows):
        before = len(errors)
        exact(
            f"curriculum TACE row {index}.lambda",
            row.get("lambda"),
            expected["fixed_lambda"],
        )
        if row.get("allow_extrapolation", False) is not expected["allow_extrapolation"]:
            errors.append(
                f"curriculum TACE row {index}.allow_extrapolation: expected "
                f"{expected['allow_extrapolation']!r}, observed "
                f"{row.get('allow_extrapolation', False)!r}"
            )
        if (row.get("physical_clamp") or []) != expected["physical_clamp"]:
            errors.append(
                f"curriculum TACE row {index}.physical_clamp: expected "
                f"{expected['physical_clamp']!r}, observed {row.get('physical_clamp')!r}"
            )
        reconcile_tace(f"curriculum TACE row {index}.tace", row.get("tace"), require_counts=False)
        if len(errors) != before:
            break

    return {"passed": not errors, "expected": expected, "errors": errors}


TRAINING_METRICS = ("Mean rewards", "Mean length", "Mean entropy")
QUALITY_METRICS = (
    "latent_p90",
    "foot_slip_per_step_m",
    "torque_saturation",
    "energy_proxy",
    "action_delay_mean_steps",
    "action_delay_nonzero_fraction",
)
#: Repo-relative motion paths. Callers that synthesise an ``args`` namespace
#: (the horizon orchestrator) need not carry them, so they are read with a
#: default rather than as required attributes.
DEFAULT_MOTION_FILE = "data/motion_lib_bones_seed/robot_filtered"
DEFAULT_SMPL_MOTION_FILE = "data/motion_lib_bones_seed/smpl_filtered"
#: The four thresholds the strict exp preset overrides, restored to the values
#: their own term files declare. Not tuning: every number here is upstream's own.
#: Needed because from scratch the strict values gate on *tracking accuracy*, not
#: on falling. Measured over 235 from-scratch iterations under tracking/base:
#: ee_body_pos ended 90.3% of episodes while anchor_pos (pelvis height) ended
#: 6.2% and anchor_ori_full 5.9% -- the robot was upright and being killed for a
#: wrist or an ankle being off in Z.
TERMINATION_DEFAULT_OVERRIDES = [
    "++manager_env.terminations.anchor_pos.params.threshold=0.5",
    "++manager_env.terminations.ee_body_pos.params.threshold=0.5",
    "++manager_env.terminations.foot_pos_xyz.params.threshold=0.5",
    "++manager_env.terminations.anchor_ori_full.params.threshold=1.0",
]
REWARD_FLOOR = 0.0333
LENGTH_FLOOR = 0.0314


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="branch origin; omit together with --from-scratch to train a fresh policy",
    )
    parser.add_argument(
        "--actuator-channel",
        choices=sorted(ACTUATOR_RANGE_PARAM),
        default="effort_limit",
        help="which actuator channel the act_* arms vary; chosen by the frozen-policy screen",
    )
    parser.add_argument(
        "--actuator-target",
        type=float,
        default=0.5,
        help=(
            "where the act_* arms end: the point act_point trains at, the upper end of "
            "act_range, and the endpoint act_ramp and act_gate schedule toward"
        ),
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help=(
            "train from a fresh initialisation instead of continuing a checkpoint. "
            "Fine-tuning the released policy is destructive at this scale -- plain "
            "no-DR continuation costs 23 profile-AUC points against the untrained "
            "origin -- so an arm comparison that starts there is measuring damage, "
            "not learning."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=None,
        help=(
            "extra iteration counts at which to export a capsule, for a convergence "
            "curve measured along one trajectory rather than across separate runs"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8600, 8601, 8602])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["lucid", "fixed", "off"])
    parser.add_argument(
        "--yoked-source-receipt",
        type=Path,
        default=None,
        help="take yoked schedules from this earlier training receipt instead of this run",
    )
    parser.add_argument(
        "--consolidation-fraction",
        type=float,
        default=0.0,
        help="curriculum arms (never 'off'): final fraction of the budget with "
        "every env on the full envelope",
    )
    parser.add_argument("--max-delay", type=int, default=8)
    parser.add_argument("--delta-target", type=float, default=0.778)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--integral-max", type=float, default=1.0)
    parser.add_argument("--return-floor", type=float, default=8.0)
    parser.add_argument(
        "--latency-cap",
        type=float,
        default=0.5,
        help="cap arms: ceiling on the actuation-latency channel's share of lambda",
    )
    parser.add_argument(
        "--margin-horizon",
        type=int,
        default=12,
        help="margin arms: prefix length K of each episode the margin is averaged over",
    )
    parser.add_argument(
        "--margin-band-lo",
        type=float,
        default=1.10,
        help="margin arms: below this focus/yardstick ratio the dose rises",
    )
    parser.add_argument(
        "--margin-band-hi",
        type=float,
        default=1.30,
        help="margin arms: above this ratio the dose falls; between, it holds",
    )
    parser.add_argument(
        "--yardstick-envs",
        type=int,
        default=64,
        help="margin arms: environments held at lambda=0 as the self-reference",
    )
    parser.add_argument(
        "--return-relative-drop",
        type=float,
        default=0.25,
        help="relative-guard arms: fractional fall below the trailing best that counts as harm",
    )
    parser.add_argument(
        "--return-window",
        type=int,
        default=8,
        help="relative-guard arms: how many epochs of its own history an arm is judged against",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.80,
        help="gate arm: probe-stratum survival the trailing window must average to expand",
    )
    parser.add_argument(
        "--gate-window",
        type=int,
        default=200,
        help="gate arm: iterations of probe survival averaged before an expansion may fire",
    )
    parser.add_argument(
        "--gate-dwell",
        type=int,
        default=200,
        help="gate arm: iterations that must pass after an expansion before the next",
    )
    parser.add_argument(
        "--gate-min-episodes",
        type=int,
        default=200,
        help="gate arm: probe episodes required in the window; below this it holds",
    )
    parser.add_argument(
        "--gate-guard-action",
        choices=("freeze", "decay"),
        default="freeze",
        help=(
            "gate arm: what the return guard does. 'freeze' halts expansion and keeps "
            "applied support; 'decay' also contracts it, which is recorded as an incident"
        ),
    )
    parser.add_argument(
        "--box-channel-budget",
        type=int,
        default=0,
        help=(
            "box arm: iterations one channel may hold the probe without a decision "
            "before the probe moves on; 0 disables the timeout"
        ),
    )
    parser.add_argument(
        "--ramp-begin-iteration",
        type=int,
        default=1000,
        help="ramp arm: iteration the open-loop frontier starts rising at",
    )
    parser.add_argument(
        "--ramp-end-iteration",
        type=int,
        default=5000,
        help="ramp arm: iteration the open-loop frontier reaches its ceiling",
    )
    parser.add_argument("--exp", default="manager/universal_token/all_modes/sonic_release")
    parser.add_argument(
        "--terminations",
        default=None,
        help=(
            "termination preset, e.g. tracking/base or tracking/eval. The stock training "
            "preset (tracking/base_adaptive_strict_ori_foot_xyz) is STRICTER than the eval "
            "preset -- 0.15 m position and 0.2 rad orientation, plus a 0.2 m foot term -- "
            "which is right for a competent policy and fatal from scratch, where 93%% of "
            "episodes die on tracking error in ~0.25 s and essentially none reach time-out."
        ),
    )
    parser.add_argument(
        "--termination-thresholds",
        choices=("strict", "default"),
        default="strict",
        help=(
            "'strict' keeps the exp preset's overrides (0.15 m anchor/ee, 0.2 m foot, "
            "0.2 rad orientation). 'default' reverts those four to the values their own "
            "terms/*.yaml files declare (0.5/0.5/0.5/1.0), keeping the same composition "
            "including the adaptive low-pelvis allowance two clips need."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="stream metrics to this Weights & Biases project; omit to stay offline",
    )
    parser.add_argument(
        "--motion-file",
        default=DEFAULT_MOTION_FILE,
        help="motion_lib pool every arm trains on",
    )
    parser.add_argument(
        "--smpl-motion-file",
        default=DEFAULT_SMPL_MOTION_FILE,
        help=(
            "SMPL pack for the SMPL observation encoder. 'dummy' substitutes zeros, "
            "which is also what a missing path does; hosts without the 32 GB pack "
            "must pass it explicitly so the receipt records the difference."
        ),
    )
    parser.add_argument(
        "--encoder",
        default=str(LUCID_ROOT / "artifacts/lucid_encoder_debug512.pt"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=LUCID_ROOT / "artifacts/curriculum_comparison",
    )
    parser.add_argument("--log-dir", type=Path, default=LUCID_ROOT / "outputs")
    parser.add_argument("--receipt-dir", type=Path, default=LUCID_ROOT / "manifests")
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= args.warmup_iterations:
        parser.error("iterations must exceed warmup iterations")
    if args.from_scratch and args.checkpoint:
        parser.error("--from-scratch and --checkpoint are mutually exclusive")
    if not args.from_scratch and not args.checkpoint:
        parser.error("pass --checkpoint, or --from-scratch to train a fresh policy")
    for horizon in args.horizons or ():
        if not 0 < horizon <= args.iterations:
            parser.error(f"horizon {horizon} must be in (0, {args.iterations}]")
    return args


def arm_order(modes: list[str], seed_index: int) -> list[str]:
    """Rotate arm order by seed to avoid confounding mode with wall-clock order."""
    offset = seed_index % len(modes)
    return modes[offset:] + modes[:offset]


def build_command(
    args,
    mode: str,
    seed: int,
    branch_id: str,
    artifact_dir: Path,
    yoked_schedule: Path | None = None,
) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}")
    curriculum_mode, anchor_ratio, source = ARMS[mode]
    if curriculum_mode == "yoked" and yoked_schedule is None:
        raise ValueError(f"arm {mode!r} needs the {source!r} schedule for seed {seed}")
    capsule_dir = artifact_dir / "capsules"
    tace = (
        [
            f"++callbacks.lucid_curriculum.anchor_ratio={anchor_ratio}",
            f"++callbacks.lucid_curriculum.anchor_seed={seed}",
        ]
        if anchor_ratio > 0.0
        else []
    )
    # Consolidation is forwarded for every curriculum arm, not only anchored
    # ones -- the anchor-gated list silently dropped it for exactly the
    # anchor_ratio = 0 arms whose collapsed final checkpoints motivated it.
    # 'off' is the no-DR control and must never receive a full-envelope phase.
    # A zero fraction emits nothing, so every existing arm's command line is
    # byte-identical to the receipts it trained under.
    consolidation_fraction = float(getattr(args, "consolidation_fraction", 0.0) or 0.0)
    consolidation = (
        [f"++callbacks.lucid_curriculum.consolidation_fraction={consolidation_fraction}"]
        if consolidation_fraction > 0.0 and curriculum_mode != "off"
        else []
    )
    yoked = (
        [f"++callbacks.lucid_curriculum.yoked_schedule_path={yoked_schedule}"]
        if curriculum_mode == "yoked"
        else []
    )
    overrides = [
        f"++callbacks.lucid_curriculum.term_lambda_overrides.{term}={value}"
        for term, value in ARM_TERM_OVERRIDES.get(mode, {}).items()
    ]
    caps = (
        [f"++callbacks.lucid_curriculum.term_lambda_caps.randomize_action_delay={args.latency_cap}"]
        if mode in CAP_ARMS
        else []
    )
    if mode in ARM_TERM_CAPS:
        # The scalar frontier climbs to the arm's ceiling; capped channels stop
        # at their own, per stratum, through the same cap path the latency-cap
        # arms use.
        caps += [
            f"++callbacks.lucid_curriculum.term_lambda_caps.{term}={cap}"
            for term, cap in sorted(ARM_TERM_CAPS[mode].items())
        ]
    margin = (
        [
            f"++callbacks.margin_observer._target_={MARGIN_OBSERVER}",
            "++callbacks.margin_observer.enabled=true",
            f"++callbacks.margin_observer.branch_id={branch_id}",
            f"++callbacks.margin_observer.output_dir={artifact_dir}",
            f"++callbacks.margin_observer.horizon={args.margin_horizon}",
            f"++callbacks.margin_observer.band_lo={args.margin_band_lo}",
            f"++callbacks.margin_observer.band_hi={args.margin_band_hi}",
            "++callbacks.lucid_curriculum.signal=margin",
            f"++callbacks.lucid_curriculum.yardstick_envs={args.yardstick_envs}",
            f"++callbacks.lucid_curriculum.margin_branch_id={branch_id}",
        ]
        if mode in MARGIN_ARMS
        else []
    )
    strata = ARM_SPREAD_STRATA.get(mode, 1)
    guard = ARM_RETURN_GUARD.get(mode, "absolute")
    fixed_lambda_value = ARM_FIXED_LAMBDA.get(mode, 1.0)
    if mode in EXPAND_ARMS:
        contract = expand_arm_contract(mode, args.num_envs)
        static_fields = {
            "curriculum_mode": curriculum_mode,
            "anchor_ratio": anchor_ratio,
            "spread_strata": strata,
            "fixed_lambda": fixed_lambda_value,
            "top_fraction": ARM_TOP_FRACTION.get(mode),
            "allow_extrapolation": mode in ARM_FIXED_LAMBDA,
        }
        expected_fields = {
            "curriculum_mode": contract["curriculum_mode"],
            "anchor_ratio": contract["anchor_ratio"],
            "spread_strata": contract["spread_strata"],
            "fixed_lambda": contract["fixed_lambda"],
            "top_fraction": 0.75,
            "allow_extrapolation": contract["allow_extrapolation"],
        }
        if static_fields != expected_fields:
            raise SystemExit(
                f"{mode} launcher contract drifted: expected {expected_fields}, "
                f"observed {static_fields}"
            )
        if args.max_delay != contract["max_delay_steps"]:
            raise SystemExit(
                f"{mode} requires exactly --max-delay {contract['max_delay_steps']} "
                f"for its frozen delay contract; observed {args.max_delay}"
            )
        if float(getattr(args, "consolidation_fraction", 0.0) or 0.0) != 0.0:
            raise SystemExit(
                f"{mode} pins its frontier mixture for the full run; "
                "--consolidation-fraction must be 0"
            )
    ceiling = ARM_LAMBDA_CEILING.get(mode, 1.0)
    extrapolation = (
        ["++callbacks.lucid_curriculum.allow_extrapolation=true"] if ceiling > 1.0 else []
    )
    if ceiling > 1.0:
        # 8 steps = the 40 ms training ceiling at 200 Hz; the delayed-actuator
        # buffer clamps any drawn delay to its allocated capacity WITHOUT error,
        # so an undersized buffer would quietly turn 1.5x latency back into 1.0x.
        # Gated on the arm's effective lambda ceiling rather than on membership
        # of the fixed-lambda table: an expansion arm reaches 1.5 too, and
        # gating on the table would have let it through silently.
        # Expansion arms cap the probe at the frontier ceiling, so their
        # maximum applied intensity is the ceiling itself, exactly as for an
        # open-loop arm at the same lambda.
        reach = min(ceiling, ARM_DELAY_CEILING.get(mode, ceiling))
        needed = reach * 8
        needed = int(needed) if float(needed).is_integer() else int(needed) + 1
        if args.max_delay < needed:
            raise SystemExit(
                f"{mode} trains latency at up to {reach}x the 0-40 ms envelope, "
                f"which needs a delay-buffer capacity of {needed} steps; the buffer "
                f"silently clamps to --max-delay ({args.max_delay}). Pass --max-delay {needed}."
            )
    actuator: list[str] = []
    if mode in ACTUATOR_ARMS:
        actuator = actuator_overrides(
            mode,
            getattr(args, "actuator_channel", "effort_limit"),
            float(getattr(args, "actuator_target", 0.5)),
        )
    spread = [f"++callbacks.lucid_curriculum.spread_strata={strata}"] if strata > 1 else []
    if strata > 1 and anchor_ratio == 0.0:
        # Strata need the cohort machinery, which the callback only installs
        # when it has a seed to draw the partition from.
        spread.append(f"++callbacks.lucid_curriculum.anchor_seed={seed}")
    if mode in ARM_TOP_FRACTION:
        sizes = expand_stratum_sizes(args.num_envs, strata, ARM_TOP_FRACTION[mode])
        spread.append(
            "++callbacks.lucid_curriculum.stratum_sizes="
            "[" + ",".join(str(size) for size in sizes) + "]"
        )
    if mode in PRACTICE_ARMS:
        # One frozen JSON literal per stratum. Hydra sees a quoted string, the
        # callback parses it, and the realized vectors land in the run's own
        # TACE telemetry, so the receipt states the exposure rather than the
        # intent.
        vectors = json.dumps(practice_vectors(mode), separators=(",", ":"))
        spread.append(f"++callbacks.lucid_curriculum.practice_vectors_json='{vectors}'")
    expansion: list[str] = []
    if mode in EXPANSION_ARMS:
        sizes = expansion_stratum_sizes(args.num_envs, strata)
        spread.append(
            "++callbacks.lucid_curriculum.stratum_sizes="
            "[" + ",".join(str(size) for size in sizes) + "]"
        )
        expansion = [
            f"++callbacks.survival_observer._target_={SURVIVAL_OBSERVER}",
            "++callbacks.survival_observer.enabled=true",
            f"++callbacks.survival_observer.branch_id={branch_id}",
            f"++callbacks.survival_observer.output_dir={artifact_dir}",
            f"++callbacks.lucid_curriculum.survival_branch_id={branch_id}",
            f"++callbacks.lucid_curriculum.gate_probe_offset={EXPANSION_STEP}",
            # Probe capped at the frontier ceiling, so this arm's maximum
            # applied intensity equals every other 1.5 arm's. A probe above the
            # ceiling would give the expansion arms strictly more support than
            # fixed_150 and fixed_u150, confounding "the gate helped" with "the
            # gate trained harder".
            f"++callbacks.lucid_curriculum.gate_probe_max={ARM_FRONTIER_MAX[mode]}",
        ]
        if mode in GATE_ARMS:
            expansion += [
                f"++callbacks.lucid_curriculum.gate_threshold={args.gate_threshold}",
                f"++callbacks.lucid_curriculum.gate_window={args.gate_window}",
                f"++callbacks.lucid_curriculum.gate_step={EXPANSION_STEP}",
                f"++callbacks.lucid_curriculum.gate_dwell={args.gate_dwell}",
                f"++callbacks.lucid_curriculum.gate_min_episodes={args.gate_min_episodes}",
                f"++callbacks.lucid_curriculum.gate_lambda_max={ARM_FRONTIER_MAX[mode]}",
                f"++callbacks.lucid_curriculum.gate_guard_action={args.gate_guard_action}",
            ]
            if mode in BOX_ARMS:
                expansion.append(
                    f"++callbacks.lucid_curriculum.box_channel_budget={args.box_channel_budget}"
                )
            if mode in ARM_BOX_CEILINGS:
                expansion += [
                    f"++callbacks.lucid_curriculum.box_lambda_max.{term}={ceiling_value}"
                    for term, ceiling_value in sorted(ARM_BOX_CEILINGS[mode].items())
                ]
        else:
            expansion += [
                "++callbacks.lucid_curriculum.ramp_start_lambda=1.0",
                f"++callbacks.lucid_curriculum.ramp_end_lambda={ARM_FRONTIER_MAX[mode]}",
                f"++callbacks.lucid_curriculum.ramp_begin_iteration={args.ramp_begin_iteration}",
                f"++callbacks.lucid_curriculum.ramp_end_iteration={args.ramp_end_iteration}",
            ]
    relative_guard = (
        [
            f"++callbacks.lucid_curriculum.return_guard={guard}",
            "++callbacks.lucid_curriculum.return_relative_drop="
            f"{ARM_RETURN_DROP.get(mode, args.return_relative_drop)}",
            f"++callbacks.lucid_curriculum.return_window={args.return_window}",
        ]
        if guard != "absolute"
        else []
    )
    ratchet = ["++callbacks.lucid_curriculum.monotonic=true"] if mode in RATCHET_ARMS else []
    origin = [] if getattr(args, "from_scratch", False) else [f"checkpoint={args.checkpoint}"]
    horizons = [
        f"++callbacks.practice_capsule.horizons.h{h:04d}={h}"
        for h in sorted(set(getattr(args, "horizons", None) or ()))
    ]
    return [
        sys.executable,
        str(REPO / "scripts" / "practice_utility" / "train_with_delay.py"),
        "--max-delay",
        str(args.max_delay),
        "--",
        f"+exp={args.exp}",
        *origin,
        f"num_envs={args.num_envs}",
        "headless=true",
        *(
            ["use_wandb=true", f"project_name={args.wandb_project}"]
            if getattr(args, "wandb_project", None)
            else ["use_wandb=false"]
        ),
        f"seed={seed}",
        f"manager_env/events={ARM_EVENT_PRESET.get(mode, 'tracking/lucid_curriculum')}",
        *(
            [f"manager_env/terminations={args.terminations}"]
            if getattr(args, "terminations", None)
            else []
        ),
        *(
            TERMINATION_DEFAULT_OVERRIDES
            if getattr(args, "termination_thresholds", "strict") == "default"
            else []
        ),
        f"++algo.config.num_learning_iterations={args.iterations}",
        "++algo.config.save_interval=100000",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file="
        f"{getattr(args, 'motion_file', DEFAULT_MOTION_FILE)}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
        f"{getattr(args, 'smpl_motion_file', DEFAULT_SMPL_MOTION_FILE)}",
        f"++callbacks.practice_observer._target_={OBSERVER}",
        "++callbacks.practice_observer.enabled=true",
        f"++callbacks.practice_observer.encoder_path={args.encoder}",
        f"++callbacks.practice_observer.branch_id={branch_id}",
        f"++callbacks.practice_observer.output_dir={artifact_dir}",
        f"++callbacks.lucid_curriculum._target_={CURRICULUM}",
        "++callbacks.lucid_curriculum.enabled=true",
        f"++callbacks.lucid_curriculum.mode={curriculum_mode}",
        *tace,
        *consolidation,
        *yoked,
        *overrides,
        *caps,
        *margin,
        *spread,
        *actuator,
        *relative_guard,
        *ratchet,
        *expansion,
        f"++callbacks.lucid_curriculum.observer_branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.branch_id={branch_id}",
        f"++callbacks.lucid_curriculum.output_dir={artifact_dir}",
        "++callbacks.lucid_curriculum.initial_lambda=0.0",
        f"++callbacks.lucid_curriculum.fixed_lambda={fixed_lambda_value}",
        *extrapolation,
        f"++callbacks.lucid_curriculum.delta_target={args.delta_target}",
        f"++callbacks.lucid_curriculum.kp={args.kp}",
        f"++callbacks.lucid_curriculum.ki={args.ki}",
        f"++callbacks.lucid_curriculum.alpha={args.alpha}",
        f"++callbacks.lucid_curriculum.integral_max={args.integral_max}",
        f"++callbacks.lucid_curriculum.return_floor={args.return_floor}",
        f"++callbacks.lucid_curriculum.warmup_iterations={args.warmup_iterations}",
        f"++callbacks.practice_capsule._target_={CAPSULE}",
        "++callbacks.practice_capsule.enabled=true",
        f"++callbacks.practice_capsule.capsule_dir={capsule_dir}",
        f"++callbacks.practice_capsule.pair_id=curriculum_seed_{seed}",
        "++callbacks.practice_capsule.role=control",
        f"++callbacks.practice_capsule.branch_id={branch_id}",
        f"++callbacks.practice_capsule.horizons.final={args.iterations}",
        *horizons,
    ]


def _logging_directory(log_path: Path) -> str | None:
    """The Hydra run directory this branch trained in, from its own log.

    Every branch of a campaign gets its own run directory and its own resolved
    config.yaml. The evaluator symlinks ONE config beside every checkpoint it
    scores, so without this the fixed and lucid checkpoints of a campaign sit
    beside a config that says mode=off. Architecture is shared so nothing loads
    wrongly, but the provenance is false, which is the class of defect this
    programme exists to catch.
    """
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    # The trainer prints a Rich table, and a long path wraps across box rows:
    #   │ Logging Directory:                          │
    #   │ logs_rl/.../sonic_release_test-             │
    #   │ 20260829_063238                             │
    # so the fragments after the label are concatenated until the row ends.
    for index, line in enumerate(lines):
        if "Logging Directory:" not in line:
            continue
        parts: list[str] = []
        for follow in lines[index + 1 : index + 6]:
            cell = follow.replace("│", "").strip()
            if not cell or cell[0] in "╰─╯" or ":" in cell:
                break
            parts.append(cell)
        joined = "".join(parts)
        if joined.startswith("logs_rl/"):
            return joined
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trailing(values: dict[int, float], window: int = 4) -> float | None:
    ordered = [values[index] for index in sorted(values)]
    return statistics.fmean(ordered[-window:]) if len(ordered) >= window else None


def aggregate(arms: dict[str, dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    result = {}
    for mode in modes:
        members = [arm for arm in arms.values() if arm["mode"] == mode]
        metrics = {}
        for metric in TRAINING_METRICS:
            values = [arm["training"][metric]["last4_mean"] for arm in members]
            values = [float(value) for value in values if value is not None]
            metrics[metric] = {
                "per_seed": {
                    str(arm["seed"]): arm["training"][metric]["last4_mean"] for arm in members
                },
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        for metric in QUALITY_METRICS:
            values = [arm["observer_last4_mean"].get(metric) for arm in members]
            values = [float(value) for value in values if value is not None]
            metrics[f"observer/{metric}"] = {
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        final_lambdas = [arm.get("final_lambda") for arm in members]
        final_lambdas = [float(value) for value in final_lambdas if value is not None]
        result[mode] = {
            "num_seeds": len(members),
            "metrics": metrics,
            "final_lambda_mean": statistics.fmean(final_lambdas) if final_lambdas else None,
            "all_complete": all(arm["complete"] for arm in members),
        }
    return result


def relative(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left == 0:
        return None
    return (right - left) / abs(left)


def comparisons(summary: dict[str, Any]) -> dict[str, Any]:
    if "lucid" not in summary:
        return {}
    output = {}
    for other in ("fixed", "off"):
        if other not in summary:
            continue
        pair = {}
        for metric, floor in (("Mean rewards", REWARD_FLOOR), ("Mean length", LENGTH_FLOOR)):
            lucid = summary["lucid"]["metrics"][metric]["mean"]
            reference = summary[other]["metrics"][metric]["mean"]
            delta = relative(reference, lucid)
            pair[metric] = {
                "lucid": lucid,
                other: reference,
                "relative_lucid_minus_other": delta,
                "outside_settled_noise_floor": delta is not None and abs(delta) > floor,
            }
        output[f"lucid_vs_{other}"] = pair
    return output


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"curriculum_comparison_ne{args.num_envs}_{stamp}"
    modes = list(dict.fromkeys(args.modes))
    source_receipt = (
        json.loads(args.yoked_source_receipt.read_text()) if args.yoked_source_receipt else None
    )
    for mode in modes:
        source = ARMS[mode][2]
        if source is not None and source not in modes and source_receipt is None:
            raise SystemExit(
                f"arm {mode!r} requires its source arm {source!r} in --modes or --yoked-source-receipt"
            )
    run_specs = []
    for seed_index, seed in enumerate(args.seeds):
        ordered = arm_order(modes, seed_index)
        # A yoked arm replays its source's schedule, so it must run after it.
        ordered = [m for m in ordered if ARMS[m][2] is None] + [m for m in ordered if ARMS[m][2]]
        for mode in ordered:
            branch_id = f"{experiment_id}_s{seed}_{mode}"
            artifact_dir = args.artifact_root / experiment_id / f"seed_{seed}" / mode
            run_specs.append((seed, mode, branch_id, artifact_dir))

    def schedule_for(seed: int, mode: str) -> Path | None:
        source = ARMS[mode][2]
        if source is None:
            return None
        source_seed = seed
        if mode in CROSS_SEED_ARMS:
            seeds = list(args.seeds)
            source_seed = seeds[(seeds.index(seed) + 1) % len(seeds)]
        if source_receipt is not None:
            for arm in source_receipt["arms"].values():
                if arm["mode"] == source and int(arm["seed"]) == source_seed:
                    return Path(arm["curriculum_path"])
            raise SystemExit(f"no {source!r} seed {source_seed} in {args.yoked_source_receipt}")
        source_branch = f"{experiment_id}_s{source_seed}_{source}"
        return (
            args.artifact_root
            / experiment_id
            / f"seed_{source_seed}"
            / source
            / f"curriculum_{source_branch}.jsonl"
        )

    commands = {
        branch_id: build_command(
            args, mode, seed, branch_id, artifact_dir, schedule_for(seed, mode)
        )
        for seed, mode, branch_id, artifact_dir in run_specs
    }
    for seed, mode, branch_id, _ in run_specs:
        print(f"[seed={seed} mode={mode} branch={branch_id}]")
        print("\n".join(commands[branch_id]))
    if not args.execute:
        print("dry run; pass --execute")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    runtime = {}
    arms: dict[str, dict[str, Any]] = {}
    for seed, mode, branch_id, artifact_dir in run_specs:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{branch_id}.log"
        runtime[branch_id] = LA.run_arm(commands[branch_id], log_path, args.min_free_mib)
        observer_path = artifact_dir / f"observer_{branch_id}.jsonl"
        arm = LA.summarize_arm(log_path, observer_path, args.iterations)
        curriculum_path = artifact_dir / f"curriculum_{branch_id}.jsonl"
        curriculum = read_jsonl(curriculum_path)
        capsule = artifact_dir / "capsules" / f"{branch_id}_final.capsule.pt"
        checkpoint = artifact_dir / "final_checkpoint.pt"
        if runtime[branch_id]["exit_code"] == 0 and capsule.is_file():
            BC.export_sonic_checkpoint(capsule, checkpoint)
        arm.update(
            {
                "seed": seed,
                "mode": mode,
                "branch_id": branch_id,
                "curriculum_path": str(curriculum_path),
                "curriculum_rows": len(curriculum),
                "final_lambda": curriculum[-1].get("lambda") if curriculum else None,
                "final_integral": curriculum[-1].get("integral") if curriculum else None,
                "mean_return_observed": any(
                    row.get("mean_return") is not None for row in curriculum
                ),
                "return_guard_trips": sum(bool(row.get("guard_tripped")) for row in curriculum),
                **(
                    {"ratchet_bind_rows": sum(bool(row.get("latch_active")) for row in curriculum)}
                    if mode in RATCHET_ARMS
                    else {}
                ),
                **(
                    {
                        # The safety claim for an expansion arm, computed from
                        # its own telemetry rather than asserted: applied
                        # support must never have moved down.
                        "final_frontier_lambda": (
                            curriculum[-1].get("frontier_lambda") if curriculum else None
                        ),
                        "expansions": sum(bool(row.get("fired")) for row in curriculum),
                        "applied_decreases": sum(
                            bool(row.get("applied_decrease")) for row in curriculum
                        ),
                        "max_frontier_drop": _max_frontier_drop(curriculum),
                        "probe_rows": sum(
                            row.get("probe_survival") is not None for row in curriculum
                        ),
                    }
                    if mode in EXPANSION_ARMS
                    else {}
                ),
                "scalable_terms": curriculum[-1].get("scalable_terms", []) if curriculum else [],
                "arm_spec": {
                    "curriculum_mode": ARMS[mode][0],
                    "anchor_ratio": ARMS[mode][1],
                    "anchor_seed": (
                        seed if ARMS[mode][1] > 0.0 or ARM_SPREAD_STRATA.get(mode, 1) > 1 else None
                    ),
                    "yoked_source": ARMS[mode][2],
                    "yoked_cross_seed": mode in CROSS_SEED_ARMS,
                    "term_lambda_overrides": ARM_TERM_OVERRIDES.get(mode, {}),
                    "run_dir": _logging_directory(log_path),
                    "spread_strata": ARM_SPREAD_STRATA.get(mode, 1),
                    "stratum_sizes": (
                        expand_arm_contract(mode, args.num_envs)["stratum_sizes"]
                        if mode in EXPAND_ARMS
                        else (
                            expansion_stratum_sizes(args.num_envs, ARM_SPREAD_STRATA[mode])
                            if mode in EXPANSION_ARMS
                            else None
                        )
                    ),
                    "stratum_lambdas": (
                        expand_arm_contract(mode, args.num_envs)["stratum_lambdas"]
                        if mode in EXPAND_ARMS
                        else (
                            curriculum[-1].get("tace", {}).get("stratum_lambdas")
                            if mode in EXPANSION_ARMS and curriculum
                            else None
                        )
                    ),
                    **(
                        {
                            "frontier_max": ARM_FRONTIER_MAX[mode],
                            "probe_offset": EXPANSION_STEP,
                            "probe_fraction": EXPANSION_PROBE_FRACTION,
                            "frontier_fraction": EXPANSION_FRONTIER_FRACTION,
                            "monotone_by_construction": True,
                            "signal": "survival" if mode in GATE_ARMS else "none",
                            "channel_ceilings": (
                                dict(ARM_BOX_CEILINGS[mode]) if mode in ARM_BOX_CEILINGS
                                else (dict(ARM_TERM_CAPS[mode]) if mode in ARM_TERM_CAPS else None)
                            ),
                            "gate": (
                                {
                                    "threshold": args.gate_threshold,
                                    "window": args.gate_window,
                                    "dwell": args.gate_dwell,
                                    "min_episodes": args.gate_min_episodes,
                                    "guard_action": args.gate_guard_action,
                                    "return_relative_drop": ARM_RETURN_DROP.get(
                                        mode, args.return_relative_drop
                                    ),
                                }
                                if mode in GATE_ARMS
                                else None
                            ),
                            "box": (
                                {
                                    "channel_budget": args.box_channel_budget,
                                    "frontier_vector_final": (
                                        curriculum[-1].get("frontier_vector") if curriculum else None
                                    ),
                                    "channel_expansions": (
                                        curriculum[-1].get("channel_expansions") if curriculum else None
                                    ),
                                }
                                if mode in BOX_ARMS
                                else None
                            ),
                            "ramp": (
                                {
                                    "start_lambda": 1.0,
                                    "end_lambda": ARM_FRONTIER_MAX[mode],
                                    "begin_iteration": args.ramp_begin_iteration,
                                    "end_iteration": args.ramp_end_iteration,
                                }
                                if mode in ("ramp_150", "ramp_asym")
                                else None
                            ),
                        }
                        if mode in EXPANSION_ARMS
                        else {}
                    ),
                    "top_fraction": ARM_TOP_FRACTION.get(mode),
                    "return_guard": ARM_RETURN_GUARD.get(mode, "absolute"),
                    **({"monotonic": True} if mode in RATCHET_ARMS else {}),
                    "fixed_lambda": ARM_FIXED_LAMBDA.get(mode, 1.0),
                    "allow_extrapolation": ARM_LAMBDA_CEILING.get(mode, 1.0) > 1.0,
                    "physical_clamp": (
                        curriculum[-1].get("physical_clamp") if curriculum else None
                    ),
                    "signal": "margin" if mode in MARGIN_ARMS else "gap",
                    "margin": (
                        {
                            "horizon": args.margin_horizon,
                            "band": [args.margin_band_lo, args.margin_band_hi],
                            "yardstick_envs": args.yardstick_envs,
                        }
                        if mode in MARGIN_ARMS
                        else None
                    ),
                    "term_lambda_caps": (
                        {"randomize_action_delay": args.latency_cap}
                        if mode in CAP_ARMS
                        else dict(ARM_TERM_CAPS.get(mode, {}))
                    ),
                    "max_delay_steps": args.max_delay,
                    "yoked_schedule_path": str(schedule_for(seed, mode)) if ARMS[mode][2] else None,
                },
                "tace_final": curriculum[-1].get("tace") if curriculum else None,
                "consolidation_rows": sum(bool(row.get("consolidation")) for row in curriculum),
                "capsule": str(capsule),
                "checkpoint": str(checkpoint),
                "checkpoint_exported": checkpoint.is_file(),
            }
        )
        if mode in EXPAND_ARMS:
            arm["expand_contract"] = validate_expand_arm_contract(
                mode,
                arm,
                curriculum,
                num_envs=args.num_envs,
                max_delay=args.max_delay,
            )
        arms[branch_id] = arm
        runtime[branch_id]["log_path"] = str(log_path)
        runtime[branch_id]["observer_path"] = str(observer_path)

    mode_summary = aggregate(arms, modes)
    comparison = comparisons(mode_summary)
    mechanics_ok = all(
        runtime[branch_id]["exit_code"] == 0
        and arm["complete"]
        and arm["actuator_groups_swapped"] == 5
        and arm["checkpoint_exported"]
        and set(arm["scalable_terms"]) == EXPECTED_SCALABLE_TERMS
        and (
            arm["mode"] not in EXPAND_ARMS or bool((arm.get("expand_contract") or {}).get("passed"))
        )
        for branch_id, arm in arms.items()
    )
    lucid_arms = [arm for arm in arms.values() if ARMS[arm["mode"]][0] == "lucid"]
    if lucid_arms:
        mechanics_ok = mechanics_ok and all(arm["mean_return_observed"] for arm in lucid_arms)
    for arm in arms.values():
        ratio = ARMS[arm["mode"]][1]
        if ratio > 0.0:
            tace = arm.get("tace_final") or {}
            expected = round(ratio * args.num_envs)
            mechanics_ok = mechanics_ok and tace.get("num_anchor") == expected

    receipt = {
        "kind": "lucid_three_arm_training_comparison",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "git_sha": TP.git_sha(),
        "git_status_short": TP.git_status(),
        "launcher_sha256": source_sha256(),
        "config": {
            "checkpoint": (None if args.from_scratch else str(Path(args.checkpoint).resolve())),
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "seeds": args.seeds,
            "modes": modes,
            "arm_order": [
                {"seed": seed, "modes": arm_order(modes, index)}
                for index, seed in enumerate(args.seeds)
            ],
            "event_preset": "tracking/lucid_curriculum",
            "termination_preset": (
                args.terminations or "tracking/base_adaptive_strict_ori_foot_xyz (exp default)"
            ),
            "termination_thresholds": args.termination_thresholds,
            "randomization_note": (
                "lambda scales EVENT-MANAGER terms only. The motion command term applies "
                "reset randomization on every training reset regardless of lambda -- root "
                "velocity +-0.5 m/s in x and y, +-0.78 rad/s yaw, pelvis +-0.05 m, joints "
                "+-0.1 rad -- because dr_scaling never touches the command manager. An arm "
                "at lambda=0 is 'no event-manager DR', NOT 'no randomization'."
            ),
            "wandb_project": args.wandb_project,
            "from_scratch": bool(args.from_scratch),
            "capsule_horizons": sorted(set(args.horizons or ())) + [args.iterations],
            "motion_file": args.motion_file,
            "smpl_motion_file": args.smpl_motion_file,
            "arms": {mode: ARMS[mode] for mode in modes},
            "consolidation_fraction": args.consolidation_fraction,
            "max_delay_steps": args.max_delay,
            "max_delay_ms": args.max_delay * 5,
            "controller": {
                "delta_target": args.delta_target,
                "kp": args.kp,
                "ki": args.ki,
                "alpha": args.alpha,
                "integral_max": args.integral_max,
                "return_floor": args.return_floor,
                "calibration": (
                    "manuscript mu+3sigma lambda=0 target; integral contribution "
                    "capped at ki*integral_max=0.02"
                ),
            },
            "training_noise_floors": {"reward": REWARD_FLOOR, "length": LENGTH_FLOOR},
        },
        "commands": commands,
        "runtime": runtime,
        "arms": arms,
        "mode_summary": mode_summary,
        "training_comparison": comparison,
        "verified": (
            [
                "every branch completed and exported a final SONIC-compatible checkpoint",
                "all five live actuator groups used delayed actuators",
                "all six DR channels were runtime-scalable",
                "LUCID received SONIC objective/rewards for its return guard",
            ]
            if mechanics_ok
            else []
        ),
        "not_yet_verified": [
            "held-out ID-clean, OOD-heavy, and 60 ms checkpoint evaluation",
            "training curves alone do not establish final policy generalization",
            *("three-seed confirmation" for _ in [0] if len(args.seeds) < 3),
        ],
    }
    receipt_path = args.receipt_dir / f"{experiment_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"mode_summary": mode_summary, "training_comparison": comparison}, indent=2))
    print(f"receipt {receipt_path}")
    return 0 if mechanics_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
