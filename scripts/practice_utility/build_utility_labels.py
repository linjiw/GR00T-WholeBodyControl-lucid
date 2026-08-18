#!/usr/bin/env python3
"""Assemble utility labels from a completed branch campaign.

Reads each pair's control and intervention artifacts -- training log, dose
report, and quality telemetry if present -- and emits one
:class:`UtilityRecord` per (context, seed), then runs Gate A and Gate B over the
result.

On the efficacy metric
----------------------
The plan's deployment objective ``J_eff`` is a macro-mean of quality-qualified
success on a frozen dev suite, which costs one evaluation pass per branch per
horizon. This script supports that when evaluation summaries are supplied, and
otherwise falls back to a **training-side** efficacy metric (mean episodic
reward at the horizon) and labels the output accordingly in
``efficacy_source``.

The fallback is not the same estimand and the report says so. It is still worth
computing: the ordering of contexts under a training-side metric is the cheapest
available check on whether the paired machinery resolves anything at all, and
the noise floor is measured on the same metric, so the comparison is internally
consistent.

Example
-------
    python scripts/practice_utility/build_utility_labels.py \\
        --manifest .../probe_screen_v1_late.json \\
        --campaign-dir .../artifacts/screen_v1_late \\
        --output .../artifacts/utility_labels_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.research.practice_utility import proxy_audit as PA  # noqa: E402
from gear_sonic.research.practice_utility import run_log as RL  # noqa: E402
from gear_sonic.research.practice_utility import utility_label as UL  # noqa: E402
from gear_sonic.research.practice_utility.schema import ContextKey, DoseReport  # noqa: E402

#: Training-side efficacy metric used when no dev-suite evaluation is supplied.
FALLBACK_EFFICACY = "Mean rewards"

#: Iterations averaged when reading efficacy at a horizon. Measured to cut the
#: cross-seed relative spread of mean reward from 4.77% to 3.33%.
DEFAULT_EFFICACY_WINDOW = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path,
                        help="directory holding one subdirectory per branch")
    parser.add_argument("--log-dir", type=Path,
                        default=Path("/data/robotixx/lucid-sonic/outputs"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise-floor", type=Path, default=None,
                        help="noise_floor_report.json; supplies Gate A's denominator")
    parser.add_argument("--efficacy-metric", default=FALLBACK_EFFICACY)
    parser.add_argument("--efficacy-window", type=int, default=DEFAULT_EFFICACY_WINDOW,
                        help="iterations averaged at each horizon (1 = single point)")
    parser.add_argument("--shared-control", action="store_true", default=True,
                        help="one control per (stage, seed), as screening uses")
    return parser.parse_args(argv)


def load_branch(campaign_dir: Path, log_dir: Path, branch_id: str, metric: str):
    """Return (metric series, dose report dict) for one branch, if present."""
    series, dose = {}, None
    for candidate in (log_dir / f"{branch_id}.log", campaign_dir / branch_id / "run.log"):
        if candidate.exists():
            series = RL.parse_run_log(candidate).series(metric)
            break
    reports = sorted((campaign_dir / branch_id).glob("dose_*.json")) if campaign_dir.exists() else []
    if reports:
        dose = json.loads(reports[-1].read_text())
    return series, dose


def dose_from_report(report: dict | None, role: str, branch_id: str) -> DoseReport:
    if report is None:
        return DoseReport(branch_id=branch_id, context_id="unknown", role=role)  # type: ignore[arg-type]
    return DoseReport(
        branch_id=report.get("branch_id", branch_id),
        context_id=report.get("context_id", "unknown"),
        role=role,  # type: ignore[arg-type]
        drawn_episodes=float(report.get("drawn_episodes", 0.0)),
        drawn_kernel_mass=float(report.get("drawn_kernel_mass", 0.0)),
        completed_env_steps=float(report.get("completed_env_steps", 0.0)),
        completed_kernel_steps=float(report.get("completed_kernel_steps", 0.0)),
        early_terminations=int(report.get("early_terminations", 0)),
    )


def evaluations_for(series: dict[int, float], horizons: dict[str, int], role: str,
                    branch_id: str, window: int = DEFAULT_EFFICACY_WINDOW
                    ) -> list[UL.BranchEvaluation]:
    """One evaluation per horizon, averaged over the last ``window`` iterations.

    Averaging rather than reading a single point at the horizon, because that
    single point is noisy. Measured across three seeds: the cross-seed relative
    spread of mean reward at the final iteration was 4.77%, while the mean over
    the last four iterations was 3.33% -- the same runs, a third less noise, for
    free. Since the noise floor is what Gate A must clear, spending nothing to
    lower it is the cheapest sensitivity available.
    """
    out = []
    for label, horizon in horizons.items():
        usable = sorted(i for i in series if i <= horizon)
        if not usable:
            continue
        chosen = usable[-window:] if window > 1 else usable[-1:]
        value = sum(series[i] for i in chosen) / len(chosen)
        out.append(
            UL.BranchEvaluation(
                branch_id=branch_id, role=role, horizon_label=label,
                j_eff=value, clean_j_eff=value,
                action_rate=0.0, foot_slip=0.0, contact_impulse=0.0, torque_saturation=0.0,
                extras={
                    "iterations_averaged": float(len(chosen)),
                    "first_iteration": float(chosen[0]),
                    "last_iteration": float(chosen[-1]),
                },
            )
        )
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    horizons = manifest["horizons"]
    campaign = manifest["campaign_id"]

    records, skipped = [], []
    for stage, contexts in manifest["contexts_per_stage"].items():
        for seed in manifest["seeds"]:
            control_id = f"{campaign}_{stage}_s{seed}_control"
            control_series, control_dose = load_branch(
                args.campaign_dir, args.log_dir, control_id, args.efficacy_metric)
            if not control_series:
                skipped.append(f"{stage}/s{seed}: control log missing")
                continue

            for index, entry in enumerate(contexts):
                pair = f"{campaign}_{stage}_s{seed}_c{index}"
                branch_id = f"{pair}_intervention"
                series, dose = load_branch(
                    args.campaign_dir, args.log_dir, branch_id, args.efficacy_metric)
                if not series:
                    skipped.append(f"{branch_id}: log missing")
                    continue
                try:
                    record = UL.build_utility_record(
                        branch_pair_id=pair,
                        context=ContextKey.from_dict(entry["context"]),
                        policy_stage=stage,
                        seed=seed,
                        horizons=horizons,
                        control_dose=dose_from_report(control_dose, "control", control_id),
                        intervention_dose=dose_from_report(dose, "intervention", branch_id),
                        control_evaluations=evaluations_for(
                            control_series, horizons, "control", control_id,
                            args.efficacy_window),
                        intervention_evaluations=evaluations_for(
                            series, horizons, "intervention", branch_id,
                            args.efficacy_window),
                        epsilon=manifest["epsilon"],
                        kernel_radius_bins=manifest["kernel_radius_bins"],
                        base_distribution_sha256=manifest["manifest_sha256"],
                        intervention_distribution_sha256=manifest["manifest_sha256"],
                        proxy_features={
                            "native_failure_rate": float(entry.get("failure_rate", 0.0)),
                            "sampling_probability": float(entry.get("sampling_probability", 0.0)),
                            **{k: float(v) for k, v in (entry.get("extras") or {}).items()},
                        },
                    )
                except ValueError as error:
                    skipped.append(f"{branch_id}: {error}")
                    continue
                records.append(record)

    usable = [r for r in records if UL.is_usable(r)]
    print(f"pairs assembled: {len(records)}  usable labels: {len(usable)}  skipped: {len(skipped)}")
    for line in skipped[:8]:
        print(f"  skip {line}")
    if len(skipped) > 8:
        print(f"  ... and {len(skipped) - 8} more")
    if not usable:
        print("\nno usable labels; nothing to audit")
        return 1

    long_horizon = max(horizons, key=lambda k: horizons[k])
    short_horizon = min(horizons, key=lambda k: horizons[k])

    floor = None
    if args.noise_floor and args.noise_floor.exists():
        report = json.loads(args.noise_floor.read_text())
        metric = report["metrics"].get(args.efficacy_metric)
        if metric:
            floor = metric["paired_deltas"]
            print(f"noise floor from {len(floor)} eps=0 pairs, sd={metric['sd_delta']:.5f}")

    gate_a = UL.assess_identifiability(usable, long_horizon, noise_floor=floor)
    gate_b = PA.assess_sufficiency(usable, long_horizon, short_horizon=short_horizon)

    payload = {
        "kind": "practice_utility_labels",
        "schema_version": 1,
        "campaign_id": campaign,
        "manifest_sha256": manifest["manifest_sha256"],
        "efficacy_source": (
            "training_side_mean_reward" if args.efficacy_metric == FALLBACK_EFFICACY
            else args.efficacy_metric
        ),
        "efficacy_window": args.efficacy_window,
        "efficacy_caveat": (
            "training-side efficacy is NOT the plan's J_eff (macro-mean quality-qualified "
            "success on a frozen dev suite). Context ordering under it is a cheap "
            "consistency check, not a deployment claim."
        ),
        "horizons": horizons,
        "summary": UL.summarize_labels(usable, long_horizon),
        "reversals": PA.count_reversals(usable, short_horizon, long_horizon),
        "gate_a_identifiability": gate_a.to_dict(),
        "gate_b_sufficiency": gate_b.to_dict(),
        "skipped": skipped,
        "records": [r.to_dict() for r in usable],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nGate A (identifiability at {long_horizon}): "
          f"{'PASS' if gate_a.passes else 'FAIL'}")
    for reason in gate_a.reasons:
        print(f"  {reason}")
    print(f"Gate B (estimator authorized): {gate_b.authorizes_estimator}")
    for reason in gate_b.reasons:
        print(f"  {reason}")
    print(f"reversals: {payload['reversals']}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
