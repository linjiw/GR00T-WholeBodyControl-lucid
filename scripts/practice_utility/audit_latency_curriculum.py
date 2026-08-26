#!/usr/bin/env python3
"""Audit the realized latency curriculum in an existing SONIC training receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = Path(
    "/data/robotixx/lucid-sonic/manifests/" "curriculum_comparison_ne128_20260820_143058.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/manifests"),
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_energy = sum((x - x_mean) ** 2 for x in xs)
    y_energy = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_energy * y_energy)
    return numerator / denominator if denominator else None


def observer_path(arm: dict[str, Any]) -> Path:
    candidates = sorted(Path(arm["curriculum_path"]).parent.glob("observer_*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"expected one observer log beside curriculum, got {candidates}")
    return candidates[0]


def applied_lambda_by_step(
    mode: str,
    curriculum_rows: list[dict[str, Any]],
    observer_rows: list[dict[str, Any]],
) -> dict[int, float]:
    """Align end-of-iteration lambda updates to the following rollout."""
    initial = 1.0 if mode == "fixed" else 0.0
    updated = {int(row["global_step"]): float(row["lambda"]) for row in curriculum_rows}
    return {
        int(row["global_step"]): updated.get(int(row["global_step"]) - 1, initial)
        for row in observer_rows
    }


def audit_arm(arm: dict[str, Any], max_delay: int, warmup: int) -> dict[str, Any]:
    curriculum_path = Path(arm["curriculum_path"])
    obs_path = observer_path(arm)
    curriculum = load_jsonl(curriculum_path)
    observer = load_jsonl(obs_path)
    applied = applied_lambda_by_step(arm["mode"], curriculum, observer)
    lambdas, means = [], []
    running_upper = 0
    envelope_violations = []
    for row in observer:
        step = int(row["global_step"])
        lam = applied[step]
        running_upper = max(running_upper, int(round(max_delay * lam)))
        observed_max = int(row.get("action_delay_max_steps", -1))
        if observed_max > running_upper:
            envelope_violations.append(
                {"global_step": step, "observed_max": observed_max, "running_upper": running_upper}
            )
        lambdas.append(lam)
        means.append(float(row.get("action_delay_mean_steps", 0.0)))

    first_warmup = [row for row in observer if int(row["global_step"]) <= warmup]
    last4 = observer[-4:]
    return {
        "seed": int(arm["seed"]),
        "mode": arm["mode"],
        "curriculum_path": str(curriculum_path),
        "curriculum_sha256": sha256(curriculum_path),
        "observer_path": str(obs_path),
        "observer_sha256": sha256(obs_path),
        "curriculum_rows": len(curriculum),
        "observer_rows": len(observer),
        "delayed_actuator_groups": int(arm["live_delay_final"]["action_delay_actuator_groups"]),
        "scalable_terms": arm["scalable_terms"],
        "warmup_all_zero": all(
            int(row.get("action_delay_max_steps", -1)) == 0 for row in first_warmup
        ),
        "observed_max_steps": max(int(row.get("action_delay_max_steps", -1)) for row in observer),
        "last4_mean_delay_steps": statistics.fmean(
            float(row.get("action_delay_mean_steps", 0.0)) for row in last4
        ),
        "lambda_delay_mean_pearson": pearson(lambdas, means),
        "envelope_violations": envelope_violations,
        "final_lambda": float(arm["final_lambda"]),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    source = json.loads(args.training_receipt.read_text())
    max_delay = int(source["config"]["max_delay_steps"])
    warmup = int(source["config"]["warmup_iterations"])
    arms = [audit_arm(arm, max_delay, warmup) for arm in source["arms"].values()]
    modes = {
        mode: [arm for arm in arms if arm["mode"] == mode] for mode in ("lucid", "fixed", "off")
    }
    expected_terms = {
        "add_joint_default_pos",
        "base_com",
        "physics_material",
        "push_robot",
        "randomize_action_delay",
        "randomize_rigid_body_mass",
    }
    mechanics_ok = (
        len(arms) == 9
        and all(arm["delayed_actuator_groups"] == 5 for arm in arms)
        and all(set(arm["scalable_terms"]) == expected_terms for arm in arms)
        and all(not arm["envelope_violations"] for arm in arms)
        and all(arm["warmup_all_zero"] for arm in modes["lucid"] + modes["off"])
        and all(arm["observed_max_steps"] == 0 for arm in modes["off"])
        and all(arm["observed_max_steps"] == max_delay for arm in modes["fixed"])
        and all((arm["lambda_delay_mean_pearson"] or 0.0) > 0.8 for arm in modes["lucid"])
    )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt_dir / f"latency_curriculum_audit_{stamp}.json"
    payload = {
        "kind": "lucid_latency_curriculum_audit",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_training_receipt": str(args.training_receipt.resolve()),
        "source_training_receipt_sha256": sha256(args.training_receipt),
        "source_experiment_id": source["experiment_id"],
        "training_contract": {
            "physics_step_ms": 5,
            "delay_envelope_steps": [0, max_delay],
            "delay_envelope_ms": [0, 5 * max_delay],
            "event_mode": "reset",
            "draw_scope": "per environment and per episode reset",
            "actuator_group_coupling": "independent (five separately sampled groups)",
            "within_episode_resampling": False,
            "curriculum_application": (
                "lambda rescales delay_range immediately; realized lags change when each environment "
                "next resets, so current buffers can contain a mixture of recent lambda values"
            ),
            "reset_order": (
                "IsaacLab scene/actuator reset first samples the buffer capacity range; the reset-mode "
                "LUCID event then overwrites it with the current curriculum-scaled draw"
            ),
            "discrete_quantization": "upper step is round(8 * lambda)",
        },
        "arms": arms,
        "cross_seed": {
            mode: {
                "last4_mean_delay_steps": statistics.fmean(
                    arm["last4_mean_delay_steps"] for arm in members
                ),
                "final_lambda_mean": statistics.fmean(arm["final_lambda"] for arm in members),
            }
            for mode, members in modes.items()
        },
        "verified": (
            [
                "all nine arms recorded five live delayed-actuator groups",
                "all six curriculum terms were runtime-scalable",
                "off remained zero-delay and fixed reached the complete 0-8-step envelope",
                "LUCID warmup remained zero-delay and realized mean delay tracked lambda",
                "no realized lag exceeded the curriculum envelope available up to that rollout",
            ]
            if mechanics_ok
            else []
        ),
        "limitations": [
            "training latency was piecewise constant within an episode, not temporal jitter",
            "actuator groups were independently delayed rather than sharing one pipeline lag",
            "the observer logged snapshots each PPO iteration, not every reset draw",
        ],
    }
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["cross_seed"], indent=2))
    print(f"receipt {receipt_path}")
    return 0 if payload["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
