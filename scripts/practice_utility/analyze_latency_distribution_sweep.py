#!/usr/bin/env python3
"""Analyze a preregistered latency discovery sweep and holdout confirmation."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DISCOVERY = Path(
    "/data/robotixx/lucid-sonic/manifests/"
    "latency_distribution_discovery_ne32_20260820_232545.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-receipt", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation-receipt", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260821)
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path("/data/robotixx/lucid-sonic/manifests"),
    )
    return parser.parse_args(argv)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def primary(row: dict[str, Any], mode: str, metric: str) -> float:
    return float(row["modes"][mode]["metrics"][metric]["mean"])


def surface(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell_id, row in discovery["cell_summary"].items():
        result = {
            "cell_id": cell_id,
            "cell": row["cell"],
            "success": {
                mode: primary(row, mode, "success_rate") for mode in ("lucid", "fixed", "off")
            },
            "progress": {
                mode: primary(row, mode, "progress_rate") for mode in ("lucid", "fixed", "off")
            },
        }
        result["lucid_min_success_margin"] = min(
            result["success"]["lucid"] - result["success"][reference]
            for reference in ("fixed", "off")
        )
        result["lucid_min_progress_margin"] = min(
            result["progress"]["lucid"] - result["progress"][reference]
            for reference in ("fixed", "off")
        )
        rows.append(result)
    return sorted(rows, key=lambda row: row["cell_id"])


def absolute_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in ("lucid", "fixed", "off"):
        result[mode] = {}
        for metric in ("success", "progress"):
            best = max(row[metric][mode] for row in rows)
            result[mode][metric] = {
                "value": best,
                "cell_ids": [row["cell_id"] for row in rows if row[metric][mode] == best],
            }
    return result


def confirmation_runs(
    receipt: dict[str, Any], cell_id: str
) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for run in receipt["runs"].values():
        if run["cell_id"] == cell_id and run["complete"]:
            grouped.setdefault(int(run["checkpoint_seed"]), {})[run["mode"]] = run
    return grouped


def paired_seed_deltas(
    grouped: dict[int, dict[str, dict[str, Any]]], reference: str, metric: str
) -> dict[int, float]:
    return {
        seed: float(modes["lucid"]["summary"][metric]) - float(modes[reference]["summary"][metric])
        for seed, modes in grouped.items()
        if "lucid" in modes and reference in modes
    }


def sign_flip_pvalue(deltas: list[float]) -> float | None:
    """Exact two-sided seed-block sign-flip test (minimum p is 0.25 at n=3)."""
    if not deltas:
        return None
    observed = abs(statistics.fmean(deltas))
    null = [
        abs(statistics.fmean(sign * value for sign, value in zip(signs, deltas)))
        for signs in itertools.product((-1, 1), repeat=len(deltas))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def hierarchical_bootstrap(
    grouped: dict[int, dict[str, dict[str, Any]]],
    reference: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample checkpoint-seed blocks, then motions within each selected block."""
    rng = random.Random(seed)
    seeds = sorted(grouped)
    samples = []
    outcome_key = "success" if metric == "success_rate" else "progress"
    for _ in range(replicates):
        seed_deltas = []
        for sampled_seed in (rng.choice(seeds) for _ in seeds):
            modes = grouped[sampled_seed]
            lucid = modes["lucid"]["summary"]["motion_outcomes"]
            other = modes[reference]["summary"]["motion_outcomes"]
            keys = sorted(set(lucid) & set(other))
            motion_deltas = []
            for _ in keys:
                key = rng.choice(keys)
                motion_deltas.append(
                    float(lucid[key][outcome_key]) - float(other[key][outcome_key])
                )
            seed_deltas.append(statistics.fmean(motion_deltas))
        samples.append(statistics.fmean(seed_deltas))
    ordered = sorted(samples)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {
        "method": "hierarchical percentile bootstrap over checkpoint seeds and motions",
        "replicates": replicates,
        "analysis_seed": seed,
        "ci95": [lower, upper],
        "probability_positive": sum(value > 0 for value in samples) / len(samples),
    }


def analyze_confirmation(
    receipt: dict[str, Any], cell_id: str, replicates: int, analysis_seed: int
) -> dict[str, Any]:
    grouped = confirmation_runs(receipt, cell_id)
    summary = receipt["cell_summary"][cell_id]
    comparisons = {}
    for reference_index, reference in enumerate(("fixed", "off")):
        comparisons[reference] = {}
        for metric_index, metric in enumerate(("success_rate", "progress_rate")):
            deltas = paired_seed_deltas(grouped, reference, metric)
            values = list(deltas.values())
            comparisons[reference][metric] = {
                "per_checkpoint_seed": deltas,
                "mean_delta": statistics.fmean(values),
                "favorable_seed_count": sum(value > 0 for value in values),
                "exact_seed_block_sign_flip_p_two_sided": sign_flip_pvalue(values),
                "bootstrap": hierarchical_bootstrap(
                    grouped,
                    reference,
                    metric,
                    replicates,
                    analysis_seed + 10 * reference_index + metric_index,
                ),
            }
    directional_replication = all(
        comparisons[reference]["success_rate"]["mean_delta"] >= 0
        and comparisons[reference]["progress_rate"]["mean_delta"] > 0
        and comparisons[reference]["progress_rate"]["favorable_seed_count"] >= 2
        for reference in ("fixed", "off")
    )
    return {
        "cell_id": cell_id,
        "mode_metrics": {
            mode: {
                metric: summary["modes"][mode]["metrics"][metric]
                for metric in ("success_rate", "progress_rate")
            }
            for mode in ("lucid", "fixed", "off")
        },
        "comparisons": comparisons,
        "directional_replication": directional_replication,
        "inference_limit": (
            "Only three independent checkpoint-seed blocks are available; the exact two-sided "
            "sign-flip test cannot attain p < 0.25. Motion-level bootstrap intervals are descriptive."
        ),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    discovery, confirmation = load(args.discovery_receipt), load(args.confirmation_receipt)
    if not discovery.get("verified") or not confirmation.get("verified"):
        raise ValueError("both sweep stages must be complete and mechanism-verified")
    selected = discovery["selected_confirmation_cells"]
    if not selected:
        raise ValueError("discovery selected no confirmation cells")
    if confirmation["protocol"]["discovery_receipt"] != str(args.discovery_receipt.resolve()):
        raise ValueError("confirmation does not link to the supplied discovery receipt")
    discovery_groups = set(discovery["protocol"]["panel"]["canonical_groups"])
    confirmation_groups = set(confirmation["protocol"]["panel"]["canonical_groups"])
    if discovery_groups & confirmation_groups:
        raise ValueError("discovery and confirmation panels overlap by canonical content")

    rows = surface(discovery)
    confirmations = {
        cell_id: analyze_confirmation(
            confirmation, cell_id, args.bootstrap_replicates, args.analysis_seed
        )
        for cell_id in selected
    }
    payload = {
        "kind": "lucid_latency_distribution_analysis",
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "discovery_receipt": str(args.discovery_receipt.resolve()),
        "confirmation_receipt": str(args.confirmation_receipt.resolve()),
        "discovery_panel": {
            "motions": discovery["protocol"]["panel"]["motion_count"],
            "canonical_groups": len(discovery_groups),
        },
        "confirmation_panel": {
            "motions": confirmation["protocol"]["panel"]["motion_count"],
            "canonical_groups": len(confirmation_groups),
        },
        "full_discovery_surface": rows,
        "absolute_best_discovery_conditions": absolute_best(rows),
        "relative_lucid_ranking": sorted(
            (
                {
                    "cell_id": row["cell_id"],
                    "min_success_margin": row["lucid_min_success_margin"],
                    "min_progress_margin": row["lucid_min_progress_margin"],
                }
                for row in rows
            ),
            key=lambda row: (row["min_success_margin"], row["min_progress_margin"]),
            reverse=True,
        ),
        "selected_confirmation_cells": selected,
        "confirmation": confirmations,
        "verified": [
            "the complete preregistered discovery surface is reported",
            "confirmation used the mechanically selected cells and disjoint canonical content",
            "confirmation comparisons are paired by checkpoint and evaluation seed",
            "all source stages passed latency-process telemetry and checkpoint-freeze checks",
        ],
        "claim_boundary": (
            "fresh-physics robustness on training-seen debug512 motions; not unseen-motion, hardware, "
            "or real-world latency-distribution evidence"
        ),
    }
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    output = args.receipt_dir / f"latency_distribution_analysis_{stamp}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(confirmations, indent=2))
    print(f"receipt {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
