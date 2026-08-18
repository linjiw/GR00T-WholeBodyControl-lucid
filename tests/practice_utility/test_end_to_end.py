"""End-to-end test of the CPU measurement chain.

Runs the whole pipeline on synthetic data with a *known* ground truth:

    scan pool -> split -> probe manifest -> simulated paired branches
             -> utility labels -> Gate A -> Gate B

Unit tests prove each stage works. This proves they compose, and -- more
usefully -- that the chain reaches the *right* verdict in both directions. Two
worlds are simulated:

* one where difficulty genuinely predicts utility, in which Gate B must refuse
  to authorize an estimator; and
* one where utility depends on something difficulty cannot see, in which Gate B
  must authorize it.

A pipeline that only ever says "build the estimator" would be worthless.
"""

import numpy as np
import pytest

from gear_sonic.research.practice_utility import motion_pool as MP
from gear_sonic.research.practice_utility import probe_manifest as PM
from gear_sonic.research.practice_utility import proxy_audit as PA
from gear_sonic.research.practice_utility import split as SP
from gear_sonic.research.practice_utility import utility_label as UL
from gear_sonic.research.practice_utility.schema import ContextKey, DoseReport, motion_hash

FAMILIES = ("walk", "run", "jump", "crawl", "dance", "carry")


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """120 clips over 40 performers, six families, no duplicates."""
    directory = tmp_path_factory.mktemp("pool")
    payload = {}
    for index in range(120):
        family = FAMILIES[index % len(FAMILIES)]
        key = f"{family}_move{index:03d}_00{index % 3}__A{index % 40:03d}"
        rng = np.random.default_rng(index)
        payload[f"{key}.pkl"] = {
            key: {"dof": rng.standard_normal((90, 29)).astype("float32"), "fps": 30}
        }
        (directory / f"{key}.pkl").write_bytes(b"")

    def loader(path):
        return payload[str(path).rsplit("/", 1)[-1]]

    return MP.scan_pool(directory, loader=loader)


def candidates_from(scan, split, stage_seed):
    """Adaptation-partition contexts, with a stage-dependent failure rate."""
    rng = np.random.default_rng(stage_seed)
    candidates = []
    for record in scan.records:
        if split.assignment[record.motion_key] != "adaptation":
            continue
        candidates.append(
            PM.ContextCandidate(
                context=ContextKey(
                    motion_key=record.motion_key,
                    motion_hash=record.content_sha256,
                    bin_index=1, bin_start_frame=50, bin_end_frame=100,
                ),
                failure_rate=float(rng.uniform(0.0, 1.0)),
                sampling_probability=0.01,
                family=record.family,
                contact_regime="rich" if record.family in ("crawl", "jump") else "light",
                partition="adaptation",
            )
        )
    return candidates


def simulate_branches(manifest, utility_of, noise=0.002, seed=0):
    """Turn a frozen manifest into utility labels under a chosen ground truth."""
    rng = np.random.default_rng(seed)
    records = []
    for stage, candidates in manifest.contexts_per_stage.items():
        for candidate in candidates:
            for branch_seed in manifest.seeds:
                true_utility = utility_of(candidate)
                control_j, kernel_gain = 0.50, 500.0
                delta = true_utility * kernel_gain + rng.normal(0.0, noise)

                horizons = manifest.horizons
                control = [
                    UL.BranchEvaluation(f"{stage}_c", "control", h, control_j, 0.80,
                                        0.010, 0.010, 50.0, 0.010)
                    for h in horizons
                ]
                treated = [
                    UL.BranchEvaluation(f"{stage}_i", "intervention", h, control_j + delta,
                                        0.80, 0.010, 0.010, 50.0, 0.010)
                    for h in horizons
                ]
                records.append(
                    UL.build_utility_record(
                        branch_pair_id=f"{stage}_{candidate.context.context_id}_{branch_seed}",
                        context=candidate.context,
                        policy_stage=stage,
                        seed=branch_seed,
                        horizons=horizons,
                        control_dose=DoseReport("c", "ctx", "control",
                                                completed_env_steps=10_000.0,
                                                completed_kernel_steps=1_000.0),
                        intervention_dose=DoseReport("i", "ctx", "intervention",
                                                     completed_env_steps=10_000.0,
                                                     completed_kernel_steps=1_000.0 + kernel_gain),
                        control_evaluations=control,
                        intervention_evaluations=treated,
                        epsilon=manifest.epsilon,
                        kernel_radius_bins=manifest.kernel_radius_bins,
                        base_distribution_sha256="a" * 64,
                        intervention_distribution_sha256="b" * 64,
                        proxy_features={
                            "native_failure_rate": candidate.failure_rate,
                            "latent_gap_p90": candidate.failure_rate * 0.5 + 0.1,
                        },
                    )
                )
    return records


@pytest.fixture(scope="module")
def frozen(pool):
    """Pool -> split -> frozen probe manifest, as a campaign would do it."""
    pool_sha = MP.pool_sha256(pool)
    split = SP.build_split(pool, pool_sha, linkage="performer")
    manifest = PM.build_probe_manifest(
        campaign_id="e2e",
        candidates_per_stage={
            stage: candidates_from(pool, split, seed)
            for seed, stage in enumerate(("early", "middle", "late"))
        },
        num_contexts=12,
        seeds=[0, 1],
        horizons={"H_s": 8, "H_l": 128},
        pool_sha256=pool_sha,
        split_sha256=split.split_sha256,
    )
    return pool, split, manifest


class TestChainComposes:
    def test_split_is_disjoint_and_on_target(self, frozen):
        _, split, _ = frozen
        assert sum(len(split.partition(p)) for p in SP.DEFAULT_RATIOS) == 120
        assert split.stats["partitions"]["adaptation"]["share"] == pytest.approx(0.6, abs=0.1)

    def test_manifest_only_probes_the_adaptation_split(self, frozen):
        _, split, manifest = frozen
        for candidates in manifest.contexts_per_stage.values():
            for candidate in candidates:
                assert split.assignment[candidate.context.motion_key] == "adaptation"

    def test_manifest_spans_the_difficulty_range(self, frozen):
        _, _, manifest = frozen
        for stage in manifest.stages:
            quartiles = {c.failure_quartile for c in manifest.contexts_per_stage[stage]}
            assert quartiles == {0, 1, 2, 3}

    def test_branch_budget_is_reported(self, frozen):
        _, _, manifest = frozen
        assert manifest.num_branches == 3 * 12 * 2
        assert manifest.num_control_branches == 3 * 2

    def test_labels_are_produced_for_every_pair(self, frozen):
        _, _, manifest = frozen
        records = simulate_branches(manifest, lambda c: 0.001 * c.failure_rate)
        assert len(records) == manifest.num_branches
        assert all(UL.is_usable(r) for r in records)


class TestVerdictWhenDifficultyPredictsUtility:
    """Ground truth: utility is a clean function of failure rate."""

    @pytest.fixture(scope="class")
    def records(self, frozen):
        _, _, manifest = frozen
        return simulate_branches(manifest, lambda c: 0.001 * c.failure_rate, noise=0.0005)

    def test_gate_a_passes(self, records):
        report = UL.assess_identifiability(records, "H_l")
        assert report.passes is True

    def test_gate_b_refuses_to_authorize_an_estimator(self, records):
        """The outcome that would strengthen the existing sampler."""
        report = PA.assess_sufficiency(records, "H_l", short_horizon="H_s")
        assert report.authorizes_estimator is False
        assert "native_failure_rate" in report.sufficient_proxies

    def test_the_proxy_scores_well(self, records):
        result = PA.audit_proxy(records, "native_failure_rate", "H_l")
        assert result.spearman > 0.9
        assert result.pairwise_accuracy > 0.9


class TestVerdictWhenUtilityIsHiddenFromDifficulty:
    """Ground truth: utility depends on family, which failure rate cannot see."""

    @pytest.fixture(scope="class")
    def records(self, frozen):
        _, _, manifest = frozen
        bonus = {"crawl": 3.0, "jump": 2.0, "dance": -2.0, "carry": -3.0}

        def utility_of(candidate):
            return 0.001 * bonus.get(candidate.family, 0.0)

        return simulate_branches(manifest, utility_of, noise=0.0005, seed=7)

    def test_gate_a_still_passes(self, records):
        """Utility is measurable; it is simply not what difficulty measures."""
        assert UL.assess_identifiability(records, "H_l").passes is True

    def test_difficulty_fails_to_predict(self, records):
        result = PA.audit_proxy(records, "native_failure_rate", "H_l")
        assert abs(result.spearman) < 0.5
        assert result.is_sufficient is False

    def test_gate_b_authorizes_the_estimator(self, records):
        report = PA.assess_sufficiency(records, "H_l", short_horizon="H_s")
        assert report.authorizes_estimator is True
        assert report.sufficient_proxies == []

    def test_utility_tracks_family_rather_than_difficulty(self, records):
        """The hidden structure is recoverable from the labels themselves."""
        bonus = {"crawl": 3.0, "jump": 2.0, "dance": -2.0, "carry": -3.0}
        by_family = {}
        for record in records:
            family = record.context.motion_key.split("_")[0]
            by_family.setdefault(family, []).append(record.utility["H_l"])

        means = {f: sum(v) / len(v) for f, v in by_family.items()}
        assert len(means) >= 3, f"too few families selected to test: {sorted(means)}"

        expected = [bonus.get(f, 0.0) for f in sorted(means)]
        observed = [means[f] for f in sorted(means)]
        assert PA.spearman(expected, observed) > 0.9

    def test_family_ordering_is_recovered_for_the_extremes(self, records):
        by_family = {}
        for record in records:
            family = record.context.motion_key.split("_")[0]
            by_family.setdefault(family, []).append(record.utility["H_l"])
        means = {f: sum(v) / len(v) for f, v in by_family.items()}
        best = max(means, key=lambda f: means[f])
        worst = min(means, key=lambda f: means[f])
        assert best in ("crawl", "jump")
        assert worst in ("carry", "dance")


class TestVerdictWhenNothingIsMeasurable:
    """Ground truth: no effect at all, only branch noise."""

    @pytest.fixture(scope="class")
    def records(self, frozen):
        _, _, manifest = frozen
        return simulate_branches(manifest, lambda c: 0.0, noise=0.02, seed=11)

    def test_gate_a_fails(self, records):
        report = UL.assess_identifiability(
            records, "H_l", noise_floor=[0.02, -0.02, 0.015, -0.018, 0.005]
        )
        assert report.passes is False

    def test_gate_a_explains_why(self, records):
        report = UL.assess_identifiability(
            records, "H_l", noise_floor=[0.02, -0.02, 0.015, -0.018, 0.005]
        )
        assert any("between/within" in reason or "noise floor" in reason
                   for reason in report.reasons)


class TestUndeliveredDoseIsExcluded:
    def test_a_pair_with_no_extra_dose_produces_no_label(self, frozen):
        _, _, manifest = frozen
        candidate = manifest.contexts_per_stage["early"][0]
        record = UL.build_utility_record(
            branch_pair_id="p", context=candidate.context, policy_stage="early", seed=0,
            horizons={"H_l": 128},
            control_dose=DoseReport("c", "ctx", "control", completed_env_steps=10_000.0,
                                    completed_kernel_steps=1_000.0),
            intervention_dose=DoseReport("i", "ctx", "intervention",
                                         completed_env_steps=10_000.0,
                                         completed_kernel_steps=1_000.0),
            control_evaluations=[UL.BranchEvaluation("c", "control", "H_l", 0.5, 0.8,
                                                     0.01, 0.01, 50.0, 0.01)],
            intervention_evaluations=[UL.BranchEvaluation("i", "intervention", "H_l", 0.9,
                                                          0.8, 0.01, 0.01, 50.0, 0.01)],
            epsilon=0.1, kernel_radius_bins=1,
            base_distribution_sha256="a" * 64, intervention_distribution_sha256="b" * 64,
        )
        # A large apparent efficacy gain, but no dose was actually delivered.
        assert record.efficacy_delta["H_l"] == pytest.approx(0.4)
        assert UL.is_usable(record) is False

    def test_such_pairs_are_counted_as_dropped(self, frozen):
        _, _, manifest = frozen
        records = simulate_branches(manifest, lambda c: 0.001)
        records.append(self._undelivered(manifest))
        summary = UL.summarize_labels(records, "H_l")
        assert summary["num_dropped_no_dose"] == 1

    def _undelivered(self, manifest):
        candidate = manifest.contexts_per_stage["late"][0]
        return UL.build_utility_record(
            branch_pair_id="p", context=candidate.context, policy_stage="late", seed=0,
            horizons={"H_l": 128},
            control_dose=DoseReport("c", "ctx", "control", completed_env_steps=10_000.0,
                                    completed_kernel_steps=1_000.0),
            intervention_dose=DoseReport("i", "ctx", "intervention",
                                         completed_env_steps=10_000.0,
                                         completed_kernel_steps=900.0),
            control_evaluations=[UL.BranchEvaluation("c", "control", "H_l", 0.5, 0.8,
                                                     0.01, 0.01, 50.0, 0.01)],
            intervention_evaluations=[UL.BranchEvaluation("i", "intervention", "H_l", 0.6,
                                                          0.8, 0.01, 0.01, 50.0, 0.01)],
            epsilon=0.1, kernel_radius_bins=1,
            base_distribution_sha256="a" * 64, intervention_distribution_sha256="b" * 64,
        )
