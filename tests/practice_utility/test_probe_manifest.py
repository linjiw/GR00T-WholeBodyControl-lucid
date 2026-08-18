"""Tests for probe-campaign context selection and plan validation.

The rule these enforce: a campaign that probes only hard contexts, or only one
family, cannot answer the question it was built for. Discovering that after the
GPU time is spent is expensive, so it is refused up front.
"""

import pytest

from gear_sonic.research.practice_utility import probe_manifest as PM
from gear_sonic.research.practice_utility.schema import ContextKey, motion_hash

FAMILIES = ("walk", "run", "jump", "crawl", "dance")


def candidate(index, failure_rate=None, family=None, regime=None, partition="adaptation"):
    key = f"m{index:03d}__A{index:03d}"
    return PM.ContextCandidate(
        context=ContextKey(key, motion_hash(key, 300, 50.0), 1, 50, 100),
        failure_rate=failure_rate if failure_rate is not None else index / 100.0,
        sampling_probability=0.01,
        family=family or FAMILIES[index % len(FAMILIES)],
        contact_regime=regime or ("rich" if index % 2 else "light"),
        partition=partition,
    )


def pool(n=100, **kwargs):
    return [candidate(i, **kwargs) for i in range(n)]


HORIZONS = {"H_s": 8, "H_m": 32, "H_l": 128}


class TestFailureQuartiles:
    def test_assigns_all_four(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        assert {c.failure_quartile for c in labelled} == {0, 1, 2, 3}

    def test_quartiles_are_balanced(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        counts = {q: sum(1 for c in labelled if c.failure_quartile == q) for q in range(4)}
        assert all(24 <= v <= 26 for v in counts.values())

    def test_lowest_failure_lands_in_quartile_zero(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        lowest = min(labelled, key=lambda c: c.failure_rate)
        assert lowest.failure_quartile == 0

    def test_rank_based_survives_a_skewed_distribution(self):
        """Fixed thresholds would leave strata empty; ranks do not."""
        skewed = [candidate(i, failure_rate=0.001 * i if i < 95 else 0.99) for i in range(100)]
        labelled = PM.assign_failure_quartiles(skewed)
        assert {c.failure_quartile for c in labelled} == {0, 1, 2, 3}

    def test_empty_is_safe(self):
        assert PM.assign_failure_quartiles([]) == []


class TestStratifiedSelect:
    def test_returns_the_requested_count(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        assert len(PM.stratified_select(labelled, 24)) == 24

    def test_selection_is_unique(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        chosen = PM.stratified_select(labelled, 24)
        assert len({c.context.context_id for c in chosen}) == 24

    def test_spans_the_difficulty_range(self):
        """The guard against a silent hard-example study."""
        labelled = PM.assign_failure_quartiles(pool(100))
        chosen = PM.stratified_select(labelled, 24)
        assert {c.failure_quartile for c in chosen} == {0, 1, 2, 3}

    def test_covers_every_family(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        chosen = PM.stratified_select(labelled, 24)
        assert {c.family for c in chosen} == set(FAMILIES)

    def test_rare_stratum_is_represented(self):
        """Round-robin, not proportional: one rare family must still appear."""
        candidates = [candidate(i, family="walk") for i in range(90)]
        candidates.append(candidate(999, family="crawl", failure_rate=0.5))
        labelled = PM.assign_failure_quartiles(candidates)
        chosen = PM.stratified_select(labelled, 12)
        assert "crawl" in {c.family for c in chosen}

    def test_is_deterministic_for_a_seed(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        a = [c.context.context_id for c in PM.stratified_select(labelled, 24, seed=5)]
        b = [c.context.context_id for c in PM.stratified_select(labelled, 24, seed=5)]
        assert a == b

    def test_seed_changes_the_selection(self):
        labelled = PM.assign_failure_quartiles(pool(100))
        a = {c.context.context_id for c in PM.stratified_select(labelled, 24, seed=1)}
        b = {c.context.context_id for c in PM.stratified_select(labelled, 24, seed=2)}
        assert a != b

    def test_refuses_to_over_request(self):
        labelled = PM.assign_failure_quartiles(pool(10))
        with pytest.raises(PM.ManifestError, match="only 10 distinct contexts"):
            PM.stratified_select(labelled, 50)

    def test_rejects_empty_pool(self):
        with pytest.raises(PM.ManifestError, match="no candidates"):
            PM.stratified_select([], 5)

    def test_rejects_nonpositive_budget(self):
        with pytest.raises(PM.ManifestError, match="must be positive"):
            PM.stratified_select(pool(10), 0)


class TestBuildProbeManifest:
    def build(self, **overrides):
        params = dict(
            campaign_id="oracle_screen_v1",
            candidates_per_stage={"early": pool(100), "middle": pool(100), "late": pool(100)},
            num_contexts=24,
            seeds=[0, 1],
            horizons=HORIZONS,
            pool_sha256="pool" + "0" * 60,
            split_sha256="split" + "0" * 59,
        )
        params.update(overrides)
        return PM.build_probe_manifest(**params)

    def test_builds_all_stages(self):
        manifest = self.build()
        assert manifest.stages == ["early", "late", "middle"]
        assert all(len(v) == 24 for v in manifest.contexts_per_stage.values())

    def test_branch_count_accounts_for_shared_controls(self):
        manifest = self.build()
        assert manifest.num_branches == 3 * 24 * 2       # stages x contexts x seeds
        assert manifest.num_control_branches == 3 * 2    # one control per stage/seed

    def test_hash_is_stable(self):
        assert self.build().manifest_sha256 == self.build().manifest_sha256

    def test_hash_tracks_the_dose(self):
        assert self.build(epsilon=0.10).manifest_sha256 != self.build(epsilon=0.20).manifest_sha256

    def test_hash_tracks_the_pool(self):
        assert self.build().manifest_sha256 != self.build(pool_sha256="other" * 12).manifest_sha256

    def test_coverage_reports_every_axis(self):
        coverage = self.build().coverage()["early"]
        assert set(coverage) >= {"failure_quartiles", "families", "contact_regimes"}
        assert set(coverage["failure_quartiles"]) == {0, 1, 2, 3}

    def test_serializes_with_contexts(self):
        payload = self.build().to_dict()
        assert payload["kind"] == "practice_utility_probe_manifest"
        assert len(payload["contexts_per_stage"]["early"]) == 24

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"candidates_per_stage": {}}, "no policy stages"),
            ({"seeds": []}, "at least one seed"),
            ({"horizons": {}}, "at least one horizon"),
            ({"epsilon": 1.5}, "epsilon"),
        ],
    )
    def test_rejects_invalid_plans(self, overrides, match):
        with pytest.raises(PM.ManifestError, match=match):
            self.build(**overrides)


class TestValidateManifest:
    def manifest_from(self, candidates, num_contexts=8):
        return PM.build_probe_manifest(
            campaign_id="c", candidates_per_stage={"middle": candidates},
            num_contexts=num_contexts, seeds=[0], horizons=HORIZONS,
            pool_sha256="p", split_sha256="s",
        )

    def test_refuses_a_hard_examples_only_campaign(self):
        """The failure this validator exists to prevent."""
        hard = [candidate(i, failure_rate=0.90 + 0.001 * i) for i in range(40)]
        # Force every selected context into the top quartile by pre-labelling.
        for c in hard:
            c.failure_quartile = 3
        manifest = PM.ProbeManifest(
            campaign_id="c", stages=["middle"], contexts_per_stage={"middle": hard[:8]},
            seeds=[0], epsilon=0.1, kernel_radius_bins=1, horizons=HORIZONS,
            pool_sha256="p", split_sha256="s",
        )
        with pytest.raises(PM.ManifestError, match="hard-example study"):
            PM.validate_manifest(manifest)

    def test_refuses_a_single_family_campaign(self):
        single = PM.assign_failure_quartiles([candidate(i, family="walk") for i in range(40)])
        manifest = PM.ProbeManifest(
            campaign_id="c", stages=["middle"],
            contexts_per_stage={"middle": PM.stratified_select(single, 8)},
            seeds=[0], epsilon=0.1, kernel_radius_bins=1, horizons=HORIZONS,
            pool_sha256="p", split_sha256="s",
        )
        with pytest.raises(PM.ManifestError, match="motion families"):
            PM.validate_manifest(manifest)

    def test_refuses_contexts_from_a_held_out_partition(self):
        """Probing dev or test would contaminate the evaluation."""
        contaminated = PM.assign_failure_quartiles(
            [candidate(i, partition="test" if i == 0 else "adaptation") for i in range(40)]
        )
        chosen = PM.stratified_select(contaminated, 8)
        chosen[0].partition = "test"
        manifest = PM.ProbeManifest(
            campaign_id="c", stages=["middle"], contexts_per_stage={"middle": chosen},
            seeds=[0], epsilon=0.1, kernel_radius_bins=1, horizons=HORIZONS,
            pool_sha256="p", split_sha256="s",
        )
        with pytest.raises(PM.ManifestError, match="adaptation split"):
            PM.validate_manifest(manifest)

    def test_refuses_duplicate_contexts(self):
        chosen = PM.stratified_select(PM.assign_failure_quartiles(pool(40)), 8)
        chosen[1] = chosen[0]
        manifest = PM.ProbeManifest(
            campaign_id="c", stages=["middle"], contexts_per_stage={"middle": chosen},
            seeds=[0], epsilon=0.1, kernel_radius_bins=1, horizons=HORIZONS,
            pool_sha256="p", split_sha256="s",
        )
        with pytest.raises(PM.ManifestError, match="duplicate"):
            PM.validate_manifest(manifest)

    def test_refuses_an_empty_stage(self):
        manifest = PM.ProbeManifest(
            campaign_id="c", stages=["middle"], contexts_per_stage={"middle": []},
            seeds=[0], epsilon=0.1, kernel_radius_bins=1, horizons=HORIZONS,
            pool_sha256="p", split_sha256="s",
        )
        with pytest.raises(PM.ManifestError, match="no contexts"):
            PM.validate_manifest(manifest)

    def test_a_balanced_campaign_passes(self):
        assert self.manifest_from(pool(100), num_contexts=24) is not None
