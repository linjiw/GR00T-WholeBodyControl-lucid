"""Group-disjoint, family-stratified splits of a motion pool.

A split that merely partitions clips at random is not evidence of
generalization. Two clips by the same performer share body proportions and
personal style; two clips with the same action name are the same behaviour
performed again. If either straddles the boundary, a "held-out" score is partly
a memorization score.

So splitting happens over **groups**, not clips. Groups are the connected
components of a graph whose edges encode the leakage channels we choose to
close:

``performer``   always on -- clips by one performer never straddle a split.
``content``     optional -- clips sharing a canonical action name never
                straddle. Turning this on is exactly the difference between
                *test-repetition* (a seen action, newly performed) and
                *test-content* (an unseen action).
``duplicate``   always on -- identical trajectories never straddle.

Reporting the two regimes separately is the point. Collapsing them into one
"OOD" number hides which generalization was actually tested, and they routinely
differ.

Assignment is greedy over groups sorted large-first, each going to whichever
partition is currently furthest below quota, with a family-balance term so a
small family is not swallowed by one partition. It is deterministic given a
seed, and :func:`verify_split` re-checks disjointness after the fact rather than
trusting the construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from gear_sonic.research.practice_utility.motion_pool import MotionRecord, PoolScan
from gear_sonic.research.practice_utility.schema import sha256_of

#: Which leakage channels create a grouping edge.
#:
#: ``performer``              new performers; actions may recur (test-repetition).
#: ``content``                unseen actions; performers may recur (test-content).
#: ``performer_and_content``  both closed at once -- usually infeasible, see
#:                            :func:`build_split` and ``max_group_share``.
LinkageMode = Literal["performer", "content", "performer_and_content"]

#: A split in which one component holds more than this share of the pool is not
#: a split. Exceeding it raises rather than silently returning a degenerate
#: partitioning.
DEFAULT_MAX_GROUP_SHARE = 0.5

#: Default partition roles and target shares (plan Part I section 4.5).
DEFAULT_RATIOS: dict[str, float] = {
    "adaptation": 0.60,   # continued training and interventions
    "dev": 0.20,          # utility labels and estimator selection
    "test": 0.20,         # frozen final evaluation; opened once
}


class SplitError(RuntimeError):
    """Raised when a split cannot be built or fails verification."""


@dataclass
class MotionSplit:
    """A completed assignment of every clip to exactly one partition."""

    assignment: dict[str, str]
    groups: dict[str, list[str]]
    group_partition: dict[str, str]
    linkage: LinkageMode
    seed: int
    ratios: dict[str, float]
    pool_sha256: str
    stats: dict[str, Any] = field(default_factory=dict)

    def partition(self, name: str) -> list[str]:
        """Clip keys in one partition, sorted."""
        if name not in self.ratios:
            raise KeyError(f"unknown partition {name!r}; have {sorted(self.ratios)}")
        return sorted(k for k, p in self.assignment.items() if p == name)

    @property
    def split_sha256(self) -> str:
        return sha256_of(
            {
                "assignment": dict(sorted(self.assignment.items())),
                "linkage": self.linkage,
                "seed": self.seed,
                "pool_sha256": self.pool_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "practice_utility_group_disjoint_split",
            "schema_version": 1,
            "linkage": self.linkage,
            "seed": self.seed,
            "ratios": self.ratios,
            "pool_sha256": self.pool_sha256,
            "split_sha256": self.split_sha256,
            "stats": self.stats,
            "assignment": dict(sorted(self.assignment.items())),
            "group_partition": dict(sorted(self.group_partition.items())),
        }


def build_groups(records: Iterable[MotionRecord], linkage: LinkageMode) -> dict[str, list[str]]:
    """Connected components of the leakage graph, as ``group_id -> clip keys``."""
    records = list(records)
    parent: dict[str, str] = {r.motion_key: r.motion_key for r in records}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)   # deterministic representative

    channels: list[dict[str, list[str]]] = []
    by_performer: dict[str, list[str]] = defaultdict(list)
    by_content: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_performer[record.parsed.performer].append(record.motion_key)
        by_content[record.content_sha256].append(record.motion_key)
        by_name[record.parsed.canonical_name].append(record.motion_key)

    channels.append(by_content)          # exact duplicates always travel together
    if linkage in ("performer", "performer_and_content"):
        channels.append(by_performer)
    if linkage in ("content", "performer_and_content"):
        channels.append(by_name)

    for channel in channels:
        for members in channel.values():
            for other in members[1:]:
                union(members[0], other)

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[find(record.motion_key)].append(record.motion_key)
    return {gid: sorted(keys) for gid, keys in sorted(groups.items())}


def build_split(
    scan: PoolScan,
    pool_sha256: str,
    linkage: LinkageMode = "performer",
    ratios: dict[str, float] | None = None,
    seed: int = 20260818,
    family_weight: float = 0.5,
    max_group_share: float = DEFAULT_MAX_GROUP_SHARE,
) -> MotionSplit:
    """Assign every clip to a partition, group-disjointly and family-balanced.

    Args:
        family_weight: how strongly to balance family shares against overall
            size. Zero optimizes only total counts; larger values protect small
            families from landing entirely in one partition.
        max_group_share: reject the split if any single group exceeds this share
            of the pool.

    On the giant component
    ---------------------
    Closing the performer and content channels *simultaneously* fails on
    BONES-SEED. Performers share actions and actions share performers, so
    transitive closure merges nearly everything: measured on our pools, the
    largest component holds 21% of the 512-clip pool and **91%** of the
    4950-clip pool. There is no partition of a 91% component into 60/20/20.

    This is a property of the dataset, not a tunable. The two generalization
    regimes must therefore be built and reported *separately* -- a
    ``performer`` split for unseen performers and a ``content`` split for unseen
    actions -- each stating plainly which channel it closes and which it leaves
    open.
    """
    # `ratios or DEFAULT_RATIOS` would silently turn an explicitly empty dict
    # into the defaults; an empty request is a caller error, not a default.
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    if not ratios:
        raise SplitError("at least one partition is required")
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise SplitError(f"partition ratios sum to {total_ratio}, expected 1.0")
    if any(v <= 0 for v in ratios.values()):
        raise SplitError(f"partition ratios must be positive: {ratios}")

    records = {r.motion_key: r for r in scan.records}
    if not records:
        raise SplitError("cannot split an empty pool")
    groups = build_groups(scan.records, linkage)

    largest = max((len(v) for v in groups.values()), default=0)
    if largest > max_group_share * len(records):
        raise SplitError(
            f"linkage {linkage!r} produced a giant component holding "
            f"{largest}/{len(records)} clips ({largest / len(records):.1%} > "
            f"{max_group_share:.0%}); it cannot be split into {sorted(ratios)}. "
            "Performers and actions are densely cross-linked in this pool, so "
            "closing both channels at once merges almost everything. Build a "
            "'performer' split and a 'content' split separately and report them "
            "as distinct generalization regimes."
        )

    families = sorted({r.family for r in scan.records})
    total = len(records)
    family_totals = {f: sum(1 for r in scan.records if r.family == f) for f in families}

    # Large groups first: they constrain the solution most, and placing them
    # late would leave no room to correct the resulting imbalance.
    order = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), _stable_key(kv[0], seed)),
    )

    counts = {p: 0 for p in ratios}
    family_counts = {p: dict.fromkeys(families, 0) for p in ratios}
    group_partition: dict[str, str] = {}
    assignment: dict[str, str] = {}

    for group_id, keys in order:
        group_families = defaultdict(int)
        for key in keys:
            group_families[records[key].family] += 1

        best = min(
            ratios,
            key=lambda p: (
                _placement_cost(
                    partition=p, size=len(keys), group_families=group_families,
                    counts=counts, family_counts=family_counts, ratios=ratios,
                    total=total, family_totals=family_totals, family_weight=family_weight,
                ),
                p,
            ),
        )
        group_partition[group_id] = best
        counts[best] += len(keys)
        for family, n in group_families.items():
            family_counts[best][family] += n
        for key in keys:
            assignment[key] = best

    split = MotionSplit(
        assignment=assignment,
        groups=groups,
        group_partition=group_partition,
        linkage=linkage,
        seed=seed,
        ratios=ratios,
        pool_sha256=pool_sha256,
        stats=_summarize(records, groups, group_partition, counts, family_counts, ratios, total),
    )
    verify_split(split, scan)
    return split


def verify_split(split: MotionSplit, scan: PoolScan) -> None:
    """Re-derive disjointness from the assignment; raise on any leak.

    Construction is not trusted. This checks the property that actually matters
    -- that no performer, duplicate trajectory, or (in content mode) action name
    appears in two partitions.
    """
    records = {r.motion_key: r for r in scan.records}
    missing = sorted(set(records) - set(split.assignment))
    if missing:
        raise SplitError(f"{len(missing)} clips were never assigned, e.g. {missing[:3]}")
    unknown = sorted(set(split.assignment) - set(records))
    if unknown:
        raise SplitError(f"assignment references clips absent from the pool: {unknown[:3]}")

    channels: dict[str, dict[str, set[str]]] = {"content_sha256": defaultdict(set)}
    if split.linkage in ("performer", "performer_and_content"):
        channels["performer"] = defaultdict(set)
    if split.linkage in ("content", "performer_and_content"):
        channels["canonical_name"] = defaultdict(set)

    for key, partition in split.assignment.items():
        record = records[key]
        channels["content_sha256"][record.content_sha256].add(partition)
        if "performer" in channels:
            channels["performer"][record.parsed.performer].add(partition)
        if "canonical_name" in channels:
            channels["canonical_name"][record.parsed.canonical_name].add(partition)

    for channel, mapping in channels.items():
        leaked = sorted(value for value, partitions in mapping.items() if len(partitions) > 1)
        if leaked:
            raise SplitError(
                f"{channel} leaks across partitions ({len(leaked)} values, "
                f"e.g. {leaked[:3]}); the split is not group-disjoint"
            )

    empty = [p for p in split.ratios if not split.partition(p)]
    if empty:
        raise SplitError(f"partitions {empty} are empty; adjust ratios or pool size")


def _placement_cost(
    *, partition, size, group_families, counts, family_counts, ratios, total,
    family_totals, family_weight,
) -> float:
    """Post-placement *fill ratio* of ``partition``; lower is a better home.

    The measure is ``(count + size) / target``, not absolute deviation. Absolute
    deviation looks correct and is not: a large partition sitting far below its
    target always shows a huge squared deficit, so every group goes to the small
    partitions instead and the large one is starved. Fill ratio is scale-free,
    so partitions fill proportionally and finish together.

    The family term applies the same reasoning per family, which is what keeps a
    small family from landing entirely in one partition.
    """
    target = ratios[partition] * total
    size_cost = (counts[partition] + size) / max(target, 1e-9)

    family_cost = 0.0
    for family, n in group_families.items():
        family_target = ratios[partition] * family_totals[family]
        family_cost += (family_counts[partition][family] + n) / max(family_target, 1e-9)
    if group_families:
        family_cost /= len(group_families)

    return size_cost + family_weight * family_cost


def _summarize(records, groups, group_partition, counts, family_counts, ratios, total):
    per_partition: dict[str, Any] = {}
    for partition in ratios:
        n = counts[partition]
        member_groups = [g for g, p in group_partition.items() if p == partition]
        per_partition[partition] = {
            "motions": n,
            "share": round(n / total, 4) if total else 0.0,
            "target_share": ratios[partition],
            "groups": len(member_groups),
            "performers": len({records[k].parsed.performer
                               for g in member_groups for k in groups[g]}),
            "family_counts": {f: c for f, c in sorted(family_counts[partition].items()) if c},
            "duration_seconds": round(
                sum(records[k].duration_seconds for g in member_groups for k in groups[g]), 2
            ),
        }
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    return {
        "total_motions": total,
        "total_groups": len(groups),
        "largest_group": sizes[0] if sizes else 0,
        "largest_group_share": round(sizes[0] / total, 4) if total and sizes else 0.0,
        "singleton_groups": sum(1 for s in sizes if s == 1),
        "partitions": per_partition,
    }


def _stable_key(value: str, seed: int) -> str:
    """Seeded, deterministic tie-break that does not depend on insertion order."""
    return sha256_of({"seed": seed, "value": value})
