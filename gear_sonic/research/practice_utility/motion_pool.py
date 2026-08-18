"""Parse a BONES-SEED motion pool into an auditable, hashable manifest.

Everything downstream -- context keys, splits, dose accounting, utility labels
-- is anchored to what this module produces, so it records *derived* facts and
keeps the evidence for each one.

BONES-SEED clip keys follow a single, exceptionless pattern in the pools we
use::

    <base_name>__A<performer>[_M]

    injured_R_leg_walk_ff_start_180_003__A173
    Neutral_throw_ball_001__A057_M          <- mirrored

Three identities are extracted, and they are not interchangeable:

``performer``
    The ``A###`` code. Splits are disjoint on this, because two clips by the
    same performer share body proportions and idiosyncratic style; letting them
    straddle a split leaks the test set into training.

``base_name`` / ``canonical_name``
    The action name, with take number and mirror suffix removed. Whether this
    may straddle a split is the difference between *test-repetition* (a seen
    action, newly performed) and *test-content* (an unseen action). Conflating
    them into one "OOD" number hides which generalization was actually tested,
    so :mod:`split` treats it as a separate, explicit linkage rule.

``content_sha256``
    A hash of the joint trajectory itself, which catches exact duplicates that
    differ only in filename.

The family taxonomy below is keyword-based and therefore approximate. It is
ordered, first-match-wins, and every assignment is reported with the token that
triggered it, so a reader can audit it rather than trust it. Families exist to
stratify sampling and to macro-average evaluation -- a large family must not be
allowed to drown a small one -- not to make semantic claims.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gear_sonic.research.practice_utility.schema import MotionPoolManifest, sha256_of

#: ``<base>__A<performer>[_M]``; the mirror suffix is optional.
KEY_PATTERN = re.compile(r"^(?P<base>.+?)__A(?P<performer>\d+)(?P<mirror>_M)?$")

#: Trailing take number, e.g. ``..._003``.
TAKE_PATTERN = re.compile(r"_(?P<take>\d+)$")

#: Ordered, first-match-wins family rules. Order encodes priority: an impaired
#: walk is filed under ``injured`` rather than ``walk`` because the impairment,
#: not the gait, dominates the dynamics regime -- and regime structure is
#: exactly what we expect to drive difficulty/utility divergence. Whole-body
#: locomotion regimes are matched before upper-body activity, because a
#: gesture performed while walking is still, dynamically, walking.
FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("crawl", ("crawl",)),
    ("jump", ("jump", "hop", "leap", "vault")),
    ("dance", ("dance", "hiphop", "vouge", "boogle", "ballet", "salsa",
               "routine", "waltz", "twerk", "moonwalk")),
    ("injured", ("injured", "limp", "hurt", "wounded")),
    ("crouch", ("crouch", "squat", "kneel", "sit", "duck", "bend", "stoop", "crouching")),
    ("run", ("jog", "run", "sprint", "dash")),
    ("walk", ("walk", "step", "march", "stride", "arc", "tiptoe", "shuffle")),
    ("turn", ("turn", "pivot", "spin", "sideway", "rotate")),
    ("carry", ("heavy", "light", "carry", "pick", "put", "throw", "reach",
               "grab", "lift", "hold", "push", "pull", "catch", "toss")),
    ("interact", ("door", "handle", "lever", "wall", "button", "knob", "switch",
                  "drawer", "keyboard", "phone", "cup", "box", "ball", "chair",
                  "table", "rail", "ladder")),
    ("exercise", ("exercise", "bicep", "pushup", "situp", "workout", "plank",
                  "lunge", "jumpingjack", "warmup", "train")),
    ("gesture", ("clap", "cheer", "triumph", "wave", "point", "salute", "greet",
                 "no", "yes", "nod", "shrug", "applaud", "show", "thumb", "know",
                 "signal", "beckon", "bow", "hail")),
    ("groom", ("dust", "rub", "brush", "itch", "scratch", "wipe", "clean",
               "wash", "comb", "adjust", "pat")),
    ("search", ("look", "search", "check", "scan", "observe", "inspect",
                "watch", "peek", "browse", "find")),
    ("idle", ("idle", "stand", "loop", "stop", "stretch", "relax", "rest",
              "wait", "breathe", "neutral", "pause")),
)

FALLBACK_FAMILY = "other"


class MotionKeyError(ValueError):
    """Raised when a clip key does not follow the expected pattern."""


@dataclass(frozen=True)
class ParsedKey:
    """The identities decoded from a clip key."""

    motion_key: str
    base_name: str
    performer: str
    take: int | None
    is_mirror: bool

    @property
    def canonical_name(self) -> str:
        """Action identity: no take number, no mirror suffix, no performer.

        Two clips sharing this are the same action, whether they are mirror
        images, different takes, or different performers.
        """
        return TAKE_PATTERN.sub("", self.base_name).lower()


@dataclass
class MotionRecord:
    """One clip, with everything needed to place it in a split."""

    motion_key: str
    path: str
    parsed: ParsedKey
    num_frames: int
    fps: float
    num_dofs: int
    content_sha256: str
    family: str
    family_evidence: str

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_key": self.motion_key,
            "path": self.path,
            "base_name": self.parsed.base_name,
            "canonical_name": self.parsed.canonical_name,
            "performer": self.parsed.performer,
            "take": self.parsed.take,
            "is_mirror": self.parsed.is_mirror,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "num_dofs": self.num_dofs,
            "duration_seconds": round(self.duration_seconds, 4),
            "content_sha256": self.content_sha256,
            "family": self.family,
            "family_evidence": self.family_evidence,
        }


@dataclass
class PoolScan:
    """The full result of scanning a pool directory."""

    records: list[MotionRecord]
    source_root: str
    duplicate_groups: list[list[str]] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)

    @property
    def num_motions(self) -> int:
        return len(self.records)

    def family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.family] = counts.get(record.family, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def performer_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.parsed.performer] = counts.get(record.parsed.performer, 0) + 1
        return counts

    def to_manifest(self, manifest_id: str) -> MotionPoolManifest:
        return MotionPoolManifest(
            manifest_id=manifest_id,
            motion_keys=[r.motion_key for r in self.records],
            motion_hashes={r.motion_key: r.content_sha256 for r in self.records},
            source_root=self.source_root,
            notes=(
                f"{self.num_motions} clips, {len(self.performer_counts())} performers, "
                f"{len(self.family_counts())} families"
            ),
        )

    def summary(self) -> dict[str, Any]:
        durations = sorted(r.duration_seconds for r in self.records)
        return {
            "num_motions": self.num_motions,
            "num_performers": len(self.performer_counts()),
            "num_canonical_names": len({r.parsed.canonical_name for r in self.records}),
            "num_mirrored": sum(1 for r in self.records if r.parsed.is_mirror),
            "family_counts": self.family_counts(),
            "duplicate_group_count": len(self.duplicate_groups),
            "unparsed_count": len(self.unparsed),
            "total_duration_seconds": round(sum(durations), 2),
            "duration_seconds_quantiles": _quantiles(durations),
        }


def parse_motion_key(motion_key: str) -> ParsedKey:
    """Decode performer, take, and mirror flag from a clip key."""
    match = KEY_PATTERN.match(motion_key)
    if not match:
        raise MotionKeyError(
            f"motion key {motion_key!r} does not match '<base>__A<performer>[_M]'; "
            "splits cannot be made performer-disjoint without this"
        )
    base = match.group("base")
    take_match = TAKE_PATTERN.search(base)
    return ParsedKey(
        motion_key=motion_key,
        base_name=base,
        performer=f"A{match.group('performer')}",
        take=int(take_match.group("take")) if take_match else None,
        is_mirror=match.group("mirror") is not None,
    )


def _token_variants(token: str) -> set[str]:
    """Cheap morphological variants so ``reaching`` matches the keyword ``reach``.

    A full stemmer would be overkill and would introduce its own errors; this
    covers the inflections that actually occur in BONES-SEED clip names
    (``-ing``, ``-ed``, plural ``-s``) and nothing else.
    """
    variants = {token}
    for suffix in ("ing", "ed"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            stem = token[: -len(suffix)]
            variants.add(stem)
            variants.add(stem + "e")
            if len(stem) > 2 and stem[-1] == stem[-2]:   # running -> run
                variants.add(stem[:-1])
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])
    return variants


def motion_family(base_name: str) -> tuple[str, str]:
    """Assign a family, returning ``(family, triggering_token)``.

    The evidence token is returned so every assignment can be audited instead of
    taken on faith. Coverage is reported by :meth:`PoolScan.summary`; a large
    ``other`` bucket is a signal that the taxonomy needs extending, not
    something to quietly tolerate, because families are what macro-averaging and
    stratification rest on.
    """
    tokens: set[str] = set()
    for raw in re.split(r"[_\s]+", base_name.lower()):
        if raw:
            tokens |= _token_variants(raw)
    for family, keywords in FAMILY_RULES:
        for keyword in keywords:
            if keyword in tokens:
                return family, keyword
    return FALLBACK_FAMILY, ""


def scan_pool(
    pool_dir: str | Path,
    loader: Any = None,
    limit: int | None = None,
) -> PoolScan:
    """Scan a ``robot_filtered`` directory into :class:`MotionRecord` objects.

    Args:
        pool_dir: directory of ``*.pkl`` clips.
        loader: callable taking a path and returning the clip dict; defaults to
            ``joblib.load``. Injected so tests need no real motion files.
        limit: stop after this many files (for smoke runs).
    """
    if loader is None:
        import joblib

        loader = joblib.load

    pool_dir = Path(pool_dir)
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"motion pool directory not found: {pool_dir}")

    paths = sorted(pool_dir.glob("*.pkl"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no .pkl clips under {pool_dir}")

    records: list[MotionRecord] = []
    unparsed: list[str] = []
    by_content: dict[str, list[str]] = {}

    for path in paths:
        payload = loader(str(path))
        for motion_key, clip in _iter_clips(payload, path):
            try:
                parsed = parse_motion_key(motion_key)
            except MotionKeyError:
                unparsed.append(motion_key)
                continue

            dof = clip["dof"]
            content_sha = hashlib.sha256(_as_bytes(dof)).hexdigest()
            family, evidence = motion_family(parsed.base_name)
            records.append(
                MotionRecord(
                    motion_key=motion_key,
                    path=str(path),
                    parsed=parsed,
                    num_frames=int(dof.shape[0]),
                    fps=float(clip.get("fps", 30)),
                    num_dofs=int(dof.shape[1]),
                    content_sha256=content_sha,
                    family=family,
                    family_evidence=evidence,
                )
            )
            by_content.setdefault(content_sha, []).append(motion_key)

    duplicates = sorted(
        (sorted(keys) for keys in by_content.values() if len(keys) > 1),
        key=lambda group: group[0],
    )
    records.sort(key=lambda r: r.motion_key)
    return PoolScan(
        records=records,
        source_root=str(pool_dir.resolve()),
        duplicate_groups=duplicates,
        unparsed=sorted(unparsed),
    )


def drop_exact_duplicates(scan: PoolScan) -> PoolScan:
    """Keep one clip per distinct trajectory, preferring the unmirrored one.

    An exact content duplicate is not extra evidence; counting it twice would
    inflate the apparent size of the pool and let the same trajectory appear on
    both sides of a split.
    """
    keep: dict[str, MotionRecord] = {}
    for record in scan.records:
        current = keep.get(record.content_sha256)
        if current is None or _prefer(record, current):
            keep[record.content_sha256] = record
    kept = sorted(keep.values(), key=lambda r: r.motion_key)
    return PoolScan(
        records=kept,
        source_root=scan.source_root,
        duplicate_groups=scan.duplicate_groups,
        unparsed=scan.unparsed,
    )


def pool_sha256(scan: PoolScan) -> str:
    """Content hash of the whole scanned pool."""
    return sha256_of(
        {
            "source_root": scan.source_root,
            "records": [
                {"motion_key": r.motion_key, "content_sha256": r.content_sha256}
                for r in sorted(scan.records, key=lambda r: r.motion_key)
            ],
        }
    )


def _iter_clips(payload: Any, path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dict of clips, got {type(payload)}")
    for motion_key, clip in payload.items():
        if not isinstance(clip, dict) or "dof" not in clip:
            raise ValueError(f"{path}:{motion_key} has no 'dof' array")
        yield str(motion_key), clip


def _prefer(candidate: MotionRecord, current: MotionRecord) -> bool:
    """Prefer unmirrored clips, then lexicographically smaller keys."""
    if candidate.parsed.is_mirror != current.parsed.is_mirror:
        return not candidate.parsed.is_mirror
    return candidate.motion_key < current.motion_key


def _as_bytes(array: Any) -> bytes:
    contiguous = array if array.flags["C_CONTIGUOUS"] else array.copy(order="C")
    return contiguous.astype("float32", copy=False).tobytes()


def _quantiles(sorted_values: list[float]) -> dict[str, float]:
    if not sorted_values:
        return {}
    def at(q: float) -> float:
        index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
        return round(sorted_values[index], 3)
    return {"p05": at(0.05), "p25": at(0.25), "p50": at(0.50), "p75": at(0.75), "p95": at(0.95)}
