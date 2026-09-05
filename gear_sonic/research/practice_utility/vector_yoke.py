"""Canonical, fail-closed schedules for exposure-matched vector yokes.

A scalar lambda trace cannot be a matched control for a per-channel box gate:
the box changes a frontier vector, rotates a one-channel-ahead probe, and sends
an absolute vector to every retained stratum.  This module converts that full
telemetry into a deterministic open-loop schedule without importing Isaac.

The runtime seam is deliberately separate.  A schedule must pass this CPU
contract before a training callback is allowed to consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
DEFAULT_MAX_INTENSITY = 3.0
_TOLERANCE = 1e-9


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_intensity(value: Any, *, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got boolean {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(number) or not 0.0 <= number <= maximum:
        raise ValueError(f"{label}={number!r} must be finite and in [0, {maximum}]")
    return number


def _vector(
    value: Any,
    *,
    label: str,
    channels: tuple[str, ...],
    maximum: float,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    observed = tuple(str(name) for name in value)
    if set(observed) != set(channels) or len(observed) != len(channels):
        raise ValueError(f"{label} channels {sorted(observed)!r} do not match {sorted(channels)!r}")
    return {
        name: _finite_intensity(value[name], label=f"{label}.{name}", maximum=maximum)
        for name in channels
    }


def _vectors_close(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return left.keys() == right.keys() and all(
        abs(float(left[name]) - float(right[name])) <= _TOLERANCE for name in left
    )


@dataclass(frozen=True)
class VectorYokeRecord:
    """One source iteration's complete applied exposure."""

    source_global_step: int
    frontier_vector: dict[str, float]
    decision_probe_vector: dict[str, float]
    applied_probe_vector: dict[str, float]
    stratum_vectors: tuple[dict[str, float], ...]
    active_channel: str | None
    applied_probe_channel: str | None
    warmup_backfill: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_global_step": self.source_global_step,
            "frontier_vector": dict(self.frontier_vector),
            "decision_probe_vector": dict(self.decision_probe_vector),
            "applied_probe_vector": dict(self.applied_probe_vector),
            "stratum_vectors": [dict(vector) for vector in self.stratum_vectors],
            "active_channel": self.active_channel,
            "applied_probe_channel": self.applied_probe_channel,
            "warmup_backfill": self.warmup_backfill,
        }


@dataclass(frozen=True)
class VectorYokeSchedule:
    """A source-hashed, absolute per-stratum vector schedule."""

    channels: tuple[str, ...]
    stratum_sizes: tuple[int, ...]
    records: tuple[VectorYokeRecord, ...]
    source_path: str
    source_sha256: str
    warmup_backfilled_records: int
    probe_transition_records: int
    max_intensity: float

    @property
    def stratum_count(self) -> int:
        return len(self.stratum_sizes)

    def record(self, index: int, *, hold_last: bool = False) -> VectorYokeRecord:
        """Return one row; exhaustion is an error unless explicitly frozen."""
        if index < 0:
            raise IndexError(f"vector-yoke index must be non-negative, got {index}")
        if index >= len(self.records):
            if hold_last and self.records:
                return self.records[-1]
            raise IndexError(
                f"vector-yoke schedule exhausted at index {index}; "
                f"source has {len(self.records)} records"
            )
        return self.records[index]

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "kind": "lucid_vector_yoke_schedule",
            "schema_version": SCHEMA_VERSION,
            "channels": list(self.channels),
            "stratum_count": self.stratum_count,
            "stratum_sizes": list(self.stratum_sizes),
            "source": {
                "path": self.source_path,
                "sha256": self.source_sha256,
                "row_count": len(self.records),
            },
            "audit": {
                "frontier_monotone": True,
                "applied_decreases": 0,
                "warmup_backfilled_records": self.warmup_backfilled_records,
                "probe_transition_records": self.probe_transition_records,
                "max_intensity": self.max_intensity,
                "exhaustion_default": "error",
            },
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._payload_without_digest()).encode())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_digest()
        payload["canonical_sha256"] = self.canonical_sha256
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


class VectorYokeCursor:
    """Stateful schedule cursor with hash-checked split/resume semantics."""

    def __init__(self, schedule: VectorYokeSchedule, *, hold_last: bool = False) -> None:
        self.schedule = schedule
        self.hold_last = bool(hold_last)
        self.next_index = 0

    def take(self) -> VectorYokeRecord:
        record = self.schedule.record(self.next_index, hold_last=self.hold_last)
        self.next_index += 1
        return record

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "schedule_sha256": self.schedule.canonical_sha256,
            "next_index": self.next_index,
            "hold_last": self.hold_last,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        observed = str(state.get("schedule_sha256", ""))
        expected = self.schedule.canonical_sha256
        if observed != expected:
            raise ValueError(
                f"vector-yoke schedule hash mismatch: state={observed!r}, expected={expected!r}"
            )
        if bool(state.get("hold_last", False)) != self.hold_last:
            raise ValueError("vector-yoke exhaustion policy changed across resume")
        index = state.get("next_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid vector-yoke next_index {index!r}")
        self.next_index = index


def _validate_schedule(schedule: VectorYokeSchedule) -> None:
    if not schedule.channels or len(set(schedule.channels)) != len(schedule.channels):
        raise ValueError("canonical vector-yoke channels must be non-empty and unique")
    if schedule.stratum_count < 2 or any(size <= 0 for size in schedule.stratum_sizes):
        raise ValueError(
            "canonical vector-yoke stratum sizes must be positive with at least two strata"
        )
    if not schedule.records:
        raise ValueError("canonical vector-yoke schedule has no records")
    steps = tuple(record.source_global_step for record in schedule.records)
    if steps != tuple(range(steps[0], steps[0] + len(steps))):
        raise ValueError("canonical vector-yoke source steps must be consecutive")
    previous: Mapping[str, float] | None = None
    seen_materialized = False
    backfilled = 0
    for index, record in enumerate(schedule.records):
        if len(record.stratum_vectors) != schedule.stratum_count:
            raise ValueError(f"canonical record {index} has the wrong stratum count")
        if not _vectors_close(record.stratum_vectors[-2], record.frontier_vector):
            raise ValueError(f"canonical record {index} frontier stratum mismatch")
        if not _vectors_close(record.stratum_vectors[-1], record.applied_probe_vector):
            raise ValueError(f"canonical record {index} probe stratum mismatch")
        if record.active_channel is not None and record.active_channel not in schedule.channels:
            raise ValueError(
                f"canonical record {index} has unknown active channel {record.active_channel!r}"
            )
        decision_channel = _probe_channel(
            record.decision_probe_vector,
            record.frontier_vector,
            label=f"canonical record {index} decision probe",
        )
        if (
            decision_channel is not None
            and record.active_channel is not None
            and decision_channel != record.active_channel
        ):
            raise ValueError(
                f"canonical record {index} decision probe raises {decision_channel!r}, "
                f"not active channel {record.active_channel!r}"
            )
        applied_channel = _probe_channel(
            record.applied_probe_vector,
            record.frontier_vector,
            label=f"canonical record {index} applied probe",
        )
        if applied_channel != record.applied_probe_channel:
            raise ValueError(
                f"canonical record {index} applied probe channel {applied_channel!r} != "
                f"declared {record.applied_probe_channel!r}"
            )
        if record.warmup_backfill:
            if seen_materialized:
                raise ValueError("canonical warm-up backfills must form a contiguous prefix")
            backfilled += 1
        else:
            seen_materialized = True
        if previous is not None:
            decreased = {
                name: (previous[name], record.frontier_vector[name])
                for name in schedule.channels
                if record.frontier_vector[name] < previous[name] - _TOLERANCE
            }
            if decreased:
                raise ValueError(f"canonical record {index} contracts its frontier: {decreased}")
        previous = record.frontier_vector
    if backfilled != schedule.warmup_backfilled_records:
        raise ValueError(
            f"canonical warm-up count {backfilled} != declared "
            f"{schedule.warmup_backfilled_records}"
        )
    transitions = sum(
        not _vectors_close(record.decision_probe_vector, record.applied_probe_vector)
        for record in schedule.records
    )
    if transitions != schedule.probe_transition_records:
        raise ValueError(
            f"canonical probe transition count {transitions} != declared "
            f"{schedule.probe_transition_records}"
        )


def _probe_channel(
    probe: Mapping[str, float],
    frontier: Mapping[str, float],
    *,
    label: str,
) -> str | None:
    below = {
        name: (frontier[name], probe[name])
        for name in frontier
        if probe[name] < frontier[name] - _TOLERANCE
    }
    if below:
        raise ValueError(f"{label} falls below the frontier: {below}")
    raised = [name for name in frontier if probe[name] > frontier[name] + _TOLERANCE]
    if len(raised) > 1:
        raise ValueError(f"{label} raises multiple channels: {raised}")
    return raised[0] if raised else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"vector-yoke source is empty: {path}")
    return rows


def _source_steps(rows: Iterable[Mapping[str, Any]]) -> tuple[int, ...]:
    steps: list[int] = []
    for index, row in enumerate(rows):
        value = row.get("global_step")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"source row {index} has invalid global_step {value!r}")
        steps.append(value)
    expected = tuple(range(steps[0], steps[0] + len(steps)))
    if tuple(steps) != expected:
        raise ValueError("source global_step values must be strictly consecutive")
    return tuple(steps)


def canonicalize_box_trace(
    path: Path,
    *,
    expected_channels: Iterable[str] | None = None,
    expected_strata: int | None = None,
    max_intensity: float = DEFAULT_MAX_INTENSITY,
) -> VectorYokeSchedule:
    """Validate a box JSONL trace and retain its complete applied vectors.

    Historical callbacks omitted TACE vectors from their warm-up log rows even
    though the initial box vectors were already applied.  A missing prefix is
    reconstructed only when every row is explicitly a warm-up hold at the same
    scalar frontier as the first complete record.  Missing vectors anywhere
    else are rejected.
    """
    path = path.resolve()
    maximum = _finite_intensity(max_intensity, label="max_intensity", maximum=float("inf"))
    if maximum <= 0.0:
        raise ValueError("max_intensity must be positive")
    rows = _read_jsonl(path)
    steps = _source_steps(rows)
    complete_indices = [
        index
        for index, row in enumerate(rows)
        if isinstance((row.get("tace") or {}).get("stratum_lambdas"), list)
        and (row.get("tace") or {}).get("stratum_lambdas")
    ]
    if not complete_indices:
        raise ValueError("box trace has no complete per-stratum vector record")
    first_complete = complete_indices[0]
    if complete_indices != list(range(first_complete, len(rows))):
        raise ValueError("per-stratum vectors may be absent only in a contiguous warm-up prefix")
    first = rows[first_complete]
    raw_channels = first.get("channels") or list((first.get("frontier_vector") or {}).keys())
    if not isinstance(raw_channels, list) or not raw_channels:
        raise ValueError("first complete box record has no channel list")
    channels = tuple(str(name) for name in raw_channels)
    if len(set(channels)) != len(channels):
        raise ValueError("box channel list contains duplicates")
    if expected_channels is not None and set(channels) != set(expected_channels):
        raise ValueError(
            f"box channels {sorted(channels)!r} do not match expected "
            f"{sorted(str(name) for name in expected_channels)!r}"
        )

    first_tace = first["tace"]
    raw_sizes = first_tace.get("stratum_sizes")
    if not isinstance(raw_sizes, list) or not raw_sizes:
        raise ValueError("first complete box record has no stratum_sizes")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in raw_sizes):
        raise ValueError(f"invalid stratum_sizes {raw_sizes!r}")
    stratum_sizes = tuple(raw_sizes)
    if expected_strata is not None and len(stratum_sizes) != int(expected_strata):
        raise ValueError(
            f"box trace has {len(stratum_sizes)} strata, expected {int(expected_strata)}"
        )
    if len(stratum_sizes) < 2:
        raise ValueError("a vector-yoked box needs a frontier and probe stratum")

    records: list[VectorYokeRecord] = []
    previous_frontier: dict[str, float] | None = None
    first_complete_record: VectorYokeRecord | None = None
    for index in complete_indices:
        row = rows[index]
        if row.get("mode") != "box":
            raise ValueError(f"source row {index} is mode {row.get('mode')!r}, expected 'box'")
        if row.get("applied_decrease") is not False:
            raise ValueError(
                f"source row {index} does not prove zero applied decrease: "
                f"{row.get('applied_decrease')!r}"
            )
        observed_channels = row.get("channels")
        if observed_channels is not None and tuple(observed_channels) != channels:
            raise ValueError(f"source row {index} changed channel order or membership")
        frontier = _vector(
            row.get("frontier_vector"),
            label=f"row[{index}].frontier_vector",
            channels=channels,
            maximum=maximum,
        )
        decision_probe = _vector(
            row.get("probe_vector"),
            label=f"row[{index}].probe_vector",
            channels=channels,
            maximum=maximum,
        )
        if previous_frontier is not None:
            decreased = {
                name: (previous_frontier[name], frontier[name])
                for name in channels
                if frontier[name] < previous_frontier[name] - _TOLERANCE
            }
            if decreased:
                raise ValueError(f"source row {index} contracts its frontier: {decreased}")
        previous_frontier = frontier
        scalar = _finite_intensity(row.get("lambda"), label=f"row[{index}].lambda", maximum=maximum)
        mean_frontier = sum(frontier.values()) / len(frontier)
        if abs(scalar - mean_frontier) > _TOLERANCE:
            raise ValueError(
                f"source row {index} scalar lambda {scalar} != frontier mean {mean_frontier}"
            )
        tace = row["tace"]
        if tuple(tace.get("stratum_sizes") or ()) != stratum_sizes:
            raise ValueError(f"source row {index} changed stratum sizes")
        if tace.get("probe_stratum") != len(stratum_sizes) - 1:
            raise ValueError(f"source row {index} does not identify the top stratum as probe")
        raw_vectors = tace.get("stratum_lambdas")
        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(stratum_sizes):
            raise ValueError(
                f"source row {index} has {len(raw_vectors) if isinstance(raw_vectors, list) else 0} "
                f"vectors for {len(stratum_sizes)} strata"
            )
        strata = tuple(
            _vector(
                vector,
                label=f"row[{index}].stratum[{stratum}]",
                channels=channels,
                maximum=maximum,
            )
            for stratum, vector in enumerate(raw_vectors)
        )
        if not _vectors_close(strata[-2], frontier):
            raise ValueError(f"source row {index} frontier stratum does not match frontier_vector")
        applied_probe = dict(strata[-1])
        _probe_channel(
            decision_probe,
            frontier,
            label=f"row[{index}].decision_probe_vector",
        )
        applied_probe_channel = _probe_channel(
            applied_probe,
            frontier,
            label=f"row[{index}].applied_probe_vector",
        )
        active = row.get("active_channel")
        if active is not None and active not in channels:
            raise ValueError(f"source row {index} has unknown active_channel {active!r}")
        record = VectorYokeRecord(
            source_global_step=steps[index],
            frontier_vector=frontier,
            decision_probe_vector=decision_probe,
            applied_probe_vector=applied_probe,
            stratum_vectors=strata,
            active_channel=active,
            applied_probe_channel=applied_probe_channel,
        )
        if first_complete_record is None:
            first_complete_record = record
        records.append(record)

    assert first_complete_record is not None
    first_mean = sum(first_complete_record.frontier_vector.values()) / len(channels)
    prefix: list[VectorYokeRecord] = []
    for index, row in enumerate(rows[:first_complete]):
        if row.get("mode") != "box" or row.get("warmup_hold") is not True:
            raise ValueError(f"source row {index} lacks vectors but is not an explicit box warm-up")
        scalar = _finite_intensity(row.get("lambda"), label=f"row[{index}].lambda", maximum=maximum)
        if abs(scalar - first_mean) > _TOLERANCE:
            raise ValueError(
                f"warm-up row {index} lambda {scalar} differs from first complete frontier "
                f"mean {first_mean}"
            )
        scalable = row.get("scalable_terms")
        if scalable is not None and set(scalable) != set(channels):
            raise ValueError(f"warm-up row {index} scalable terms differ from box channels")
        prefix.append(
            VectorYokeRecord(
                source_global_step=steps[index],
                frontier_vector=dict(first_complete_record.frontier_vector),
                decision_probe_vector=dict(first_complete_record.decision_probe_vector),
                applied_probe_vector=dict(first_complete_record.applied_probe_vector),
                stratum_vectors=tuple(
                    dict(vector) for vector in first_complete_record.stratum_vectors
                ),
                active_channel=first_complete_record.active_channel,
                applied_probe_channel=first_complete_record.applied_probe_channel,
                warmup_backfill=True,
            )
        )

    all_records = tuple(prefix + records)
    schedule = VectorYokeSchedule(
        channels=channels,
        stratum_sizes=stratum_sizes,
        records=all_records,
        source_path=str(path),
        source_sha256=_sha256_path(path),
        warmup_backfilled_records=len(prefix),
        probe_transition_records=sum(
            not _vectors_close(record.decision_probe_vector, record.applied_probe_vector)
            for record in all_records
        ),
        max_intensity=maximum,
    )
    _validate_schedule(schedule)
    return schedule


def load_schedule(path: Path) -> VectorYokeSchedule:
    """Load a canonical schedule and verify its embedded content digest."""
    payload = json.loads(path.read_text())
    if payload.get("kind") != "lucid_vector_yoke_schedule":
        raise ValueError(f"not a LUCID vector-yoke schedule: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported vector-yoke schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    observed_digest = payload.pop("canonical_sha256", None)
    computed_digest = _sha256_bytes(_canonical_json(payload).encode())
    if observed_digest != computed_digest:
        raise ValueError(
            f"canonical vector-yoke digest mismatch: stored={observed_digest!r}, "
            f"computed={computed_digest!r}"
        )
    channels = tuple(str(name) for name in payload["channels"])
    maximum = float(payload["audit"]["max_intensity"])
    if payload.get("stratum_count") != len(payload.get("stratum_sizes") or ()):
        raise ValueError("canonical vector-yoke stratum_count does not match stratum_sizes")
    if payload["source"].get("row_count") != len(payload.get("records") or ()):
        raise ValueError("canonical vector-yoke source row count does not match records")
    if payload["audit"].get("frontier_monotone") is not True:
        raise ValueError("canonical vector-yoke does not assert a monotone frontier")
    if payload["audit"].get("applied_decreases") != 0:
        raise ValueError("canonical vector-yoke reports an applied decrease")
    if payload["audit"].get("exhaustion_default") != "error":
        raise ValueError("canonical vector-yoke must fail closed on schedule exhaustion")
    records = tuple(
        VectorYokeRecord(
            source_global_step=int(record["source_global_step"]),
            frontier_vector=_vector(
                record["frontier_vector"],
                label=f"record[{index}].frontier_vector",
                channels=channels,
                maximum=maximum,
            ),
            decision_probe_vector=_vector(
                record["decision_probe_vector"],
                label=f"record[{index}].decision_probe_vector",
                channels=channels,
                maximum=maximum,
            ),
            applied_probe_vector=_vector(
                record["applied_probe_vector"],
                label=f"record[{index}].applied_probe_vector",
                channels=channels,
                maximum=maximum,
            ),
            stratum_vectors=tuple(
                _vector(
                    vector,
                    label=f"record[{index}].stratum[{stratum}]",
                    channels=channels,
                    maximum=maximum,
                )
                for stratum, vector in enumerate(record["stratum_vectors"])
            ),
            active_channel=record.get("active_channel"),
            applied_probe_channel=record.get("applied_probe_channel"),
            warmup_backfill=bool(record.get("warmup_backfill", False)),
        )
        for index, record in enumerate(payload["records"])
    )
    schedule = VectorYokeSchedule(
        channels=channels,
        stratum_sizes=tuple(int(size) for size in payload["stratum_sizes"]),
        records=records,
        source_path=str(payload["source"]["path"]),
        source_sha256=str(payload["source"]["sha256"]),
        warmup_backfilled_records=int(payload["audit"]["warmup_backfilled_records"]),
        probe_transition_records=int(payload["audit"]["probe_transition_records"]),
        max_intensity=maximum,
    )
    _validate_schedule(schedule)
    if schedule.canonical_sha256 != observed_digest:
        raise ValueError("loaded vector-yoke schedule does not round-trip to its digest")
    return schedule
