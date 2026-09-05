import copy
import json

import pytest

from gear_sonic.research.practice_utility import vector_yoke as VY

CHANNELS = ("joint", "push")


def row(step, *, joint=1.0, push=1.0, active="joint", warmup=False):
    frontier = {"joint": joint, "push": push}
    probe = dict(frontier)
    probe[active] = min(3.0, probe[active] + 0.125)
    if warmup:
        return {
            "global_step": step,
            "mode": "box",
            "lambda": 0.5 * (joint + push),
            "warmup_hold": True,
            "scalable_terms": list(CHANNELS),
        }
    return {
        "global_step": step,
        "mode": "box",
        "lambda": 0.5 * (joint + push),
        "frontier_vector": frontier,
        "probe_vector": probe,
        "active_channel": active,
        "channels": list(CHANNELS),
        "applied_decrease": False,
        "tace": {
            "stratum_sizes": [2, 6, 2],
            "stratum_lambdas": [
                {name: value * 0.5 for name, value in frontier.items()},
                dict(frontier),
                dict(probe),
            ],
            "probe_stratum": 2,
        },
    }


def write_trace(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return path


def valid_schedule(tmp_path):
    path = write_trace(
        tmp_path / "box.jsonl",
        [
            row(1, warmup=True),
            row(2),
            row(3, joint=1.125, active="push"),
        ],
    )
    return VY.canonicalize_box_trace(
        path,
        expected_channels=CHANNELS,
        expected_strata=3,
    )


def test_canonical_schedule_preserves_every_applied_stratum_vector(tmp_path):
    schedule = valid_schedule(tmp_path)
    assert len(schedule.records) == 3
    assert schedule.stratum_sizes == (2, 6, 2)
    assert schedule.records[2].frontier_vector == {"joint": 1.125, "push": 1.0}
    assert schedule.records[2].decision_probe_vector == {"joint": 1.125, "push": 1.125}
    assert schedule.records[2].stratum_vectors[-1] == schedule.records[2].applied_probe_vector


def test_missing_warmup_vectors_are_backfilled_only_from_proved_initial_hold(tmp_path):
    schedule = valid_schedule(tmp_path)
    assert schedule.warmup_backfilled_records == 1
    assert schedule.records[0].warmup_backfill is True
    assert schedule.records[0].stratum_vectors == schedule.records[1].stratum_vectors


def test_probe_rotation_transition_keeps_decision_and_applied_vectors(tmp_path):
    first = row(1)
    transition = row(2, joint=1.125)
    transition["probe_vector"] = {"joint": 1.25, "push": 1.0}
    transition["tace"]["stratum_lambdas"][-1] = {"joint": 1.125, "push": 1.125}
    schedule = VY.canonicalize_box_trace(
        write_trace(tmp_path / "rotation.jsonl", [first, transition])
    )
    record = schedule.records[-1]
    assert record.active_channel == "joint"
    assert record.applied_probe_channel == "push"
    assert record.decision_probe_vector != record.applied_probe_vector
    assert schedule.probe_transition_records == 1


def test_non_warmup_missing_vectors_are_rejected(tmp_path):
    rows = [row(1, warmup=True), row(2), row(3)]
    rows[0].pop("warmup_hold")
    with pytest.raises(ValueError, match="not an explicit box warm-up"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", rows))


def test_missing_vectors_after_materialization_are_rejected(tmp_path):
    rows = [row(1), row(2, warmup=True), row(3)]
    with pytest.raises(ValueError, match="contiguous warm-up prefix"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", rows))


def test_frontier_contraction_is_rejected(tmp_path):
    rows = [row(1, joint=1.125), row(2, joint=1.0)]
    with pytest.raises(ValueError, match="contracts its frontier"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", rows))


def test_missing_no_decrease_proof_is_rejected(tmp_path):
    rows = [row(1)]
    rows[0]["applied_decrease"] = None
    with pytest.raises(ValueError, match="does not prove zero applied decrease"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", rows))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item["frontier_vector"].pop("push"), "channels"),
        (lambda item: item["tace"].update(probe_stratum=1), "top stratum as probe"),
        (lambda item: item["tace"]["stratum_lambdas"].pop(), "vectors for 3 strata"),
    ],
)
def test_malformed_exposure_contract_is_rejected(tmp_path, mutation, message):
    item = row(1)
    mutation(item)
    with pytest.raises(ValueError, match=message):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", [item]))


def test_stratum_sizes_must_stay_fixed(tmp_path):
    rows = [row(1), row(2)]
    rows[1]["tace"]["stratum_sizes"] = [3, 5, 2]
    with pytest.raises(ValueError, match="changed stratum sizes"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", rows))


def test_intensity_must_be_inside_the_declared_bound(tmp_path):
    item = row(1)
    item["lambda"] = 9.0
    with pytest.raises(ValueError, match="must be finite and in"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", [item]))


def test_global_steps_must_be_consecutive(tmp_path):
    with pytest.raises(ValueError, match="strictly consecutive"):
        VY.canonicalize_box_trace(write_trace(tmp_path / "bad.jsonl", [row(1), row(3)]))


def test_schedule_round_trip_is_digest_stable(tmp_path):
    schedule = valid_schedule(tmp_path)
    path = tmp_path / "schedule.json"
    schedule.write(path)
    loaded = VY.load_schedule(path)
    assert loaded.to_dict() == schedule.to_dict()
    assert loaded.canonical_sha256 == schedule.canonical_sha256


def test_tampered_schedule_is_rejected(tmp_path):
    schedule = valid_schedule(tmp_path)
    path = tmp_path / "schedule.json"
    schedule.write(path)
    payload = json.loads(path.read_text())
    payload["records"][-1]["frontier_vector"]["push"] = 2.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        VY.load_schedule(path)


def test_rehashed_but_contract_invalid_schedule_is_rejected(tmp_path):
    schedule = valid_schedule(tmp_path)
    payload = schedule.to_dict()
    payload["records"][-1]["frontier_vector"]["joint"] = 0.5
    unsigned = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    payload["canonical_sha256"] = VY._sha256_bytes(VY._canonical_json(unsigned).encode())
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frontier stratum mismatch|contracts its frontier"):
        VY.load_schedule(path)


def test_exhaustion_is_fail_closed_unless_hold_is_explicit(tmp_path):
    schedule = valid_schedule(tmp_path)
    with pytest.raises(IndexError, match="exhausted"):
        schedule.record(len(schedule.records))
    assert schedule.record(100, hold_last=True) == schedule.records[-1]


def test_cursor_resume_matches_uninterrupted_sequence(tmp_path):
    schedule = valid_schedule(tmp_path)
    uninterrupted = VY.VectorYokeCursor(schedule)
    expected = [uninterrupted.take() for _ in range(3)]

    first = VY.VectorYokeCursor(schedule)
    observed = [first.take()]
    state = first.state_dict()
    resumed = VY.VectorYokeCursor(schedule)
    resumed.load_state_dict(state)
    observed.extend(resumed.take() for _ in range(2))
    assert observed == expected


def test_cursor_refuses_a_different_schedule_on_resume(tmp_path):
    schedule = valid_schedule(tmp_path)
    state = VY.VectorYokeCursor(schedule).state_dict()
    changed = copy.deepcopy(schedule.to_dict())
    changed["source"]["path"] = "/different/source"
    changed_payload = {k: v for k, v in changed.items() if k != "canonical_sha256"}
    changed["canonical_sha256"] = VY._sha256_bytes(VY._canonical_json(changed_payload).encode())
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    other = VY.VectorYokeCursor(VY.load_schedule(path))
    with pytest.raises(ValueError, match="schedule hash mismatch"):
        other.load_state_dict(state)
