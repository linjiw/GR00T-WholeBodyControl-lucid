import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.research.practice_utility import motion_pool as MP
from gear_sonic.research.practice_utility.schema import ContextKey, sha256_of
from scripts.practice_utility import create_probe_manifest as C


def row(name: str, failure: float, probability: float):
    context = ContextKey(
        motion_key=name,
        motion_hash=(name[0] * 64),
        bin_index=0,
        bin_start_frame=0,
        bin_end_frame=50,
        perturbation_group="native",
        severity_level=0,
        encoder_mode="g1",
    )
    return {
        **context.to_dict(),
        "context_id": context.context_id,
        "failure_rate": failure,
        "sampling_probability": probability,
    }


def test_group_snapshot_paths_preserves_replicate_origins():
    grouped = C.group_snapshot_paths(["late=/a.json", "late=/b.json", "early=/c.json"])
    assert [str(path) for path in grouped["late"]] == ["/a.json", "/b.json"]
    assert [str(path) for path in grouped["early"]] == ["/c.json"]


def test_group_snapshot_paths_rejects_malformed_argument():
    with pytest.raises(SystemExit, match="STAGE=PATH"):
        C.group_snapshot_paths(["missing-separator"])


def test_intersection_keeps_only_contexts_present_in_every_origin():
    shared_a = row("alpha", 0.2, 0.1)
    shared_b = {**shared_a, "failure_rate": 0.6, "sampling_probability": 0.3}
    first_contexts = [shared_a, row("beta", 0.9, 0.2)]
    second_contexts = [shared_b, row("gamma", 0.1, 0.4)]
    first = {"num_active_bins": len(first_contexts), "contexts": first_contexts}
    second = {"num_active_bins": len(second_contexts), "contexts": second_contexts}
    merged = C.intersect_snapshots([first, second])
    assert merged["num_origins"] == 2
    assert len(merged["contexts"]) == 1
    assert merged["contexts"][0]["context_id"] == shared_a["context_id"]
    assert merged["contexts"][0]["failure_rate"] == pytest.approx(0.4)
    assert merged["contexts"][0]["sampling_probability"] == pytest.approx(0.2)
    assert merged["contexts"][0]["resident_multiplicity_by_origin"] == [1, 1]
    assert merged["contexts"][0]["origin_count"] == 2


def test_intersection_canonicalizes_identical_with_replacement_copies():
    duplicate = row("alpha", 0.2, 0.1)
    duplicate.update(global_bin_id=7, num_episodes=8.0, num_failures=3.0)
    merged = C.intersect_snapshots(
        [{"num_active_bins": 2, "contexts": [duplicate, dict(duplicate)]}]
    )
    assert len(merged["contexts"]) == 1
    canonical = merged["contexts"][0]
    assert canonical["sampling_probability"] == pytest.approx(0.2)
    assert canonical["resident_multiplicity_by_origin"] == [2]
    assert canonical["num_episodes"] == 8.0
    assert canonical["num_failures"] == 3.0


def test_intersection_rejects_conflicting_with_replacement_copies():
    duplicate = row("alpha", 0.2, 0.1)
    conflict = {**duplicate, "num_failures": 4.0}
    with pytest.raises(ValueError, match="conflicting serialized rows"):
        C.intersect_snapshots([{"num_active_bins": 2, "contexts": [duplicate, conflict]}])


def test_intersection_binds_num_active_bins_to_raw_rows_not_unique_contexts():
    duplicate = row("alpha", 0.2, 0.1)
    with pytest.raises(ValueError, match="raw serialized context rows"):
        C.intersect_snapshots([{"num_active_bins": 1, "contexts": [duplicate, dict(duplicate)]}])


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_grade_fixture(tmp_path: Path, seeds=(9300, 9301)):
    records = []
    contexts = []
    for index in range(8):
        key = f"motion_{index}__A{index:03d}"
        content_sha256 = f"{index + 1:x}" * 64
        records.append(
            {
                "motion_key": key,
                "content_sha256": content_sha256,
                "path": str(tmp_path / f"{key}.pkl"),
                "num_frames": 120,
                "fps": 30.0,
                "family": ("walk", "jump", "dance")[index % 3],
            }
        )
        context = ContextKey(
            motion_key=key,
            motion_hash=content_sha256,
            bin_index=0,
            bin_start_frame=0,
            bin_end_frame=50,
        )
        contexts.append(
            {
                **context.to_dict(),
                "context_id": context.context_id,
                "failure_rate": (index + 1) / 10.0,
                "sampling_probability": 1.0 / 8.0,
            }
        )

    pool_sha256 = sha256_of(
        {
            "source_root": "/frozen/pool",
            "records": [
                {
                    "motion_key": record["motion_key"],
                    "content_sha256": record["content_sha256"],
                }
                for record in records
            ],
        }
    )
    pool_path = write_json(
        tmp_path / "pool.json",
        {
            "kind": "practice_utility_motion_pool",
            "schema_version": 1,
            "pool_sha256": pool_sha256,
            "source_root": "/frozen/pool",
            "motions": records,
        },
    )
    assignment = {record["motion_key"]: "adaptation" for record in records}
    split_sha256 = sha256_of(
        {
            "assignment": dict(sorted(assignment.items())),
            "linkage": "performer",
            "seed": 20260818,
            "pool_sha256": pool_sha256,
        }
    )
    split_path = write_json(
        tmp_path / "split.json",
        {
            "kind": "practice_utility_group_disjoint_split",
            "schema_version": 1,
            "linkage": "performer",
            "seed": 20260818,
            "pool_sha256": pool_sha256,
            "split_sha256": split_sha256,
            "assignment": assignment,
        },
    )

    origin_step = 36
    origins = {}
    for seed in seeds:
        snapshot_path = write_json(
            tmp_path / f"snapshot_{seed}.json",
            {
                "kind": "practice_utility_sampler_snapshot",
                "schema_version": 1,
                "seed": seed,
                "branch_id": f"origin_s{seed}",
                "global_step": origin_step,
                "snapshot_timeline_fps": 50.0,
                "num_active_bins": len(contexts),
                "contexts": contexts,
            },
        )
        origins[str(seed)] = {
            "seed": seed,
            "origin_step": origin_step,
            "snapshot": str(snapshot_path),
            "snapshot_sha256": digest(snapshot_path),
            "resident_context_ids": sorted(item["context_id"] for item in contexts),
            "num_resident_contexts": len(contexts),
            "num_active_context_rows": len(contexts),
            "num_duplicate_active_context_rows": 0,
            "settled": True,
            "blockers": [],
        }
    common = sorted(item["context_id"] for item in contexts)
    origin_map_path = write_json(
        tmp_path / "origin_map.json",
        {
            "kind": "practice_utility_probe_origin_map",
            "schema_version": 1,
            "experiment_id": "origin",
            "stage": "late",
            "origin_step": origin_step,
            "motion_pool_manifest_sha256": pool_sha256,
            "dev_suite_sha256": split_sha256,
            "seeds": list(seeds),
            "origins": origins,
            "common_resident_context_ids": common,
            "num_common_resident_contexts": len(common),
            "usable_for_manifest_selection": True,
        },
    )
    return {
        "pool": pool_path,
        "split": split_path,
        "origin_map": origin_map_path,
        "pool_sha256": pool_sha256,
        "split_sha256": split_sha256,
        "contexts": contexts,
        "origins": origins,
    }


def load_fixture_origins(fixture, seeds=(9300, 9301)):
    pool, pool_sha256, _ = C.verify_pool_manifest(fixture["pool"])
    clips = C.load_clip_index(pool)
    references = C.parse_origin_map_references(
        [["late", str(fixture["origin_map"]), digest(fixture["origin_map"])]]
    )
    return C.load_origin_stages(
        references,
        seeds=list(seeds),
        clips=clips,
        pool_sha256=pool_sha256,
        split_sha256=fixture["split_sha256"],
    )


def test_source_bin_bounds_converts_50hz_snapshot_to_30hz_clip():
    assert C.source_bin_bounds(
        50,
        100,
        source_fps=30.0,
        snapshot_timeline_fps=50.0,
        source_num_frames=180,
    ) == (30, 60)


def test_claim_grade_origin_map_rejects_snapshot_timeline_mismatch(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    snapshot_path = Path(fixture["origins"]["9300"]["snapshot"])
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["snapshot_timeline_fps"] = 60.0
    write_json(snapshot_path, snapshot)
    fixture["origins"]["9300"]["snapshot_sha256"] = digest(snapshot_path)
    origin_map = json.loads(fixture["origin_map"].read_text())
    origin_map["origins"] = fixture["origins"]
    write_json(fixture["origin_map"], origin_map)

    with pytest.raises(ValueError, match="timeline FPS"):
        load_fixture_origins(fixture)


def test_claim_grade_origin_map_requires_embedded_snapshot_timeline(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    snapshot_path = Path(fixture["origins"]["9300"]["snapshot"])
    snapshot = json.loads(snapshot_path.read_text())
    snapshot.pop("snapshot_timeline_fps")
    write_json(snapshot_path, snapshot)
    fixture["origins"]["9300"]["snapshot_sha256"] = digest(snapshot_path)
    origin_map = json.loads(fixture["origin_map"].read_text())
    origin_map["origins"] = fixture["origins"]
    write_json(fixture["origin_map"], origin_map)

    with pytest.raises(ValueError, match="timeline FPS"):
        load_fixture_origins(fixture)


def test_claim_grade_origin_map_requires_exactly_one_origin_per_seed(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    payload = json.loads(fixture["origin_map"].read_text())
    payload["origins"].pop("9301")
    write_json(fixture["origin_map"], payload)
    with pytest.raises(ValueError, match="exactly one origin per declared seed"):
        load_fixture_origins(fixture)


def test_claim_grade_origin_map_rejects_snapshot_reused_across_seeds(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    payload = json.loads(fixture["origin_map"].read_text())
    payload["origins"]["9301"]["snapshot"] = payload["origins"]["9300"]["snapshot"]
    payload["origins"]["9301"]["snapshot_sha256"] = payload["origins"]["9300"]["snapshot_sha256"]
    write_json(fixture["origin_map"], payload)
    with pytest.raises(ValueError, match="reuses one snapshot path"):
        load_fixture_origins(fixture)


def test_claim_grade_origin_map_rejects_unverified_snapshot_bytes(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    snapshot = Path(fixture["origins"]["9300"]["snapshot"])
    snapshot.write_text(snapshot.read_text() + "\n")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        load_fixture_origins(fixture)


def test_claim_grade_origin_map_rejects_snapshot_from_another_seed(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    snapshot = Path(fixture["origins"]["9301"]["snapshot"])
    payload = json.loads(snapshot.read_text())
    payload["branch_id"] = "origin_s9300"
    write_json(snapshot, payload)
    origin_map = json.loads(fixture["origin_map"].read_text())
    origin_map["origins"]["9301"]["snapshot_sha256"] = digest(snapshot)
    write_json(fixture["origin_map"], origin_map)
    with pytest.raises(ValueError, match="branch_id must be 'origin_s9301'"):
        load_fixture_origins(fixture)


def test_claim_grade_origin_map_rejects_context_motion_hash_outside_pool(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    snapshot = Path(fixture["origins"]["9300"]["snapshot"])
    payload = json.loads(snapshot.read_text())
    original = payload["contexts"][0]
    context = ContextKey.from_dict({**original, "motion_hash": "f" * 64})
    payload["contexts"][0].update(context.to_dict(), context_id=context.context_id)
    write_json(snapshot, payload)
    origin_map = json.loads(fixture["origin_map"].read_text())
    origin_map["origins"]["9300"]["snapshot_sha256"] = digest(snapshot)
    origin_map["origins"]["9300"]["resident_context_ids"] = sorted(
        item["context_id"] for item in payload["contexts"]
    )
    write_json(fixture["origin_map"], origin_map)
    with pytest.raises(ValueError, match="ContextKey motion hash mismatch"):
        load_fixture_origins(fixture)


def test_verify_pool_manifest_rejects_tampered_logical_hash(tmp_path):
    fixture = claim_grade_fixture(tmp_path)
    payload = json.loads(fixture["pool"].read_text())
    payload["motions"][0]["content_sha256"] = "f" * 64
    write_json(fixture["pool"], payload)
    with pytest.raises(ValueError, match="pool logical hash mismatch"):
        C.verify_pool_manifest(fixture["pool"])


def live_pool_payload(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    key = "walk_forward_001__A001"
    path = source / f"{key}.pkl"
    joblib.dump(
        {key: {"dof": np.arange(24, dtype=np.float32).reshape(8, 3), "fps": 30.0}},
        path,
    )
    scan = MP.scan_pool(source)
    return {
        "source_root": scan.source_root,
        "deduplicated": False,
        "pool_sha256": MP.pool_sha256(scan),
        "motions": [record.to_dict() for record in scan.records],
    }, path


def test_pool_source_bytes_rescan_and_detect_mid_selection_mutation(tmp_path):
    pool, path = live_pool_payload(tmp_path)
    binding = C.verify_pool_source_bytes(pool)
    assert binding["source_file_count"] == 1
    assert len(binding["source_files_sha256"]) == 64
    C.reverify_pool_source_bytes(binding)

    key = pool["motions"][0]["motion_key"]
    joblib.dump({key: {"dof": np.ones((8, 3), dtype=np.float32), "fps": 30.0}}, path)
    with pytest.raises(ValueError, match="changed during manifest creation"):
        C.reverify_pool_source_bytes(binding)
    with pytest.raises(ValueError, match="source bytes differ"):
        C.verify_pool_source_bytes(pool)


def test_pool_source_rescan_rejects_mutable_record_path(tmp_path):
    pool, path = live_pool_payload(tmp_path)
    alternate = tmp_path / "alternate.pkl"
    alternate.write_bytes(path.read_bytes())
    pool["motions"][0]["path"] = str(alternate)
    with pytest.raises(ValueError, match="path is not its rescanned source file"):
        C.verify_pool_source_bytes(pool)


def test_claim_grade_main_refuses_dirty_tree_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "git_status", lambda: [" M claim_bearing.py"])
    output = tmp_path / "manifest.json"
    with pytest.raises(SystemExit, match="requires a clean committed tree"):
        C.main(
            [
                "--origin-map",
                "late",
                str(tmp_path / "origin.json"),
                "a" * 64,
                "--pool-manifest",
                str(tmp_path / "pool.json"),
                "--split-manifest",
                str(tmp_path / "split.json"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_main_writes_separate_hash_complete_claim_grade_receipt(tmp_path, monkeypatch):
    fixture = claim_grade_fixture(tmp_path)
    monkeypatch.setattr(C, "git_status", lambda: [])
    monkeypatch.setattr(C, "git_sha", lambda: "c" * 40)
    monkeypatch.setattr(
        C,
        "verify_pool_source_bytes",
        lambda pool: {
            "source_root": "/frozen/pool",
            "pool_sha256": pool["pool_sha256"],
            "source_file_count": 8,
            "source_files_sha256": "d" * 64,
            "files": {"bound": {"path": "/frozen/bound", "file_sha256": "e" * 64}},
        },
    )
    monkeypatch.setattr(C, "reverify_pool_source_bytes", lambda binding: None)
    output = tmp_path / "probe_manifest.json"
    receipt = tmp_path / "probe_creation.json"
    result = C.main(
        [
            "--origin-map",
            "late",
            str(fixture["origin_map"]),
            digest(fixture["origin_map"]),
            "--pool-manifest",
            str(fixture["pool"]),
            "--split-manifest",
            str(fixture["split"]),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
            "--campaign-id",
            "screen_v2_test",
            "--contexts-per-stage",
            "8",
            "--seeds",
            "9300",
            "9301",
            "--skip-motion-features",
        ]
    )
    assert result == 0
    manifest = json.loads(output.read_text())
    creation = json.loads(receipt.read_text())
    assert manifest["campaign_id"] == "screen_v2_test"
    assert creation["claim_grade_inputs"] is True
    assert creation["selection"]["snapshot_timeline_fps"] == 50.0
    assert creation["selection"]["counts_per_stage"]["late"] == {
        "source_origin_count": 2,
        "common_resident_contexts": 8,
        "partition_candidates": 8,
        "skipped_outside_partition": 0,
        "skipped_missing_from_pool": 0,
        "selected_contexts": 8,
    }
    assert creation["inputs"]["origin_maps"]["late"]["file_sha256"] == digest(fixture["origin_map"])
    assert set(creation["inputs"]["origin_maps"]["late"]["snapshots"]) == {
        "9300",
        "9301",
    }
    assert creation["manifest"]["file_sha256"] == digest(output)
    assert len(creation["launcher_sha256"]) == 64
    assert creation["git_sha"] == "c" * 40
    assert creation["git_status_short"] == []
    assert creation["inputs"]["pool_manifest"]["source_bytes"] == {
        "source_root": "/frozen/pool",
        "source_file_count": 8,
        "source_files_sha256": "d" * 64,
    }
