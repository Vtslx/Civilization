from __future__ import annotations

import json

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage151_helix_path_weight_model import (
    HelixWeightStore,
    path_id_for_link,
)
from experiments.civilization_transformer_qwen3.analysis.stage157_helix_weight_snapshot import (
    build_snapshot,
    canonical_json,
    compute_sha256,
    read_snapshot,
    run_stage157_helix_weight_snapshot_smoke,
    validate_weight_table,
    write_snapshot,
)
from experiments.civilization_transformer_qwen3.analysis.stage73_orion_memory_kernel import (
    MemoryLink,
    MemoryLinkType,
)


def _store() -> tuple[HelixWeightStore, set[str]]:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    a = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    b = MemoryLink("c-3", "c-4", MemoryLinkType.PROCEDURE, 0.5, 0.0)
    store.register_link(a)
    store.register_link(b)
    store.approve(path_id_for_link(a), 1.5, reason="boost", feedback_source="t")
    return store, {path_id_for_link(a), path_id_for_link(b)}


def test_stage157_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage157_helix_weight_snapshot_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage157_helix_weight_snapshot"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()


def test_stage157_snapshot_hash_is_deterministic() -> None:
    store, known = _store()
    s1 = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    s2 = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    assert s1.sha256 == s2.sha256
    assert compute_sha256(s1.payload()) == s1.sha256


def test_stage157_hash_changes_when_weight_changes() -> None:
    store, known = _store()
    before = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    store.approve(sorted(known)[0], 0.2, reason="change", feedback_source="t")
    after = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    assert before.sha256 != after.sha256


def test_stage157_atomic_write_and_round_trip(tmp_path) -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    path = write_snapshot(snapshot, tmp_path / "snap.json")
    assert path.exists()
    assert not (tmp_path / "snap.json.tmp").exists()  # no tmp leftover
    restored = read_snapshot(path)
    assert restored.sha256 == snapshot.sha256
    assert restored.weights == snapshot.weights
    assert restored.update_events == snapshot.update_events


def test_stage157_tamper_detection(tmp_path) -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    path = tmp_path / "snap.json"
    write_snapshot(snapshot, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["weights"][0]["value"] = 999.0  # tamper without updating hash
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        read_snapshot(path)


def test_stage157_validated_rejects_dangling() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    real = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    ghost = MemoryLink("g-1", "g-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    store.register_link(real)
    store.register_link(ghost)
    # ghost is not in known_path_ids -> dangling -> fail-closed.
    with pytest.raises(KeyError):
        build_snapshot(store, known_path_ids={path_id_for_link(real)}, now_fn=lambda: 0.0)


def test_stage157_snapshot_contains_weights_and_events() -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, last_n_events=10, now_fn=lambda: 0.0)
    assert snapshot.weight_count == 2
    assert snapshot.max_version >= 1
    assert len(snapshot.weights) == 2
    assert len(snapshot.update_events) >= 1


def test_stage157_canonical_json_is_sorted_and_compact() -> None:
    payload = {"b": 2, "a": 1, "nested": {"z": 0, "a": 1}}
    cj = canonical_json(payload)
    assert cj == '{"a":1,"b":2,"nested":{"a":1,"z":0}}'  # sorted keys, compact
