from __future__ import annotations

import json

import pytest

from civilization.engine.stages.stage151_helix_path_weight_model import (
    HelixWeightStore,
    path_id_for_link,
)
from civilization.engine.stages.stage157_helix_weight_snapshot import (
    build_snapshot,
    write_snapshot,
)
from civilization.engine.stages.stage158_helix_weight_recovery import (
    HelixWeightRecovery,
    run_stage158_helix_weight_recovery_smoke,
)
from civilization.engine.stages.stage73_orion_memory_kernel import (
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


def test_stage158_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage158_helix_weight_recovery_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage158_helix_weight_recovery"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()


def test_stage158_clean_verify_no_drift_no_dangling() -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    recovery = HelixWeightRecovery(now_fn=lambda: 0.0)
    report = recovery.verify(snapshot, store, known)
    assert report.hash_matches is True
    assert report.drifted_paths == ()
    assert report.dangling_paths == ()


def test_stage158_drift_detected_after_live_change() -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    # Mutate the live store after snapshot.
    store.approve(sorted(known)[0], 0.2, reason="drift", feedback_source="t")
    recovery = HelixWeightRecovery(now_fn=lambda: 0.0)
    report = recovery.verify(snapshot, store, known)
    assert len(report.drifted_paths) == 1


def test_stage158_tamper_detection_on_read(tmp_path) -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    path = tmp_path / "snap.json"
    write_snapshot(snapshot, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["weights"][0]["value"] = 999.0
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    recovery = HelixWeightRecovery(now_fn=lambda: 0.0)
    # Reading a tampered snapshot file raises (delegates to stage157 read_snapshot).
    from civilization.engine.stages.stage157_helix_weight_snapshot import read_snapshot
    with pytest.raises(ValueError):
        read_snapshot(path)


def test_stage158_recovery_is_idempotent_and_matches_snapshot() -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    recovery = HelixWeightRecovery(now_fn=lambda: 0.0)
    restored_a, _ = recovery.recover(snapshot)
    restored_b, _ = recovery.recover(snapshot)
    assert restored_a.to_dict() == restored_b.to_dict()
    assert all(restored_a.weights[w.path_id] == w for w in snapshot.weights)


def test_stage158_dangling_reported_and_fail_closed() -> None:
    store = HelixWeightStore(now_fn=lambda: 0.0)
    real = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    ghost = MemoryLink("g-1", "g-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    store.register_link(real)
    store.register_link(ghost)
    known_all = {path_id_for_link(real), path_id_for_link(ghost)}
    snapshot = build_snapshot(store, known_path_ids=known_all, now_fn=lambda: 0.0)

    recovery = HelixWeightRecovery(now_fn=lambda: 0.0)
    # Verify against a live link set that omits the ghost link.
    report = recovery.verify(snapshot, store, {path_id_for_link(real)})
    assert path_id_for_link(ghost) in report.dangling_paths
    # fail_closed raises on dangling.
    with pytest.raises(KeyError):
        recovery.fail_closed_on_dangling(snapshot, {path_id_for_link(real)})


def test_stage158_recovery_does_not_modify_live_store() -> None:
    store, known = _store()
    snapshot = build_snapshot(store, known_path_ids=known, now_fn=lambda: 0.0)
    before = store.to_dict()
    HelixWeightRecovery(now_fn=lambda: 0.0).recover(snapshot)
    assert store.to_dict() == before  # live store untouched by recovery
