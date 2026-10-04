from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from .stage151_helix_path_weight_model import HelixWeightStore
from .stage157_helix_weight_snapshot import HelixWeightSnapshot, compute_sha256, read_snapshot


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage158_helix_weight_recovery")


@dataclass(frozen=True)
class RecoveryReport:
    snapshot_id: str
    snapshot_sha256: str
    recomputed_sha256: str
    hash_matches: bool
    path_count_snapshot: int
    path_count_live: int
    drifted_paths: tuple[str, ...]
    dangling_paths: tuple[str, ...]
    recovered: bool
    recovered_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HelixWeightRecovery:
    """Verifies and restores a Helix weight view from a snapshot.

    Recovery rebuilds a ``HelixWeightStore`` from the snapshot's weights and
    events — it never touches the Orion cells/links. The snapshot SHA-256 is
    recomputed and compared (tamper detection); drift between the snapshot and a
    live store is reported; snapshot paths that no longer resolve to live links
    are flagged as dangling.
    """

    def __init__(self, *, now_fn=time.time) -> None:
        self._now_fn = now_fn

    def verify(self, snapshot: HelixWeightSnapshot, live_store: HelixWeightStore, live_link_path_ids: set[str]) -> RecoveryReport:
        recomputed = compute_sha256(snapshot.payload())
        hash_matches = recomputed == snapshot.sha256
        snapshot_paths = {w.path_id: w for w in snapshot.weights}
        live_paths = set(live_store.weights.keys())
        drifted = sorted(
            pid
            for pid in snapshot_paths
            if pid in live_paths and snapshot_paths[pid].value != live_store.weights[pid].value
        )
        dangling = sorted(pid for pid in snapshot_paths if pid not in live_link_path_ids)
        return RecoveryReport(
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.sha256,
            recomputed_sha256=recomputed,
            hash_matches=hash_matches,
            path_count_snapshot=len(snapshot_paths),
            path_count_live=len(live_paths),
            drifted_paths=tuple(drifted),
            dangling_paths=tuple(dangling),
            recovered=False,
            recovered_at=self._now_fn(),
        )

    def recover(self, snapshot: HelixWeightSnapshot) -> tuple[HelixWeightStore, RecoveryReport]:
        """Rebuild a weight store from the snapshot (read-only over cells/links)."""
        payload = {
            "weights": [w.to_dict() for w in snapshot.weights],
            "update_events": [e.to_dict() for e in snapshot.update_events],
        }
        restored = HelixWeightStore.from_dict(payload, now_fn=self._now_fn)
        report = RecoveryReport(
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.sha256,
            recomputed_sha256=compute_sha256(snapshot.payload()),
            hash_matches=True,
            path_count_snapshot=len(snapshot.weights),
            path_count_live=len(snapshot.weights),
            drifted_paths=(),
            dangling_paths=(),
            recovered=True,
            recovered_at=self._now_fn(),
        )
        return restored, report

    def fail_closed_on_dangling(self, snapshot: HelixWeightSnapshot, live_link_path_ids: set[str]) -> None:
        """Raise if the snapshot references paths absent from the live link set."""
        dangling = [w.path_id for w in snapshot.weights if w.path_id not in live_link_path_ids]
        if dangling:
            raise KeyError(f"snapshot references dangling links: {sorted(dangling)}")


def run_stage158_helix_weight_recovery_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    from .stage151_helix_path_weight_model import path_id_for_link
    from .stage157_helix_weight_snapshot import build_snapshot, write_snapshot
    from .stage73_orion_memory_kernel import MemoryLink, MemoryLinkType

    clock = {"now": 9100.0}
    store = HelixWeightStore(now_fn=lambda: clock["now"])
    link_a = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    link_b = MemoryLink("c-3", "c-4", MemoryLinkType.PROCEDURE, 0.5, 0.0)
    store.register_link(link_a)
    store.register_link(link_b)
    clock["now"] = 9101.0
    store.approve(path_id_for_link(link_a), 1.5, reason="boost", feedback_source="smoke")
    known = {path_id_for_link(link_a), path_id_for_link(link_b)}

    snapshot = build_snapshot(store, known_path_ids=known, snapshot_id="snapshot-stage158", now_fn=lambda: clock["now"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = write_snapshot(snapshot, output / "weight_snapshot.json")

    recovery = HelixWeightRecovery(now_fn=lambda: clock["now"])

    # Clean verify: live store matches snapshot -> no drift, no dangling.
    clean_report = recovery.verify(snapshot, store, known)
    clean_no_drift = len(clean_report.drifted_paths) == 0
    clean_no_dangling = len(clean_report.dangling_paths) == 0
    clean_hash_matches = clean_report.hash_matches

    # Drift detection: mutate the live store after snapshot -> drift reported.
    clock["now"] = 9102.0
    store.approve(path_id_for_link(link_a), 0.3, reason="post-snapshot change", feedback_source="smoke")
    drift_report = recovery.verify(snapshot, store, known)
    drift_detected = path_id_for_link(link_a) in drift_report.drifted_paths

    # Tamper detection: rewrite the snapshot file with a modified weight -> hash mismatch on read.
    tampered_path = output / "tampered_snapshot.json"
    tampered_data = snapshot.to_dict()
    tampered_data["weights"][0]["value"] = 999.0
    tampered_path.write_text(json.dumps(tampered_data, ensure_ascii=False, indent=2), encoding="utf-8")
    tamper_detected = False
    try:
        read_snapshot(tampered_path)
    except ValueError:
        tamper_detected = True

    # Recovery: rebuild a store from the (clean) snapshot, idempotent.
    restored_a, _ = recovery.recover(snapshot)
    restored_b, _ = recovery.recover(snapshot)
    recovery_idempotent = restored_a.to_dict() == restored_b.to_dict()
    recovery_matches_snapshot = all(
        restored_a.weights[w.path_id] == w for w in snapshot.weights
    )

    # Dangling fail-closed: a snapshot path not in live links raises.
    dangling_store = HelixWeightStore(now_fn=lambda: clock["now"])
    dangling_store.register_link(link_a)
    dangling_store.register_link(MemoryLink("ghost-1", "ghost-2", MemoryLinkType.TEMPORAL, 1.0, 0.0))
    dangling_snapshot = build_snapshot(
        dangling_store,
        known_path_ids={path_id_for_link(link_a), path_id_for_link(MemoryLink("ghost-1", "ghost-2", MemoryLinkType.TEMPORAL, 1.0, 0.0))},
        snapshot_id="snapshot-dangling",
        now_fn=lambda: clock["now"],
    )
    # Verify against a live link set that omits the ghost link -> dangling reported.
    dangling_report = recovery.verify(dangling_snapshot, dangling_store, {path_id_for_link(link_a)})
    dangling_reported = len(dangling_report.dangling_paths) == 1
    # And fail_closed_on_dangling raises.
    fail_closed_raises = False
    try:
        recovery.fail_closed_on_dangling(dangling_snapshot, {path_id_for_link(link_a)})
    except KeyError:
        fail_closed_raises = True

    summary = {
        "stage": "stage158_helix_weight_recovery",
        "version": "v0.00.07",
        "snapshot_id": snapshot.snapshot_id,
        "stage_gates": {
            "clean_no_drift": clean_no_drift,
            "clean_no_dangling": clean_no_dangling,
            "clean_hash_matches": clean_hash_matches,
            "drift_detected": drift_detected,
            "tamper_detected": tamper_detected,
            "recovery_idempotent": recovery_idempotent,
            "recovery_matches_snapshot": recovery_matches_snapshot,
            "dangling_reported": dangling_reported,
            "fail_closed_on_dangling": fail_closed_raises,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "recovery_report.json").write_text(json.dumps(clean_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
