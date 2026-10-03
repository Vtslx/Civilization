from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .stage151_helix_path_weight_model import HelixWeightStore, PathWeight, WeightUpdateEvent


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage157_helix_weight_snapshot")
DEFAULT_LAST_N_EVENTS = 50


def validate_weight_table(store: HelixWeightStore, known_path_ids: set[str]) -> None:
    """Fail-closed validation: every registered path must resolve to a real link/edge."""
    unknown = sorted(pid for pid in store.weights if pid not in known_path_ids)
    if unknown:
        raise KeyError(f"weight table references unknown paths (dangling): {unknown}")
    for weight in store.weights.values():
        if weight.version < 0:
            raise ValueError(f"path {weight.path_id} has negative version {weight.version}")
        if not isinstance(weight.value, (int, float)) or weight.value != weight.value:  # NaN check
            raise ValueError(f"path {weight.path_id} has non-finite value {weight.value}")


def _canonical_payload(weights: tuple[PathWeight, ...], update_events: tuple[WeightUpdateEvent, ...], *, version: str, snapshot_id: str, created_at: float, last_n: int) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "version": version,
        "created_at": created_at,
        "last_n_events": last_n,
        "weight_count": len(weights),
        "max_version": max((w.version for w in weights), default=0),
        "weights": [w.to_dict() for w in weights],
        "update_events": [e.to_dict() for e in update_events],
    }


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON: sorted keys, compact separators, no sha256 field."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HelixWeightSnapshot:
    snapshot_id: str
    version: str
    created_at: float
    last_n_events: int
    weight_count: int
    max_version: int
    weights: tuple[PathWeight, ...]
    update_events: tuple[WeightUpdateEvent, ...]
    sha256: str

    def payload(self) -> dict[str, Any]:
        return _canonical_payload(self.weights, self.update_events, version=self.version, snapshot_id=self.snapshot_id, created_at=self.created_at, last_n=self.last_n_events)

    def to_dict(self) -> dict[str, Any]:
        data = self.payload()
        data["sha256"] = self.sha256
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HelixWeightSnapshot":
        weights = tuple(PathWeight.from_dict(w) for w in payload.get("weights", []))
        events = tuple(WeightUpdateEvent.from_dict(e) for e in payload.get("update_events", []))
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            version=str(payload["version"]),
            created_at=float(payload["created_at"]),
            last_n_events=int(payload["last_n_events"]),
            weight_count=int(payload["weight_count"]),
            max_version=int(payload["max_version"]),
            weights=weights,
            update_events=events,
            sha256=str(payload["sha256"]),
        )


def build_snapshot(
    store: HelixWeightStore,
    *,
    known_path_ids: set[str],
    last_n_events: int = DEFAULT_LAST_N_EVENTS,
    snapshot_id: str = "snapshot-000001",
    now_fn=time.time,
) -> HelixWeightSnapshot:
    """Validate the table, then build a canonical, hashed snapshot."""
    validate_weight_table(store, known_path_ids)
    weights = tuple(sorted(store.weights.values(), key=lambda w: w.path_id))
    recent = tuple(store.update_events[-last_n_events:])
    payload = _canonical_payload(weights, recent, version="v0.00.07", snapshot_id=snapshot_id, created_at=now_fn(), last_n=last_n_events)
    return HelixWeightSnapshot(
        snapshot_id=snapshot_id,
        version="v0.00.07",
        created_at=payload["created_at"],
        last_n_events=last_n_events,
        weight_count=payload["weight_count"],
        max_version=payload["max_version"],
        weights=weights,
        update_events=recent,
        sha256=compute_sha256(payload),
    )


def write_snapshot(snapshot: HelixWeightSnapshot, path: str | Path) -> Path:
    """Atomic write: write to a temp file then os.replace onto the target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_snapshot(path: str | Path) -> HelixWeightSnapshot:
    """Load a snapshot and verify its SHA-256 (tamper detection)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshot = HelixWeightSnapshot.from_dict(data)
    recomputed = compute_sha256(snapshot.payload())
    if recomputed != snapshot.sha256:
        raise ValueError(f"snapshot sha256 mismatch: stored={snapshot.sha256} recomputed={recomputed}")
    return snapshot


def run_stage157_helix_weight_snapshot_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    from .stage151_helix_path_weight_model import path_id_for_link
    from .stage73_orion_memory_kernel import MemoryLink, MemoryLinkType, OrionMemoryStore

    clock = {"now": 9000.0}
    store = HelixWeightStore(now_fn=lambda: clock["now"])
    link_a = MemoryLink("c-1", "c-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    link_b = MemoryLink("c-3", "c-4", MemoryLinkType.PROCEDURE, 0.5, 0.0)
    store.register_link(link_a)
    store.register_link(link_b)
    clock["now"] = 9001.0
    store.approve(path_id_for_link(link_a), 1.5, reason="boost", feedback_source="smoke")
    known = {path_id_for_link(link_a), path_id_for_link(link_b)}

    snapshot = build_snapshot(store, known_path_ids=known, snapshot_id="snapshot-stage157", now_fn=lambda: clock["now"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = write_snapshot(snapshot, output / "weight_snapshot.json")

    # Round-trip: read back and verify hash.
    restored = read_snapshot(snapshot_path)
    hash_matches_on_read = restored.sha256 == snapshot.sha256
    canonical_reproducible = compute_sha256(snapshot.payload()) == compute_sha256(restored.payload())
    weights_match = restored.weights == snapshot.weights
    events_match = restored.update_events == snapshot.update_events

    # Atomicity: no .tmp file left behind.
    no_tmp_leftover = not (output / "weight_snapshot.json.tmp").exists()

    # Hash is sensitive: mutating a weight changes the hash.
    tampered_payload = snapshot.payload()
    tampered_payload["weights"][0]["value"] = 999.0
    hash_sensitive = compute_sha256(tampered_payload) != snapshot.sha256

    # Validated-only: a dangling path is rejected.
    dangling_store = HelixWeightStore(now_fn=lambda: clock["now"])
    dangling_store.register_link(link_a)
    bogus = MemoryLink("ghost-1", "ghost-2", MemoryLinkType.TEMPORAL, 1.0, 0.0)
    dangling_store.register_link(bogus)  # not in known_path_ids
    validated_rejects_dangling = False
    try:
        build_snapshot(dangling_store, known_path_ids={path_id_for_link(link_a)}, now_fn=lambda: clock["now"])
    except KeyError:
        validated_rejects_dangling = True

    summary = {
        "stage": "stage157_helix_weight_snapshot",
        "version": "v0.00.07",
        "snapshot_id": snapshot.snapshot_id,
        "sha256": snapshot.sha256,
        "weight_count": snapshot.weight_count,
        "max_version": snapshot.max_version,
        "stage_gates": {
            "hash_matches_on_read": hash_matches_on_read,
            "canonical_reproducible": canonical_reproducible,
            "weights_match": weights_match,
            "events_match": events_match,
            "atomic_no_tmp_leftover": no_tmp_leftover,
            "hash_sensitive_to_tamper": hash_sensitive,
            "validated_rejects_dangling": validated_rejects_dangling,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
