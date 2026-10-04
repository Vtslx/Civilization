from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import _create_lagoon_schema
from .stage145_rosette_honeycomb_graph import RosetteHoneycombGraph, RosetteHoneycombGraphBuilder, RosetteHoneycombNode
from .stage147_rosette_honeycomb_validator import RosetteHoneycombValidator, _regraph
from .stage148_rosette_honeycomb_snapshot import ROSETTE_HONEYCOMB_SCHEMA_VERSION, RosetteHoneycombSnapshot
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage149_rosette_honeycomb_recovery")


@dataclass(frozen=True)
class RosetteHoneycombRecoveryResult:
    recovered: bool
    reason: str
    graph: RosetteHoneycombGraph | None
    validation_errors: tuple[str, ...] = ()


class RosetteHoneycombRecovery:
    """Recovers checksummed graphs only after live topology validation."""

    def recover(self, path: str | Path, store: OrionMemoryStore, binder: TrifidEpisodeBinder) -> RosetteHoneycombRecoveryResult:
        try:
            envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return RosetteHoneycombRecoveryResult(False, "invalid_json", None)
        try:
            manifest = envelope["manifest"]
            if int(manifest["schema_version"]) != ROSETTE_HONEYCOMB_SCHEMA_VERSION:
                return RosetteHoneycombRecoveryResult(False, "unsupported_schema_version", None)
            graph = RosetteHoneycombGraph.from_dict(envelope["graph"])
            if manifest["graph_id"] != graph.graph_id:
                return RosetteHoneycombRecoveryResult(False, "graph_id_mismatch", None)
            digest = hashlib.sha256(RosetteHoneycombSnapshot.canonical_graph_bytes(graph)).hexdigest()
            if manifest["sha256"] != digest:
                return RosetteHoneycombRecoveryResult(False, "checksum_mismatch", None)
        except (KeyError, TypeError, ValueError):
            return RosetteHoneycombRecoveryResult(False, "invalid_schema", None)
        validation = RosetteHoneycombValidator().validate(store, binder, graph)
        if not validation.valid:
            return RosetteHoneycombRecoveryResult(False, "graph_validation_failed", None, validation.errors)
        return RosetteHoneycombRecoveryResult(True, "ok", graph)


def run_stage149_rosette_honeycomb_recovery_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    graph = RosetteHoneycombGraphBuilder().build(store, binder, "laboratory collect")
    snapshot = RosetteHoneycombSnapshot()
    valid_path = output / "valid-honeycomb.json"
    snapshot.save(store, binder, graph, valid_path)
    recovery = RosetteHoneycombRecovery()
    cell_count, link_count = len(store.cells), len(store.links)
    validation_trace_count = sum(bool(event.details.get("rosette_honeycomb_graph")) for event in store.trace_events)
    valid = recovery.recover(valid_path, store, binder)
    recovery_validation_traced = sum(bool(event.details.get("rosette_honeycomb_graph")) for event in store.trace_events) > validation_trace_count

    tampered_path = output / "tampered-honeycomb.json"
    tampered_payload = json.loads(valid_path.read_text(encoding="utf-8"))
    tampered_payload["graph"]["query"] = "tampered"
    tampered_path.write_text(json.dumps(tampered_payload, ensure_ascii=False), encoding="utf-8")
    tampered = recovery.recover(tampered_path, store, binder)

    dangling_node = RosetteHoneycombNode("node:episodic-999999", "episodic-999999", "episodic", "episode", "support", "missing", 1.0, 1.0)
    dangling_graph = _regraph(graph, nodes=(*graph.nodes, dangling_node))
    dangling_digest = hashlib.sha256(snapshot.canonical_graph_bytes(dangling_graph)).hexdigest()
    dangling_path = output / "dangling-honeycomb.json"
    dangling_path.write_text(json.dumps({"manifest": {"schema_version": 1, "graph_id": dangling_graph.graph_id, "sha256": dangling_digest}, "graph": dangling_graph.to_dict()}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    dangling = recovery.recover(dangling_path, store, binder)

    invalid_path = output / "invalid-honeycomb.json"
    invalid_path.write_text("{invalid", encoding="utf-8")
    invalid = recovery.recover(invalid_path, store, binder)
    summary = {
        "stage": "stage149_rosette_honeycomb_recovery",
        "stage_gates": {
            "valid_graph_recovered": valid.recovered and valid.graph == graph and valid.reason == "ok",
            "tampering_rejected": not tampered.recovered and tampered.reason == "checksum_mismatch",
            "live_dangling_reference_rejected": not dangling.recovered and dangling.reason == "graph_validation_failed" and any("missing_or_inactive_cell" in error for error in dangling.validation_errors),
            "invalid_json_rejected": not invalid.recovered and invalid.reason == "invalid_json",
            "recovery_is_read_only": len(store.cells) == cell_count and len(store.links) == link_count,
            "live_topology_revalidated": recovery_validation_traced,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output, summary=summary)
    return summary
