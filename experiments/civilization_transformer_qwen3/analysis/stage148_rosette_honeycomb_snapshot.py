from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import _create_lagoon_schema
from .stage145_rosette_honeycomb_graph import RosetteHoneycombGraph, RosetteHoneycombGraphBuilder
from .stage147_rosette_honeycomb_validator import RosetteHoneycombValidator, _regraph
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage148_rosette_honeycomb_snapshot")
ROSETTE_HONEYCOMB_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RosetteHoneycombSnapshotManifest:
    schema_version: int
    graph_id: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RosetteHoneycombSnapshot:
    """Saves validated honeycomb graphs as canonical, atomic snapshot envelopes."""

    def save(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, graph: RosetteHoneycombGraph, path: str | Path) -> RosetteHoneycombSnapshotManifest:
        validation = RosetteHoneycombValidator().validate(store, binder, graph)
        if not validation.valid:
            raise ValueError(f"invalid honeycomb graph: {validation.errors}")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        graph_bytes = self.canonical_graph_bytes(graph)
        manifest = RosetteHoneycombSnapshotManifest(ROSETTE_HONEYCOMB_SCHEMA_VERSION, graph.graph_id, hashlib.sha256(graph_bytes).hexdigest())
        encoded = json.dumps({"manifest": manifest.to_dict(), "graph": graph.to_dict()}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
        return manifest

    @staticmethod
    def canonical_graph_bytes(graph: RosetteHoneycombGraph) -> bytes:
        return json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run_stage148_rosette_honeycomb_snapshot_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    output = Path(output_dir)
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    graph = RosetteHoneycombGraphBuilder().build(store, binder, "laboratory collect")
    snapshot = RosetteHoneycombSnapshot()
    first_path, second_path = output / "honeycomb.json", output / "honeycomb-repeat.json"
    first = snapshot.save(store, binder, graph, first_path)
    second = snapshot.save(store, binder, graph, second_path)
    invalid_nodes = list(graph.nodes)
    invalid_nodes[0] = replace(invalid_nodes[0], scale="invalid")
    rejected = False
    try:
        snapshot.save(store, binder, _regraph(graph, nodes=invalid_nodes), output / "invalid.json")
    except ValueError:
        rejected = True
    summary = {
        "stage": "stage148_rosette_honeycomb_snapshot",
        "manifest": first.to_dict(),
        "stage_gates": {
            "validated_graph_written": first_path.exists() and second_path.exists(),
            "invalid_graph_rejected": rejected and not (output / "invalid.json").exists(),
            "schema_version_frozen": first.schema_version == 1,
            "graph_id_preserved": first.graph_id == graph.graph_id,
            "snapshot_reproducible": first.sha256 == second.sha256 and first_path.read_bytes() == second_path.read_bytes(),
            "canonical_hash_matches": first.sha256 == hashlib.sha256(snapshot.canonical_graph_bytes(graph)).hexdigest(),
            "temporary_file_removed": not first_path.with_name(f".{first_path.name}.tmp").exists(),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output, summary=summary)
    return summary
