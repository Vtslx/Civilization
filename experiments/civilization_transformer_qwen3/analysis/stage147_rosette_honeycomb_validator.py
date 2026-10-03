from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import _create_lagoon_schema
from .stage145_rosette_honeycomb_graph import RosetteHoneycombGraph, RosetteHoneycombGraphBuilder, RosetteHoneycombNode
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage147_rosette_honeycomb_validator")


@dataclass(frozen=True)
class RosetteHoneycombValidationReport:
    valid: bool
    errors: tuple[str, ...]


class RosetteHoneycombValidator:
    """Fail-closed graph validation against live memory cells, links, and binder provenance."""

    VALID_SCALES = {"episode", "schema", "control"}
    VALID_ADJACENCIES = {"consolidation", "conflict", "decision"}

    def validate(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, graph: RosetteHoneycombGraph) -> RosetteHoneycombValidationReport:
        errors: list[str] = []
        node_ids = [node.node_id for node in graph.nodes]
        cell_ids = [node.cell_id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)) or len(cell_ids) != len(set(cell_ids)):
            errors.append("duplicate_node")
        if graph.graph_id != RosetteHoneycombGraphBuilder.graph_id(graph.packet_id, graph.query, graph.nodes, graph.edges):
            errors.append("graph_id_mismatch")
        for node in graph.nodes:
            cell = store.cells.get(node.cell_id)
            if cell is None or cell.decay_state != "active":
                errors.append(f"{node.node_id}:missing_or_inactive_cell")
                continue
            if cell.memory_system.value != node.memory_system:
                errors.append(f"{node.node_id}:memory_system_mismatch")
            expected_scale = {"semantic": "schema", "episodic": "episode", "procedural": "control"}.get(node.memory_system)
            if node.scale not in self.VALID_SCALES or node.scale != expected_scale:
                errors.append(f"{node.node_id}:scale_mismatch")
        node_set = set(node_ids)
        for edge in graph.edges:
            if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
                errors.append("dangling_edge")
            if edge.adjacency_type not in self.VALID_ADJACENCIES:
                errors.append("unknown_adjacency")
        try:
            expected = RosetteHoneycombGraphBuilder().build(store, binder, graph.query)
        except ValueError:
            errors.append("live_graph_rebuild_failed")
        else:
            if expected.packet_id != graph.packet_id:
                errors.append("packet_id_mismatch")
            if expected.nodes != graph.nodes:
                errors.append("node_topology_mismatch")
            if expected.edges != graph.edges:
                errors.append("edge_topology_mismatch")
        return RosetteHoneycombValidationReport(not errors, tuple(errors))


def _regraph(graph: RosetteHoneycombGraph, *, nodes=None, edges=None) -> RosetteHoneycombGraph:
    next_nodes = tuple(nodes if nodes is not None else graph.nodes)
    next_edges = tuple(edges if edges is not None else graph.edges)
    graph_id = RosetteHoneycombGraphBuilder.graph_id(graph.packet_id, graph.query, next_nodes, next_edges)
    return RosetteHoneycombGraph(graph_id, graph.packet_id, graph.query, next_nodes, next_edges)


def run_stage147_rosette_honeycomb_validator_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    graph = RosetteHoneycombGraphBuilder().build(store, binder, "laboratory collect")
    validator = RosetteHoneycombValidator()
    cell_count, link_count = len(store.cells), len(store.links)
    valid = validator.validate(store, binder, graph)
    schema_index = next(index for index, node in enumerate(graph.nodes) if node.role == "semantic")
    bad_nodes = list(graph.nodes)
    bad_nodes[schema_index] = replace(bad_nodes[schema_index], scale="episode")
    scale_mismatch = validator.validate(store, binder, _regraph(graph, nodes=bad_nodes))
    dangling_node = RosetteHoneycombNode("node:episodic-999999", "episodic-999999", "episodic", "episode", "support", "missing", 1.0, 1.0)
    dangling = validator.validate(store, binder, _regraph(graph, nodes=(*graph.nodes, dangling_node)))
    bad_edges = list(graph.edges)
    bad_edges[0] = replace(bad_edges[0], adjacency_type="unknown")
    bad_adjacency = validator.validate(store, binder, _regraph(graph, edges=bad_edges))
    summary = {
        "stage": "stage147_rosette_honeycomb_validator",
        "stage_gates": {
            "valid_graph_accepted": valid.valid and valid.errors == (),
            "scale_mismatch_rejected": not scale_mismatch.valid and any("scale_mismatch" in error for error in scale_mismatch.errors),
            "dangling_node_rejected": not dangling.valid and any("missing_or_inactive_cell" in error for error in dangling.errors),
            "unknown_adjacency_rejected": not bad_adjacency.valid and "unknown_adjacency" in bad_adjacency.errors,
            "live_topology_rechecked": "node_topology_mismatch" in scale_mismatch.errors and "edge_topology_mismatch" in bad_adjacency.errors,
            "validator_is_read_only": len(store.cells) == cell_count and len(store.links) == link_count,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
