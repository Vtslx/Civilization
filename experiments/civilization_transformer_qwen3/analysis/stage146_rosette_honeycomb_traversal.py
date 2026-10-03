from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import _create_lagoon_schema
from .stage145_rosette_honeycomb_graph import RosetteHoneycombEdge, RosetteHoneycombGraph, RosetteHoneycombGraphBuilder
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage146_rosette_honeycomb_traversal")


@dataclass(frozen=True)
class RosetteTraversalResult:
    start_node_id: str
    visited_node_ids: tuple[str, ...]
    traversed_edges: tuple[RosetteHoneycombEdge, ...]
    depth_by_node: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["traversed_edges"] = [edge.to_dict() for edge in self.traversed_edges]
        return payload


class RosetteHoneycombTraverser:
    """Deterministically traverses honeycomb adjacencies as bidirectional memory relations."""

    def traverse(self, graph: RosetteHoneycombGraph, start_cell_id: str, *, allowed_adjacencies: set[str] | None = None, max_depth: int = 3, max_nodes: int | None = None) -> RosetteTraversalResult:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if max_nodes is not None and max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        node_by_cell = {node.cell_id: node for node in graph.nodes}
        if start_cell_id not in node_by_cell:
            raise ValueError("start_cell_id is not in graph")
        start = node_by_cell[start_cell_id].node_id
        adjacency: dict[str, list[tuple[str, RosetteHoneycombEdge]]] = {node.node_id: [] for node in graph.nodes}
        for edge in graph.edges:
            if allowed_adjacencies is not None and edge.adjacency_type not in allowed_adjacencies:
                continue
            adjacency[edge.source_node_id].append((edge.target_node_id, edge))
            adjacency[edge.target_node_id].append((edge.source_node_id, edge))
        for neighbors in adjacency.values():
            neighbors.sort(key=lambda item: (item[1].adjacency_type, item[0]))
        limit = max_nodes or len(graph.nodes)
        queue = [start]
        visited = [start]
        depths = {start: 0}
        used_edges: list[RosetteHoneycombEdge] = []
        edge_keys: set[tuple[str, str, str]] = set()
        while queue and len(visited) < limit:
            current = queue.pop(0)
            if depths[current] >= max_depth:
                continue
            for neighbor, edge in adjacency[current]:
                if neighbor in depths:
                    continue
                depths[neighbor] = depths[current] + 1
                visited.append(neighbor)
                queue.append(neighbor)
                key = (edge.source_node_id, edge.target_node_id, edge.adjacency_type)
                if key not in edge_keys:
                    used_edges.append(edge)
                    edge_keys.add(key)
                if len(visited) >= limit:
                    break
        return RosetteTraversalResult(start, tuple(visited), tuple(used_edges), depths)


def run_stage146_rosette_honeycomb_traversal_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    semantic_id, _conflict, _decision = _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    graph = RosetteHoneycombGraphBuilder().build(store, binder, "laboratory collect")
    traverser = RosetteHoneycombTraverser()
    full = traverser.traverse(graph, semantic_id)
    consolidation = traverser.traverse(graph, semantic_id, allowed_adjacencies={"consolidation"})
    conflict = traverser.traverse(graph, semantic_id, allowed_adjacencies={"conflict"})
    bounded = traverser.traverse(graph, semantic_id, max_nodes=3)
    role_by_node = {node.node_id: node.role for node in graph.nodes}
    summary = {
        "stage": "stage146_rosette_honeycomb_traversal",
        "full": full.to_dict(),
        "stage_gates": {
            "full_traversal_reaches_all_scales": len(full.visited_node_ids) == len(graph.nodes) and {role_by_node[item] for item in full.visited_node_ids} == {"semantic", "support", "conflict", "decision"},
            "consolidation_filter_isolated": {role_by_node[item] for item in consolidation.visited_node_ids} == {"semantic", "support"},
            "conflict_filter_isolated": {role_by_node[item] for item in conflict.visited_node_ids} == {"semantic", "conflict"},
            "node_budget_respected": len(bounded.visited_node_ids) == 3,
            "traversal_is_deterministic": bounded == traverser.traverse(graph, semantic_id, max_nodes=3),
            "depths_are_bounded": max(full.depth_by_node.values()) <= 3,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
