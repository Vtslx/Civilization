from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .stage101_trifid_episode_binding import TrifidEpisodeBinder
from .stage139_rosette_context_packet import RosetteContextAssembler, _create_lagoon_schema
from .stage73_orion_memory_kernel import MemoryLinkType, MemorySystem, OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage145_rosette_honeycomb_graph")


@dataclass(frozen=True)
class RosetteHoneycombNode:
    node_id: str
    cell_id: str
    memory_system: str
    scale: str
    role: str
    group_id: str
    confidence: float
    importance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RosetteHoneycombNode":
        return cls(str(payload["node_id"]), str(payload["cell_id"]), str(payload["memory_system"]), str(payload["scale"]), str(payload["role"]), str(payload["group_id"]), float(payload["confidence"]), float(payload["importance"]))


@dataclass(frozen=True)
class RosetteHoneycombEdge:
    source_node_id: str
    target_node_id: str
    adjacency_type: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RosetteHoneycombEdge":
        return cls(str(payload["source_node_id"]), str(payload["target_node_id"]), str(payload["adjacency_type"]), float(payload["weight"]))


@dataclass(frozen=True)
class RosetteHoneycombGraph:
    graph_id: str
    packet_id: str
    query: str
    nodes: tuple[RosetteHoneycombNode, ...]
    edges: tuple[RosetteHoneycombEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"graph_id": self.graph_id, "packet_id": self.packet_id, "query": self.query, "nodes": [node.to_dict() for node in self.nodes], "edges": [edge.to_dict() for edge in self.edges]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RosetteHoneycombGraph":
        return cls(str(payload["graph_id"]), str(payload["packet_id"]), str(payload["query"]), tuple(RosetteHoneycombNode.from_dict(item) for item in payload["nodes"]), tuple(RosetteHoneycombEdge.from_dict(item) for item in payload["edges"]))


class RosetteHoneycombGraphBuilder:
    """Builds a multi-scale, multi-adjacency graph from a verified Rosette context packet."""

    LINK_ADJACENCY = {MemoryLinkType.REPLAY: "consolidation", MemoryLinkType.CONFLICT: "conflict", MemoryLinkType.TASK: "decision"}

    def build(self, store: OrionMemoryStore, binder: TrifidEpisodeBinder, query: str) -> RosetteHoneycombGraph:
        packet = RosetteContextAssembler().assemble(store, binder, query)
        nodes_by_cell: dict[str, RosetteHoneycombNode] = {}
        for group in packet.groups:
            semantic_id = group.factual_cell_ids[0]
            self._add_node(store, nodes_by_cell, semantic_id, "schema", "semantic", group.group_id)
            for episode_id in group.support_episode_ids:
                frame = binder.frames.get(episode_id)
                if frame is None:
                    raise ValueError("support episode is missing from binder")
                self._add_node(store, nodes_by_cell, frame.anchor_cell_id, "episode", "support", group.group_id)
            for cell_id in group.factual_cell_ids[1:]:
                self._add_node(store, nodes_by_cell, cell_id, "episode", "conflict", group.group_id)
            for cell_id in group.control_cell_ids:
                self._add_node(store, nodes_by_cell, cell_id, "control", "decision", group.group_id)
        cell_to_node = {node.cell_id: node.node_id for node in nodes_by_cell.values()}
        edges = [
            RosetteHoneycombEdge(cell_to_node[link.source_cell_id], cell_to_node[link.target_cell_id], self.LINK_ADJACENCY[link.link_type], link.weight)
            for link in store.links
            if link.link_type in self.LINK_ADJACENCY and link.source_cell_id in cell_to_node and link.target_cell_id in cell_to_node
        ]
        edges.sort(key=lambda edge: (edge.source_node_id, edge.target_node_id, edge.adjacency_type))
        nodes = tuple(sorted(nodes_by_cell.values(), key=lambda node: node.node_id))
        evidence = store.read_cells([node.cell_id for node in nodes], trace_details={"rosette_honeycomb_graph": True, "packet_id": packet.packet_id, "query": query})
        if len(evidence) != len(nodes):
            raise ValueError("honeycomb graph contains missing or inactive cells")
        graph_id = self.graph_id(packet.packet_id, query, nodes, tuple(edges))
        return RosetteHoneycombGraph(graph_id, packet.packet_id, query, nodes, tuple(edges))

    @staticmethod
    def _add_node(store: OrionMemoryStore, nodes: dict[str, RosetteHoneycombNode], cell_id: str, scale: str, role: str, group_id: str) -> None:
        if cell_id in nodes:
            return
        cell = store.cells.get(cell_id)
        if cell is None:
            raise ValueError(f"missing graph cell: {cell_id}")
        nodes[cell_id] = RosetteHoneycombNode(f"node:{cell_id}", cell_id, cell.memory_system.value, scale, role, group_id, cell.confidence, cell.importance)

    @staticmethod
    def graph_id(packet_id: str, query: str, nodes: tuple[RosetteHoneycombNode, ...], edges: tuple[RosetteHoneycombEdge, ...]) -> str:
        payload = {"packet_id": packet_id, "query": query, "nodes": [node.to_dict() for node in nodes], "edges": [edge.to_dict() for edge in edges]}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return f"rosette-graph-{digest[:20]}"


def run_stage145_rosette_honeycomb_graph_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    store, binder = OrionMemoryStore(), TrifidEpisodeBinder()
    _create_lagoon_schema(store, binder, scene="laboratory", offset=0.0, with_conflict=True, approved=True)
    builder = RosetteHoneycombGraphBuilder()
    graph = builder.build(store, binder, "laboratory collect")
    scales = {node.scale for node in graph.nodes}
    adjacencies = {edge.adjacency_type for edge in graph.edges}
    summary = {
        "stage": "stage145_rosette_honeycomb_graph",
        "graph": graph.to_dict(),
        "stage_gates": {
            "multi_scale_nodes_present": scales == {"episode", "schema", "control"},
            "multi_adjacency_edges_present": adjacencies == {"consolidation", "conflict", "decision"},
            "all_graph_cells_active": all(store.cells[node.cell_id].decay_state == "active" for node in graph.nodes),
            "support_and_conflict_nodes_preserved": sum(node.role == "support" for node in graph.nodes) == 2 and sum(node.role == "conflict" for node in graph.nodes) == 1,
            "decision_is_control_scale": sum(node.role == "decision" and node.scale == "control" for node in graph.nodes) == 1,
            "graph_id_is_deterministic": graph.graph_id == builder.build(store, binder, "laboratory collect").graph_id,
            "graph_read_traced": any(event.details.get("rosette_honeycomb_graph") for event in store.trace_events),
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output_dir, summary=summary)
    return summary
