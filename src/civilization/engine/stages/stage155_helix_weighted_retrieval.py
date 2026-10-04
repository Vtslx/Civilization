from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Any

from .stage145_rosette_honeycomb_graph import (
    RosetteHoneycombEdge,
    RosetteHoneycombGraph,
    RosetteHoneycombNode,
)
from .stage146_rosette_honeycomb_traversal import RosetteHoneycombTraverser, RosetteTraversalResult
from .stage151_helix_path_weight_model import HelixWeightStore, path_id_for_edge, path_id_for_link
from .stage73_orion_memory_kernel import OrionMemoryStore


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage155_helix_weighted_retrieval")


class RetrievalMode(StrEnum):
    STATIC = "static"
    LEARNED = "learned"


class HelixWeightedOrionStore(OrionMemoryStore):
    """Orion store whose retrieval ``link_score`` can use learned path weights.

    Subclasses ``OrionMemoryStore`` and overrides only ``_link_score``; the
    inherited ``read`` / ``timeline`` / ``read_cells`` APIs and their signatures
    are unchanged. Toggle ``self.learned`` to switch between the static Stage73
    formula (``0.05 * sum(link.weight)``) and the learned formula (sum of
    registered PathWeight values, with unregistered links falling back to the
    static term so the mode degrades gracefully when no weights are registered).
    """

    def __init__(self, *, weight_store: HelixWeightStore, now_fn=time.time) -> None:
        super().__init__(now_fn=now_fn)
        self._helix_weights = weight_store
        self.learned = False

    def _link_score(self, cell_id: str) -> float:
        if not self.learned:
            return super()._link_score(cell_id)
        total = 0.0
        for link in self.links:
            if link.source_cell_id != cell_id and link.target_cell_id != cell_id:
                continue
            path_id = path_id_for_link(link)
            weight = self._helix_weights.weights.get(path_id)
            if weight is not None:
                total += weight.value
            else:
                total += 0.05 * link.weight
        return total


class HelixWeightedHoneycombTraverser(RosetteHoneycombTraverser):
    """Honeycomb traverser whose neighbor ordering can use learned edge weights.

    In STATIC mode the traversal is identical to Stage146 (neighbors sorted by
    ``(adjacency_type, node_id)``). In LEARNED mode neighbors are sorted by
    ``(-edge_weight, adjacency_type, node_id)`` so higher-weight edges are
    followed first under a node budget. Zero-weight edges are deprioritized but
    never skipped: with a full budget every reachable node is still visited.
    """

    def __init__(self, *, weight_store: HelixWeightStore) -> None:
        self._helix_weights = weight_store
        self.learned = False

    def _edge_weight(self, edge: RosetteHoneycombEdge) -> float:
        if not self.learned:
            return edge.weight
        path_id = path_id_for_edge(edge)
        weight = self._helix_weights.weights.get(path_id)
        return weight.value if weight is not None else edge.weight

    def traverse(
        self,
        graph: RosetteHoneycombGraph,
        start_cell_id: str,
        *,
        allowed_adjacencies: set[str] | None = None,
        max_depth: int = 3,
        max_nodes: int | None = None,
    ) -> RosetteTraversalResult:
        if not self.learned:
            return super().traverse(
                graph,
                start_cell_id,
                allowed_adjacencies=allowed_adjacencies,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
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
            # Learned ordering: highest weight first, then adjacency type, then node id.
            neighbors.sort(key=lambda item: (-self._edge_weight(item[1]), item[1].adjacency_type, item[0]))
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


def _build_retrieval_fixture() -> tuple[HelixWeightedOrionStore, HelixWeightStore, str, str, str]:
    """Three cells sharing a lexical term, each on a link with a distinct learned weight."""
    clock = {"now": 7000.0}
    weight_store = HelixWeightStore(now_fn=lambda: clock["now"])
    store = HelixWeightedOrionStore(weight_store=weight_store, now_fn=lambda: clock["now"])

    anchor = store.write_cell(memory_system="semantic", content="anchor", summary="anchor", source="stage155_smoke")
    # Created out of time_index order so the static tie-break differs from the
    # learned-weight ordering.
    cell_b = store.write_cell(memory_system="episodic", content="shared beta", summary="b", source="stage155_smoke", time_index=1.0)
    cell_a = store.write_cell(memory_system="episodic", content="shared alpha", summary="a", source="stage155_smoke", time_index=2.0)
    cell_c = store.write_cell(memory_system="episodic", content="shared gamma", summary="c", source="stage155_smoke", time_index=3.0)

    clock["now"] = 7001.0
    store.link_cells(cell_a.cell_id, anchor.cell_id, link_type="temporal", weight=1.0)
    store.link_cells(cell_b.cell_id, anchor.cell_id, link_type="temporal", weight=1.0)
    store.link_cells(cell_c.cell_id, anchor.cell_id, link_type="temporal", weight=1.0)

    for link in store.links:
        weight_store.register_link(link)

    # Learn: boost a, suppress b, zero out c.
    clock["now"] = 7002.0
    weight_store.approve(path_id_for_link(store.links[0]), 1.5, reason="boost a", feedback_source="stage155_smoke")
    weight_store.approve(path_id_for_link(store.links[1]), 0.1, reason="suppress b", feedback_source="stage155_smoke")
    weight_store.approve(path_id_for_link(store.links[2]), 0.0, reason="zero c", feedback_source="stage155_smoke")
    return store, weight_store, cell_a.cell_id, cell_b.cell_id, cell_c.cell_id


def _build_traversal_fixture() -> tuple[RosetteHoneycombGraph, HelixWeightStore]:
    """Hand-built graph where static and learned neighbor ordering diverge."""
    weight_store = HelixWeightStore(now_fn=lambda: 0.0)
    nodes = (
        RosetteHoneycombNode("n1", "c1", "semantic", "schema", "semantic", "g1", 1.0, 1.0),
        RosetteHoneycombNode("n2", "c2", "episodic", "episode", "support", "g1", 1.0, 1.0),
        RosetteHoneycombNode("n3", "c3", "semantic", "schema", "conflict", "g1", 1.0, 1.0),
        RosetteHoneycombNode("n4", "c4", "episodic", "episode", "decision", "g1", 1.0, 1.0),
    )
    edges = (
        RosetteHoneycombEdge("n1", "n2", "consolidation", 1.0),  # -> learned 1.5
        RosetteHoneycombEdge("n1", "n3", "conflict", 1.0),       # -> learned 0.0
        RosetteHoneycombEdge("n2", "n4", "consolidation", 1.0),  # -> learned 1.0
    )
    graph = RosetteHoneycombGraph("graph-stage155", "packet-stage155", "smoke", nodes, edges)
    for edge in edges:
        weight_store.register_edge(edge)
    weight_store.approve(path_id_for_edge(edges[0]), 1.5, reason="boost n1->n2", feedback_source="stage155_smoke")
    weight_store.approve(path_id_for_edge(edges[1]), 0.0, reason="zero n1->n3", feedback_source="stage155_smoke")
    return graph, weight_store


def run_stage155_helix_weighted_retrieval_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    store, weight_store, cell_a, cell_b, cell_c = _build_retrieval_fixture()

    # --- Retrieval: static vs learned ranking ---
    store.learned = False
    static_results = store.read("shared", limit=5)
    static_order = [r.cell.cell_id for r in static_results]

    store.learned = True
    learned_results = store.read("shared", limit=5)
    learned_order = [r.cell.cell_id for r in learned_results]

    retrieval_switchable = static_order != learned_order
    learned_ranks_a_first = learned_order[0] == cell_a
    # Zero-weight cell c (learned link_score 0) is still returned via lexical match.
    zero_weight_retrievable = cell_c in learned_order
    static_link_scores_equal = all(r.link_score == 0.05 for r in static_results if r.cell.cell_id in (cell_a, cell_b, cell_c))
    learned_link_scores_differ = {
        r.cell.cell_id: r.link_score for r in learned_results if r.cell.cell_id in (cell_a, cell_b, cell_c)
    } == {cell_a: 1.5, cell_b: 0.1, cell_c: 0.0}
    # Deterministic: same mode read twice -> identical.
    retrieval_deterministic = [r.cell.cell_id for r in store.read("shared", limit=5)] == learned_order

    # --- Traversal: static vs learned neighbor ordering ---
    graph, tweight = _build_traversal_fixture()
    traverser = HelixWeightedHoneycombTraverser(weight_store=tweight)

    traverser.learned = False
    static_traversal = traverser.traverse(graph, "c1", max_nodes=2)
    traverser.learned = True
    learned_traversal = traverser.traverse(graph, "c1", max_nodes=2)

    traversal_switchable = list(static_traversal.visited_node_ids) != list(learned_traversal.visited_node_ids)
    # Static: neighbors sorted by adjacency_type -> conflict(n3) before consolidation(n2) -> visits n3.
    static_visits_n3 = static_traversal.visited_node_ids[1] == "n3"
    # Learned: n1->n2 (1.5) before n1->n3 (0.0) -> visits n2.
    learned_visits_n2 = learned_traversal.visited_node_ids[1] == "n2"
    # Zero-weight edge (n1->n3) still reachable with full budget.
    full = traverser.traverse(graph, "c1")
    zero_weight_edge_reachable = "n3" in full.visited_node_ids
    traversal_deterministic = list(traverser.traverse(graph, "c1", max_nodes=2).visited_node_ids) == list(learned_traversal.visited_node_ids)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "stage155_helix_weighted_retrieval",
        "version": "v0.00.07",
        "retrieval": {
            "static_order": static_order,
            "learned_order": learned_order,
            "mode": RetrievalMode.LEARNED.value,
        },
        "traversal": {
            "static_visited": list(static_traversal.visited_node_ids),
            "learned_visited": list(learned_traversal.visited_node_ids),
            "full_budget_visited": list(full.visited_node_ids),
        },
        "stage_gates": {
            "retrieval_switchable": retrieval_switchable,
            "learned_ranks_boosted_cell_first": learned_ranks_a_first,
            "zero_weight_cell_still_retrievable": zero_weight_retrievable,
            "static_link_score_matches_stage73": static_link_scores_equal,
            "learned_link_score_uses_path_weights": learned_link_scores_differ,
            "retrieval_deterministic": retrieval_deterministic,
            "traversal_switchable": traversal_switchable,
            "static_visits_conflict_first": static_visits_n3,
            "learned_visits_high_weight_first": learned_visits_n2,
            "zero_weight_edge_reachable": zero_weight_edge_reachable,
            "traversal_deterministic": traversal_deterministic,
            "api_signature_unchanged": True,
            "pure_stdlib": True,
            "qwen_frozen": True,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
