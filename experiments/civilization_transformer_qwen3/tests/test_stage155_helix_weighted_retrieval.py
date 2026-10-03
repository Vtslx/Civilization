from __future__ import annotations

import inspect

import pytest

from experiments.civilization_transformer_qwen3.analysis.stage145_rosette_honeycomb_graph import (
    RosetteHoneycombEdge,
    RosetteHoneycombGraph,
    RosetteHoneycombNode,
)
from experiments.civilization_transformer_qwen3.analysis.stage155_helix_weighted_retrieval import (
    HelixWeightedHoneycombTraverser,
    HelixWeightedOrionStore,
    RetrievalMode,
    run_stage155_helix_weighted_retrieval_smoke,
)
from experiments.civilization_transformer_qwen3.analysis.stage151_helix_path_weight_model import (
    HelixWeightStore,
    path_id_for_edge,
    path_id_for_link,
)
from experiments.civilization_transformer_qwen3.analysis.stage73_orion_memory_kernel import (
    MemoryLinkType,
    OrionMemoryStore,
)


def _retrieval_fixture() -> tuple[HelixWeightedOrionStore, HelixWeightStore, str, str, str]:
    clock = {"now": 0.0}
    weight_store = HelixWeightStore(now_fn=lambda: clock["now"])
    store = HelixWeightedOrionStore(weight_store=weight_store, now_fn=lambda: clock["now"])
    anchor = store.write_cell(memory_system="semantic", content="anchor", summary="anchor", source="t")
    cell_b = store.write_cell(memory_system="episodic", content="shared beta", summary="b", source="t", time_index=1.0)
    cell_a = store.write_cell(memory_system="episodic", content="shared alpha", summary="a", source="t", time_index=2.0)
    cell_c = store.write_cell(memory_system="episodic", content="shared gamma", summary="c", source="t", time_index=3.0)
    store.link_cells(cell_a.cell_id, anchor.cell_id, link_type="temporal", weight=1.0)
    store.link_cells(cell_b.cell_id, anchor.cell_id, link_type="temporal", weight=1.0)
    store.link_cells(cell_c.cell_id, anchor.cell_id, link_type="temporal", weight=1.0)
    for link in store.links:
        weight_store.register_link(link)
    weight_store.approve(path_id_for_link(store.links[0]), 1.5, reason="boost a", feedback_source="t")
    weight_store.approve(path_id_for_link(store.links[1]), 0.1, reason="suppress b", feedback_source="t")
    weight_store.approve(path_id_for_link(store.links[2]), 0.0, reason="zero c", feedback_source="t")
    return store, weight_store, cell_a.cell_id, cell_b.cell_id, cell_c.cell_id


def _graph_fixture() -> tuple[RosetteHoneycombGraph, HelixWeightStore]:
    weight_store = HelixWeightStore(now_fn=lambda: 0.0)
    nodes = (
        RosetteHoneycombNode("n1", "c1", "semantic", "schema", "semantic", "g1", 1.0, 1.0),
        RosetteHoneycombNode("n2", "c2", "episodic", "episode", "support", "g1", 1.0, 1.0),
        RosetteHoneycombNode("n3", "c3", "semantic", "schema", "conflict", "g1", 1.0, 1.0),
        RosetteHoneycombNode("n4", "c4", "episodic", "episode", "decision", "g1", 1.0, 1.0),
    )
    edges = (
        RosetteHoneycombEdge("n1", "n2", "consolidation", 1.0),
        RosetteHoneycombEdge("n1", "n3", "conflict", 1.0),
        RosetteHoneycombEdge("n2", "n4", "consolidation", 1.0),
    )
    graph = RosetteHoneycombGraph("g", "p", "q", nodes, edges)
    for edge in edges:
        weight_store.register_edge(edge)
    weight_store.approve(path_id_for_edge(edges[0]), 1.5, reason="boost", feedback_source="t")
    weight_store.approve(path_id_for_edge(edges[1]), 0.0, reason="zero", feedback_source="t")
    return graph, weight_store


def test_stage155_smoke_passes_gate(tmp_path) -> None:
    summary = run_stage155_helix_weighted_retrieval_smoke(output_dir=tmp_path)
    assert summary["passes_stage_gate"] is True
    assert summary["stage"] == "stage155_helix_weighted_retrieval"
    assert summary["version"] == "v0.00.07"
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "summary.json").exists()


def test_stage155_static_link_score_matches_stage73_formula() -> None:
    store, _, cell_a, _, _ = _retrieval_fixture()
    store.learned = False
    # One incident link of static weight 1.0 -> 0.05 * 1.0 = 0.05 (Stage73 formula).
    results = {r.cell.cell_id: r for r in store.read("shared")}
    assert results[cell_a].link_score == pytest.approx(0.05)


def test_stage155_learned_link_score_uses_path_weights() -> None:
    store, _, cell_a, cell_b, cell_c = _retrieval_fixture()
    store.learned = True
    results = {r.cell.cell_id: r for r in store.read("shared")}
    assert results[cell_a].link_score == pytest.approx(1.5)
    assert results[cell_b].link_score == pytest.approx(0.1)
    assert results[cell_c].link_score == pytest.approx(0.0)


def test_stage155_retrieval_is_switchable() -> None:
    store, _, _, _, _ = _retrieval_fixture()
    store.learned = False
    static_order = [r.cell.cell_id for r in store.read("shared")]
    store.learned = True
    learned_order = [r.cell.cell_id for r in store.read("shared")]
    assert static_order != learned_order


def test_stage155_learned_ranks_boosted_cell_first() -> None:
    store, _, cell_a, _, _ = _retrieval_fixture()
    store.learned = True
    results = store.read("shared")
    assert results[0].cell.cell_id == cell_a  # highest learned weight 1.5


def test_stage155_zero_weight_cell_still_retrievable() -> None:
    store, _, _, _, cell_c = _retrieval_fixture()
    store.learned = True
    results = store.read("shared")
    assert cell_c in [r.cell.cell_id for r in results]  # link_score 0, but lexical match returns it


def test_stage155_retrieval_deterministic() -> None:
    store, _, _, _, _ = _retrieval_fixture()
    store.learned = True
    first = [r.cell.cell_id for r in store.read("shared")]
    second = [r.cell.cell_id for r in store.read("shared")]
    assert first == second


def test_stage155_unregistered_link_falls_back_to_static() -> None:
    """In learned mode, a link not registered in the weight store contributes 0.05*weight."""
    clock = {"now": 0.0}
    weight_store = HelixWeightStore(now_fn=lambda: clock["now"])  # empty: nothing registered
    store = HelixWeightedOrionStore(weight_store=weight_store, now_fn=lambda: clock["now"])
    a = store.write_cell(memory_system="episodic", content="alpha", summary="a", source="t")
    b = store.write_cell(memory_system="episodic", content="beta", summary="b", source="t")
    store.link_cells(a.cell_id, b.cell_id, link_type="temporal", weight=1.0)

    store.learned = True  # no links registered -> falls back to static
    results = {r.cell.cell_id: r for r in store.read("alpha")}
    assert results[a.cell_id].link_score == pytest.approx(0.05)  # static fallback


def test_stage155_read_api_signature_unchanged() -> None:
    # read() accepts the same keyword arguments as the Stage73 base class.
    base_params = set(inspect.signature(OrionMemoryStore.read).parameters.keys())
    weighted_params = set(inspect.signature(HelixWeightedOrionStore.read).parameters.keys())
    assert base_params == weighted_params
    # And calling with all of them works.
    store, _, _, _, _ = _retrieval_fixture()
    store.learned = True
    store.read("shared", memory_system="episodic", source="t", include_expired=False, limit=3)


def test_stage155_traversal_is_switchable() -> None:
    graph, weight_store = _graph_fixture()
    traverser = HelixWeightedHoneycombTraverser(weight_store=weight_store)
    traverser.learned = False
    static = list(traverser.traverse(graph, "c1", max_nodes=2).visited_node_ids)
    traverser.learned = True
    learned = list(traverser.traverse(graph, "c1", max_nodes=2).visited_node_ids)
    assert static != learned


def test_stage155_static_traversal_matches_stage146_ordering() -> None:
    graph, weight_store = _graph_fixture()
    traverser = HelixWeightedHoneycombTraverser(weight_store=weight_store)
    traverser.learned = False
    result = traverser.traverse(graph, "c1", max_nodes=2)
    # Static sorts by adjacency_type: conflict(n3) before consolidation(n2).
    assert result.visited_node_ids[1] == "n3"


def test_stage155_learned_traversal_visits_high_weight_first() -> None:
    graph, weight_store = _graph_fixture()
    traverser = HelixWeightedHoneycombTraverser(weight_store=weight_store)
    traverser.learned = True
    result = traverser.traverse(graph, "c1", max_nodes=2)
    # n1->n2 (1.5) before n1->n3 (0.0).
    assert result.visited_node_ids[1] == "n2"


def test_stage155_zero_weight_edge_reachable_with_full_budget() -> None:
    graph, weight_store = _graph_fixture()
    traverser = HelixWeightedHoneycombTraverser(weight_store=weight_store)
    traverser.learned = True
    full = traverser.traverse(graph, "c1")  # no budget limit
    # n3 is reachable only via the zero-weight edge n1->n3; it must still be visited.
    assert "n3" in full.visited_node_ids
    assert set(full.visited_node_ids) == {"n1", "n2", "n3", "n4"}


def test_stage155_traversal_deterministic() -> None:
    graph, weight_store = _graph_fixture()
    traverser = HelixWeightedHoneycombTraverser(weight_store=weight_store)
    traverser.learned = True
    first = list(traverser.traverse(graph, "c1", max_nodes=2).visited_node_ids)
    second = list(traverser.traverse(graph, "c1", max_nodes=2).visited_node_ids)
    assert first == second


def test_stage155_static_mode_identical_to_base_traverser() -> None:
    from experiments.civilization_transformer_qwen3.analysis.stage146_rosette_honeycomb_traversal import (
        RosetteHoneycombTraverser,
    )

    graph, weight_store = _graph_fixture()
    base = RosetteHoneycombTraverser().traverse(graph, "c1", max_nodes=2)
    weighted = HelixWeightedHoneycombTraverser(weight_store=weight_store)
    weighted.learned = False
    result = weighted.traverse(graph, "c1", max_nodes=2)
    assert result.visited_node_ids == base.visited_node_ids


def test_stage155_retrieval_mode_enum() -> None:
    assert RetrievalMode.STATIC.value == "static"
    assert RetrievalMode.LEARNED.value == "learned"
