from civilization.engine.stages.stage73_orion_memory_kernel import OrionMemoryStore
from civilization.engine.stages.stage79_orion_global_retrieval import OrionGlobalRetrievalRouter, run_stage79_orion_global_retrieval_smoke


def test_stage79_global_retrieval_is_opt_in() -> None:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    session.write_cell(memory_system="semantic", content="session evidence", summary="session", source="s")
    global_store.write_cell(memory_system="semantic", content="global evidence", summary="global", source="g")
    router = OrionGlobalRetrievalRouter()
    assert [item.tier for item in router.read(session, global_store, query="evidence", include_global=False)] == ["session"]
    assert {item.tier for item in router.read(session, global_store, query="evidence", include_global=True)} == {"session", "global"}


def test_stage79_relevance_comparison_uses_shared_corpus() -> None:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    session.write_cell(
        memory_system="episodic",
        summary="Plan overview",
        content="The project plan discusses the budget and timeline. " * 20,
        source="session",
    )
    precise_global = global_store.write_cell(
        memory_system="semantic",
        summary="Project budget decision: approve",
        content="The approved project budget decision is 12,000.",
        source="global",
    )

    results = OrionGlobalRetrievalRouter().read(
        session,
        global_store,
        query="What is the project budget decision?",
        include_global=True,
        limit=2,
    )

    assert results[0].tier == "global"
    assert results[0].result.cell.cell_id == precise_global.cell_id
    assert results[0].result.score_breakdown["bm25f"] > results[1].result.score_breakdown["bm25f"]


def test_stage79_session_priority_breaks_equal_relevance_ties() -> None:
    session, global_store = OrionMemoryStore(now_fn=lambda: 100.0), OrionMemoryStore(now_fn=lambda: 100.0)
    session.write_cell(memory_system="semantic", content="shared evidence", summary="shared evidence", source="test")
    global_store.write_cell(memory_system="semantic", content="shared evidence", summary="shared evidence", source="test")

    results = OrionGlobalRetrievalRouter().read(
        session, global_store, query="shared evidence", include_global=True, limit=2
    )

    assert [item.tier for item in results] == ["session", "global"]


def test_stage79_latest_intent_can_rank_newer_global_fact_first() -> None:
    session, global_store = OrionMemoryStore(now_fn=lambda: 100.0), OrionMemoryStore(now_fn=lambda: 100.0)
    session.write_cell(
        memory_system="episodic", content="travel plan Kyoto", summary="travel plan", source="test", time_index=1.0
    )
    newer = global_store.write_cell(
        memory_system="episodic", content="travel plan Kyoto", summary="travel plan", source="test", time_index=2.0
    )

    results = OrionGlobalRetrievalRouter().read(
        session, global_store, query="What is the latest travel plan?", include_global=True, limit=2
    )

    assert results[0].tier == "global"
    assert results[0].result.cell.cell_id == newer.cell_id


def test_stage79_smoke(tmp_path) -> None:
    assert run_stage79_orion_global_retrieval_smoke(output_dir=tmp_path)["passes_stage_gate"]
