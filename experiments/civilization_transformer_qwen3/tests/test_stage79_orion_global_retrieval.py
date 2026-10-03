from experiments.civilization_transformer_qwen3.analysis.stage73_orion_memory_kernel import OrionMemoryStore
from experiments.civilization_transformer_qwen3.analysis.stage79_orion_global_retrieval import OrionGlobalRetrievalRouter, run_stage79_orion_global_retrieval_smoke


def test_stage79_global_retrieval_is_opt_in() -> None:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    session.write_cell(memory_system="semantic", content="session evidence", summary="session", source="s")
    global_store.write_cell(memory_system="semantic", content="global evidence", summary="global", source="g")
    router = OrionGlobalRetrievalRouter()
    assert [item.tier for item in router.read(session, global_store, query="evidence", include_global=False)] == ["session"]
    assert {item.tier for item in router.read(session, global_store, query="evidence", include_global=True)} == {"session", "global"}


def test_stage79_smoke(tmp_path) -> None:
    assert run_stage79_orion_global_retrieval_smoke(output_dir=tmp_path)["passes_stage_gate"]
