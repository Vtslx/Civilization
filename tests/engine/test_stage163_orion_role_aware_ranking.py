from __future__ import annotations

from civilization.engine.stages.stage73_orion_memory_kernel import OrionMemoryStore
from civilization.engine.stages.stage163_orion_role_aware_ranking import OrionRoleAwareRanker


def test_stage163_personal_query_prefers_user_statement_to_assistant_advice() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    user_fact = "Conversation date: 2024-03-01\nuser: My hotel destination is Kyoto.\nassistant: Tokyo might also be a nice destination."
    assistant_advice = "Conversation date: 2024-03-02\nassistant: Your hotel destination is Kyoto."
    target = store.write_cell(memory_system="episodic", content=user_fact, summary=user_fact, source="conversation")
    distractor = store.write_cell(
        memory_system="episodic", content=assistant_advice, summary=assistant_advice, source="conversation"
    )

    results = store.read("Where is my hotel destination?", limit=2)

    assert results[0].cell.cell_id == target.cell_id
    assert results[0].score_breakdown["user_role_weight"] == 2.0
    assert results[0].score_breakdown["assistant_role_weight"] == 0.25
    assert results[1].cell.cell_id == distractor.cell_id
    assert store.trace_events[-1].details["ranking_version"] == OrionRoleAwareRanker.version


def test_stage163_assistant_recommendation_query_prefers_assistant_turn() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    user_turn = store.write_cell(
        memory_system="episodic",
        content="user: The hotel destination was Kyoto.",
        summary="user: The hotel destination was Kyoto.",
        source="conversation",
    )
    assistant_turn = store.write_cell(
        memory_system="episodic",
        content="assistant: I recommended Kyoto for the hotel destination.",
        summary="assistant: I recommended Kyoto for the hotel destination.",
        source="conversation",
    )

    results = store.read("What did the assistant recommend for the hotel destination?", limit=2)

    assert results[0].cell.cell_id == assistant_turn.cell_id
    assert results[0].score_breakdown["assistant_role_weight"] == 2.0
    assert results[0].score_breakdown["user_role_weight"] == 0.25
    assert user_turn.cell_id in {result.cell.cell_id for result in results}


def test_stage163_unstructured_memories_keep_neutral_role_weights() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    cell = store.write_cell(
        memory_system="episodic",
        content="Kyoto hotel destination preference",
        summary="Kyoto hotel destination preference",
        source="test",
    )

    result = store.read("Where is my hotel destination?")[0]

    assert result.cell.cell_id == cell.cell_id
    assert result.score_breakdown["user_role_weight"] == 1.0
    assert result.score_breakdown["assistant_role_weight"] == 1.0


def test_stage163_metadata_speaker_role_is_supported() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    user_fact = store.write_cell(
        memory_system="episodic",
        content="Kyoto is my preferred hotel destination",
        summary="Kyoto is my preferred hotel destination",
        source="test",
        metadata={"speaker": "user"},
    )
    assistant_advice = store.write_cell(
        memory_system="episodic",
        content="Kyoto could be a good hotel destination",
        summary="Kyoto could be a good hotel destination",
        source="test",
        metadata={"speaker": "assistant"},
    )

    results = store.read("What is my preferred hotel destination?", limit=2)

    assert results[0].cell.cell_id == user_fact.cell_id
    assert results[1].cell.cell_id == assistant_advice.cell_id
