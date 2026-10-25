from __future__ import annotations

import pytest

from civilization.engine.stages.stage161_orion_relevance_ranking import OrionRelevanceRanker
from civilization.engine.stages.stage73_orion_memory_kernel import OrionMemoryStore


def test_stage161_rare_evidence_terms_outrank_long_generic_text() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    target = store.write_cell(
        memory_system="episodic",
        summary="Coffee creamer coupon redeemed at Target",
        content="I used a five dollar coupon on coffee creamer at Target.",
        source="conversation",
    )
    distractor = store.write_cell(
        memory_system="episodic",
        summary="Assistant shopping suggestions",
        content=("A coupon may be useful. " * 80) + "Coffee is popular for breakfast.",
        source="assistant",
        confidence=1.0,
        importance=1.0,
    )

    results = store.read("Where did I redeem the $5 coupon on coffee creamer?", limit=2)

    assert [result.cell.cell_id for result in results] == [target.cell_id, distractor.cell_id]
    assert results[0].score_breakdown["bm25f"] > results[1].score_breakdown["bm25f"]
    assert results[0].to_dict()["score_breakdown"]["bm25f"] == results[0].score_breakdown["bm25f"]


def test_stage161_summary_field_counters_long_content_and_priors() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    target = store.write_cell(
        memory_system="episodic",
        summary="Eight days camping in the United States",
        content="The trip lasted eight days.",
        source="user",
        confidence=0.2,
        importance=0.1,
    )
    noisy = store.write_cell(
        memory_system="episodic",
        summary="Travel notes",
        content=("Camping is a favorite travel activity throughout the United States. " * 50),
        source="assistant",
        confidence=1.0,
        importance=1.0,
    )

    result = store.read("How many days did I spend camping in the United States this year?", limit=2)

    assert result[0].cell.cell_id == target.cell_id
    assert result[0].score_breakdown["importance_prior"] <= 0.05
    assert result[0].score_breakdown["confidence_prior"] <= 0.05
    assert result[1].cell.cell_id == noisy.cell_id


def test_stage161_latest_intent_reverses_only_temporal_tie_break() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    older = store.write_cell(memory_system="episodic", summary="travel plan", content="visit Kyoto", source="test", time_index=1)
    newer = store.write_cell(memory_system="episodic", summary="travel plan", content="visit Kyoto", source="test", time_index=2)

    latest = store.read("What is my latest travel plan?", limit=2)
    general = store.read("What is my travel plan?", limit=2)

    assert [item.cell.cell_id for item in latest] == [newer.cell_id, older.cell_id]
    assert [item.cell.cell_id for item in general] == [older.cell_id, newer.cell_id]


def test_stage161_chinese_latest_intent_is_recognized() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    older = store.write_cell(memory_system="episodic", summary="计划", content="京都", source="test", time_index=1)
    newer = store.write_cell(memory_system="episodic", summary="计划", content="京都", source="test", time_index=2)

    results = store.read("我最近的计划是什么", limit=2)

    assert [item.cell.cell_id for item in results] == [newer.cell_id, older.cell_id]


def test_stage161_filters_and_empty_query_preserve_store_contract() -> None:
    store = OrionMemoryStore(now_fn=lambda: 100.0)
    semantic = store.write_cell(memory_system="semantic", summary="shared fact", content="shared fact", source="global")
    episodic = store.write_cell(memory_system="episodic", summary="shared event", content="shared event", source="session")

    assert [item.cell.cell_id for item in store.read("shared", memory_system="semantic")] == [semantic.cell_id]
    assert [item.cell.cell_id for item in store.read("", source="session")] == [episodic.cell_id]
    assert store.read("missing query") == []


def test_stage161_ranker_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="k1"):
        OrionRelevanceRanker(k1=0)
    with pytest.raises(ValueError, match="between"):
        OrionRelevanceRanker(b=1.1)
    with pytest.raises(ValueError, match="phrase_bonus"):
        OrionRelevanceRanker(phrase_bonus=-0.1)
