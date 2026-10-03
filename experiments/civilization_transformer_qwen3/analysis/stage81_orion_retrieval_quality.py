from __future__ import annotations

import json
from pathlib import Path

from .stage73_orion_memory_kernel import OrionMemoryStore
from .stage79_orion_global_retrieval import OrionGlobalRetrievalRouter


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage81_orion_retrieval_quality")


def run_stage81_orion_retrieval_quality(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    session, global_store = OrionMemoryStore(), OrionMemoryStore()
    session.write_cell(memory_system="semantic", content="session alpha evidence", summary="alpha", source="session")
    global_store.write_cell(memory_system="semantic", content="global beta evidence", summary="beta", source="stage78")
    router = OrionGlobalRetrievalRouter()
    cases = [("alpha evidence", "session"), ("beta evidence", "global")]
    rows = []
    for query, expected in cases:
        local = router.read(session, global_store, query=query, include_global=False)
        combined = router.read(session, global_store, query=query, include_global=True)
        rows.append({"query": query, "expected_tier": expected, "session_only_hit": any(item.tier == expected for item in local), "global_opt_in_hit": any(item.tier == expected for item in combined), "combined_first_tier": combined[0].tier if combined else None})
    session_only = sum(row["session_only_hit"] for row in rows) / len(rows)
    global_opt_in = sum(row["global_opt_in_hit"] for row in rows) / len(rows)
    summary = {"stage": "stage81_orion_retrieval_quality", "rows": rows, "session_only_hit_rate": session_only, "global_opt_in_hit_rate": global_opt_in, "stage_gates": {"global_improves_cross_session_hit": global_opt_in > session_only, "session_query_remains_session_first": rows[0]["combined_first_tier"] == "session", "global_is_opt_in": not rows[1]["session_only_hit"] and rows[1]["global_opt_in_hit"]}}
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
