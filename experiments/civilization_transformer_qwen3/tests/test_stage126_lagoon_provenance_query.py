from experiments.civilization_transformer_qwen3.analysis.stage126_lagoon_provenance_query import run_stage126_lagoon_provenance_query_smoke


def test_stage126_queries_verified_lagoon_schema_provenance(tmp_path) -> None:
    summary = run_stage126_lagoon_provenance_query_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert all(summary["stage_gates"].values())
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "memory_links.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
